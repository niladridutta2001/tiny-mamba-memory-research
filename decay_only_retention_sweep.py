"""Frozen-model decay-only intervention over a 511-step retention gap.

The first d_conv-1 NOOPs after SET run normally so the SET's convolutional
response can finish. Only the decay exponent is then multiplied by alpha;
the recurrent input-write term, convolution, weights, and readout are unchanged.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from generate_dataset import KEYS, encode_event, random_event
from hypothesis_pilot import load_model
from mechanistic_analysis import log_retention


ALPHAS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25)
PREFIX_LENGTH = 32
NOOP_COUNT = 510
SET_INDEX = PREFIX_LENGTH
NOOP_START = SET_INDEX + 1
QUERY_INDEX = NOOP_START + NOOP_COUNT


def make_examples(contexts_per_pair: int, seed: int) -> tuple[list[list[int]], list[int], list[int], list[int]]:
    rng = random.Random(seed)
    noop = encode_event(["NOOP", None, None])
    sequences: list[list[int]] = []
    labels: list[int] = []
    keys: list[int] = []
    clusters: list[int] = []
    for key_index, key in enumerate(KEYS):
        for value in range(1, 5):
            cluster = key_index * 4 + value - 1
            for _ in range(contexts_per_pair):
                prefix = [random_event(rng, exclude_key=key) for _ in range(PREFIX_LENGTH)]
                sequence = [encode_event(event) for event in prefix]
                sequence.append(encode_event(["SET", key, value]))
                sequence.extend([noop] * NOOP_COUNT)
                sequence.append(encode_event(["QUERY", key, None]))
                sequences.append(sequence)
                labels.append(value)
                keys.append(key_index)
                clusters.append(cluster)
    return sequences, labels, keys, clusters


def intervention_scale(model: torch.nn.Module, input_ids: torch.Tensor, alpha: float) -> torch.Tensor:
    natural = model.event_retention_scale(input_ids)
    if natural is None:
        natural = torch.ones((*input_ids.shape, model.backbone.layers[0].mixer.args.d_inner), device=input_ids.device)
    scale = natural.clone()
    washout_start = NOOP_START + model.d_conv - 1
    scale[:, washout_start:QUERY_INDEX] *= alpha
    return scale


@torch.no_grad()
def extract(
    model: torch.nn.Module,
    sequences: list[list[int]],
    labels: list[int],
    keys: list[int],
    alpha: float,
    device: torch.device,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    features, query_hidden, logits, cumulative = [], [], [], []
    for start in range(0, len(sequences), batch_size):
        stop = start + batch_size
        inputs = torch.tensor(sequences[start:stop], dtype=torch.long, device=device)
        scale = intervention_scale(model, inputs, alpha)
        hidden = model.hidden_states(inputs, decay_scale=scale)
        query_hidden.append(hidden[:, QUERY_INDEX].cpu())
        logits.append(model.head(hidden[:, QUERY_INDEX]).cpu())
        key_ids = torch.tensor(keys[start:stop], device=device)
        features.append(torch.cat([hidden[:, QUERY_INDEX - 1], F.one_hot(key_ids, num_classes=8).float()], dim=-1).cpu())
        raw_log_decay = log_retention(model, inputs)
        effective_log_decay = raw_log_decay * scale.unsqueeze(-1)
        cumulative.append(effective_log_decay[:, NOOP_START:QUERY_INDEX].sum(dim=1).div(math.log(10)).cpu())
    return {
        "features": torch.cat(features),
        "query_hidden": torch.cat(query_hidden),
        "logits": torch.cat(logits),
        "labels": torch.tensor(labels, dtype=torch.long),
        "cumulative_log10_retention": torch.cat(cumulative),
    }


def fit_probe(
    train: dict[str, torch.Tensor], test: dict[str, torch.Tensor], device: torch.device
) -> tuple[float, np.ndarray]:
    train_x = train["features"].to(device).clone()
    test_x = test["features"].to(device).clone()
    train_y = (train["labels"] - 1).to(device)
    test_y = (test["labels"] - 1).to(device)
    mean = train_x[:, :32].mean(0)
    scale = train_x[:, :32].std(0).clamp_min(1e-5)
    train_x[:, :32] = (train_x[:, :32] - mean) / scale
    test_x[:, :32] = (test_x[:, :32] - mean) / scale
    torch.manual_seed(424242)
    probe = torch.nn.Linear(40, 4).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=0.03, weight_decay=1e-4)
    for _ in range(300):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(probe(train_x), train_y)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        correct = probe(test_x).argmax(-1).eq(test_y)
        return float(correct.float().mean().item()), correct.cpu().numpy().astype(np.int8)


def task_metrics(extracted: dict[str, torch.Tensor]) -> tuple[dict[str, float], np.ndarray]:
    logits = extracted["logits"]
    labels = extracted["labels"]
    probabilities = logits.softmax(-1)
    predictions = logits.argmax(-1)
    correct = predictions.eq(labels)
    target_probability = probabilities.gather(1, labels[:, None]).squeeze(1)
    wrong_logits = logits.clone()
    wrong_logits[torch.arange(len(labels)), labels] = -torch.inf
    margins = logits.gather(1, labels[:, None]).squeeze(1) - wrong_logits.max(-1).values
    cumulative = extracted["cumulative_log10_retention"].flatten()
    return {
        "task_accuracy": float(correct.float().mean().item()),
        "mean_nll": float(F.cross_entropy(logits, labels).item()),
        "mean_correct_class_probability": float(target_probability.mean().item()),
        "mean_logit_margin": float(margins.mean().item()),
        "median_cumulative_log10_retention": float(cumulative.median().item()),
        "p10_cumulative_log10_retention": float(torch.quantile(cumulative, 0.1).item()),
        "p90_cumulative_log10_retention": float(torch.quantile(cumulative, 0.9).item()),
    }, correct.numpy().astype(np.int8)


def cluster_bootstrap_difference(
    treatment: np.ndarray,
    baseline: np.ndarray,
    clusters: np.ndarray,
    seed: int,
    draws: int = 10000,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    unique = np.unique(clusters)
    cluster_effects = np.asarray([(treatment[clusters == item] - baseline[clusters == item]).mean() for item in unique])
    samples = rng.integers(0, len(unique), size=(draws, len(unique)))
    bootstrapped = cluster_effects[samples].mean(axis=1)
    return (
        float(cluster_effects.mean()),
        float(np.quantile(bootstrapped, 0.025)),
        float(np.quantile(bootstrapped, 0.975)),
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict], output: Path) -> None:
    alphas = [row["alpha"] for row in rows]
    figure, axes = plt.subplots(1, 3, figsize=(14.2, 4.6))
    axes[0].plot(alphas, [100 * row["task_accuracy"] for row in rows], "o-", label="Task head")
    axes[0].plot(alphas, [100 * row["probe_accuracy"] for row in rows], "s-", label="Linear probe")
    axes[0].axhline(25, color="#6b7280", linestyle="--", linewidth=1.3, label="Chance")
    axes[0].set(xlabel=r"Decay multiplier $\alpha$", ylabel="Accuracy (%)", title="Value recovery after 511 steps", ylim=(0, 105))
    axes[0].legend(frameon=False)

    axes[1].plot(alphas, [row["mean_correct_class_probability"] for row in rows], "o-", label="Correct-class probability")
    axes[1].plot(alphas, [row["mean_logit_margin"] for row in rows], "s-", label="Target logit margin")
    axes[1].axhline(0, color="#6b7280", linewidth=1)
    axes[1].set(xlabel=r"Decay multiplier $\alpha$", title="Frozen readout evidence")
    axes[1].legend(frameon=False)

    axes[2].plot(alphas, [row["median_cumulative_log10_retention"] for row in rows], "o-", label="Median")
    axes[2].fill_between(
        alphas,
        [row["p10_cumulative_log10_retention"] for row in rows],
        [row["p90_cumulative_log10_retention"] for row in rows],
        alpha=0.2,
        label="10th-90th percentile",
    )
    axes[2].set(xlabel=r"Decay multiplier $\alpha$", ylabel=r"Cumulative $\log_{10}$ retention", title="Effective recurrent attenuation")
    axes[2].set_yscale("symlog", linthresh=1.0)
    axes[2].legend(frameon=False)
    for axis in axes:
        axis.axvline(1.0, color="#dc2626", linestyle=":", linewidth=1.5)
        axis.set_xticks(alphas)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Decay-only intervention after convolutional washout (frozen seed-42 model)")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(output, dpi=180)
    plt.close(figure)


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_model(args.checkpoint, device)
    train_examples = make_examples(args.train_contexts_per_pair, args.train_seed)
    test_examples = make_examples(args.test_contexts_per_pair, args.test_seed)
    test_clusters = np.asarray(test_examples[3], dtype=np.int16)
    rows: list[dict] = []
    predictions: dict[float, np.ndarray] = {}
    probe_predictions: dict[float, np.ndarray] = {}
    for alpha in ALPHAS:
        train = extract(model, *train_examples[:3], alpha, device, args.batch_size)
        test = extract(model, *test_examples[:3], alpha, device, args.batch_size)
        metrics, correct = task_metrics(test)
        predictions[alpha] = correct
        probe_accuracy, probe_correct = fit_probe(train, test, device)
        probe_predictions[alpha] = probe_correct
        rows.append({
            "alpha": alpha,
            "normal_washout_noops": model.d_conv - 1,
            "intervened_noops": NOOP_COUNT - (model.d_conv - 1),
            "n_task_test": len(test_examples[0]),
            "probe_accuracy": probe_accuracy,
            **metrics,
        })

    contrasts = []
    for index, alpha in enumerate(ALPHAS):
        effect, low, high = cluster_bootstrap_difference(predictions[alpha], predictions[1.0], test_clusters, args.bootstrap_seed + index)
        probe_effect, probe_low, probe_high = cluster_bootstrap_difference(
            probe_predictions[alpha], probe_predictions[1.0], test_clusters, args.bootstrap_seed + 100 + index
        )
        contrasts.append({
            "alpha": alpha,
            "comparison": "alpha_vs_trained_alpha_1",
            "task_accuracy_effect_pp": 100 * effect,
            "ci_low_pp": 100 * low,
            "ci_high_pp": 100 * high,
            "probe_accuracy_effect_pp": 100 * probe_effect,
            "probe_ci_low_pp": 100 * probe_low,
            "probe_ci_high_pp": 100 * probe_high,
            "cluster_count": 32,
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "metrics.csv", rows)
    write_csv(args.output_dir / "contrasts.csv", contrasts)
    np.savez_compressed(
        args.output_dir / "paired_predictions.npz",
        labels=np.asarray(test_examples[1], dtype=np.int8),
        keys=np.asarray(test_examples[2], dtype=np.int8),
        clusters=test_clusters,
        **{f"alpha_{str(alpha).replace('.', '_')}": values for alpha, values in predictions.items()},
        **{f"probe_alpha_{str(alpha).replace('.', '_')}": values for alpha, values in probe_predictions.items()},
    )
    summary = {
        "scope": "Frozen seed-42 independently gated width-4 model; diagnostic OOD intervention",
        "checkpoint": str(args.checkpoint),
        "sequence": {
            "prefix_events": PREFIX_LENGTH,
            "set_index": SET_INDEX,
            "noop_count": NOOP_COUNT,
            "query_index": QUERY_INDEX,
            "set_to_query_delay": QUERY_INDEX - SET_INDEX,
            "normal_convolution_washout_noops": model.d_conv - 1,
            "decay_intervened_noops": NOOP_COUNT - (model.d_conv - 1),
        },
        "intervention": "Multiply only the recurrent decay exponent during post-washout NOOPs; preserve delta*B*u writes, convolution, weights, and readout.",
        "metrics": rows,
        "contrasts": contrasts,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot(rows, args.output_dir / "decay_only_retention_sweep.png")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("results/selective_retention/seed42/event_decay/u1500/baseline.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/decay_only_retention/seed42"))
    parser.add_argument("--train-contexts-per-pair", type=int, default=12)
    parser.add_argument("--test-contexts-per-pair", type=int, default=8)
    parser.add_argument("--train-seed", type=int, default=41042)
    parser.add_argument("--test-seed", type=int, default=42042)
    parser.add_argument("--bootstrap-seed", type=int, default=43042)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({"metrics": result["metrics"], "contrasts": result["contrasts"]}, indent=2))
