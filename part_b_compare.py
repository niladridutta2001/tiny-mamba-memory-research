"""Evaluate key-conditioned routing against baseline and tied-row control."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from data.dataset import StateTrackingDataset
from evaluate import EVALUATION_SPLITS
from h2_multiseed import DELAYS, DENSITIES, KINDS, make_design, predict, sequences_for
from h3_multiseed import retention_examples, write_csv
from hypothesis_pilot import load_model
from train import build_loader, evaluate


MODEL_PATHS = {
    "Baseline": Path("results/extended/seed42/u1500/baseline.pt"),
    "Key-conditioned": Path("results/part_b/seed42/key_routing/u1500/baseline.pt"),
    "Tied routing": Path("results/part_b/seed42/tied_routing/u1500/baseline.pt"),
}


def paired_bootstrap(values: np.ndarray, rng: np.random.Generator, draws: int) -> tuple[float, float, float]:
    observed = float(100 * values.mean())
    bootstrap = np.empty(draws, dtype=np.float64)
    chunk = 500
    for start in range(0, draws, chunk):
        stop = min(start + chunk, draws)
        indices = rng.integers(0, len(values), (stop - start, len(values)))
        bootstrap[start:stop] = 100 * values[indices].mean(axis=1)
    low, high = np.quantile(bootstrap, (0.025, 0.975))
    return observed, float(low), float(high)


@torch.no_grad()
def simple_predict(model: torch.nn.Module, sequences: list[list[int]], device: torch.device, batch_size: int) -> np.ndarray:
    outputs = []
    for start in range(0, len(sequences), batch_size):
        inputs = torch.tensor(sequences[start : start + batch_size], dtype=torch.long, device=device)
        outputs.append(model(inputs)[:, -1].argmax(-1).cpu().numpy())
    return np.concatenate(outputs).astype(np.int8)


def routing_diagnostics(model: torch.nn.Module) -> tuple[list[dict], list[dict], list[dict], dict]:
    table = model.routing_table().detach().cpu().numpy()
    centered = table - table.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(table, axis=1, keepdims=True)
    cosine = table @ table.T / np.maximum(norms @ norms.T, 1e-12)
    off_diagonal = cosine[~np.eye(len(table), dtype=bool)]
    distance = np.linalg.norm(table[:, None, :] - table[None, :, :], axis=-1) / np.maximum(norms, 1e-12)
    distance_off_diagonal = distance[~np.eye(len(table), dtype=bool)]
    channel_variance = table.var(axis=0)
    rows = [
        {"key": chr(ord("A") + key), **{f"channel_{channel}": float(table[key, channel]) for channel in range(64)}}
        for key in range(8)
    ]
    pairwise_rows = [
        {
            "key_i": chr(ord("A") + i),
            "key_j": chr(ord("A") + j),
            "normalized_l2": float(distance[i, j]),
            "l2": float(np.linalg.norm(table[i] - table[j])),
            "norm_i": float(np.linalg.norm(table[i])),
            "norm_j": float(np.linalg.norm(table[j])),
            "cosine_similarity": float(cosine[i, j]),
        }
        for i in range(8) for j in range(8) if i != j
    ]
    variance_rows = [
        {
            "channel": channel,
            "mean_across_keys": float(table[:, channel].mean()),
            "variance_across_keys": float(channel_variance[channel]),
            "std_across_keys": float(np.sqrt(channel_variance[channel])),
            "min_across_keys": float(table[:, channel].min()),
            "max_across_keys": float(table[:, channel].max()),
        }
        for channel in range(64)
    ]
    summary = {
        "gate_mean": float(table.mean()),
        "gate_min": float(table.min()),
        "gate_max": float(table.max()),
        "mean_channel_std_across_keys": float(table.std(axis=0).mean()),
        "max_channel_std_across_keys": float(table.std(axis=0).max()),
        "mean_pairwise_cosine_similarity": float(off_diagonal.mean()),
        "mean_pairwise_normalized_l2": float(distance_off_diagonal.mean()),
        "median_pairwise_normalized_l2": float(np.median(distance_off_diagonal)),
        "min_pairwise_normalized_l2": float(distance_off_diagonal.min()),
        "max_pairwise_normalized_l2": float(distance_off_diagonal.max()),
        "mean_channel_variance_across_keys": float(channel_variance.mean()),
        "median_channel_variance_across_keys": float(np.median(channel_variance)),
        "max_channel_variance_across_keys": float(channel_variance.max()),
        "mean_absolute_centered_gate": float(np.abs(centered).mean()),
    }
    return rows, pairwise_rows, variance_rows, summary


def make_plots(metrics: list[dict], contrasts: list[dict], routing_rows: list[dict], output_dir: Path) -> None:
    colors = {"Baseline": "#333333", "Key-conditioned": "#2ca02c", "Tied routing": "#9467bd"}
    markers = {"Baseline": "o", "Key-conditioned": "^", "Tied routing": "s"}
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 8), sharex=True, sharey=True)
    for axis, delay in zip(axes.flat, DELAYS):
        for model_name in MODEL_PATHS:
            rows = [
                row for row in metrics
                if row["test"] == "H2" and int(row["delay"]) == delay
                and row["kind"] == "set" and row["model"] == model_name
            ]
            rows.sort(key=lambda row: float(row["density"]))
            axis.plot(
                [100 * float(row["density"]) for row in rows],
                [100 * float(row["accuracy"]) for row in rows],
                marker=markers[model_name], color=colors[model_name], label=model_name,
            )
        axis.axhline(25, color="0.55", linestyle=":", linewidth=1)
        axis.set(title=f"Fixed delay = {delay}", ylim=(15, 100))
        axis.grid(axis="y", alpha=0.25)
    axes[0, 0].legend(frameon=False, fontsize=8)
    for axis in axes[1]:
        axis.set_xlabel("Unrelated SET density (%)")
    for axis in axes[:, 0]:
        axis.set_ylabel("Accuracy (%)")
    figure.suptitle("Part B: key-conditioned routing on the H2 interference test")
    figure.tight_layout()
    figure.savefig(output_dir / "part_b_h2_accuracy.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    x = np.arange(len(DELAYS))
    for offset, comparison, label, color, marker in (
        (-0.12, "proposed_vs_tied", "Raw accuracy difference", "#2ca02c", "o"),
        (0.12, "interference_did", "Reduction in SET interference", "#1f77b4", "s"),
    ):
        rows = [row for row in contrasts if row["comparison"] == comparison and float(row["density"]) == 1.0]
        rows.sort(key=lambda row: int(row["delay"]))
        effect = np.asarray([float(row["effect_pp"]) for row in rows])
        low = np.asarray([float(row["ci_low"]) for row in rows])
        high = np.asarray([float(row["ci_high"]) for row in rows])
        axis.errorbar(
            x + offset, effect, yerr=np.vstack((effect - low, high - effect)),
            fmt=marker, capsize=4, color=color, label=label,
        )
    axis.axhline(0, color="0.35", linewidth=1)
    axis.set(
        xticks=x, xticklabels=[str(delay) for delay in DELAYS],
        xlabel="Fixed SET-to-QUERY delay", ylabel="Key-conditioned minus tied (pp)",
        title="Key-selectivity effect at 100% unrelated-SET density",
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "part_b_h2_effects.png", dpi=180)
    plt.close(figure)

    table = np.asarray([[float(row[f"channel_{channel}"]) for channel in range(64)] for row in routing_rows])
    centered = table - table.mean(axis=0, keepdims=True)
    limit = max(float(np.abs(centered).max()), 1e-5)
    norms = np.linalg.norm(table, axis=1, keepdims=True)
    distance = np.linalg.norm(table[:, None, :] - table[None, :, :], axis=-1) / np.maximum(norms, 1e-12)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": (2.2, 1)})
    image = axes[0].imshow(centered, aspect="auto", cmap="coolwarm", vmin=-limit, vmax=limit)
    axes[0].set(
        yticks=range(8), yticklabels=list("ABCDEFGH"), xlabel="Recurrent channel",
        ylabel="Logical key", title="Learned key-specific gate deviation from channel mean",
    )
    figure.colorbar(image, ax=axes[0], label="Gate deviation")
    distance_image = axes[1].imshow(distance, cmap="viridis", vmin=0)
    axes[1].set(
        xticks=range(8), xticklabels=list("ABCDEFGH"), yticks=range(8), yticklabels=list("ABCDEFGH"),
        xlabel="Key j", ylabel="Key i", title=r"$\|r_i-r_j\|_2/(\|r_i\|_2+\epsilon)$",
    )
    figure.colorbar(distance_image, ax=axes[1], label="Normalized distance")
    figure.tight_layout()
    figure.savefig(output_dir / "routing_heatmap.png", dpi=180)
    plt.close(figure)


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    models = {name: load_model(path, device) for name, path in MODEL_PATHS.items()}
    rng = np.random.default_rng(args.bootstrap_seed)
    metrics, contrasts = [], []

    h2_data, h2_targets = {}, {}
    for delay in DELAYS:
        designs = make_design(delay, args.examples, args.evaluation_seed)
        for density in DENSITIES:
            for kind in KINDS:
                sequences, targets, _ = sequences_for(designs, delay, density, kind)
                h2_data[(delay, kind, density)] = sequences
                h2_targets[delay] = targets

    predictions = {}
    for model_name, model in models.items():
        for condition, sequences in h2_data.items():
            predictions[(model_name, *condition)] = predict(model, sequences, device, args.batch_size)
            delay, kind, density = condition
            correct = predictions[(model_name, *condition)] == h2_targets[delay]
            metrics.append(
                {
                    "test": "H2", "model": model_name, "delay": delay, "kind": kind,
                    "density": density, "accuracy": float(correct.mean()), "condition": "",
                }
            )

    for delay in DELAYS:
        target = h2_targets[delay]
        for density in DENSITIES:
            proposed = (predictions[("Key-conditioned", delay, "set", density)] == target).astype(float)
            tied = (predictions[("Tied routing", delay, "set", density)] == target).astype(float)
            effect, low, high = paired_bootstrap(proposed - tied, rng, args.bootstrap_draws)
            contrasts.append(
                {"delay": delay, "density": density, "comparison": "proposed_vs_tied", "effect_pp": effect, "ci_low": low, "ci_high": high}
            )
            proposed_noop = (predictions[("Key-conditioned", delay, "set", 0.0)] == target).astype(float)
            tied_noop = (predictions[("Tied routing", delay, "set", 0.0)] == target).astype(float)
            did = (proposed - proposed_noop) - (tied - tied_noop)
            effect, low, high = paired_bootstrap(did, rng, args.bootstrap_draws)
            contrasts.append(
                {"delay": delay, "density": density, "comparison": "interference_did", "effect_pp": effect, "ci_low": low, "ci_high": high}
            )

    for delay in DELAYS:
        sequences, target, _ = retention_examples(delay, args.examples, args.evaluation_seed)
        for model_name, model in models.items():
            prediction = simple_predict(model, sequences, device, args.batch_size)
            metrics.append(
                {
                    "test": "retention", "model": model_name, "delay": delay, "kind": "",
                    "density": "", "accuracy": float((prediction == target).mean()), "condition": "isolated",
                }
            )

    fixed_split_rows = []
    for model_name, model in models.items():
        for split, (condition, length) in EVALUATION_SPLITS.items():
            path = args.data_dir / f"{split}.jsonl.gz"
            result = evaluate(model, build_loader(path, args.fixed_batch_size, False, 42), device)
            fixed_split_rows.append(
                {
                    "model": model_name, "split": split, "condition": condition, "length": length,
                    "query_accuracy": result["query_accuracy"], "query_count": result["query_count"],
                }
            )

    routing_rows, routing_pairs, routing_variance, routing_summary = routing_diagnostics(models["Key-conditioned"])
    tied_rows, tied_pairs, tied_variance, tied_summary = routing_diagnostics(models["Tied routing"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "metrics.csv", metrics)
    write_csv(args.output_dir / "contrasts.csv", contrasts)
    write_csv(args.output_dir / "fixed_splits.csv", fixed_split_rows)
    write_csv(args.output_dir / "key_routing_table.csv", routing_rows)
    write_csv(args.output_dir / "tied_routing_table.csv", tied_rows)
    write_csv(args.output_dir / "key_routing_pairwise_distances.csv", routing_pairs)
    write_csv(args.output_dir / "tied_routing_pairwise_distances.csv", tied_pairs)
    write_csv(args.output_dir / "key_routing_channel_variance.csv", routing_variance)
    write_csv(args.output_dir / "tied_routing_channel_variance.csv", tied_variance)
    make_plots(metrics, contrasts, routing_rows, args.output_dir)
    summary = {
        "checkpoints": {name: str(path) for name, path in MODEL_PATHS.items()},
        "examples_per_h2_condition": args.examples,
        "bootstrap_draws": args.bootstrap_draws,
        "routing_summary": routing_summary,
        "tied_routing_summary": tied_summary,
        "metrics": metrics,
        "contrasts": contrasts,
        "fixed_splits": fixed_split_rows,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/part_b/seed42/comparison"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/generated"))
    parser.add_argument("--examples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--fixed-batch-size", type=int, default=16)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--evaluation-seed", type=int, default=20261023)
    parser.add_argument("--bootstrap-seed", type=int, default=424242)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({key: result[key] for key in ("routing_summary", "tied_routing_summary")}, indent=2))
