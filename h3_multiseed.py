"""High-precision, multi-seed test of the H3 retention-overwrite trade-off.

The evaluation uses paired examples for every alpha, randomized contexts that
exclude the tracked key, and a cluster bootstrap over model seeds and examples.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from generate_dataset import KEYS, encode_event
from hypothesis_pilot import load_model


ALPHAS = (0.25, 0.5, 1.0, 2.0)
MODES = ("global", "noop_only", "set_only")
RETENTION_DELAYS = (32, 64, 128)
OVERWRITE_COUNTS = (1, 4, 8)
OVERWRITE_DELAYS = (32, 64)
PRACTICAL_MARGIN_PP = 2.0
NOOP = encode_event(["NOOP", None, None])


def random_context(rng: random.Random, tracked_key: str, length: int = 32) -> list[int]:
    """Create a realistic prefix without touching the key being measured."""
    other_keys = [key for key in KEYS if key != tracked_key]
    context = []
    for _ in range(length):
        draw = rng.random()
        if draw < 0.55:
            event = ["NOOP", None, None]
        elif draw < 0.85:
            event = ["SET", rng.choice(other_keys), rng.randint(1, 4)]
        else:
            event = ["QUERY", rng.choice(other_keys), None]
        context.append(encode_event(event))
    return context


def balanced_assignments(n: int, rng: random.Random) -> list[tuple[str, int]]:
    cells = [(key, value) for key in KEYS for value in range(1, 5)]
    assignments = (cells * ((n + len(cells) - 1) // len(cells)))[:n]
    rng.shuffle(assignments)
    return assignments


def retention_examples(delay: int, n: int, seed: int) -> tuple[list[list[int]], np.ndarray, np.ndarray]:
    rng = random.Random(seed + 1009 * delay)
    sequences, targets = [], []
    for key, value in balanced_assignments(n, rng):
        sequence = random_context(rng, key)
        sequence.append(encode_event(["SET", key, value]))
        sequence.extend([NOOP] * delay)
        sequence.append(encode_event(["QUERY", key, None]))
        sequences.append(sequence)
        targets.append(value)
    target_array = np.asarray(targets, dtype=np.int8)
    return sequences, target_array, np.full(n, -1, dtype=np.int8)


def overwrite_examples(
    overwrite_count: int, final_delay: int, n: int, seed: int, spacing: int = 4
) -> tuple[list[list[int]], np.ndarray, np.ndarray]:
    """Generate old-value -> replacement chains followed by a delayed query."""
    rng = random.Random(seed + 1009 * overwrite_count + 9173 * final_delay)
    sequences, targets, obsolete = [], [], []
    for key, final_value in balanced_assignments(n, rng):
        # Work backwards so the final assignment is balanced and every update
        # differs from the value it replaces.
        values = [final_value]
        for _ in range(overwrite_count):
            choices = [value for value in range(1, 5) if value != values[-1]]
            values.append(rng.choice(choices))
        values.reverse()

        sequence = random_context(rng, key)
        for index, value in enumerate(values):
            sequence.append(encode_event(["SET", key, value]))
            if index < len(values) - 1:
                sequence.extend([NOOP] * spacing)
        sequence.extend([NOOP] * final_delay)
        sequence.append(encode_event(["QUERY", key, None]))
        sequences.append(sequence)
        targets.append(values[-1])
        obsolete.append(values[-2])
    return sequences, np.asarray(targets, dtype=np.int8), np.asarray(obsolete, dtype=np.int8)


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    sequences: list[list[int]],
    alpha: float,
    mode: str,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    outputs = []
    for start in range(0, len(sequences), batch_size):
        inputs = torch.tensor(sequences[start : start + batch_size], dtype=torch.long, device=device)
        outputs.append(model.scaled_logits(inputs, alpha, mode)[:, -1].argmax(-1).cpu().numpy())
    return np.concatenate(outputs).astype(np.int8)


def cluster_bootstrap_difference(
    values: np.ndarray, baseline: np.ndarray, rng: np.random.Generator, draws: int
) -> tuple[float, float, float]:
    """Paired difference with resampling over model seeds and examples."""
    seed_count, example_count = values.shape
    observed = float(100 * (values - baseline).mean())
    differences = values - baseline
    # Vectorize the inner example bootstrap in bounded chunks. Each column of
    # seed_boot is one independent within-model resample; whole model seeds are
    # then resampled on top of it.
    seed_boot = np.empty((seed_count, draws), dtype=np.float64)
    chunk_size = 500
    for seed_index in range(seed_count):
        for start in range(0, draws, chunk_size):
            stop = min(start + chunk_size, draws)
            example_indices = rng.integers(0, example_count, (stop - start, example_count))
            seed_boot[seed_index, start:stop] = differences[seed_index, example_indices].mean(axis=1)
    seed_indices = rng.integers(0, seed_count, (draws, seed_count))
    draw_indices = np.arange(draws)[:, None]
    boot = 100 * seed_boot[seed_indices, draw_indices].mean(axis=1)
    low, high = np.quantile(boot, (0.025, 0.975))
    return observed, float(low), float(high)


def summarize(
    outcomes: dict[tuple, np.ndarray], draws: int, bootstrap_seed: int
) -> tuple[list[dict], list[dict]]:
    metrics, contrasts = [], []
    rng = np.random.default_rng(bootstrap_seed)
    conditions = sorted({key[:-2] for key in outcomes if len(key) == 5})
    for condition in conditions:
        for mode in MODES:
            baseline_predictions = outcomes[(*condition, mode, 1.0)]
            targets = outcomes[(*condition, "targets")]
            obsolete = outcomes[(*condition, "obsolete")]
            baseline_correct = (baseline_predictions == targets).astype(np.float64)
            baseline_stale = (baseline_predictions == obsolete).astype(np.float64) if np.all(obsolete >= 0) else None
            for alpha in ALPHAS:
                predictions = outcomes[(*condition, mode, alpha)]
                correct = (predictions == targets).astype(np.float64)
                row = {
                    "test": condition[0],
                    "condition_a": condition[1],
                    "condition_b": condition[2],
                    "mode": mode,
                    "alpha": alpha,
                    "accuracy": float(correct.mean()),
                    "n_models": correct.shape[0],
                    "n_examples_per_model": correct.shape[1],
                }
                if baseline_stale is not None:
                    stale = (predictions == obsolete).astype(np.float64)
                    row["obsolete_error_rate"] = float(stale.mean())
                else:
                    stale = None
                    row["obsolete_error_rate"] = ""
                metrics.append(row)

                if alpha == 1.0:
                    continue
                effect, low, high = cluster_bootstrap_difference(correct, baseline_correct, rng, draws)
                contrast = {
                    "test": condition[0],
                    "condition_a": condition[1],
                    "condition_b": condition[2],
                    "mode": mode,
                    "alpha_vs_1": alpha,
                    "accuracy_effect_pp": effect,
                    "accuracy_ci_low": low,
                    "accuracy_ci_high": high,
                }
                if stale is not None and baseline_stale is not None:
                    stale_effect, stale_low, stale_high = cluster_bootstrap_difference(stale, baseline_stale, rng, draws)
                    contrast.update(
                        obsolete_effect_pp=stale_effect,
                        obsolete_ci_low=stale_low,
                        obsolete_ci_high=stale_high,
                    )
                else:
                    contrast.update(obsolete_effect_pp="", obsolete_ci_low="", obsolete_ci_high="")
                contrasts.append(contrast)
    return metrics, contrasts


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def make_plot(metrics: list[dict], contrasts: list[dict], output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    colors = {0.25: "#9467bd", 0.5: "#1f77b4", 1.0: "#333333", 2.0: "#ff7f0e"}
    for alpha in ALPHAS:
        rows = [
            row for row in metrics
            if row["test"] == "retention" and row["mode"] == "noop_only" and float(row["alpha"]) == alpha
        ]
        rows.sort(key=lambda row: int(row["condition_a"]))
        axes[0].plot(
            [int(row["condition_a"]) for row in rows],
            [100 * float(row["accuracy"]) for row in rows],
            "o-", color=colors[alpha], label=f"alpha={alpha:g}",
        )
    axes[0].axhline(25, color="0.55", linestyle=":", linewidth=1, label="chance")
    axes[0].set(xlabel="NOOP delay", ylabel="Accuracy (%)", title="Passive retention (NOOP-only scaling)", ylim=(20, 100))
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)

    x = np.arange(len(OVERWRITE_COUNTS))
    for offset, delay, color, marker in ((-0.08, 32, "#d62728", "s"), (0.08, 64, "#1f77b4", "o")):
        selected = [
            row for row in contrasts
            if row["test"] == "overwrite" and row["mode"] == "noop_only"
            and float(row["alpha_vs_1"]) == 0.5 and int(row["condition_b"]) == delay
        ]
        selected.sort(key=lambda row: int(row["condition_a"]))
        effects = np.asarray([float(row["accuracy_effect_pp"]) for row in selected])
        low = np.asarray([float(row["accuracy_ci_low"]) for row in selected])
        high = np.asarray([float(row["accuracy_ci_high"]) for row in selected])
        axes[1].errorbar(
            x + offset, effects, yerr=np.vstack((effects - low, high - effects)),
            fmt=marker, capsize=4, color=color, label=f"final delay={delay}",
        )
    axes[1].axhline(0, color="0.35", linewidth=1)
    axes[1].axhline(-PRACTICAL_MARGIN_PP, color="#d62728", linestyle="--", linewidth=1, label="-2 pp penalty margin")
    axes[1].set(
        xticks=x,
        xticklabels=[str(count) for count in OVERWRITE_COUNTS],
        xlabel="Number of overwrites",
        ylabel="Accuracy change vs alpha=1 (pp)",
        title="Overwrite cost of alpha=0.5",
    )
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoints = sorted(args.checkpoint_root.glob("seed*/u1500/baseline.pt"))
    if len(checkpoints) < 3:
        raise RuntimeError(f"expected at least three checkpoints, found {len(checkpoints)}")

    datasets: dict[tuple, tuple[list[list[int]], np.ndarray, np.ndarray]] = {}
    for delay in RETENTION_DELAYS:
        datasets[("retention", delay, 0)] = retention_examples(delay, args.examples, args.evaluation_seed)
    for count in OVERWRITE_COUNTS:
        for delay in OVERWRITE_DELAYS:
            datasets[("overwrite", count, delay)] = overwrite_examples(
                count, delay, args.examples, args.evaluation_seed
            )

    outcomes: dict[tuple, np.ndarray] = {}
    for condition, (_, targets, obsolete) in datasets.items():
        outcomes[(*condition, "targets")] = np.broadcast_to(targets, (len(checkpoints), len(targets)))
        outcomes[(*condition, "obsolete")] = np.broadcast_to(obsolete, (len(checkpoints), len(obsolete)))

    for model_index, checkpoint in enumerate(checkpoints):
        model = load_model(checkpoint, device)
        for condition, (sequences, _, _) in datasets.items():
            baseline = predict(model, sequences, 1.0, "global", device, args.batch_size)
            for mode in MODES:
                for alpha in ALPHAS:
                    key = (*condition, mode, alpha)
                    if key not in outcomes:
                        outcomes[key] = np.empty((len(checkpoints), args.examples), dtype=np.int8)
                    outcomes[key][model_index] = (
                        baseline if alpha == 1.0
                        else predict(model, sequences, alpha, mode, device, args.batch_size)
                    )
        del model

    metrics, contrasts = summarize(outcomes, args.bootstrap_draws, args.bootstrap_seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "metrics.csv", metrics)
    write_csv(args.output_dir / "contrasts.csv", contrasts)
    np.savez_compressed(args.output_dir / "paired_predictions.npz", **{str(key): value for key, value in outcomes.items()})
    make_plot(metrics, contrasts, args.output_dir / "h3_multiseed.png")

    summary = {
        "checkpoints": [str(path) for path in checkpoints],
        "n_models": len(checkpoints),
        "examples_per_condition_per_model": args.examples,
        "bootstrap_draws": args.bootstrap_draws,
        "practical_penalty_margin_pp": PRACTICAL_MARGIN_PP,
        "metrics": metrics,
        "contrasts": contrasts,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, default=Path("results/extended"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/h3_multiseed"))
    parser.add_argument("--examples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--evaluation-seed", type=int, default=20261023)
    parser.add_argument("--bootstrap-seed", type=int, default=314159)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({key: result[key] for key in result if key not in {"metrics", "contrasts"}}, indent=2))
