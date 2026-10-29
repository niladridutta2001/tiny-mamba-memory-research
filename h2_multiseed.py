"""Definitive multi-seed test of cross-key SET interference (H2)."""

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
from h3_multiseed import cluster_bootstrap_difference, random_context, write_csv
from hypothesis_pilot import load_model


DELAYS = (10, 32, 64, 128)
DENSITIES = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
KINDS = ("set", "query")
PRACTICAL_MARGIN_PP = 2.0
NOOP = encode_event(["NOOP", None, None])


def balanced_assignments(n: int, rng: random.Random) -> list[tuple[str, int]]:
    cells = [(key, value) for key in KEYS for value in range(1, 5)]
    assignments = (cells * ((n + len(cells) - 1) // len(cells)))[:n]
    rng.shuffle(assignments)
    return assignments


def make_design(delay: int, n: int, seed: int) -> list[dict]:
    """Create paired base examples with a nested order of distractor positions."""
    rng = random.Random(seed + 1009 * delay)
    designs = []
    for tracked_key, target in balanced_assignments(n, rng):
        positions = list(range(delay))
        rng.shuffle(positions)
        other_keys = [key for key in KEYS if key != tracked_key]
        distractors = [(rng.choice(other_keys), rng.randint(1, 4)) for _ in range(delay)]
        designs.append(
            {
                "tracked_key": tracked_key,
                "target": target,
                "context": random_context(rng, tracked_key),
                "position_order": positions,
                "distractors": distractors,
            }
        )
    return designs


def sequences_for(
    designs: list[dict], delay: int, density: float, kind: str
) -> tuple[list[list[int]], np.ndarray, np.ndarray]:
    distractor_count = round(delay * density)
    sequences, targets, latest_values = [], [], []
    for design in designs:
        selected = set(design["position_order"][:distractor_count])
        middle = []
        latest_value = -1
        for position in range(delay):
            if position not in selected:
                middle.append(NOOP)
                continue
            key, value = design["distractors"][position]
            if kind == "set":
                middle.append(encode_event(["SET", key, value]))
                latest_value = value
            elif kind == "query":
                middle.append(encode_event(["QUERY", key, None]))
            else:
                raise ValueError(f"unknown distractor kind: {kind}")
        sequence = [
            *design["context"],
            encode_event(["SET", design["tracked_key"], design["target"]]),
            *middle,
            encode_event(["QUERY", design["tracked_key"], None]),
        ]
        sequences.append(sequence)
        targets.append(design["target"])
        latest_values.append(latest_value)
    return sequences, np.asarray(targets, dtype=np.int8), np.asarray(latest_values, dtype=np.int8)


@torch.no_grad()
def predict(model: torch.nn.Module, sequences: list[list[int]], device: torch.device, batch_size: int) -> np.ndarray:
    outputs = []
    for start in range(0, len(sequences), batch_size):
        inputs = torch.tensor(sequences[start : start + batch_size], dtype=torch.long, device=device)
        outputs.append(model(inputs)[:, -1].argmax(-1).cpu().numpy())
    return np.concatenate(outputs).astype(np.int8)


def summarize(
    predictions: dict[tuple[int, str, float], np.ndarray],
    targets: dict[int, np.ndarray],
    latest_values: dict[tuple[int, float], np.ndarray],
    draws: int,
    bootstrap_seed: int,
) -> tuple[list[dict], list[dict]]:
    metrics, contrasts = [], []
    rng = np.random.default_rng(bootstrap_seed)
    for delay in DELAYS:
        target = targets[delay][None, :]
        noop_prediction = predictions[(delay, "set", 0.0)]
        noop_correct = (noop_prediction == target).astype(np.float64)
        for density in DENSITIES:
            for kind in KINDS:
                prediction = predictions[(delay, kind, density)]
                correct = (prediction == target).astype(np.float64)
                capture_rate: float | str = ""
                capture_n = 0
                if kind == "set" and density > 0:
                    latest = latest_values[(delay, density)][None, :]
                    eligible = latest != target
                    capture = (prediction == latest) & (prediction != target) & eligible
                    capture_n = int(eligible.sum() * prediction.shape[0])
                    capture_rate = float(capture.sum() / capture_n)
                metrics.append(
                    {
                        "delay": delay,
                        "density": density,
                        "kind": kind,
                        "accuracy": float(correct.mean()),
                        "distractor_capture_rate": capture_rate,
                        "capture_eligible_n": capture_n,
                        "n_models": prediction.shape[0],
                        "n_examples_per_model": prediction.shape[1],
                    }
                )

            if density == 0:
                continue
            set_correct = (predictions[(delay, "set", density)] == target).astype(np.float64)
            query_correct = (predictions[(delay, "query", density)] == target).astype(np.float64)
            for comparison, values, baseline in (
                ("set_vs_noop", set_correct, noop_correct),
                ("query_vs_noop", query_correct, noop_correct),
                ("set_vs_query", set_correct, query_correct),
            ):
                effect, low, high = cluster_bootstrap_difference(values, baseline, rng, draws)
                contrasts.append(
                    {
                        "delay": delay,
                        "density": density,
                        "comparison": comparison,
                        "accuracy_effect_pp": effect,
                        "ci_low": low,
                        "ci_high": high,
                    }
                )
    return metrics, contrasts


def make_plots(metrics: list[dict], contrasts: list[dict], output_dir: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 8), sharex=True, sharey=True)
    for axis, delay in zip(axes.flat, DELAYS):
        for kind, marker, color in (("set", "o", "#d62728"), ("query", "s", "#1f77b4")):
            rows = [row for row in metrics if int(row["delay"]) == delay and row["kind"] == kind]
            rows.sort(key=lambda row: float(row["density"]))
            axis.plot(
                [100 * float(row["density"]) for row in rows],
                [100 * float(row["accuracy"]) for row in rows],
                marker=marker, color=color, label=f"unrelated {kind.upper()}",
            )
        axis.axhline(25, color="0.55", linestyle=":", linewidth=1)
        axis.set(title=f"Fixed delay = {delay}", ylim=(20, 100))
        axis.grid(axis="y", alpha=0.25)
    axes[0, 0].legend(frameon=False, fontsize=8)
    for axis in axes[1, :]:
        axis.set_xlabel("Distractor density (%)")
    for axis in axes[:, 0]:
        axis.set_ylabel("Accuracy (%)")
    figure.suptitle("H2: cross-key interference with matched active-token control")
    figure.tight_layout()
    figure.savefig(output_dir / "h2_accuracy.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    x = np.arange(len(DELAYS))
    for offset, comparison, color, marker, label in (
        (-0.12, "set_vs_noop", "#d62728", "o", "SET vs NOOP"),
        (0.12, "set_vs_query", "#1f77b4", "s", "SET vs QUERY"),
    ):
        rows = [
            row for row in contrasts
            if row["comparison"] == comparison and float(row["density"]) == 1.0
        ]
        rows.sort(key=lambda row: int(row["delay"]))
        effect = np.asarray([float(row["accuracy_effect_pp"]) for row in rows])
        low = np.asarray([float(row["ci_low"]) for row in rows])
        high = np.asarray([float(row["ci_high"]) for row in rows])
        axis.errorbar(
            x + offset, effect, yerr=np.vstack((effect - low, high - effect)),
            fmt=marker, capsize=4, color=color, label=label,
        )
    axis.axhline(0, color="0.35", linewidth=1)
    axis.axhline(-PRACTICAL_MARGIN_PP, color="#d62728", linestyle="--", linewidth=1, label="-2 pp margin")
    axis.set(
        xticks=x,
        xticklabels=[str(delay) for delay in DELAYS],
        xlabel="Fixed SET-to-QUERY delay",
        ylabel="Accuracy effect at 100% density (pp)",
        title="Cross-key SET effect with 95% hierarchical bootstrap intervals",
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "h2_effects.png", dpi=180)
    plt.close(figure)


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoints = sorted(args.checkpoint_root.glob("seed*/u1500/baseline.pt"))
    if len(checkpoints) < 5:
        raise RuntimeError(f"expected five checkpoints, found {len(checkpoints)}")

    datasets: dict[tuple[int, str, float], list[list[int]]] = {}
    targets: dict[int, np.ndarray] = {}
    latest_values: dict[tuple[int, float], np.ndarray] = {}
    for delay in DELAYS:
        designs = make_design(delay, args.examples, args.evaluation_seed)
        for density in DENSITIES:
            for kind in KINDS:
                sequences, target, latest = sequences_for(designs, delay, density, kind)
                datasets[(delay, kind, density)] = sequences
                targets[delay] = target
                if kind == "set":
                    latest_values[(delay, density)] = latest

    predictions: dict[tuple[int, str, float], np.ndarray] = {
        key: np.empty((len(checkpoints), args.examples), dtype=np.int8) for key in datasets
    }
    for model_index, checkpoint in enumerate(checkpoints):
        model = load_model(checkpoint, device)
        for delay in DELAYS:
            baseline = predict(model, datasets[(delay, "set", 0.0)], device, args.batch_size)
            predictions[(delay, "set", 0.0)][model_index] = baseline
            predictions[(delay, "query", 0.0)][model_index] = baseline
            for density in DENSITIES[1:]:
                for kind in KINDS:
                    predictions[(delay, kind, density)][model_index] = predict(
                        model, datasets[(delay, kind, density)], device, args.batch_size
                    )
        del model

    metrics, contrasts = summarize(predictions, targets, latest_values, args.bootstrap_draws, args.bootstrap_seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "metrics.csv", metrics)
    write_csv(args.output_dir / "contrasts.csv", contrasts)
    np.savez_compressed(
        args.output_dir / "paired_predictions.npz",
        **{str(key): value for key, value in predictions.items()},
    )
    make_plots(metrics, contrasts, args.output_dir)
    summary = {
        "checkpoints": [str(path) for path in checkpoints],
        "n_models": len(checkpoints),
        "examples_per_condition_per_model": args.examples,
        "bootstrap_draws": args.bootstrap_draws,
        "practical_margin_pp": PRACTICAL_MARGIN_PP,
        "metrics": metrics,
        "contrasts": contrasts,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, default=Path("results/extended"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/h2_multiseed"))
    parser.add_argument("--examples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--evaluation-seed", type=int, default=20261023)
    parser.add_argument("--bootstrap-seed", type=int, default=271828)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({key: result[key] for key in result if key not in {"metrics", "contrasts"}}, indent=2))
