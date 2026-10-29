"""Supplementary d_conv sweep for post-overwrite settling behavior."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from h3_multiseed import write_csv
from hypothesis_pilot import load_model
from part_b_compare import paired_bootstrap
from stale_value_diagnostic import balanced_designs, make_sequences, predict


MODEL_PATHS = {
    1: Path("results/selective_retention/seed42/dconv_sweep/dconv1/u1500/baseline.pt"),
    2: Path("results/selective_retention/seed42/dconv_sweep/dconv2/u1500/baseline.pt"),
    4: Path("results/selective_retention/seed42/event_decay/u1500/baseline.pt"),
    8: Path("results/selective_retention/seed42/dconv_sweep/dconv8/u1500/baseline.pt"),
}
POST_DELAYS = (0, 1, 2, 4, 8, 16, 32)
PRE_DELAY = 100
VALIDATION = {1: 0.940201818861343, 2: 0.7771271957144636, 4: 0.8029151613305096, 8: 0.8065279681076367}
PARAMETERS = {1: 10149, 2: 10213, 4: 10341, 8: 10597}


def make_plot(metrics: list[dict], output_dir: Path) -> None:
    colors = {1: "#9467bd", 2: "#2ca02c", 4: "#d62728", 8: "#1f77b4"}
    markers = {1: "D", 2: "o", 4: "^", 8: "s"}
    x = np.arange(len(POST_DELAYS))
    figure, axes = plt.subplots(1, 3, figsize=(15.2, 4.7), sharex=True)
    for width in MODEL_PATHS:
        rows = sorted(
            (row for row in metrics if int(row["d_conv"]) == width),
            key=lambda row: int(row["post_delay"]),
        )
        accuracy = np.asarray([100 * float(row["overwrite_accuracy"]) for row in rows])
        stale = np.asarray([100 * float(row["stale_value_error_rate"]) for row in rows])
        label = f"d_conv={width}" + (" (mandatory baseline)" if width == 4 else "")
        axes[0].plot(x, accuracy, marker=markers[width], color=colors[width], label=label)
        axes[1].plot(x, stale, marker=markers[width], color=colors[width], label=label)
        axes[2].plot(x, accuracy - accuracy[0], marker=markers[width], color=colors[width], label=label)
    axes[0].set(ylabel="Overwrite accuracy (%)", title="Predict new value", ylim=(0, 100))
    axes[1].set(ylabel="Stale-value error rate (%)", title="Predict obsolete value", ylim=(0, 25))
    axes[2].axhline(0, color="0.35", linewidth=1)
    axes[2].set(ylabel="Accuracy change from immediate query (pp)", title="Post-overwrite settling")
    for axis in axes:
        axis.set_xlabel("NOOPs after overwrite")
        axis.set_xticks(x, [str(delay) for delay in POST_DELAYS])
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    figure.suptitle("Supplementary convolution-width diagnostic: 100 NOOPs before overwrite")
    figure.tight_layout()
    figure.savefig(output_dir / "dconv_overwrite_sweep.png", dpi=180)
    plt.close(figure)


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    models = {width: load_model(path, device) for width, path in MODEL_PATHS.items()}
    designs = balanced_designs(args.examples, args.evaluation_seed, PRE_DELAY)
    metrics, predictions = [], {}
    targets = {}
    for post_delay in POST_DELAYS:
        overwrite, retention, old_target, new_target = make_sequences(designs, PRE_DELAY, post_delay)
        targets[post_delay] = (old_target, new_target)
        for width, model in models.items():
            overwrite_prediction = predict(model, overwrite, device, args.batch_size)
            retention_prediction = predict(model, retention, device, args.batch_size)
            predictions[(width, post_delay, "overwrite")] = overwrite_prediction
            predictions[(width, post_delay, "retention")] = retention_prediction
            correct = overwrite_prediction == new_target
            stale = overwrite_prediction == old_target
            metrics.append(
                {
                    "d_conv": width,
                    "post_delay": post_delay,
                    "overwrite_accuracy": float(correct.mean()),
                    "stale_value_error_rate": float(stale.mean()),
                    "other_error_rate": float((~correct & ~stale).mean()),
                    "retention_accuracy": float((retention_prediction == old_target).mean()),
                    "validation_accuracy": VALIDATION[width],
                    "parameter_count": PARAMETERS[width],
                    "n_examples": args.examples,
                }
            )

    peaks, contrasts = [], []
    rng = np.random.default_rng(args.bootstrap_seed)
    for width in MODEL_PATHS:
        rows = sorted(
            (row for row in metrics if int(row["d_conv"]) == width),
            key=lambda row: int(row["post_delay"]),
        )
        best = max(rows, key=lambda row: float(row["overwrite_accuracy"]))
        immediate = predictions[(width, 0, "overwrite")] == targets[0][1]
        peak_delay = int(best["post_delay"])
        peak_correct = predictions[(width, peak_delay, "overwrite")] == targets[peak_delay][1]
        effect, low, high = paired_bootstrap(
            peak_correct.astype(float) - immediate.astype(float), rng, args.bootstrap_draws
        )
        peaks.append(
            {
                "d_conv": width,
                "peak_post_delay": peak_delay,
                "peak_accuracy": float(best["overwrite_accuracy"]),
                "immediate_accuracy": float(immediate.mean()),
                "peak_gain_pp": effect,
                "peak_gain_ci_low": low,
                "peak_gain_ci_high": high,
                "validation_accuracy": VALIDATION[width],
                "parameter_count": PARAMETERS[width],
            }
        )
        for post_delay in POST_DELAYS:
            width_correct = predictions[(width, post_delay, "overwrite")] == targets[post_delay][1]
            baseline_correct = predictions[(4, post_delay, "overwrite")] == targets[post_delay][1]
            effect, low, high = paired_bootstrap(
                width_correct.astype(float) - baseline_correct.astype(float), rng, args.bootstrap_draws
            )
            contrasts.append(
                {
                    "d_conv_vs_4": width,
                    "post_delay": post_delay,
                    "accuracy_effect_pp": effect,
                    "ci_low": low,
                    "ci_high": high,
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "metrics.csv", metrics)
    write_csv(args.output_dir / "peaks.csv", peaks)
    write_csv(args.output_dir / "contrasts.csv", contrasts)
    np.savez_compressed(
        args.output_dir / "paired_predictions.npz",
        **{str(key): value for key, value in predictions.items()},
    )
    make_plot(metrics, args.output_dir)
    summary = {
        "supplementary_diagnostic": True,
        "mandatory_baseline_d_conv": 4,
        "caveat": "Changing d_conv changes parameter count, initialization trajectory, and optimization behavior.",
        "pre_delay": PRE_DELAY,
        "post_delays": POST_DELAYS,
        "examples_per_condition": args.examples,
        "bootstrap_draws": args.bootstrap_draws,
        "checkpoints": {str(width): str(path) for width, path in MODEL_PATHS.items()},
        "peaks": peaks,
        "metrics": metrics,
        "contrasts": contrasts,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/selective_retention/seed42/dconv_sweep/comparison"))
    parser.add_argument("--examples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--evaluation-seed", type=int, default=20261023)
    parser.add_argument("--bootstrap-seed", type=int, default=173205)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({"caveat": result["caveat"], "peaks": result["peaks"]}, indent=2))
