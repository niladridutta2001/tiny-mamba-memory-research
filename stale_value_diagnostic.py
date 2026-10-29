"""Measure pure retention, overwrite accuracy, and stale-value errors."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from generate_dataset import KEYS, encode_event
from h3_multiseed import random_context, write_csv
from hypothesis_pilot import load_model
from part_b_compare import paired_bootstrap


MODEL_PATHS = {
    "Baseline": Path("results/extended/seed42/u1500/baseline.pt"),
    "Independent decay": Path("results/selective_retention/seed42/event_decay/u1500/baseline.pt"),
    "Coupled gate": Path("results/selective_retention/seed42/coupled_event_gate/u1500/baseline.pt"),
}
PRE_DELAYS = (10, 32, 64, 100, 128)
POST_DELAYS = (0, 1, 4, 16, 32, 64)
PRIMARY_PRE_DELAY = 100
NOOP = encode_event(["NOOP", None, None])


def balanced_designs(n: int, seed: int, pre_delay: int) -> list[dict]:
    rng = random.Random(seed + 1009 * pre_delay)
    cells = [
        (key, old_value, new_value)
        for key in KEYS
        for old_value in range(1, 5)
        for new_value in range(1, 5)
        if new_value != old_value
    ]
    assignments = (cells * ((n + len(cells) - 1) // len(cells)))[:n]
    rng.shuffle(assignments)
    return [
        {
            "key": key,
            "old_value": old_value,
            "new_value": new_value,
            "context": random_context(rng, key),
        }
        for key, old_value, new_value in assignments
    ]


def make_sequences(
    designs: list[dict], pre_delay: int, post_delay: int
) -> tuple[list[list[int]], list[list[int]], np.ndarray, np.ndarray]:
    overwrite, retention, old_targets, new_targets = [], [], [], []
    for design in designs:
        key = design["key"]
        old_value = design["old_value"]
        new_value = design["new_value"]
        overwrite.append(
            [
                *design["context"],
                encode_event(["SET", key, old_value]),
                *([NOOP] * pre_delay),
                encode_event(["SET", key, new_value]),
                *([NOOP] * post_delay),
                encode_event(["QUERY", key, None]),
            ]
        )
        # The no-overwrite control exactly matches the user's delay-100 example
        # at post_delay=0 and extends its passive interval as post_delay grows.
        retention.append(
            [
                *design["context"],
                encode_event(["SET", key, old_value]),
                *([NOOP] * (pre_delay + post_delay)),
                encode_event(["QUERY", key, None]),
            ]
        )
        old_targets.append(old_value)
        new_targets.append(new_value)
    return (
        overwrite,
        retention,
        np.asarray(old_targets, dtype=np.int8),
        np.asarray(new_targets, dtype=np.int8),
    )


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    sequences: list[list[int]],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    parts = []
    for start in range(0, len(sequences), batch_size):
        inputs = torch.tensor(sequences[start : start + batch_size], dtype=torch.long, device=device)
        parts.append(model(inputs)[:, -1].argmax(-1).cpu().numpy())
    return np.concatenate(parts).astype(np.int8)


def bootstrap_metric(
    values: np.ndarray, rng: np.random.Generator, draws: int
) -> tuple[float, float, float]:
    return paired_bootstrap(values.astype(np.float64), rng, draws)


def make_plots(metrics: list[dict], output_dir: Path) -> None:
    styles = {
        "Baseline": ("#333333", "o"),
        "Independent decay": ("#d62728", "^"),
        "Coupled gate": ("#1f77b4", "s"),
    }
    figure, axes = plt.subplots(1, 3, figsize=(15.2, 4.7), sharex=True)
    panels = (
        ("retention_accuracy", "Pure-retention accuracy (%)", "No overwrite"),
        ("overwrite_accuracy", "Overwrite accuracy (%)", "Predict new value"),
        ("stale_value_error_rate", "Stale-value error rate (%)", "Predict obsolete value"),
    )
    x = np.arange(len(POST_DELAYS))
    for axis, (field, ylabel, title) in zip(axes, panels):
        for model, (color, marker) in styles.items():
            rows = [
                row for row in metrics
                if row["model"] == model and int(row["pre_delay"]) == PRIMARY_PRE_DELAY
            ]
            rows.sort(key=lambda row: int(row["post_delay"]))
            axis.plot(
                x,
                [100 * float(row[field]) for row in rows],
                marker=marker,
                color=color,
                label=model,
            )
        axis.set(xlabel="NOOPs after overwrite", ylabel=ylabel, title=title, ylim=(0, 100))
        axis.set_xticks(x, [str(value) for value in POST_DELAYS])
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    figure.suptitle("Long-retention overwrite diagnostic: 100 NOOPs before overwrite")
    figure.tight_layout()
    figure.savefig(output_dir / "stale_value_primary.png", dpi=180)
    plt.close(figure)

    independent = [row for row in metrics if row["model"] == "Independent decay"]
    fields = (
        ("retention_accuracy", "Pure retention accuracy"),
        ("overwrite_accuracy", "Overwrite accuracy"),
        ("stale_value_error_rate", "Stale-value error rate"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(15.2, 4.6), sharex=True, sharey=True)
    for axis, (field, title) in zip(axes, fields):
        matrix = np.asarray(
            [
                [
                    100 * float(next(
                        row[field] for row in independent
                        if int(row["pre_delay"]) == pre and int(row["post_delay"]) == post
                    ))
                    for post in POST_DELAYS
                ]
                for pre in PRE_DELAYS
            ]
        )
        image = axis.imshow(matrix, aspect="auto", vmin=0, vmax=100, cmap="viridis")
        axis.set(
            xticks=range(len(POST_DELAYS)),
            xticklabels=[str(value) for value in POST_DELAYS],
            yticks=range(len(PRE_DELAYS)),
            yticklabels=[str(value) for value in PRE_DELAYS],
            xlabel="NOOPs after overwrite",
            title=title,
        )
        for row_index in range(len(PRE_DELAYS)):
            for column_index in range(len(POST_DELAYS)):
                color = "white" if matrix[row_index, column_index] < 50 else "black"
                axis.text(column_index, row_index, f"{matrix[row_index, column_index]:.0f}", ha="center", va="center", color=color, fontsize=8)
    axes[0].set_ylabel("NOOPs before overwrite")
    figure.suptitle("Independent-decay model across pre/post-overwrite delays")
    figure.subplots_adjust(left=0.07, right=0.88, bottom=0.13, top=0.82, wspace=0.12)
    color_axis = figure.add_axes([0.90, 0.18, 0.012, 0.58])
    figure.colorbar(image, cax=color_axis, label="Percent")
    figure.savefig(output_dir / "stale_value_delay_grid.png", dpi=180)
    plt.close(figure)


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    models = {name: load_model(path, device) for name, path in MODEL_PATHS.items()}
    rng = np.random.default_rng(args.bootstrap_seed)
    metrics, contrasts = [], []
    predictions: dict[tuple, np.ndarray] = {}

    for pre_delay in PRE_DELAYS:
        designs = balanced_designs(args.examples, args.evaluation_seed, pre_delay)
        for post_delay in POST_DELAYS:
            overwrite, retention, old_target, new_target = make_sequences(
                designs, pre_delay, post_delay
            )
            model_outcomes = {}
            for model_name, model in models.items():
                overwrite_prediction = predict(model, overwrite, device, args.batch_size)
                retention_prediction = predict(model, retention, device, args.batch_size)
                predictions[(model_name, pre_delay, post_delay, "overwrite")] = overwrite_prediction
                predictions[(model_name, pre_delay, post_delay, "retention")] = retention_prediction
                overwrite_correct = overwrite_prediction == new_target
                stale = overwrite_prediction == old_target
                retention_correct = retention_prediction == old_target
                model_outcomes[model_name] = {
                    "overwrite": overwrite_correct.astype(float),
                    "stale": stale.astype(float),
                    "retention": retention_correct.astype(float),
                }
                metrics.append(
                    {
                        "model": model_name,
                        "pre_delay": pre_delay,
                        "post_delay": post_delay,
                        "retention_accuracy": float(retention_correct.mean()),
                        "overwrite_accuracy": float(overwrite_correct.mean()),
                        "stale_value_error_rate": float(stale.mean()),
                        "other_error_rate": float((~overwrite_correct & ~stale).mean()),
                        "stale_fraction_of_errors": float(stale.sum() / max((~overwrite_correct).sum(), 1)),
                        "n_examples": args.examples,
                    }
                )

            for metric_name in ("overwrite", "stale", "retention"):
                for comparison, left, right in (
                    ("independent_vs_baseline", "Independent decay", "Baseline"),
                    ("independent_vs_coupled", "Independent decay", "Coupled gate"),
                ):
                    effect, low, high = paired_bootstrap(
                        model_outcomes[left][metric_name] - model_outcomes[right][metric_name],
                        rng,
                        args.bootstrap_draws,
                    )
                    contrasts.append(
                        {
                            "pre_delay": pre_delay,
                            "post_delay": post_delay,
                            "metric": metric_name,
                            "comparison": comparison,
                            "effect_pp": effect,
                            "ci_low": low,
                            "ci_high": high,
                        }
                    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "metrics.csv", metrics)
    write_csv(args.output_dir / "contrasts.csv", contrasts)
    np.savez_compressed(
        args.output_dir / "paired_predictions.npz",
        **{str(key): value for key, value in predictions.items()},
    )
    make_plots(metrics, args.output_dir)
    summary = {
        "scope": "matched seed-42 diagnostic",
        "old_and_new_values_always_differ": True,
        "stale_value_error_definition": "prediction equals obsolete value / all overwritten queries",
        "pre_delays": PRE_DELAYS,
        "post_delays": POST_DELAYS,
        "models": {name: str(path) for name, path in MODEL_PATHS.items()},
        "examples_per_condition": args.examples,
        "bootstrap_draws": args.bootstrap_draws,
        "metrics": metrics,
        "contrasts": contrasts,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/selective_retention/seed42/stale_value"))
    parser.add_argument("--examples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--evaluation-seed", type=int, default=20261023)
    parser.add_argument("--bootstrap-seed", type=int, default=141421)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({key: result[key] for key in result if key not in {"metrics", "contrasts"}}, indent=2))
