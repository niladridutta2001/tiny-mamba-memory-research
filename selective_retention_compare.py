"""Compare learned independent retention gating with baseline and coupled control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from evaluate import EVALUATION_SPLITS
from h2_multiseed import DELAYS as H2_DELAYS, DENSITIES, make_design, sequences_for
from h3_multiseed import OVERWRITE_COUNTS, overwrite_examples, retention_examples, write_csv
from hypothesis_pilot import load_model
from part_b_compare import paired_bootstrap
from train import build_loader, evaluate


MODEL_PATHS = {
    "Baseline": Path("results/extended/seed42/u1500/baseline.pt"),
    "Independent decay": Path("results/selective_retention/seed42/event_decay/u1500/baseline.pt"),
    "Coupled gate": Path("results/selective_retention/seed42/coupled_event_gate/u1500/baseline.pt"),
}
RETENTION_DELAYS = (10, 32, 64, 128, 256, 500)
OVERWRITE_DELAYS = (32, 64)
EVENT_NAMES = ("NOOP", "SET", "QUERY")


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


def add_contrasts(
    rows: list[dict],
    predictions: dict[tuple, np.ndarray],
    targets: dict[tuple, np.ndarray],
    conditions: list[tuple],
    rng: np.random.Generator,
    draws: int,
) -> None:
    for condition in conditions:
        target = targets[condition]
        correct = {
            name: (predictions[(name, *condition)] == target).astype(np.float64)
            for name in MODEL_PATHS
        }
        for comparison, left, right in (
            ("independent_vs_baseline", "Independent decay", "Baseline"),
            ("independent_vs_coupled", "Independent decay", "Coupled gate"),
            ("coupled_vs_baseline", "Coupled gate", "Baseline"),
        ):
            effect, low, high = paired_bootstrap(correct[left] - correct[right], rng, draws)
            rows.append(
                {
                    "test": condition[0],
                    "condition_a": condition[1],
                    "condition_b": condition[2] if len(condition) > 2 else "",
                    "comparison": comparison,
                    "effect_pp": effect,
                    "ci_low": low,
                    "ci_high": high,
                }
            )


def gate_diagnostics(models: dict[str, torch.nn.Module]) -> tuple[list[dict], list[dict]]:
    channel_rows, summary_rows = [], []
    for model_name in ("Independent decay", "Coupled gate"):
        raw = models[model_name].event_gate.detach().cpu().numpy()
        multipliers = np.exp(np.tanh(raw))
        for event_index, event in enumerate(EVENT_NAMES):
            values = multipliers[event_index]
            summary_rows.append(
                {
                    "model": model_name,
                    "event": event,
                    "mean_multiplier": float(values.mean()),
                    "std_multiplier": float(values.std()),
                    "min_multiplier": float(values.min()),
                    "max_multiplier": float(values.max()),
                    "fraction_below_one": float((values < 1).mean()),
                }
            )
            for channel, value in enumerate(values):
                channel_rows.append(
                    {"model": model_name, "event": event, "channel": channel, "multiplier": float(value)}
                )
    return channel_rows, summary_rows


def make_plots(metrics: list[dict], gate_rows: list[dict], output_dir: Path) -> None:
    styles = {
        "Baseline": ("#333333", "o"),
        "Independent decay": ("#d62728", "^"),
        "Coupled gate": ("#1f77b4", "s"),
    }
    figure, axes = plt.subplots(1, 3, figsize=(15.2, 4.6))
    for model, (color, marker) in styles.items():
        rows = [row for row in metrics if row["test"] == "retention" and row["model"] == model]
        rows.sort(key=lambda row: int(row["condition_a"]))
        axes[0].plot(
            [int(row["condition_a"]) for row in rows],
            [100 * float(row["accuracy"]) for row in rows],
            marker=marker,
            color=color,
            label=model,
        )
    axes[0].axhline(25, color="0.55", linestyle=":", linewidth=1)
    axes[0].set(xlabel="NOOP delay", ylabel="Query accuracy (%)", title="Isolated retention", ylim=(0, 100))
    axes[0].set_xscale("log")
    axes[0].set_xticks(RETENTION_DELAYS, [str(delay) for delay in RETENTION_DELAYS])
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)

    for model, (color, marker) in styles.items():
        rows = [
            row for row in metrics
            if row["test"] == "interference" and row["model"] == model
            and int(row["condition_a"]) == 32
        ]
        rows.sort(key=lambda row: float(row["condition_b"]))
        axes[1].plot(
            [100 * float(row["condition_b"]) for row in rows],
            [100 * float(row["accuracy"]) for row in rows],
            marker=marker,
            color=color,
            label=model,
        )
    axes[1].axhline(25, color="0.55", linestyle=":", linewidth=1)
    axes[1].set(
        xlabel="Unrelated-SET density (%)",
        ylabel="Query accuracy (%)",
        title="Interference at delay 32",
        ylim=(0, 100),
    )
    axes[1].grid(axis="y", alpha=0.25)

    for model, (color, marker) in styles.items():
        rows = [
            row for row in metrics
            if row["test"] == "overwrite" and row["model"] == model
            and int(row["condition_b"]) == 32
        ]
        rows.sort(key=lambda row: int(row["condition_a"]))
        axes[2].plot(
            [int(row["condition_a"]) for row in rows],
            [100 * float(row["accuracy"]) for row in rows],
            marker=marker,
            color=color,
            label=model,
        )
    axes[2].set(
        xlabel="Number of overwrites",
        ylabel="Query accuracy (%)",
        title="Overwrite test, final delay 32",
        xticks=list(OVERWRITE_COUNTS),
        ylim=(0, 100),
    )
    axes[2].grid(axis="y", alpha=0.25)
    figure.suptitle("Learned selective retention: matched seed-42 comparison")
    figure.tight_layout()
    figure.savefig(output_dir / "selective_retention_summary.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(10.5, 8), sharex=True, sharey=True)
    for axis, delay in zip(axes.flat, H2_DELAYS):
        for model, (color, marker) in styles.items():
            rows = [
                row for row in metrics
                if row["test"] == "interference" and row["model"] == model
                and int(row["condition_a"]) == delay
            ]
            rows.sort(key=lambda row: float(row["condition_b"]))
            axis.plot(
                [100 * float(row["condition_b"]) for row in rows],
                [100 * float(row["accuracy"]) for row in rows],
                marker=marker,
                color=color,
                label=model,
            )
        axis.axhline(25, color="0.55", linestyle=":", linewidth=1)
        axis.set(title=f"Fixed delay = {delay}", ylim=(0, 100))
        axis.grid(axis="y", alpha=0.25)
    axes[0, 0].legend(frameon=False, fontsize=8)
    for axis in axes[1]:
        axis.set_xlabel("Unrelated-SET density (%)")
    for axis in axes[:, 0]:
        axis.set_ylabel("Query accuracy (%)")
    figure.suptitle("Selective retention under cross-key interference")
    figure.tight_layout()
    figure.savefig(output_dir / "selective_retention_interference.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), sharey=True)
    x = np.arange(len(EVENT_NAMES))
    for axis, model in zip(axes, ("Independent decay", "Coupled gate")):
        values = [
            [float(row["multiplier"]) for row in gate_rows if row["model"] == model and row["event"] == event]
            for event in EVENT_NAMES
        ]
        parts = axis.violinplot(values, positions=x, showmeans=True, showextrema=True)
        for body in parts["bodies"]:
            body.set_facecolor(styles[model][0])
            body.set_alpha(0.35)
        for key in ("cmeans", "cmins", "cmaxes", "cbars"):
            parts[key].set_color(styles[model][0])
        axis.axhline(1, color="0.35", linewidth=1)
        axis.set(xticks=x, xticklabels=EVENT_NAMES, xlabel="Event type", title=model)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel(r"Learned multiplier $r_{e,d}$")
    figure.suptitle("Learned event-conditioned retention multipliers across 64 channels")
    figure.tight_layout()
    figure.savefig(output_dir / "learned_retention_multipliers.png", dpi=180)
    plt.close(figure)


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    models = {name: load_model(path, device) for name, path in MODEL_PATHS.items()}
    predictions: dict[tuple, np.ndarray] = {}
    targets: dict[tuple, np.ndarray] = {}
    metrics: list[dict] = []
    conditions: list[tuple] = []

    for delay in RETENTION_DELAYS:
        condition = ("retention", delay, "")
        sequences, target, _ = retention_examples(delay, args.examples, args.evaluation_seed)
        conditions.append(condition)
        targets[condition] = target
        for model_name, model in models.items():
            result = predict(model, sequences, device, args.batch_size)
            predictions[(model_name, *condition)] = result
            metrics.append(
                {"test": "retention", "model": model_name, "condition_a": delay, "condition_b": "", "accuracy": float((result == target).mean())}
            )

    for delay in H2_DELAYS:
        designs = make_design(delay, args.examples, args.evaluation_seed)
        for density in DENSITIES:
            condition = ("interference", delay, density)
            sequences, target, _ = sequences_for(designs, delay, density, "set")
            conditions.append(condition)
            targets[condition] = target
            for model_name, model in models.items():
                result = predict(model, sequences, device, args.batch_size)
                predictions[(model_name, *condition)] = result
                metrics.append(
                    {"test": "interference", "model": model_name, "condition_a": delay, "condition_b": density, "accuracy": float((result == target).mean())}
                )

    for overwrite_count in OVERWRITE_COUNTS:
        for delay in OVERWRITE_DELAYS:
            condition = ("overwrite", overwrite_count, delay)
            sequences, target, _ = overwrite_examples(
                overwrite_count, delay, args.examples, args.evaluation_seed
            )
            conditions.append(condition)
            targets[condition] = target
            for model_name, model in models.items():
                result = predict(model, sequences, device, args.batch_size)
                predictions[(model_name, *condition)] = result
                metrics.append(
                    {"test": "overwrite", "model": model_name, "condition_a": overwrite_count, "condition_b": delay, "accuracy": float((result == target).mean())}
                )

    contrasts: list[dict] = []
    add_contrasts(
        contrasts,
        predictions,
        targets,
        conditions,
        np.random.default_rng(args.bootstrap_seed),
        args.bootstrap_draws,
    )

    fixed_rows = []
    for model_name, model in models.items():
        for split, (label, length) in EVALUATION_SPLITS.items():
            result = evaluate(
                model,
                build_loader(args.data_dir / f"{split}.jsonl.gz", args.fixed_batch_size, False, 42),
                device,
            )
            fixed_rows.append(
                {"model": model_name, "split": split, "condition": label, "length": length, "query_accuracy": result["query_accuracy"], "query_count": result["query_count"]}
            )

    gate_rows, gate_summary = gate_diagnostics(models)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "metrics.csv", metrics)
    write_csv(args.output_dir / "contrasts.csv", contrasts)
    write_csv(args.output_dir / "fixed_splits.csv", fixed_rows)
    write_csv(args.output_dir / "learned_gate_channels.csv", gate_rows)
    write_csv(args.output_dir / "learned_gate_summary.csv", gate_summary)
    np.savez_compressed(
        args.output_dir / "paired_predictions.npz",
        **{str(key): value for key, value in predictions.items()},
    )
    make_plots(metrics, gate_rows, args.output_dir)
    summary = {
        "scope": "matched seed-42 experiment",
        "checkpoints": {name: str(path) for name, path in MODEL_PATHS.items()},
        "examples_per_condition": args.examples,
        "bootstrap_draws": args.bootstrap_draws,
        "metrics": metrics,
        "contrasts": contrasts,
        "fixed_splits": fixed_rows,
        "learned_gates": gate_summary,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/selective_retention/seed42/comparison"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/generated"))
    parser.add_argument("--examples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--fixed-batch-size", type=int, default=16)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--evaluation-seed", type=int, default=20261023)
    parser.add_argument("--bootstrap-seed", type=int, default=57721)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({"scope": result["scope"], "learned_gates": result["learned_gates"]}, indent=2))
