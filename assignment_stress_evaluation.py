"""Assignment stress evaluation with query-level diagnostic metrics.

Uses the fixed 500-example, length-512 delay and overwrite stress files from
the assignment dataset, plus the three fixed length-generalization splits.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from data.dataset import IGNORE_INDEX, StateTrackingDataset, collate_batch
from evaluate import EVALUATION_SPLITS, load_model


MODEL_SPECS = {
    "Original Mamba": {
        "role": "Baseline",
        "checkpoint": Path("results/extended/seed42/u1500/baseline.pt"),
        "metadata": Path("results/extended/seed42/u1500/baseline_validation.json"),
        "training_seconds": 77.17721230001189 + 78.4752487000078 + 126.58114390005358 + 167.41071540000848,
    },
    "Fixed width 8": {
        "role": "Supplementary reference",
        "checkpoint": Path("results/adaptive_kernel_kernel_only/seed42/fixed8/u1500/baseline.pt"),
        "metadata": Path("results/adaptive_kernel_kernel_only/seed42/fixed8/u1500/baseline_validation.json"),
    },
    "Static mixture": {
        "role": "Ablation",
        "checkpoint": Path("results/adaptive_kernel_kernel_only/seed42/static/u1500/baseline.pt"),
        "metadata": Path("results/adaptive_kernel_kernel_only/seed42/static/u1500/baseline_validation.json"),
    },
    "Adaptive mixture": {
        "role": "Proposed",
        "checkpoint": Path("results/adaptive_kernel_kernel_only/seed42/adaptive/u1500/baseline.pt"),
        "metadata": Path("results/adaptive_kernel_kernel_only/seed42/adaptive/u1500/baseline_validation.json"),
    },
}

DELAY_BINS = ((0, 1, "0-1"), (2, 4, "2-4"), (5, 16, "5-16"), (17, 64, "17-64"),
              (65, 128, "65-128"), (129, 256, "129-256"), (257, math.inf, "257+"))
OVERWRITE_BINS = ((0, 0, "0"), (1, 1, "1"), (2, 3, "2-3"), (4, 7, "4-7"), (8, math.inf, "8+"))
INTERFERENCE_BINS = ((0, 0, "0"), (1, 4, "1-4"), (5, 16, "5-16"), (17, 64, "17-64"),
                     (65, math.inf, "65+"))


def bin_value(value: int | None, bins: tuple[tuple[float, float, str], ...], unset: str | None = None) -> str:
    if value is None:
        if unset is None:
            raise ValueError("A bin label is required for None")
        return unset
    for low, high, label in bins:
        if low <= value <= high:
            return label
    raise ValueError(value)


def query_contexts(record: dict) -> dict[int, dict]:
    state: dict[str, int] = {}
    histories: dict[str, list[int]] = defaultdict(list)
    last_set: dict[str, int] = {}
    contexts: dict[int, dict] = {}
    for index, (op, key, value) in enumerate(record["events"]):
        if op == "SET":
            state[key] = int(value)
            histories[key].append(int(value))
            last_set[key] = index
        elif op == "QUERY":
            start = last_set.get(key, -1) + 1
            interval = record["events"][start:index]
            cross_sets = sum(event[0] == "SET" and event[1] != key for event in interval)
            noops = sum(event[0] == "NOOP" for event in interval)
            history = histories.get(key, [])
            current = state.get(key, 0)
            obsolete = sorted(set(history[:-1]) - {current}) if len(history) > 1 else []
            contexts[index] = {
                "initialized": key in last_set,
                "set_to_query_delay": index - last_set[key] if key in last_set else None,
                "overwrite_count_at_query": max(0, len(history) - 1),
                "cross_key_sets_since_update": cross_sets,
                "noops_since_update": noops,
                "obsolete_values": obsolete,
                "query_key": key,
            }
    return contexts


def expected_calibration_error(rows: list[dict], bins: int = 10) -> float:
    if not rows:
        return float("nan")
    total = len(rows)
    ece = 0.0
    for lower in np.linspace(0.0, 0.9, bins):
        upper = lower + 0.1
        selected = [row for row in rows if lower <= row["confidence"] < upper or (upper >= 1 and row["confidence"] == 1)]
        if selected:
            accuracy = np.mean([row["correct"] for row in selected])
            confidence = np.mean([row["confidence"] for row in selected])
            ece += len(selected) / total * abs(accuracy - confidence)
    return float(ece)


def summarize(rows: list[dict]) -> dict:
    targets = sorted({row["target"] for row in rows})
    recalls = []
    for target in targets:
        selected = [row for row in rows if row["target"] == target]
        recalls.append(np.mean([row["correct"] for row in selected]))
    initialized = [row for row in rows if row["initialized"]]
    unset = [row for row in rows if not row["initialized"]]
    stale_eligible = [row for row in rows if row["obsolete_values"]]
    return {
        "query_count": len(rows),
        "accuracy": float(np.mean([row["correct"] for row in rows])),
        "mean_nll": float(np.mean([row["nll"] for row in rows])),
        "brier_score": float(np.mean([row["brier_score"] for row in rows])),
        "mean_confidence": float(np.mean([row["confidence"] for row in rows])),
        "ece_10bin": expected_calibration_error(rows),
        "macro_recall": float(np.mean(recalls)),
        "initialized_accuracy": float(np.mean([row["correct"] for row in initialized])) if initialized else None,
        "unset_accuracy": float(np.mean([row["correct"] for row in unset])) if unset else None,
        "stale_value_error_rate": float(np.mean([row["stale_error"] for row in stale_eligible])) if stale_eligible else None,
        "stale_eligible_queries": len(stale_eligible),
    }


def grouped(rows: list[dict], field: str, ordered_labels: list[str], model: str, split: str) -> list[dict]:
    output = []
    for label in ordered_labels:
        selected = [row for row in rows if row[field] == label]
        if selected:
            output.append({"model": model, "split": split, "grouping": field, "bin": label, **summarize(selected)})
    return output


@torch.no_grad()
def evaluate_split(model: torch.nn.Module, dataset: StateTrackingDataset, batch_size: int, device: torch.device) -> tuple[list[dict], dict]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    contexts = {record["sample_id"]: query_contexts(record) for record in dataset.records}
    records_by_id = {record["sample_id"]: record for record in dataset.records}
    rows: list[dict] = []
    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    token_count = 0
    for batch in loader:
        inputs = batch["input_ids"].to(device)
        targets = batch["targets"].to(device)
        logits = model(inputs)
        probabilities = logits.softmax(dim=-1)
        token_count += int(batch["lengths"].sum())
        for row_index, sample_id in enumerate(batch["sample_ids"]):
            record = records_by_id[sample_id]
            for timestep, target in enumerate(record["query_targets"]):
                if target is None:
                    continue
                probs = probabilities[row_index, timestep].detach().cpu().numpy()
                prediction = int(probs.argmax())
                target = int(target)
                context = contexts[sample_id][timestep]
                one_hot = np.zeros(5, dtype=np.float64)
                one_hot[target] = 1.0
                row = {
                    "sample_id": sample_id,
                    "timestep": timestep,
                    "is_probe": timestep == record["probe_index"],
                    "designed_probe_gap": str(record["probe_gap"]) if record["probe_gap"] is not None else "",
                    "designed_overwrite_count": str(record["overwrite_count"]) if record["overwrite_count"] is not None else "",
                    "target": target,
                    "prediction": prediction,
                    "correct": int(prediction == target),
                    "confidence": float(probs[prediction]),
                    "nll": float(-math.log(max(float(probs[target]), 1e-12))),
                    "brier_score": float(np.square(probs - one_hot).sum()),
                    **context,
                }
                row["stale_error"] = int(prediction in context["obsolete_values"])
                row["delay_bin"] = bin_value(context["set_to_query_delay"], DELAY_BINS, "UNSET")
                row["overwrite_bin"] = bin_value(context["overwrite_count_at_query"], OVERWRITE_BINS)
                row["interference_bin"] = bin_value(context["cross_key_sets_since_update"], INTERFERENCE_BINS)
                rows.append(row)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    performance = {
        "evaluation_seconds": elapsed,
        "tokens_per_second": token_count / elapsed,
        "queries_per_second": len(rows) / elapsed,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
    }
    return rows, performance


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def paired_bootstrap(a: np.ndarray, b: np.ndarray, seed: int = 20261026, draws: int = 10000) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    differences = a.astype(float) - b.astype(float)
    effects = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, 500):
        count = min(500, draws - start)
        indices = rng.integers(0, len(differences), size=(count, len(differences)))
        effects[start:start + count] = differences[indices].mean(axis=1)
    return float(differences.mean()), float(np.quantile(effects, 0.025)), float(np.quantile(effects, 0.975))


def make_plots(overall: list[dict], grouped_rows: list[dict], output_dir: Path) -> None:
    models = list(MODEL_SPECS)
    colors = plt.cm.tab10(np.linspace(0, 0.8, len(models)))
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    stress_splits = ["stress_gap_512", "stress_overwrite_512"]
    width = 0.15
    for model_index, (model, color) in enumerate(zip(models, colors)):
        values = [100 * next(row["accuracy"] for row in overall if row["model"] == model and row["split"] == split) for split in stress_splits]
        positions = np.arange(2) + (model_index - 2) * width
        axes[0, 0].bar(positions, values, width, label=model, color=color)
    axes[0, 0].set_xticks(range(2), ["Delay stress", "Overwrite stress"])
    axes[0, 0].set_ylabel("All-query accuracy (%)")
    axes[0, 0].set_title("Assignment stress sets")
    axes[0, 0].grid(axis="y", alpha=0.25)

    probe_rows = [row for row in overall if row["split"] in stress_splits]
    for model_index, (model, color) in enumerate(zip(models, colors)):
        values = [100 * next(row["probe_accuracy"] for row in probe_rows if row["model"] == model and row["split"] == split) for split in stress_splits]
        positions = np.arange(2) + (model_index - 2) * width
        axes[0, 1].bar(positions, values, width, label=model, color=color)
    axes[0, 1].set_xticks(range(2), ["Final gap probe", "Final overwrite probe"])
    axes[0, 1].set_ylabel("Probe accuracy (%)")
    axes[0, 1].set_title("Designed final-query probes")
    axes[0, 1].grid(axis="y", alpha=0.25)

    for model, color in zip(models, colors):
        selected = [row for row in grouped_rows if row["model"] == model and row["split"] == "stress_gap_512" and row["grouping"] == "delay_bin"]
        mapping = {row["bin"]: row for row in selected}
        labels = [label for label in ["UNSET", "0-1", "2-4", "5-16", "17-64", "65-128", "129-256", "257+"] if label in mapping]
        axes[1, 0].plot(labels, [100 * mapping[label]["accuracy"] for label in labels], marker="o", label=model, color=color)
    axes[1, 0].set_ylabel("Accuracy (%)")
    axes[1, 0].set_xlabel("SET-to-QUERY delay bin")
    axes[1, 0].set_title("Delay stress: accuracy by delay")
    axes[1, 0].tick_params(axis="x", rotation=30)
    axes[1, 0].grid(axis="y", alpha=0.25)

    for model, color in zip(models, colors):
        selected = [row for row in grouped_rows if row["model"] == model and row["split"] == "stress_overwrite_512" and row["grouping"] == "designed_overwrite_count"]
        mapping = {row["bin"]: row for row in selected}
        labels = [label for label in ["1", "2", "4", "8"] if label in mapping]
        axes[1, 1].plot(labels, [100 * mapping[label]["accuracy"] for label in labels], marker="o", label=model, color=color)
    axes[1, 1].set_ylabel("Accuracy (%)")
    axes[1, 1].set_xlabel("Designed focus-key overwrite count")
    axes[1, 1].set_title("Final overwrite probe by assigned stress level")
    axes[1, 1].grid(axis="y", alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False)
    fig.suptitle("Assignment-defined stress evaluation (fixed seed-42 data)")
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(output_dir / "assignment_stress_metrics.png", dpi=180)
    plt.close(fig)


def make_resource_plot(tradeoffs: list[dict], output: Path) -> None:
    """Plot measured training time against peak GPU allocation."""
    styles = {
        "Original Mamba": ("#1f77b4", "o", (-20, 11)),
        "Fixed width 8": ("#2ca02c", "s", (10, -19)),
        "Static mixture": ("#8c564b", "D", (10, 11)),
        "Adaptive mixture": ("#bcbd22", "^", (-118, 11)),
    }
    figure, axis = plt.subplots(figsize=(8.2, 5.1))
    for row in tradeoffs:
        model = str(row["model"])
        seconds = float(row["training_seconds"])
        memory_mib = float(row["recorded_training_peak_gpu_memory_bytes"]) / (1024 ** 2)
        color, marker, offset = styles[model]
        axis.scatter(
            seconds, memory_mib, s=90, color=color, marker=marker,
            edgecolor="white", linewidth=0.9, zorder=3,
        )
        axis.annotate(
            f"{model}\n{seconds:.1f} s, {memory_mib:.1f} MiB",
            (seconds, memory_mib), xytext=offset, textcoords="offset points",
            fontsize=9, color="#1f2937",
            arrowprops={"arrowstyle": "-", "color": "#9ca3af", "linewidth": 0.8},
        )
    axis.set(
        xlabel="Training time for 1,500 updates (seconds)",
        ylabel="Recorded peak GPU allocation (MiB)",
        title="Gate-free kernel experiment: training cost",
    )
    axis.grid(alpha=0.25)
    axis.margins(x=0.12, y=0.22)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    overall_rows: list[dict] = []
    grouped_rows: list[dict] = []
    class_rows: list[dict] = []
    probe_predictions: dict[tuple[str, str], np.ndarray] = {}
    tradeoffs: list[dict] = []
    split_names = list(EVALUATION_SPLITS)

    for model_name, spec in MODEL_SPECS.items():
        metadata = json.loads(spec["metadata"].read_text(encoding="utf-8"))
        model = load_model(spec["checkpoint"], device)
        tradeoffs.append({
            "model": model_name,
            "role": spec["role"],
            "parameter_count": metadata["parameter_count"],
            "training_updates": metadata["training"].get("ending_update", metadata["training"]["updates"]),
            "training_seconds": spec.get("training_seconds", metadata["training"]["elapsed_seconds"]),
            "hardware": metadata["device_name"],
            "recorded_training_peak_gpu_memory_bytes": metadata["peak_gpu_memory_bytes"],
        })
        for split in split_names:
            dataset = StateTrackingDataset(args.data_dir / f"{split}.jsonl.gz")
            query_rows, performance = evaluate_split(model, dataset, args.batch_size, device)
            summary = summarize(query_rows)
            probes = [row for row in query_rows if row["is_probe"]]
            overall_rows.append({
                "model": model_name,
                "role": spec["role"],
                "split": split,
                "condition": EVALUATION_SPLITS[split][0],
                "sequence_length": EVALUATION_SPLITS[split][1],
                **summary,
                "probe_count": len(probes),
                "probe_accuracy": float(np.mean([row["correct"] for row in probes])) if probes else None,
                "probe_stale_value_error_rate": float(np.mean([row["stale_error"] for row in probes])) if probes else None,
                **performance,
            })
            delay_labels = ["UNSET", "0-1", "2-4", "5-16", "17-64", "65-128", "129-256", "257+"]
            grouped_rows.extend(grouped(query_rows, "delay_bin", delay_labels, model_name, split))
            grouped_rows.extend(grouped(query_rows, "overwrite_bin", ["0", "1", "2-3", "4-7", "8+"], model_name, split))
            grouped_rows.extend(grouped(query_rows, "interference_bin", ["0", "1-4", "5-16", "17-64", "65+"], model_name, split))
            grouped_rows.extend(grouped(probes, "designed_overwrite_count", ["1", "2", "4", "8"], model_name, split))
            probe_gaps = sorted({row["designed_probe_gap"] for row in probes if row["designed_probe_gap"]}, key=int)
            grouped_rows.extend(grouped(probes, "designed_probe_gap", probe_gaps, model_name, split))
            for target in range(5):
                selected = [row for row in query_rows if row["target"] == target]
                class_rows.append({
                    "model": model_name,
                    "split": split,
                    "target_class": target,
                    "support": len(selected),
                    "recall": float(np.mean([row["correct"] for row in selected])) if selected else None,
                })
            if probes:
                probe_predictions[(model_name, split)] = np.asarray([row["correct"] for row in probes], dtype=np.int8)

    contrasts = []
    for split in ("stress_gap_512", "stress_overwrite_512"):
        for comparison, first, second in (
            ("adaptive_vs_static", "Adaptive mixture", "Static mixture"),
            ("adaptive_vs_original", "Adaptive mixture", "Original Mamba"),
        ):
            effect, low, high = paired_bootstrap(probe_predictions[(first, split)], probe_predictions[(second, split)])
            contrasts.append({
                "split": split,
                "comparison": comparison,
                "probe_accuracy_effect_pp": 100 * effect,
                "ci_low_pp": 100 * low,
                "ci_high_pp": 100 * high,
                "n_paired_probes": len(probe_predictions[(first, split)]),
            })

    write_csv(args.output_dir / "overall_metrics.csv", overall_rows)
    write_csv(args.output_dir / "grouped_metrics.csv", grouped_rows)
    write_csv(args.output_dir / "class_metrics.csv", class_rows)
    write_csv(args.output_dir / "tradeoffs.csv", tradeoffs)
    write_csv(args.output_dir / "paired_probe_contrasts.csv", contrasts)
    summary = {
        "scope": "Assignment fixed evaluation data; matched seed-42 kernel-only checkpoints without retention gating",
        "assignment_stress_definition": {
            "stress_gap_512": "500 length-512 sequences; focus key set at t=0, unrelated events intervene, focus query at t=511",
            "stress_overwrite_512": "500 length-512 sequences; focus key receives 1, 2, 4, or 8 overwrites, focus query at t=511",
        },
        "models": {name: {key: str(value) if isinstance(value, Path) else value for key, value in spec.items()} for name, spec in MODEL_SPECS.items()},
        "overall_metrics": overall_rows,
        "grouped_metrics": grouped_rows,
        "class_metrics": class_rows,
        "tradeoffs": tradeoffs,
        "paired_probe_contrasts": contrasts,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    make_plots(overall_rows, grouped_rows, args.output_dir)
    make_resource_plot(tradeoffs, args.output_dir / "kernel_resource_tradeoff.png")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/generated"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/adaptive_kernel_kernel_only/seed42/assignment_stress"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({"paired_probe_contrasts": result["paired_probe_contrasts"]}, indent=2))
