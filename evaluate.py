"""Evaluate A3 baselines on the fixed assignment splits and create record plots."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from data.dataset import IGNORE_INDEX, StateTrackingDataset, collate_batch
from task_model import StateTracker
from train import build_loader, evaluate


EVALUATION_SPLITS = {
    "test_id": ("ID", 128),
    "test_ood_256": ("OOD", 256),
    "test_ood_512": ("OOD", 512),
    "stress_gap_512": ("Delay stress", 512),
    "stress_overwrite_512": ("Overwrite stress", 512),
}


def target_counts(dataset: StateTrackingDataset) -> Counter:
    counts: Counter = Counter()
    for sample in dataset:
        counts.update(int(value) for value in sample["targets"] if int(value) != IGNORE_INDEX)
    return counts


def constant_accuracy(dataset: StateTrackingDataset, prediction: int) -> tuple[float, int]:
    counts = target_counts(dataset)
    total = sum(counts.values())
    return counts[prediction] / total, total


def load_model(checkpoint: Path, device: torch.device) -> StateTracker:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model_config = payload.get("config", {}).get("model", {})
    model = StateTracker(
        variant=str(model_config.get("variant", "baseline")),
        route_init=float(model_config.get("route_init", 0.95)),
        d_conv=int(model_config.get("d_conv", 4)),
    ).to(device)
    model.load_state_dict(payload["model_state_dict"])
    return model


def plot_length_accuracy(rows: list[dict], output: Path) -> None:
    length_rows = [row for row in rows if row["condition"] in {"ID", "OOD"}]
    fig, axis = plt.subplots(figsize=(7.2, 4.5))
    styles = {
        "Neural": ("o-", "#2563eb"),
        "Always UNSET": ("s--", "#dc2626"),
        "Train majority": ("^--", "#d97706"),
        "Oracle": ("D-", "#059669"),
    }
    for baseline, (style, color) in styles.items():
        selected = sorted((row for row in length_rows if row["baseline"] == baseline), key=lambda row: row["length"])
        axis.plot(
            [row["length"] for row in selected],
            [100 * row["query_accuracy"] for row in selected],
            style,
            color=color,
            linewidth=2,
            markersize=6,
            label=baseline,
        )
    axis.set(xlabel="Sequence length", ylabel="Query accuracy (%)", xticks=[128, 256, 512], ylim=(0, 103))
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_stress_accuracy(rows: list[dict], output: Path) -> None:
    stress = [row for row in rows if "stress" in row["condition"].lower()]
    baselines = ["Neural", "Always UNSET", "Train majority", "Oracle"]
    conditions = ["Delay stress", "Overwrite stress"]
    width = 0.19
    fig, axis = plt.subplots(figsize=(7.2, 4.5))
    colors = ["#2563eb", "#dc2626", "#d97706", "#059669"]
    for index, (baseline, color) in enumerate(zip(baselines, colors)):
        values = []
        for condition in conditions:
            row = next(item for item in stress if item["condition"] == condition and item["baseline"] == baseline)
            values.append(100 * row["query_accuracy"])
        positions = [position + (index - 1.5) * width for position in range(len(conditions))]
        axis.bar(positions, values, width, label=baseline, color=color)
    axis.set_xticks(range(len(conditions)), conditions)
    axis.set_ylabel("Query accuracy (%)")
    axis.set_ylim(0, 105)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def run(args: argparse.Namespace) -> list[dict]:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    train_dataset = StateTrackingDataset(args.data_dir / "train.jsonl.gz")
    majority_class, majority_count = target_counts(train_dataset).most_common(1)[0]
    model = load_model(args.checkpoint, device)
    rows: list[dict] = []
    for split, (condition, length) in EVALUATION_SPLITS.items():
        path = args.data_dir / f"{split}.jsonl.gz"
        dataset = StateTrackingDataset(path)
        neural = evaluate(model, build_loader(path, args.batch_size, False, 42), device)
        unset_accuracy, count = constant_accuracy(dataset, 0)
        majority_accuracy, _ = constant_accuracy(dataset, majority_class)
        shared = {"split": split, "condition": condition, "length": length, "query_count": count}
        rows.extend(
            [
                {**shared, "baseline": "Neural", "query_accuracy": neural["query_accuracy"]},
                {**shared, "baseline": "Always UNSET", "query_accuracy": unset_accuracy},
                {**shared, "baseline": "Train majority", "query_accuracy": majority_accuracy},
                {**shared, "baseline": "Oracle", "query_accuracy": 1.0},
            ]
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "a3_metrics.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "train_majority_class": majority_class,
        "train_majority_count": majority_count,
        "rows": rows,
    }
    (args.output_dir / "a3_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_length_accuracy(rows, args.output_dir / "a3_accuracy_by_length.png")
    plot_stress_accuracy(rows, args.output_dir / "a3_stress_accuracy.png")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("results/a3/checkpoints/baseline.pt"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/generated"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/a3"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", choices=["cpu", "cuda"])
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))
