"""Inference-time delta interventions for the H3 retention-overwrite test."""

from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from generate_dataset import encode_event
from hypothesis_pilot import DELAYS, retention_sequences
from task_model import StateTracker

MODES = ("global", "noop_only", "set_only")
ALPHAS = (0.25, 0.5, 1.0, 2.0)


def load_model(device: torch.device) -> StateTracker:
    checkpoint = Path("results/extended/seed42/u1500/baseline.pt")
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = StateTracker().to(device)
    model.load_state_dict(payload["model_state_dict"])
    return model.eval()


@torch.no_grad()
def retention_accuracy(model: StateTracker, delay: int, alpha: float, mode: str, device: torch.device) -> float:
    sequences, labels = retention_sequences(delay)
    inputs = torch.tensor(sequences, dtype=torch.long, device=device)
    predictions = model.scaled_logits(inputs, alpha, mode)[:, -1].argmax(-1).cpu()
    return float((predictions == torch.tensor(labels)).float().mean().item())


def load_overwrite_records() -> list[dict]:
    path = Path("data/generated/stress_overwrite_512.jsonl.gz")
    with gzip.open(path, "rt", encoding="utf-8") as source:
        return [json.loads(line) for line in source]


@torch.no_grad()
def overwrite_metrics(
    model: StateTracker, records: list[dict], alpha: float, mode: str, device: torch.device, batch_size: int = 16
) -> dict[int | str, dict[str, float | int]]:
    totals = {count: {"correct": 0, "obsolete": 0, "total": 0} for count in (1, 2, 4, 8)}
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        inputs = torch.tensor([[encode_event(event) for event in record["events"]] for record in batch], device=device)
        predictions = model.scaled_logits(inputs, alpha, mode).argmax(-1).cpu()
        for row, record in enumerate(batch):
            probe_index = int(record["probe_index"])
            target = int(record["query_targets"][probe_index])
            query_key = record["events"][probe_index][1]
            assigned_values = [
                event[2]
                for event in record["events"][:probe_index]
                if event[0] == "SET" and event[1] == query_key
            ]
            previous_value = assigned_values[-2]
            prediction = int(predictions[row, probe_index])
            count = int(record["overwrite_count"])
            totals[count]["total"] += 1
            totals[count]["correct"] += int(prediction == target)
            totals[count]["obsolete"] += int(prediction == previous_value and prediction != target)
    result = {}
    aggregate = {"correct": 0, "obsolete": 0, "total": 0}
    for count, values in totals.items():
        result[count] = {
            "accuracy": values["correct"] / values["total"],
            "obsolete_error_rate": values["obsolete"] / values["total"],
            "n": values["total"],
        }
        for key in aggregate:
            aggregate[key] += values[key]
    result["all"] = {
        "accuracy": aggregate["correct"] / aggregate["total"],
        "obsolete_error_rate": aggregate["obsolete"] / aggregate["total"],
        "n": aggregate["total"],
    }
    return result


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    overwrite_records = load_overwrite_records()
    rows = []
    for mode in MODES:
        for alpha in ALPHAS:
            for delay in DELAYS:
                rows.append(
                    {
                        "mode": mode,
                        "alpha": alpha,
                        "test": "retention",
                        "condition": delay,
                        "accuracy": retention_accuracy(model, delay, alpha, mode, device),
                        "obsolete_error_rate": "",
                        "n": 32,
                    }
                )
            overwrite = overwrite_metrics(model, overwrite_records, alpha, mode, device)
            for count, metrics in overwrite.items():
                rows.append(
                    {
                        "mode": mode,
                        "alpha": alpha,
                        "test": "overwrite",
                        "condition": count,
                        **metrics,
                    }
                )

    output_dir = Path("results/h3_delta_sweep")
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "delta_sweep.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "delta_sweep.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    figure, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharey=True)
    for axis, mode in zip(axes, MODES):
        retention = [
            row for row in rows if row["mode"] == mode and row["test"] == "retention" and row["condition"] == 128
        ]
        overwrite = [
            row for row in rows if row["mode"] == mode and row["test"] == "overwrite" and row["condition"] == "all"
        ]
        obsolete = overwrite
        axis.plot(ALPHAS, [100 * row["accuracy"] for row in retention], "o-", label="Retention d=128")
        axis.plot(ALPHAS, [100 * row["accuracy"] for row in overwrite], "s-", label="Overwrite accuracy")
        axis.plot(ALPHAS, [100 * row["obsolete_error_rate"] for row in obsolete], "^--", label="Obsolete-value errors")
        axis.set_xscale("log", base=2)
        axis.set_xticks(ALPHAS, [str(alpha) for alpha in ALPHAS])
        axis.set(xlabel="Delta scale alpha", title=mode.replace("_", " ").title(), ylim=(0, 105))
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Rate (%)")
    axes[-1].legend(frameon=False, loc="best")
    figure.subplots_adjust(left=0.07, right=0.98, bottom=0.16, top=0.9, wspace=0.12)
    figure.savefig(output_dir / "delta_tradeoff.png", dpi=180)
    plt.close(figure)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
