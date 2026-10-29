"""Aggregate the three independently trained retention pilots."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEEDS = (42, 43, 44)
T_CRITICAL_DF2_95 = 4.302652729


def mean_ci(values: list[float]) -> tuple[float, float, float, float]:
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    standard_deviation = math.sqrt(variance)
    margin = T_CRITICAL_DF2_95 * standard_deviation / math.sqrt(len(values))
    return mean, standard_deviation, mean - margin, mean + margin


def main() -> None:
    root = Path("results/extended")
    output_dir = root / "multiseed"
    output_dir.mkdir(parents=True, exist_ok=True)
    by_seed: dict[int, dict[int, float]] = {}
    validation = {}
    for seed in SEEDS:
        pilot_path = root / f"seed{seed}" / "u1500" / "pilot" / "hypothesis_pilot.csv"
        with pilot_path.open(encoding="utf-8") as source:
            rows = list(csv.DictReader(source))
        by_seed[seed] = {int(float(row["x"])): float(row["accuracy"]) for row in rows if row["hypothesis"] == "H1"}
        validation_path = root / f"seed{seed}" / "u1500" / "baseline_validation.json"
        validation[seed] = json.loads(validation_path.read_text(encoding="utf-8"))["validation"]["query_accuracy"]

    delays = sorted(next(iter(by_seed.values())))
    summary = []
    for delay in delays:
        values = [by_seed[seed][delay] for seed in SEEDS]
        mean, standard_deviation, lower, upper = mean_ci(values)
        summary.append(
            {
                "delay": delay,
                "mean_accuracy": mean,
                "sample_sd": standard_deviation,
                "ci95_lower": lower,
                "ci95_upper": upper,
                **{f"seed_{seed}": by_seed[seed][delay] for seed in SEEDS},
            }
        )

    paired_drops = {}
    for long_delay in (64, 128, 256, 500):
        values = [by_seed[seed][10] - by_seed[seed][long_delay] for seed in SEEDS]
        mean, standard_deviation, lower, upper = mean_ci(values)
        paired_drops[f"delay_10_minus_{long_delay}"] = {
            "mean_drop": mean,
            "sample_sd": standard_deviation,
            "ci95_lower": lower,
            "ci95_upper": upper,
            "per_seed": dict(zip(SEEDS, values)),
        }

    with (output_dir / "retention_multiseed.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)
    record = {"model_seeds": SEEDS, "validation_accuracy": validation, "retention": summary, "paired_drops": paired_drops}
    (output_dir / "retention_multiseed.json").write_text(json.dumps(record, indent=2), encoding="utf-8")

    figure, axis = plt.subplots(figsize=(8.6, 5.0))
    means = [100 * row["mean_accuracy"] for row in summary]
    lower_errors = [100 * (row["mean_accuracy"] - max(0.0, row["ci95_lower"])) for row in summary]
    upper_errors = [100 * (min(1.0, row["ci95_upper"]) - row["mean_accuracy"]) for row in summary]
    axis.errorbar(delays, means, yerr=[lower_errors, upper_errors], fmt="o-", capsize=4, color="#2563eb")
    axis.axhline(25, linestyle="--", linewidth=1.5, color="#6b7280", label="4-value chance (25%)")
    axis.set_xscale("symlog", linthresh=4)
    axis.set_xlim(-1, 550)
    axis.set_xticks(delays, [str(delay) for delay in delays])
    axis.set(xlabel="Intervening NOOPs (symlog scale)", ylabel="Accuracy (%)", ylim=(0, 105))
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    figure.subplots_adjust(left=0.12, right=0.97, bottom=0.18, top=0.95)
    figure.savefig(output_dir / "retention_multiseed.png", dpi=180)
    plt.close(figure)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
