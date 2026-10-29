"""Balanced seed-42 diagnostic pilots for the A4 hypotheses."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from generate_dataset import KEYS, encode_event
from task_model import StateTracker


DELAYS = (0, 1, 2, 4, 10, 32, 64, 128, 256, 500)
DENSITIES = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
BURN_IN = 32


def load_model(checkpoint: Path, device: torch.device) -> StateTracker:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model_config = payload.get("config", {}).get("model", {})
    model = StateTracker(
        variant=str(model_config.get("variant", "baseline")),
        route_init=float(model_config.get("route_init", 0.95)),
        d_conv=int(model_config.get("d_conv", 4)),
    ).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def accuracy(model: StateTracker, sequences: list[list[int]], labels: list[int], device: torch.device) -> float:
    inputs = torch.tensor(sequences, dtype=torch.long, device=device)
    predictions = model(inputs)[:, -1].argmax(-1).cpu()
    return float((predictions == torch.tensor(labels)).float().mean().item())


def retention_sequences(delay: int) -> tuple[list[list[int]], list[int]]:
    sequences, labels = [], []
    noop = encode_event(["NOOP", None, None])
    for key in KEYS:
        for value in range(1, 5):
            sequence = [noop] * BURN_IN
            sequence.append(encode_event(["SET", key, value]))
            sequence.extend([noop] * delay)
            sequence.append(encode_event(["QUERY", key, None]))
            sequences.append(sequence)
            labels.append(value)
    return sequences, labels


def interference_sequences(density: float, seed: int, delay: int = 4) -> tuple[list[list[int]], list[int]]:
    rng = random.Random(seed)
    sequences, labels = [], []
    noop = encode_event(["NOOP", None, None])
    distractor_count = round(delay * density)
    for tracked_key in KEYS:
        other_keys = [key for key in KEYS if key != tracked_key]
        for value in range(1, 5):
            positions = set(rng.sample(range(delay), distractor_count))
            middle = []
            for position in range(delay):
                if position in positions:
                    middle.append(encode_event(["SET", rng.choice(other_keys), rng.randint(1, 4)]))
                else:
                    middle.append(noop)
            sequences.append(
                [noop] * BURN_IN
                + [encode_event(["SET", tracked_key, value]), *middle, encode_event(["QUERY", tracked_key, None])]
            )
            labels.append(value)
    return sequences, labels


def run(args: argparse.Namespace) -> list[dict]:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_model(args.checkpoint, device)
    rows = []
    for delay in DELAYS:
        sequences, labels = retention_sequences(delay)
        rows.append({"hypothesis": "H1", "x": delay, "seed": 42, "accuracy": accuracy(model, sequences, labels, device)})
    for density in DENSITIES:
        for seed in range(5):
            sequences, labels = interference_sequences(density, seed)
            rows.append(
                {"hypothesis": "H2", "x": density, "seed": seed, "accuracy": accuracy(model, sequences, labels, device)}
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "hypothesis_pilot.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "hypothesis_pilot.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    h1 = [row for row in rows if row["hypothesis"] == "H1"]
    axes[0].plot([row["x"] for row in h1], [100 * row["accuracy"] for row in h1], "o-", color="#2563eb")
    axes[0].set(xlabel="Intervening NOOPs", ylabel="Accuracy (%)", title="H1: passive retention", ylim=(0, 105))
    h2_means = []
    for density in DENSITIES:
        values = [row["accuracy"] for row in rows if row["hypothesis"] == "H2" and row["x"] == density]
        h2_means.append(100 * sum(values) / len(values))
    axes[1].plot([100 * density for density in DENSITIES], h2_means, "s-", color="#dc2626")
    axes[1].set(xlabel="Cross-key SET density (%)", ylabel="Accuracy (%)", title="H2: update interference", ylim=(0, 105))
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(args.output_dir / "hypothesis_pilot.png", dpi=180)
    plt.close(figure)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("results/a3/checkpoints/baseline.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/a4"))
    parser.add_argument("--device", choices=["cpu", "cuda"])
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))
