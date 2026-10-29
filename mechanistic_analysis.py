"""Trace effective SSM retention and measure hidden-state value decodability."""

from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from generate_dataset import KEYS, encode_event, random_event
from hypothesis_pilot import BURN_IN, DELAYS, retention_sequences
from task_model import StateTracker

MODEL_SEEDS = (42, 43, 44)
T_CRITICAL_DF2_95 = 4.302652729


def mean_ci(values: list[float]) -> tuple[float, float, float]:
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    margin = T_CRITICAL_DF2_95 * math.sqrt(variance) / math.sqrt(len(values))
    return mean, mean - margin, mean + margin


def load_model(seed: int, device: torch.device) -> StateTracker:
    path = Path(f"results/extended/seed{seed}/u1500/baseline.pt")
    payload = torch.load(path, map_location=device, weights_only=False)
    model = StateTracker().to(device)
    model.load_state_dict(payload["model_state_dict"])
    return model.eval()


@torch.no_grad()
def log_retention(model: StateTracker, input_ids: torch.Tensor) -> torch.Tensor:
    """Return exact log(exp(delta*A)) for the single Mamba layer."""
    layer = model.backbone.layers[0]
    mixer = layer.mixer
    block_input = layer.norm(model.backbone.embedding(input_ids))
    projected = mixer.in_proj(block_input)
    x, _ = projected.split([mixer.args.d_inner, mixer.args.d_inner], dim=-1)
    length = x.shape[1]
    x = mixer.conv1d(x.transpose(1, 2))[:, :, :length].transpose(1, 2)
    x = F.silu(x)
    x_dbl = mixer.x_proj(x)
    delta, _, _ = x_dbl.split([mixer.args.dt_rank, mixer.args.d_state, mixer.args.d_state], dim=-1)
    delta = F.softplus(mixer.dt_proj(delta))
    state_matrix = -torch.exp(mixer.A_log.float())
    return delta.unsqueeze(-1) * state_matrix.unsqueeze(0).unsqueeze(0)


def retention_metrics(model: StateTracker, delay: int, device: torch.device) -> dict[str, float]:
    sequences, _ = retention_sequences(delay)
    inputs = torch.tensor(sequences, dtype=torch.long, device=device)
    if delay == 0:
        cumulative_log10 = torch.zeros((len(sequences), 64, 8), device=device)
    else:
        per_step_log = log_retention(model, inputs)
        start = BURN_IN + 1
        cumulative_log10 = per_step_log[:, start : start + delay].sum(dim=1) / math.log(10)
    flattened = cumulative_log10.flatten().cpu()
    return {
        "median_log10_retention": float(flattened.median().item()),
        "p10_log10_retention": float(torch.quantile(flattened, 0.1).item()),
        "p90_log10_retention": float(torch.quantile(flattened, 0.9).item()),
    }


def make_probe_examples(delay: int, contexts_per_pair: int, seed: int) -> tuple[list[list[int]], list[int], list[int]]:
    rng = random.Random(seed)
    noop = encode_event(["NOOP", None, None])
    sequences, labels, keys = [], [], []
    for key_index, key in enumerate(KEYS):
        for value in range(1, 5):
            for _ in range(contexts_per_pair):
                prefix = [random_event(rng, exclude_key=key) for _ in range(BURN_IN)]
                tokens = [encode_event(event) for event in prefix]
                tokens.append(encode_event(["SET", key, value]))
                tokens.extend([noop] * delay)
                sequences.append(tokens)
                labels.append(value - 1)
                keys.append(key_index)
    return sequences, labels, keys


@torch.no_grad()
def extract_features(
    model: StateTracker, sequences: list[list[int]], keys: list[int], device: torch.device, batch_size: int = 64
) -> torch.Tensor:
    pieces = []
    for start in range(0, len(sequences), batch_size):
        batch = torch.tensor(sequences[start : start + batch_size], dtype=torch.long, device=device)
        hidden = model.hidden_states(batch)[:, -1]
        key_ids = torch.tensor(keys[start : start + batch_size], device=device)
        pieces.append(torch.cat([hidden, F.one_hot(key_ids, num_classes=8).float()], dim=-1).cpu())
    return torch.cat(pieces)


def probe_accuracy(model: StateTracker, delay: int, model_seed: int, device: torch.device) -> float:
    train_sequences, train_labels, train_keys = make_probe_examples(delay, 12, 10_000 + model_seed)
    test_sequences, test_labels, test_keys = make_probe_examples(delay, 8, 20_000 + model_seed)
    train_x = extract_features(model, train_sequences, train_keys, device).to(device)
    test_x = extract_features(model, test_sequences, test_keys, device).to(device)
    train_y = torch.tensor(train_labels, dtype=torch.long, device=device)
    test_y = torch.tensor(test_labels, dtype=torch.long, device=device)
    mean, scale = train_x[:, :32].mean(0), train_x[:, :32].std(0).clamp_min(1e-5)
    train_x[:, :32] = (train_x[:, :32] - mean) / scale
    test_x[:, :32] = (test_x[:, :32] - mean) / scale
    torch.manual_seed(30_000 + model_seed + delay)
    probe = torch.nn.Linear(40, 4).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=0.03, weight_decay=1e-4)
    for _ in range(250):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(probe(train_x), train_y)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        return float((probe(test_x).argmax(-1) == test_y).float().mean().item())


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for model_seed in MODEL_SEEDS:
        model = load_model(model_seed, device)
        pilot_path = Path(f"results/extended/seed{model_seed}/u1500/pilot/hypothesis_pilot.csv")
        with pilot_path.open(encoding="utf-8") as source:
            behavior = {int(float(row["x"])): float(row["accuracy"]) for row in csv.DictReader(source) if row["hypothesis"] == "H1"}
        for delay in DELAYS:
            retention = retention_metrics(model, delay, device)
            rows.append(
                {
                    "model_seed": model_seed,
                    "delay": delay,
                    "behavior_accuracy": behavior[delay],
                    "probe_accuracy": probe_accuracy(model, delay, model_seed, device),
                    **retention,
                }
            )

    output_dir = Path("results/mechanistic")
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "mechanistic_metrics.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summary = []
    for delay in DELAYS:
        selected = [row for row in rows if row["delay"] == delay]
        record = {"delay": delay}
        for metric in ("behavior_accuracy", "probe_accuracy", "median_log10_retention"):
            mean, lower, upper = mean_ci([row[metric] for row in selected])
            record[f"{metric}_mean"] = mean
            record[f"{metric}_ci95_lower"] = lower
            record[f"{metric}_ci95_upper"] = upper
        summary.append(record)
    (output_dir / "mechanistic_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.7))
    axes[0].plot([row["delay"] for row in summary], [row["median_log10_retention_mean"] for row in summary], "o-")
    axes[0].set(xlabel="Intervening NOOPs", ylabel="Median cumulative log10 retention")
    for metric, label, marker in (("behavior_accuracy", "Task head", "o-"), ("probe_accuracy", "Linear probe", "s-")):
        axes[1].plot([row["delay"] for row in summary], [100 * row[f"{metric}_mean"] for row in summary], marker, label=label)
    axes[1].axhline(25, linestyle="--", color="#6b7280", linewidth=1.5, label="Chance")
    axes[1].set(xlabel="Intervening NOOPs", ylabel="Accuracy (%)", ylim=(0, 105))
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.set_xscale("symlog", linthresh=4)
        axis.set_xlim(-1, 550)
        axis.set_xticks(DELAYS, [str(delay) for delay in DELAYS])
        axis.grid(axis="y", alpha=0.25)
    figure.subplots_adjust(left=0.08, right=0.98, bottom=0.18, top=0.96, wspace=0.28)
    figure.savefig(output_dir / "mechanistic_analysis.png", dpi=180)
    plt.close(figure)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
