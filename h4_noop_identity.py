"""Test whether learned NOOP transitions drift and causally impair retrieval."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from h3_multiseed import cluster_bootstrap_difference, retention_examples, write_csv
from hypothesis_pilot import BURN_IN, load_model


DELAYS = (10, 32, 64, 128)
EPSILON = 1e-8


@torch.no_grad()
def trace_batch(model: torch.nn.Module, inputs: torch.Tensor, delay: int) -> dict[str, np.ndarray]:
    """Return exact SSM-state and post-backbone displacement for each example."""
    layer = model.backbone.layers[0]
    mixer = layer.mixer
    hidden = model.hidden_states(inputs)
    noop_start = BURN_IN + 1
    noop_stop = noop_start + delay

    hidden_previous = hidden[:, noop_start - 1 : noop_stop - 1]
    hidden_current = hidden[:, noop_start:noop_stop]
    hidden_step = torch.linalg.vector_norm(hidden_current - hidden_previous, dim=-1) / (
        torch.linalg.vector_norm(hidden_previous, dim=-1) + EPSILON
    )
    hidden_net = torch.linalg.vector_norm(hidden[:, noop_stop - 1] - hidden[:, noop_start - 1], dim=-1) / (
        torch.linalg.vector_norm(hidden[:, noop_start - 1], dim=-1) + EPSILON
    )

    block_input = layer.norm(model.backbone.embedding(inputs))
    projected = mixer.in_proj(block_input)
    u, _ = projected.split([mixer.args.d_inner, mixer.args.d_inner], dim=-1)
    length = u.shape[1]
    u = mixer.conv1d(u.transpose(1, 2))[:, :, :length].transpose(1, 2)
    u = F.silu(u)
    projected_parameters = mixer.x_proj(u)
    delta, b_vector, _ = projected_parameters.split(
        [mixer.args.dt_rank, mixer.args.d_state, mixer.args.d_state], dim=-1
    )
    delta = F.softplus(mixer.dt_proj(delta))
    state_matrix = -torch.exp(mixer.A_log.float())
    delta_a = torch.exp(delta.unsqueeze(-1) * state_matrix.unsqueeze(0).unsqueeze(0))
    delta_b_u = delta.unsqueeze(-1) * b_vector.unsqueeze(2) * u.unsqueeze(-1)

    state = torch.zeros(
        (inputs.shape[0], mixer.args.d_inner, mixer.args.d_state),
        device=inputs.device,
        dtype=delta_a.dtype,
    )
    state_after_set = None
    recurrent_steps = []
    for timestep in range(noop_stop):
        previous = state
        state = delta_a[:, timestep] * state + delta_b_u[:, timestep]
        if timestep == noop_start - 1:
            state_after_set = state.clone()
        elif timestep >= noop_start:
            recurrent_steps.append(
                torch.linalg.vector_norm((state - previous).flatten(1), dim=-1)
                / (torch.linalg.vector_norm(previous.flatten(1), dim=-1) + EPSILON)
            )
    assert state_after_set is not None
    recurrent_step = torch.stack(recurrent_steps, dim=1)
    recurrent_net = torch.linalg.vector_norm((state - state_after_set).flatten(1), dim=-1) / (
        torch.linalg.vector_norm(state_after_set.flatten(1), dim=-1) + EPSILON
    )

    baseline_prediction = model.head(hidden[:, -1]).argmax(-1)
    # Freeze only the controlled post-SET NOOP interval. Prefix NOOPs remain
    # unchanged so the intervention cannot alter the state entering the SET.
    identity_scale = torch.ones_like(inputs, dtype=torch.float32)
    identity_scale[:, noop_start:noop_stop] = 0.0
    identity_prediction = model.head(model.hidden_states(inputs, identity_scale))[:, -1].argmax(-1)
    # The depthwise convolution spreads the SET token into the next d_conv-1
    # positions. A second intervention preserves those completion steps and
    # freezes only NOOPs after the convolutional receptive field has washed out.
    washout_scale = torch.ones_like(inputs, dtype=torch.float32)
    washout_start = noop_start + mixer.args.d_conv - 1
    washout_scale[:, washout_start:noop_stop] = 0.0
    washout_identity_prediction = model.head(model.hidden_states(inputs, washout_scale))[:, -1].argmax(-1)
    return {
        "baseline_prediction": baseline_prediction.cpu().numpy().astype(np.int8),
        "identity_prediction": identity_prediction.cpu().numpy().astype(np.int8),
        "washout_identity_prediction": washout_identity_prediction.cpu().numpy().astype(np.int8),
        "recurrent_cumulative": recurrent_step.sum(dim=1).cpu().numpy(),
        "recurrent_mean_step": recurrent_step.mean(dim=1).cpu().numpy(),
        "recurrent_net": recurrent_net.cpu().numpy(),
        "hidden_cumulative": hidden_step.sum(dim=1).cpu().numpy(),
        "hidden_mean_step": hidden_step.mean(dim=1).cpu().numpy(),
        "hidden_net": hidden_net.cpu().numpy(),
    }


def bootstrap_standardized_slope(
    drift: np.ndarray,
    errors: np.ndarray,
    rng: np.random.Generator,
    draws: int,
) -> tuple[float, float, float]:
    """Error-probability slope per one-SD drift, resampling seeds and examples."""
    seed_count, example_count = drift.shape

    def slope(x: np.ndarray, y: np.ndarray, axis: int = -1) -> np.ndarray:
        x_centered = x - x.mean(axis=axis, keepdims=True)
        y_centered = y - y.mean(axis=axis, keepdims=True)
        numerator = (x_centered * y_centered).mean(axis=axis)
        denominator = (x_centered * x_centered).mean(axis=axis)
        raw = numerator / np.maximum(denominator, 1e-12)
        return raw * x.std(axis=axis)

    observed = float(100 * np.mean([slope(drift[i], errors[i]) for i in range(seed_count)]))
    seed_boot = np.empty((seed_count, draws), dtype=np.float64)
    chunk_size = 250
    for seed_index in range(seed_count):
        for start in range(0, draws, chunk_size):
            stop = min(start + chunk_size, draws)
            indices = rng.integers(0, example_count, (stop - start, example_count))
            x = drift[seed_index, indices]
            y = errors[seed_index, indices]
            seed_boot[seed_index, start:stop] = 100 * slope(x, y, axis=1)
    seed_indices = rng.integers(0, seed_count, (draws, seed_count))
    draw_indices = np.arange(draws)[:, None]
    bootstrap = seed_boot[seed_indices, draw_indices].mean(axis=1)
    low, high = np.quantile(bootstrap, (0.025, 0.975))
    return observed, float(low), float(high)


def summarize(
    arrays: dict[tuple[int, str], np.ndarray],
    targets: dict[int, np.ndarray],
    draws: int,
    bootstrap_seed: int,
) -> tuple[list[dict], list[dict]]:
    metrics, effects = [], []
    rng = np.random.default_rng(bootstrap_seed)
    seed_count = next(iter(arrays.values())).shape[0]
    for delay in DELAYS:
        target = targets[delay][None, :]
        baseline_correct = (arrays[(delay, "baseline_prediction")] == target).astype(np.float64)
        identity_correct = (arrays[(delay, "identity_prediction")] == target).astype(np.float64)
        washout_correct = (arrays[(delay, "washout_identity_prediction")] == target).astype(np.float64)
        for model_index in range(seed_count):
            metrics.append(
                {
                    "model_index": model_index,
                    "delay": delay,
                    "baseline_accuracy": float(baseline_correct[model_index].mean()),
                    "identity_accuracy": float(identity_correct[model_index].mean()),
                    "washout_identity_accuracy": float(washout_correct[model_index].mean()),
                    **{
                        f"{name}_mean": float(arrays[(delay, name)][model_index].mean())
                        for name in (
                            "recurrent_cumulative", "recurrent_mean_step", "recurrent_net",
                            "hidden_cumulative", "hidden_mean_step", "hidden_net",
                        )
                    },
                }
            )

        accuracy_effect, accuracy_low, accuracy_high = cluster_bootstrap_difference(
            identity_correct, baseline_correct, rng, draws
        )
        washout_effect, washout_low, washout_high = cluster_bootstrap_difference(
            washout_correct, baseline_correct, rng, draws
        )
        errors = 1.0 - baseline_correct
        effect_row = {
            "delay": delay,
            "baseline_accuracy": float(baseline_correct.mean()),
            "identity_accuracy": float(identity_correct.mean()),
            "identity_effect_pp": accuracy_effect,
            "identity_ci_low": accuracy_low,
            "identity_ci_high": accuracy_high,
            "washout_identity_accuracy": float(washout_correct.mean()),
            "washout_identity_effect_pp": washout_effect,
            "washout_identity_ci_low": washout_low,
            "washout_identity_ci_high": washout_high,
        }
        for name in ("recurrent_cumulative", "recurrent_net", "hidden_cumulative", "hidden_net"):
            slope_effect, slope_low, slope_high = bootstrap_standardized_slope(
                arrays[(delay, name)], errors, rng, draws
            )
            effect_row[f"{name}_error_slope_pp_per_sd"] = slope_effect
            effect_row[f"{name}_slope_ci_low"] = slope_low
            effect_row[f"{name}_slope_ci_high"] = slope_high
            effect_row[f"{name}_median"] = float(np.median(arrays[(delay, name)]))
        effects.append(effect_row)
    return metrics, effects


def make_plot(effects: list[dict], output: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))
    delays = [int(row["delay"]) for row in effects]

    for name, label, marker, color in (
        ("recurrent_cumulative", "Selective-scan state", "o-", "#d62728"),
        ("hidden_cumulative", "Post-backbone hidden", "s-", "#1f77b4"),
    ):
        axes[0, 0].plot(delays, [float(row[f"{name}_median"]) for row in effects], marker, color=color, label=label)
    axes[0, 0].set(xlabel="NOOP delay", ylabel="Median cumulative relative displacement", title="Accumulated NOOP drift")
    axes[0, 0].set_yscale("log")
    axes[0, 0].legend(frameon=False, fontsize=8)

    axes[0, 1].plot(delays, [100 * float(row["baseline_accuracy"]) for row in effects], "o-", label="Learned NOOP")
    axes[0, 1].plot(delays, [100 * float(row["identity_accuracy"]) for row in effects], "s-", label="Forced recurrent identity")
    axes[0, 1].plot(
        delays, [100 * float(row["washout_identity_accuracy"]) for row in effects],
        "^-", label="Identity after conv washout",
    )
    axes[0, 1].axhline(25, color="0.55", linestyle=":", linewidth=1, label="chance")
    axes[0, 1].set(xlabel="NOOP delay", ylabel="Accuracy (%)", title="Causal no-update intervention", ylim=(15, 100))
    axes[0, 1].legend(frameon=False, fontsize=8)

    for offset, prefix, label, color, marker in (
        (-1.2, "identity", "Freeze all gap NOOPs", "#9467bd", "o"),
        (1.2, "washout_identity", "Freeze after conv washout", "#2ca02c", "^"),
    ):
        accuracy = np.asarray([float(row[f"{prefix}_effect_pp"]) for row in effects])
        accuracy_low = np.asarray([float(row[f"{prefix}_ci_low"]) for row in effects])
        accuracy_high = np.asarray([float(row[f"{prefix}_ci_high"]) for row in effects])
        axes[1, 0].errorbar(
            np.asarray(delays) + offset, accuracy,
            yerr=np.vstack((accuracy - accuracy_low, accuracy_high - accuracy)),
            fmt=marker, capsize=4, color=color, label=label,
        )
    axes[1, 0].axhline(0, color="0.35", linewidth=1)
    axes[1, 0].set(xlabel="NOOP delay", ylabel="Identity minus baseline (pp)", title="Accuracy effect of freezing NOOP updates")
    axes[1, 0].legend(frameon=False, fontsize=8)

    for offset, name, label, color, marker in (
        (-1.2, "recurrent_cumulative", "Selective-scan state", "#d62728", "o"),
        (1.2, "hidden_cumulative", "Post-backbone hidden", "#1f77b4", "s"),
    ):
        values = np.asarray([float(row[f"{name}_error_slope_pp_per_sd"]) for row in effects])
        low = np.asarray([float(row[f"{name}_slope_ci_low"]) for row in effects])
        high = np.asarray([float(row[f"{name}_slope_ci_high"]) for row in effects])
        axes[1, 1].errorbar(
            np.asarray(delays) + offset, values, yerr=np.vstack((values - low, high - values)),
            fmt=marker, capsize=4, color=color, label=label,
        )
    axes[1, 1].axhline(0, color="0.35", linewidth=1)
    axes[1, 1].set(
        xlabel="NOOP delay", ylabel="Error change per 1-SD drift (pp)",
        title="Within-delay drift/error association",
    )
    axes[1, 1].legend(frameon=False, fontsize=8)

    for axis in axes.flat:
        axis.set_xticks(DELAYS, [str(delay) for delay in DELAYS])
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoints = sorted(args.checkpoint_root.glob("seed*/u1500/baseline.pt"))
    if len(checkpoints) < 5:
        raise RuntimeError(f"expected five checkpoints, found {len(checkpoints)}")

    sequences, targets = {}, {}
    for delay in DELAYS:
        sequence, target, _ = retention_examples(delay, args.examples, args.evaluation_seed)
        sequences[delay], targets[delay] = sequence, target

    names = (
        "baseline_prediction", "identity_prediction", "washout_identity_prediction",
        "recurrent_cumulative", "recurrent_mean_step",
        "recurrent_net", "hidden_cumulative", "hidden_mean_step", "hidden_net",
    )
    arrays = {
        (delay, name): np.empty((len(checkpoints), args.examples), dtype=np.int8 if "prediction" in name else np.float32)
        for delay in DELAYS for name in names
    }
    for model_index, checkpoint in enumerate(checkpoints):
        model = load_model(checkpoint, device)
        for delay in DELAYS:
            pieces = {name: [] for name in names}
            for start in range(0, args.examples, args.batch_size):
                inputs = torch.tensor(
                    sequences[delay][start : start + args.batch_size], dtype=torch.long, device=device
                )
                batch = trace_batch(model, inputs, delay)
                for name in names:
                    pieces[name].append(batch[name])
            for name in names:
                arrays[(delay, name)][model_index] = np.concatenate(pieces[name])
        del model

    metrics, effects = summarize(arrays, targets, args.bootstrap_draws, args.bootstrap_seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "seed_metrics.csv", metrics)
    write_csv(args.output_dir / "effects.csv", effects)
    np.savez_compressed(args.output_dir / "per_example_metrics.npz", **{str(key): value for key, value in arrays.items()})
    make_plot(effects, args.output_dir / "h4_noop_identity.png")
    summary = {
        "checkpoints": [str(path) for path in checkpoints],
        "n_models": len(checkpoints),
        "examples_per_condition_per_model": args.examples,
        "bootstrap_draws": args.bootstrap_draws,
        "effects": effects,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, default=Path("results/extended"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/h4_noop_identity"))
    parser.add_argument("--examples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--evaluation-seed", type=int, default=20261023)
    parser.add_argument("--bootstrap-seed", type=int, default=161803)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({key: result[key] for key in result if key != "effects"}, indent=2))
