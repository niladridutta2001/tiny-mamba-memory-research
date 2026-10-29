"""Diagnostic ablation of write injection and decay at unrelated SET events."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from h2_multiseed import DELAYS, DENSITIES, make_design, sequences_for
from h3_multiseed import cluster_bootstrap_difference, write_csv
from hypothesis_pilot import load_model


MODES = ("normal", "suppress_write", "suppress_decay")
BURN_IN = 32


@torch.no_grad()
def predict_modes(
    model: torch.nn.Module,
    sequences: list[list[int]],
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    outputs: dict[str, list[np.ndarray]] = {mode: [] for mode in MODES}
    for start in range(0, len(sequences), batch_size):
        inputs = torch.tensor(sequences[start : start + batch_size], dtype=torch.long, device=device)
        middle = torch.zeros_like(inputs, dtype=torch.bool)
        middle[:, BURN_IN + 1 : -1] = True
        unrelated_set = middle & (inputs >= 1) & (inputs <= 32)

        normal_logits = model(inputs)

        write_route = torch.ones(
            (*inputs.shape, model.backbone.args.d_inner), dtype=torch.float32, device=device
        )
        write_route[unrelated_set] = 0.0
        no_write_logits = model.head(
            model.hidden_states(inputs, write_route_override=write_route)
        )

        decay_scale = torch.ones_like(inputs, dtype=torch.float32)
        decay_scale[unrelated_set] = 0.0
        no_decay_logits = model.head(model.hidden_states(inputs, decay_scale=decay_scale))

        for mode, logits in (
            ("normal", normal_logits),
            ("suppress_write", no_write_logits),
            ("suppress_decay", no_decay_logits),
        ):
            outputs[mode].append(logits[:, -1].argmax(-1).cpu().numpy())
    return {
        mode: np.concatenate(parts).astype(np.int8) for mode, parts in outputs.items()
    }


def summarize(
    predictions: dict[tuple[int, float, str], np.ndarray],
    targets: dict[int, np.ndarray],
    draws: int,
    bootstrap_seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    metrics, per_model, contrasts = [], [], []
    rng = np.random.default_rng(bootstrap_seed)
    comparisons = (
        ("suppress_write_vs_normal", "suppress_write", "normal"),
        ("suppress_decay_vs_normal", "suppress_decay", "normal"),
        ("suppress_decay_vs_write", "suppress_decay", "suppress_write"),
    )
    for delay in DELAYS:
        target = targets[delay][None, :]
        for density in DENSITIES:
            correctness = {}
            for mode in MODES:
                correct = (predictions[(delay, density, mode)] == target).astype(np.float64)
                correctness[mode] = correct
                metrics.append(
                    {
                        "delay": delay,
                        "density": density,
                        "mode": mode,
                        "accuracy": float(correct.mean()),
                        "n_models": correct.shape[0],
                        "n_examples_per_model": correct.shape[1],
                    }
                )
                for model_index, value in enumerate(correct.mean(axis=1)):
                    per_model.append(
                        {
                            "model_index": model_index,
                            "delay": delay,
                            "density": density,
                            "mode": mode,
                            "accuracy": float(value),
                        }
                    )
            for name, intervention, baseline in comparisons:
                effect, low, high = cluster_bootstrap_difference(
                    correctness[intervention], correctness[baseline], rng, draws
                )
                contrasts.append(
                    {
                        "delay": delay,
                        "density": density,
                        "comparison": name,
                        "accuracy_effect_pp": effect,
                        "ci_low": low,
                        "ci_high": high,
                    }
                )
    return metrics, per_model, contrasts


def make_plots(metrics: list[dict], contrasts: list[dict], output_dir: Path) -> None:
    styles = {
        "normal": ("#333333", "o", "Normal scan"),
        "suppress_write": ("#1f77b4", "s", "Suppress unrelated writes"),
        "suppress_decay": ("#d62728", "^", "Suppress unrelated decay"),
    }
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 8), sharex=True, sharey=True)
    for axis, delay in zip(axes.flat, DELAYS):
        for mode in MODES:
            rows = [
                row for row in metrics
                if int(row["delay"]) == delay and row["mode"] == mode
            ]
            rows.sort(key=lambda row: float(row["density"]))
            color, marker, label = styles[mode]
            axis.plot(
                [100 * float(row["density"]) for row in rows],
                [100 * float(row["accuracy"]) for row in rows],
                marker=marker,
                color=color,
                label=label,
            )
        axis.axhline(25, color="0.55", linestyle=":", linewidth=1)
        axis.set(title=f"Fixed delay = {delay}", ylim=(0, 100))
        axis.grid(axis="y", alpha=0.25)
    axes[0, 0].legend(frameon=False, fontsize=8)
    for axis in axes[1, :]:
        axis.set_xlabel("Unrelated-SET density (%)")
    for axis in axes[:, 0]:
        axis.set_ylabel("Query accuracy (%)")
    figure.suptitle("Frozen-model diagnostic: which SET transition term causes interference?")
    figure.tight_layout()
    figure.savefig(output_dir / "term_ablation_accuracy.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8.4, 4.8))
    x = np.arange(len(DELAYS))
    for offset, comparison, color, marker, label in (
        (-0.10, "suppress_write_vs_normal", "#1f77b4", "s", "Suppress write − normal"),
        (0.10, "suppress_decay_vs_normal", "#d62728", "^", "Suppress decay − normal"),
    ):
        rows = [
            row for row in contrasts
            if row["comparison"] == comparison and float(row["density"]) == 1.0
        ]
        rows.sort(key=lambda row: int(row["delay"]))
        effect = np.asarray([float(row["accuracy_effect_pp"]) for row in rows])
        low = np.asarray([float(row["ci_low"]) for row in rows])
        high = np.asarray([float(row["ci_high"]) for row in rows])
        axis.errorbar(
            x + offset,
            effect,
            yerr=np.vstack((effect - low, high - effect)),
            fmt=marker,
            capsize=4,
            color=color,
            label=label,
        )
    axis.axhline(0, color="0.35", linewidth=1)
    axis.set(
        xticks=x,
        xticklabels=[str(delay) for delay in DELAYS],
        xlabel="Fixed SET-to-QUERY delay",
        ylabel="Accuracy change (percentage points)",
        title="Intervention effects at 100% unrelated-SET density (95% CI)",
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "term_ablation_effects.png", dpi=180)
    plt.close(figure)


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoints = sorted(args.checkpoint_root.glob("seed*/u1500/baseline.pt"))
    if len(checkpoints) < 5:
        raise RuntimeError(f"expected five checkpoints, found {len(checkpoints)}")

    datasets: dict[tuple[int, float], list[list[int]]] = {}
    targets: dict[int, np.ndarray] = {}
    for delay in DELAYS:
        designs = make_design(delay, args.examples, args.evaluation_seed)
        for density in DENSITIES:
            sequences, target, _ = sequences_for(designs, delay, density, "set")
            datasets[(delay, density)] = sequences
            targets[delay] = target

    predictions = {
        (delay, density, mode): np.empty((len(checkpoints), args.examples), dtype=np.int8)
        for delay in DELAYS for density in DENSITIES for mode in MODES
    }
    for model_index, checkpoint in enumerate(checkpoints):
        model = load_model(checkpoint, device)
        for delay in DELAYS:
            for density in DENSITIES:
                batch_predictions = predict_modes(
                    model, datasets[(delay, density)], device, args.batch_size
                )
                for mode in MODES:
                    predictions[(delay, density, mode)][model_index] = batch_predictions[mode]
        del model

    metrics, per_model, contrasts = summarize(
        predictions, targets, args.bootstrap_draws, args.bootstrap_seed
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "metrics.csv", metrics)
    write_csv(args.output_dir / "per_model_metrics.csv", per_model)
    write_csv(args.output_dir / "contrasts.csv", contrasts)
    np.savez_compressed(
        args.output_dir / "paired_predictions.npz",
        **{str(key): value for key, value in predictions.items()},
    )
    make_plots(metrics, contrasts, args.output_dir)

    summary = {
        "diagnostic_only": True,
        "intervention_definition": {
            "suppress_write": "At unrelated SET events only, set delta*B*u to zero; retain exp(delta*A), token processing, C readout, and D*u.",
            "suppress_decay": "At unrelated SET events only, set exp(delta*A) to one; retain delta*B*u, token processing, C readout, and D*u.",
        },
        "distribution_shift_caveat": "Both interventions perturb a frozen model away from its trained hidden-state distribution.",
        "checkpoints": [str(path) for path in checkpoints],
        "n_models": len(checkpoints),
        "examples_per_condition_per_model": args.examples,
        "bootstrap_draws": args.bootstrap_draws,
        "metrics": metrics,
        "contrasts": contrasts,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, default=Path("results/extended"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/interference_term_ablation")
    )
    parser.add_argument("--examples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--evaluation-seed", type=int, default=20261023)
    parser.add_argument("--bootstrap-seed", type=int, default=161803)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({key: result[key] for key in result if key not in {"metrics", "contrasts"}}, indent=2))
