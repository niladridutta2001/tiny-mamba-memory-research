"""Reset only local convolutional history at an overwrite boundary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dconv_overwrite_sweep import POST_DELAYS, PRE_DELAY
from h3_multiseed import write_csv
from hypothesis_pilot import load_model
from part_b_compare import paired_bootstrap
from stale_value_diagnostic import balanced_designs, make_sequences, predict


CHECKPOINT = Path("results/selective_retention/seed42/event_decay/u1500/baseline.pt")
OVERWRITE_POSITION = 32 + 1 + PRE_DELAY


@torch.no_grad()
def predict_with_reset(
    model: torch.nn.Module,
    sequences: list[list[int]],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    parts = []
    for start in range(0, len(sequences), batch_size):
        inputs = torch.tensor(sequences[start : start + batch_size], dtype=torch.long, device=device)
        reset = torch.zeros_like(inputs, dtype=torch.bool)
        reset[:, OVERWRITE_POSITION] = True
        logits = model.head(model.hidden_states(inputs, conv_reset=reset))
        parts.append(logits[:, -1].argmax(-1).cpu().numpy())
    return np.concatenate(parts).astype(np.int8)


def make_plot(metrics: list[dict], output_dir: Path) -> None:
    styles = {
        "Normal history": ("#d62728", "^", "-"),
        "Reset at overwrite": ("#333333", "o", "--"),
    }
    x = np.arange(len(POST_DELAYS))
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.6), sharex=True)
    for mode, (color, marker, linestyle) in styles.items():
        rows = sorted(
            (row for row in metrics if row["mode"] == mode),
            key=lambda row: int(row["post_delay"]),
        )
        axes[0].plot(
            x,
            [100 * float(row["overwrite_accuracy"]) for row in rows],
            marker=marker,
            linestyle=linestyle,
            color=color,
            label=mode,
        )
        axes[1].plot(
            x,
            [100 * float(row["stale_value_error_rate"]) for row in rows],
            marker=marker,
            linestyle=linestyle,
            color=color,
            label=mode,
        )
    axes[0].set(ylabel="Overwrite accuracy (%)", title="Predict new value", ylim=(0, 100))
    axes[1].set(ylabel="Stale-value error rate (%)", title="Predict obsolete value", ylim=(0, 25))
    for axis in axes:
        axis.set_xlabel("NOOPs after overwrite")
        axis.set_xticks(x, [str(delay) for delay in POST_DELAYS])
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)
    figure.suptitle("Frozen d_conv=4 diagnostic: reset local convolutional history only")
    figure.tight_layout()
    figure.savefig(output_dir / "conv_history_reset.png", dpi=180)
    plt.close(figure)


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_model(CHECKPOINT, device)
    designs = balanced_designs(args.examples, args.evaluation_seed, PRE_DELAY)
    metrics, contrasts, predictions = [], [], {}
    rng = np.random.default_rng(args.bootstrap_seed)
    for post_delay in POST_DELAYS:
        overwrite, _, old_target, new_target = make_sequences(designs, PRE_DELAY, post_delay)
        normal = predict(model, overwrite, device, args.batch_size)
        reset = predict_with_reset(model, overwrite, device, args.batch_size)
        predictions[(post_delay, "normal")] = normal
        predictions[(post_delay, "reset")] = reset
        outcomes = {}
        for mode, result in (("Normal history", normal), ("Reset at overwrite", reset)):
            correct = result == new_target
            stale = result == old_target
            outcomes[mode] = {"accuracy": correct.astype(float), "stale": stale.astype(float)}
            metrics.append(
                {
                    "mode": mode,
                    "post_delay": post_delay,
                    "overwrite_accuracy": float(correct.mean()),
                    "stale_value_error_rate": float(stale.mean()),
                    "other_error_rate": float((~correct & ~stale).mean()),
                    "n_examples": args.examples,
                }
            )
        for metric in ("accuracy", "stale"):
            effect, low, high = paired_bootstrap(
                outcomes["Reset at overwrite"][metric] - outcomes["Normal history"][metric],
                rng,
                args.bootstrap_draws,
            )
            contrasts.append(
                {
                    "post_delay": post_delay,
                    "metric": metric,
                    "reset_effect_pp": effect,
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
    make_plot(metrics, args.output_dir)
    summary = {
        "diagnostic_only": True,
        "checkpoint": str(CHECKPOINT),
        "intervention": "Reset only causal convolutional input history immediately before the overwrite SET; retain recurrent SSM state.",
        "distribution_shift_caveat": "The frozen model was not trained with convolution-history resets.",
        "pre_delay": PRE_DELAY,
        "overwrite_position": OVERWRITE_POSITION,
        "post_delays": POST_DELAYS,
        "examples_per_condition": args.examples,
        "metrics": metrics,
        "contrasts": contrasts,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/selective_retention/seed42/dconv_sweep/conv_reset"))
    parser.add_argument("--examples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--evaluation-seed", type=int, default=20261023)
    parser.add_argument("--bootstrap-seed", type=int, default=223607)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({key: result[key] for key in result if key not in {"metrics", "contrasts"}}, indent=2))
