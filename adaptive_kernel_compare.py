"""Evaluate adaptive causal kernel routing against matched and fixed controls."""

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
from evaluate import EVALUATION_SPLITS
from h3_multiseed import write_csv
from hypothesis_pilot import load_model
from part_b_compare import paired_bootstrap
from stale_value_diagnostic import balanced_designs, make_sequences, predict
from train import build_loader, evaluate


MODEL_PATHS = {
    "Fixed width 4": Path("results/extended/seed42/u1500/baseline.pt"),
    "Fixed width 8": Path("results/adaptive_kernel_kernel_only/seed42/fixed8/u1500/baseline.pt"),
    "Static mixture": Path("results/adaptive_kernel_kernel_only/seed42/static/u1500/baseline.pt"),
    "Adaptive mixture": Path("results/adaptive_kernel_kernel_only/seed42/adaptive/u1500/baseline.pt"),
}
WIDTHS = np.asarray([1, 2, 4, 8], dtype=np.float64)


@torch.no_grad()
def router_diagnostics(model: torch.nn.Module, device: torch.device) -> tuple[list[dict], list[dict]]:
    input_ids = torch.arange(41, device=device).view(1, -1)
    mixer = model.backbone.layers[0].mixer
    mixer.capture_conv_alpha = True
    model(input_ids)
    alpha = mixer.captured_conv_alpha[0].cpu().numpy()
    mixer.capture_conv_alpha = False
    token_rows = []
    for token_id, probabilities in enumerate(alpha):
        event = "NOOP" if token_id == 0 else "SET" if token_id <= 32 else "QUERY"
        token_rows.append(
            {
                "token_id": token_id,
                "event": event,
                **{f"prob_width_{width}": float(probabilities[index]) for index, width in enumerate(WIDTHS.astype(int))},
                "expected_width": float(probabilities @ WIDTHS),
            }
        )
    event_rows = []
    for event in ("NOOP", "SET", "QUERY"):
        selected = alpha[[row["event"] == event for row in token_rows]]
        expected = selected @ WIDTHS
        event_rows.append(
            {
                "event": event,
                **{f"mean_prob_width_{width}": float(selected[:, index].mean()) for index, width in enumerate(WIDTHS.astype(int))},
                "mean_expected_width": float(expected.mean()),
                "std_expected_width_across_tokens": float(expected.std()),
            }
        )
    return token_rows, event_rows


def make_plots(metrics: list[dict], router_events: list[dict], output_dir: Path) -> None:
    styles = {
        "Fixed width 4": ("#333333", "o", "-"),
        "Fixed width 8": ("#9467bd", "D", "-"),
        "Static mixture": ("#1f77b4", "s", "--"),
        "Adaptive mixture": ("#d62728", "^", "-"),
    }
    x = np.arange(len(POST_DELAYS))
    figure, axes = plt.subplots(1, 3, figsize=(15.2, 4.7))
    for model_name, (color, marker, linestyle) in styles.items():
        rows = sorted(
            (row for row in metrics if row["model"] == model_name),
            key=lambda row: int(row["post_delay"]),
        )
        axes[0].plot(x, [100 * float(row["overwrite_accuracy"]) for row in rows], marker=marker, linestyle=linestyle, color=color, label=model_name)
        axes[1].plot(x, [100 * float(row["stale_value_error_rate"]) for row in rows], marker=marker, linestyle=linestyle, color=color, label=model_name)
    axes[0].set(ylabel="Overwrite accuracy (%)", title="Predict new value", ylim=(0, 100))
    axes[1].set(ylabel="Stale-value error rate (%)", title="Predict obsolete value", ylim=(0, 25))
    for axis in axes[:2]:
        axis.set_xlabel("NOOPs after overwrite")
        axis.set_xticks(x, [str(delay) for delay in POST_DELAYS])
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)

    events = ("NOOP", "SET", "QUERY")
    positions = np.arange(len(events))
    bar_width = 0.34
    for offset, model_name, color in ((-bar_width / 2, "Static mixture", "#1f77b4"), (bar_width / 2, "Adaptive mixture", "#d62728")):
        rows = [row for row in router_events if row["model"] == model_name]
        rows.sort(key=lambda row: events.index(row["event"]))
        axes[2].bar(
            positions + offset,
            [float(row["mean_expected_width"]) for row in rows],
            bar_width,
            color=color,
            label=model_name,
        )
    axes[2].set(
        xticks=positions,
        xticklabels=events,
        ylabel="Expected effective width",
        title="Learned router behavior",
        ylim=(0, 8),
    )
    axes[2].grid(axis="y", alpha=0.25)
    axes[2].legend(frameon=False, fontsize=8)
    figure.suptitle("Adaptive causal kernel without retention gating: matched seed-42 comparison")
    figure.tight_layout()
    figure.savefig(output_dir / "adaptive_kernel_comparison.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.5), sharey=True)
    colors = ["#9467bd", "#2ca02c", "#d62728", "#1f77b4"]
    for axis, model_name in zip(axes, ("Static mixture", "Adaptive mixture")):
        rows = [row for row in router_events if row["model"] == model_name]
        rows.sort(key=lambda row: ("NOOP", "SET", "QUERY").index(row["event"]))
        bottom = np.zeros(3)
        for width, color in zip(WIDTHS.astype(int), colors):
            values = np.asarray([float(row[f"mean_prob_width_{width}"]) for row in rows])
            axis.bar(range(3), values, bottom=bottom, color=color, label=f"width {width}")
            bottom += values
        axis.set(xticks=range(3), xticklabels=("NOOP", "SET", "QUERY"), xlabel="Event type", title=model_name, ylim=(0, 1))
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Mean router probability")
    axes[1].legend(frameon=False, fontsize=8)
    figure.suptitle("Learned receptive-field mixture by event type")
    figure.tight_layout()
    figure.savefig(output_dir / "adaptive_kernel_router.png", dpi=180)
    plt.close(figure)


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    models = {name: load_model(path, device) for name, path in MODEL_PATHS.items()}
    validation = {
        name: float(torch.load(path, map_location="cpu", weights_only=False)["validation"]["query_accuracy"])
        for name, path in MODEL_PATHS.items()
    }
    designs = balanced_designs(args.examples, args.evaluation_seed, PRE_DELAY)
    metrics, contrasts, predictions = [], [], {}
    rng = np.random.default_rng(args.bootstrap_seed)
    for post_delay in POST_DELAYS:
        overwrite, retention, old_target, new_target = make_sequences(designs, PRE_DELAY, post_delay)
        outcomes = {}
        for model_name, model in models.items():
            overwrite_prediction = predict(model, overwrite, device, args.batch_size)
            retention_prediction = predict(model, retention, device, args.batch_size)
            predictions[(model_name, post_delay, "overwrite")] = overwrite_prediction
            predictions[(model_name, post_delay, "retention")] = retention_prediction
            correct = overwrite_prediction == new_target
            stale = overwrite_prediction == old_target
            outcomes[model_name] = {"accuracy": correct.astype(float), "stale": stale.astype(float)}
            metrics.append(
                {
                    "model": model_name,
                    "post_delay": post_delay,
                    "overwrite_accuracy": float(correct.mean()),
                    "stale_value_error_rate": float(stale.mean()),
                    "other_error_rate": float((~correct & ~stale).mean()),
                    "retention_accuracy": float((retention_prediction == old_target).mean()),
                    "validation_accuracy": validation[model_name],
                    "n_examples": args.examples,
                }
            )
        for metric in ("accuracy", "stale"):
            for comparison, left, right in (
                ("adaptive_vs_static", "Adaptive mixture", "Static mixture"),
                ("adaptive_vs_fixed4", "Adaptive mixture", "Fixed width 4"),
            ):
                effect, low, high = paired_bootstrap(outcomes[left][metric] - outcomes[right][metric], rng, args.bootstrap_draws)
                contrasts.append(
                    {"post_delay": post_delay, "metric": metric, "comparison": comparison, "effect_pp": effect, "ci_low": low, "ci_high": high}
                )

    fixed_rows = []
    for model_name, model in models.items():
        for split, (condition, length) in EVALUATION_SPLITS.items():
            result = evaluate(model, build_loader(args.data_dir / f"{split}.jsonl.gz", args.fixed_batch_size, False, 42), device)
            fixed_rows.append(
                {"model": model_name, "split": split, "condition": condition, "length": length, "query_accuracy": result["query_accuracy"], "query_count": result["query_count"]}
            )

    router_tokens, router_events = [], []
    for model_name in ("Static mixture", "Adaptive mixture"):
        token_rows, event_rows = router_diagnostics(models[model_name], device)
        router_tokens.extend({"model": model_name, **row} for row in token_rows)
        router_events.extend({"model": model_name, **row} for row in event_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "metrics.csv", metrics)
    write_csv(args.output_dir / "contrasts.csv", contrasts)
    write_csv(args.output_dir / "fixed_splits.csv", fixed_rows)
    write_csv(args.output_dir / "router_by_token.csv", router_tokens)
    write_csv(args.output_dir / "router_by_event.csv", router_events)
    np.savez_compressed(args.output_dir / "paired_predictions.npz", **{str(key): value for key, value in predictions.items()})
    make_plots(metrics, router_events, args.output_dir)
    summary = {
        "scope": "matched seed-42 kernel-only experiment; no model has an event-conditioned retention gate",
        "architectural_limitation": "The adaptive router sees the current token-side representation, not recurrent state; it cannot directly identify an overwrite.",
        "checkpoints": {name: str(path) for name, path in MODEL_PATHS.items()},
        "validation_accuracy": validation,
        "metrics": metrics,
        "contrasts": contrasts,
        "fixed_splits": fixed_rows,
        "router_by_event": router_events,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/adaptive_kernel_kernel_only/seed42/comparison"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/generated"))
    parser.add_argument("--examples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--fixed-batch-size", type=int, default=16)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--evaluation-seed", type=int, default=20261023)
    parser.add_argument("--bootstrap-seed", type=int, default=244949)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({"limitation": result["architectural_limitation"], "router_by_event": result["router_by_event"]}, indent=2))
