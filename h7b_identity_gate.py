"""Evaluate identity-default slow-state gating and its static control."""

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

from h7a_utils import (
    DELAYS,
    evaluate_probe,
    extract_query_features,
    fit_probe,
    load_checkpoint,
    make_retention_examples,
    paired_bootstrap,
    probe_inputs,
    trace_query,
)
from stale_value_diagnostic import balanced_designs, make_sequences, predict


MODEL_SPECS = {
    "M1a_constrained": Path("results/h7a/checkpoints/M1_fast_slow/baseline.pt"),
    "H7b_adaptive": Path("results/h7b/checkpoints/adaptive/baseline.pt"),
    "H7b_static": Path("results/h7b/checkpoints/static/baseline.pt"),
}
COLORS = {
    "M1a_constrained": "#334155",
    "H7b_adaptive": "#2563eb",
    "H7b_static": "#d97706",
}
DISPLAY = {
    "M1a_constrained": "M1a constrained",
    "H7b_adaptive": "H7b adaptive",
    "H7b_static": "Static control",
}
PRE_DELAYS = (32, 100, 256)
POST_DELAYS = (0, 1, 4, 16, 32)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def combine(parts: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.cat([part[key] for part in parts]) for key in parts[0]}


def cached_extract(
    path: Path,
    model: torch.nn.Module,
    delay: int,
    contexts: int,
    seed: int,
    device: torch.device,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=False)
    values = extract_query_features(model, make_retention_examples(delay, contexts, seed), device, batch_size)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(values, path)
    return values


def probe_suite(
    model_name: str,
    model: torch.nn.Module,
    cache_dir: Path,
    device: torch.device,
    batch_size: int,
    model_index: int,
) -> tuple[list[dict], dict[int, np.ndarray]]:
    short_train = combine([
        cached_extract(cache_dir / f"{model_name}_short_train_d{delay}.pt", model, delay, 6, 91042 + delay, device, batch_size)
        for delay in (10, 32, 64)
    ])
    short_validation = combine([
        cached_extract(cache_dir / f"{model_name}_short_val_d{delay}.pt", model, delay, 2, 92042 + delay, device, batch_size)
        for delay in (10, 32, 64)
    ])
    long_train = combine([
        cached_extract(cache_dir / f"{model_name}_long_train_d{delay}.pt", model, delay, 6, 93042 + delay, device, batch_size)
        for delay in (128, 256, 511)
    ])
    long_validation = combine([
        cached_extract(cache_dir / f"{model_name}_long_val_d{delay}.pt", model, delay, 2, 94042 + delay, device, batch_size)
        for delay in (128, 256, 511)
    ])
    test = {
        delay: cached_extract(cache_dir / f"{model_name}_test_d{delay}.pt", model, delay, 8, 95042 + delay, device, batch_size)
        for delay in DELAYS
    }
    probes = {"short": {}, "long": {}}
    for feature_index, feature in enumerate(("fast", "slow", "full")):
        probes["short"][feature] = fit_probe(
            probe_inputs(short_train, feature), short_train["labels"],
            probe_inputs(short_validation, feature), short_validation["labels"],
            101042 + model_index * 20 + feature_index, device,
        )
        probes["long"][feature] = fit_probe(
            probe_inputs(long_train, feature), long_train["labels"],
            probe_inputs(long_validation, feature), long_validation["labels"],
            102042 + model_index * 20 + feature_index, device,
        )

    rows: list[dict] = []
    correctness: dict[int, np.ndarray] = {}
    for delay, data in test.items():
        original = data["original_logits"].argmax(-1).eq(data["labels"])
        correctness[delay] = original.numpy().astype(np.int8)
        rows.append({
            "model": model_name,
            "delay": delay,
            "measurement": "original_head",
            "accuracy": original.float().mean().item(),
            "training_horizon": "backbone_32_128",
        })
        for regime, fitted in probes.items():
            for feature, probe in fitted.items():
                if regime == "long" and delay < 128:
                    continue
                result = evaluate_probe(probe, probe_inputs(data, feature), data["labels"], device)
                rows.append({
                    "model": model_name,
                    "delay": delay,
                    "measurement": f"{feature}_{regime}_probe",
                    "accuracy": result["accuracy"],
                    "training_horizon": "10_32_64" if regime == "short" else "128_256_511_diagnostic",
                })
    return rows, correctness


@torch.no_grad()
def gate_statistics(model_name: str, model: torch.nn.Module, device: torch.device, batch_size: int) -> list[dict]:
    examples = make_retention_examples(128, 2, 106042)
    sequences = examples[0]
    captured: dict[str, list[torch.Tensor]] = {"NOOP": [], "SET": [], "QUERY": []}
    for start in range(0, len(sequences), batch_size):
        inputs = torch.tensor(sequences[start : start + batch_size], dtype=torch.long, device=device)
        traced = trace_query(model, inputs)
        gate = traced["slow_gate"]
        if gate is None:
            continue
        event_masks = {
            "NOOP": inputs == 0,
            "SET": (inputs >= 1) & (inputs <= 32),
            "QUERY": (inputs >= 33) & (inputs <= 40),
        }
        for event, mask in event_masks.items():
            captured[event].append(gate[mask].detach().cpu().reshape(-1))
    rows = []
    for event, pieces in captured.items():
        if not pieces:
            continue
        values = torch.cat(pieces)
        rows.append({
            "model": model_name,
            "event": event,
            "mean_gate": values.mean().item(),
            "std_gate": values.std().item(),
            "min_gate": values.min().item(),
            "max_gate": values.max().item(),
            "values": values.numel(),
        })
    return rows


@torch.no_grad()
def drift_statistics(
    model_name: str,
    model: torch.nn.Module,
    delay: int,
    device: torch.device,
    batch_size: int,
) -> dict:
    sequences = make_retention_examples(delay, 2, 107042 + delay)[0]
    cumulative, net, mean_step = [], [], []
    for start in range(0, len(sequences), batch_size):
        inputs = torch.tensor(sequences[start : start + batch_size], dtype=torch.long, device=device)
        states = trace_query(model, inputs)["state_sequence"][:, :, :, 4:].flatten(2)
        after_set = states[:, 8]
        noop_states = states[:, 9 : 9 + delay]
        previous = torch.cat((after_set.unsqueeze(1), noop_states[:, :-1]), dim=1)
        step = torch.linalg.vector_norm(noop_states - previous, dim=-1) / (
            torch.linalg.vector_norm(previous, dim=-1) + 1e-8
        )
        cumulative.append(step.sum(-1).cpu())
        mean_step.append(step.mean(-1).cpu())
        net.append(
            (
                torch.linalg.vector_norm(noop_states[:, -1] - after_set, dim=-1)
                / (torch.linalg.vector_norm(after_set, dim=-1) + 1e-8)
            ).cpu()
        )
    return {
        "model": model_name,
        "delay": delay,
        "mean_step_relative_drift": torch.cat(mean_step).mean().item(),
        "cumulative_relative_drift": torch.cat(cumulative).mean().item(),
        "net_relative_displacement": torch.cat(net).mean().item(),
        "examples": len(sequences),
    }


def overwrite_diagnostic(models: dict[str, torch.nn.Module], device: torch.device, batch_size: int, examples: int) -> list[dict]:
    rows = []
    for pre_delay in PRE_DELAYS:
        designs = balanced_designs(examples, 108042, pre_delay)
        for post_delay in POST_DELAYS:
            overwrite, retention, old_target, new_target = make_sequences(designs, pre_delay, post_delay)
            for model_name, model in models.items():
                overwrite_prediction = predict(model, overwrite, device, batch_size)
                retention_prediction = predict(model, retention, device, batch_size)
                rows.append({
                    "model": model_name,
                    "pre_delay": pre_delay,
                    "post_delay": post_delay,
                    "retention_accuracy": float((retention_prediction == old_target).mean()),
                    "overwrite_accuracy": float((overwrite_prediction == new_target).mean()),
                    "stale_value_error_rate": float((overwrite_prediction == old_target).mean()),
                    "examples": len(designs),
                })
    return rows


def plots(metrics: list[dict], gates: list[dict], drift: list[dict], overwrite: list[dict], metadata: list[dict], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8.4, 4.8))
    positions = np.arange(len(DELAYS))
    for model in MODEL_SPECS:
        selected = [r for r in metrics if r["model"] == model and r["measurement"] == "original_head"]
        axis.plot(positions, [100 * r["accuracy"] for r in selected], "o-", color=COLORS[model], label=DISPLAY[model])
    axis.axhline(25, color="#64748b", linestyle=":", label="Chance")
    axis.set(xlabel="Intervening NOOP steps", ylabel="Original-head QUERY accuracy (%)", ylim=(0, 105))
    axis.set_xticks(positions, [str(x) for x in DELAYS])
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output / "retention_accuracy.png", dpi=200)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), sharey=True)
    features = ("fast", "slow", "full")
    x = np.arange(len(features))
    width = 0.24
    for axis, regime in zip(axes, ("short", "long")):
        for offset, model in enumerate(MODEL_SPECS):
            values = [
                next(100 * r["accuracy"] for r in metrics if r["model"] == model and r["delay"] == 511 and r["measurement"] == f"{feature}_{regime}_probe")
                for feature in features
            ]
            axis.bar(x + (offset - 1) * width, values, width, color=COLORS[model], label=DISPLAY[model])
        axis.axhline(25, color="#64748b", linestyle=":")
        axis.set(title=f"{regime.capitalize()}-trained probes", ylabel="Held-out accuracy (%)", ylim=(0, 105))
        axis.set_xticks(x, ("Fast state", "Slow state", "Full state"))
        axis.grid(axis="y", alpha=0.2)
    axes[1].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "probe_localization_511.png", dpi=200)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12.6, 4.7))
    events = ("NOOP", "SET", "QUERY")
    x = np.arange(len(events))
    all_gate_values = []
    for offset, model in enumerate(("H7b_adaptive", "H7b_static")):
        values = [next(100 * r["mean_gate"] for r in gates if r["model"] == model and r["event"] == event) for event in events]
        all_gate_values.extend(values)
        bars = axes[0].bar(x + (offset - 0.5) * 0.34, values, 0.34, color=COLORS[model], label=DISPLAY[model])
        axes[0].bar_label(bars, fmt="%.2f", padding=2, fontsize=8)
    axes[0].set(title="Learned slow-update gate", ylabel="Mean gate (%)", ylim=(0, max(all_gate_values) * 1.35))
    axes[0].set_xticks(x, events)
    axes[0].grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False)
    for model in MODEL_SPECS:
        selected = [r for r in drift if r["model"] == model]
        axes[1].plot([r["delay"] for r in selected], [r["cumulative_relative_drift"] for r in selected], "o-", color=COLORS[model], label=DISPLAY[model])
    axes[1].set(title="Accumulated passive slow-state drift", xlabel="NOOP steps", ylabel="Sum of relative step displacement")
    axes[1].set_xticks((128, 256, 511))
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "gate_and_drift.png", dpi=200)
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.7), sharex=True)
    fields = (
        ("retention_accuracy", "Pure retention accuracy (%)"),
        ("overwrite_accuracy", "Overwrite accuracy (%)"),
        ("stale_value_error_rate", "Stale-value error rate (%)"),
    )
    x = np.arange(len(POST_DELAYS))
    for axis, (field, ylabel) in zip(axes, fields):
        for model in MODEL_SPECS:
            selected = [r for r in overwrite if r["model"] == model and r["pre_delay"] == 100]
            axis.plot(x, [100 * r[field] for r in selected], "o-", color=COLORS[model], label=DISPLAY[model])
        axis.set(xlabel="NOOPs after overwrite", ylabel=ylabel, ylim=(0, 105))
        axis.set_xticks(x, [str(v) for v in POST_DELAYS])
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "overwrite_tradeoff.png", dpi=200)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
    labels = [DISPLAY[r["model"]] for r in metadata]
    colors = [COLORS[r["model"]] for r in metadata]
    axes[0].bar(labels, [r["training_seconds"] for r in metadata], color=colors)
    axes[0].set(title="1,500 matched updates", ylabel="Training time (seconds)")
    axes[1].bar(labels, [r["peak_gpu_memory_bytes"] / 2**20 for r in metadata], color=colors)
    axes[1].set(title="Recorded allocation", ylabel="Peak allocated GPU memory (MiB)")
    for axis in axes:
        axis.tick_params(axis="x", rotation=15)
        axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output / "resource_comparison.png", dpi=200)
    plt.close(figure)


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / "probes" / "cache"
    models = {}
    metadata = []
    metrics, gates, drift = [], [], []
    correctness = {}
    for model_index, (model_name, checkpoint) in enumerate(MODEL_SPECS.items()):
        model, payload = load_checkpoint(checkpoint, device)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        models[model_name] = model
        record = json.loads((checkpoint.parent / "baseline_validation.json").read_text(encoding="utf-8"))
        metadata.append({
            "model": model_name,
            "parameters": sum(p.numel() for p in model.parameters()),
            "validation_accuracy": payload["validation"]["query_accuracy"],
            "training_seconds": record["training"]["elapsed_seconds"],
            "peak_gpu_memory_bytes": record["peak_gpu_memory_bytes"],
        })
        model_rows, model_correctness = probe_suite(model_name, model, cache_dir, device, args.batch_size, model_index)
        metrics.extend(model_rows)
        for delay, values in model_correctness.items():
            correctness[(model_name, delay)] = values
        if model_name != "M1a_constrained":
            gates.extend(gate_statistics(model_name, model, device, args.batch_size))
        for delay in (128, 256, 511):
            drift.append(drift_statistics(model_name, model, delay, device, args.batch_size))

    overwrite = overwrite_diagnostic(models, device, args.batch_size, args.overwrite_examples)
    contrasts = []
    for comparator in ("M1a_constrained", "H7b_static"):
        effect, low, high = paired_bootstrap(
            correctness[("H7b_adaptive", 511)], correctness[(comparator, 511)], 109042 + len(contrasts)
        )
        contrasts.append({
            "contrast": f"H7b_adaptive_minus_{comparator}",
            "delay": 511,
            "effect_pp": 100 * effect,
            "ci_low_pp": 100 * low,
            "ci_high_pp": 100 * high,
            "examples": len(correctness[("H7b_adaptive", 511)]),
        })

    write_csv(args.output_dir / "metrics.csv", metrics)
    write_csv(args.output_dir / "gate_statistics.csv", gates)
    write_csv(args.output_dir / "drift.csv", drift)
    write_csv(args.output_dir / "overwrite.csv", overwrite)
    write_csv(args.output_dir / "contrasts.csv", contrasts)
    write_csv(args.output_dir / "model_metadata.csv", metadata)
    plots(metrics, gates, drift, overwrite, metadata, args.output_dir / "plots")
    result = {
        "scope": "Seed-42 pilot; all backbones trained only on assignment lengths 32-128.",
        "models": metadata,
        "metrics": metrics,
        "gate_statistics": gates,
        "drift": drift,
        "overwrite": overwrite,
        "contrasts": contrasts,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_report(result, args.output_dir / "report.md")
    return result


def write_report(result: dict, path: Path) -> None:
    def metric(model: str, measurement: str, delay: int = 511) -> float:
        return next(r["accuracy"] for r in result["metrics"] if r["model"] == model and r["measurement"] == measurement and r["delay"] == delay)

    def gate(model: str, event: str) -> float:
        return next(r["mean_gate"] for r in result["gate_statistics"] if r["model"] == model and r["event"] == event)

    def drift(model: str) -> float:
        return next(r["cumulative_relative_drift"] for r in result["drift"] if r["model"] == model and r["delay"] == 511)

    def overwrite(model: str, post: int, field: str) -> float:
        return next(
            r[field] for r in result["overwrite"]
            if r["model"] == model and r["pre_delay"] == 100 and r["post_delay"] == post
        )

    primary = result["contrasts"][0]
    control = result["contrasts"][1]
    rows = []
    for model in MODEL_SPECS:
        rows.append(
            f"| {DISPLAY[model]} | {100*metric(model, 'original_head'):.1f}% | "
            f"{100*metric(model, 'slow_short_probe'):.1f}% | {100*metric(model, 'slow_long_probe'):.1f}% | {drift(model):.3f} |"
        )
    text = f"""# H7b seed-42 pilot: identity-default slow-state updates

## Design

H7b keeps M1a's four constrained fast and four constrained slow modes, but replaces the slow recurrence with an interpolation between exact identity and the ordinary candidate update. The adaptive gate depends channel-wise on the current convolution-processed input; the static control has the same 128 additional parameters but receives an input-independent constant. Both start at sigmoid(-4)={float(torch.sigmoid(torch.tensor(-4.0))):.4f}.

## Delay-511 diagnosis

| Model | Original head | Slow short probe | Slow long probe | Cumulative NOOP drift |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

Adaptive gate means: NOOP {100*gate('H7b_adaptive','NOOP'):.2f}%, SET {100*gate('H7b_adaptive','SET'):.2f}%, QUERY {100*gate('H7b_adaptive','QUERY'):.2f}%. The diagnostic mechanism is supported only if the adaptive gate suppresses irrelevant NOOP updates relative to useful SET updates and improves the outcome over both M1a and the static control.

Original-head effect versus M1a: {primary['effect_pp']:+.1f} pp (paired 95% bootstrap CI [{primary['ci_low_pp']:.1f}, {primary['ci_high_pp']:.1f}], n={primary['examples']}). Effect versus the parameter-matched static control: {control['effect_pp']:+.1f} pp (CI [{control['ci_low_pp']:.1f}, {control['ci_high_pp']:.1f}]).

## Mandatory overwrite check: 100 NOOPs before overwrite

| Model | Immediate overwrite | Immediate stale error | Overwrite after 32 NOOPs | Stale error after 32 NOOPs |
|---|---:|---:|---:|---:|
| M1a constrained | {100*overwrite('M1a_constrained',0,'overwrite_accuracy'):.1f}% | {100*overwrite('M1a_constrained',0,'stale_value_error_rate'):.1f}% | {100*overwrite('M1a_constrained',32,'overwrite_accuracy'):.1f}% | {100*overwrite('M1a_constrained',32,'stale_value_error_rate'):.1f}% |
| H7b adaptive | {100*overwrite('H7b_adaptive',0,'overwrite_accuracy'):.1f}% | {100*overwrite('H7b_adaptive',0,'stale_value_error_rate'):.1f}% | {100*overwrite('H7b_adaptive',32,'overwrite_accuracy'):.1f}% | {100*overwrite('H7b_adaptive',32,'stale_value_error_rate'):.1f}% |
| Static control | {100*overwrite('H7b_static',0,'overwrite_accuracy'):.1f}% | {100*overwrite('H7b_static',0,'stale_value_error_rate'):.1f}% | {100*overwrite('H7b_static',32,'overwrite_accuracy'):.1f}% | {100*overwrite('H7b_static',32,'stale_value_error_rate'):.1f}% |

## Conclusion

The adaptive identity gate is a successful extreme-horizon diagnostic but not a successful replacement model in its current form. It reduces delay-511 cumulative slow-state drift from {drift('M1a_constrained'):.3f} to {drift('H7b_adaptive'):.3f} and raises original-head accuracy from {100*metric('M1a_constrained','original_head'):.1f}% to {100*metric('H7b_adaptive','original_head'):.1f}%. However, the learned event separation is small (SET minus NOOP = {100*(gate('H7b_adaptive','SET')-gate('H7b_adaptive','NOOP')):.2f} percentage points), short-distribution validation falls, and obsolete values persist much more often. The current hypothesis is therefore only partially supported: identity-default dynamics help extreme retention/readout stability, but a single coupled gate suppresses useful replacement writes too aggressively.

## Interpretation boundaries

- Long-trained probes are diagnostic and receive long-horizon supervision unavailable to the original task head.
- The static control isolates input dependence from additional gate capacity, but its redundant channel-wise weights collapse to an input-independent parameterization during computation.
- Overwrite accuracy and stale-value errors are mandatory: a retention gain is not useful if obsolete assignments cannot be replaced.
- This is one training seed; paired example-level intervals are not a multi-seed architecture conclusion.
"""
    path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/h7b"))
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--overwrite-examples", type=int, default=192)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({"contrasts": result["contrasts"]}, indent=2))
