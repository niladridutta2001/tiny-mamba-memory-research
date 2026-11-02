"""Run the frozen seed-42 H7a comparison and diagnostic probes."""

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

from h7a_timescale_audit import audit
from h7a_utils import (
    DELAYS,
    evaluate_probe,
    extract_query_features,
    fit_probe,
    load_checkpoint,
    make_retention_examples,
    paired_bootstrap,
    probe_inputs,
)


MODEL_SPECS = {
    "M0_original": Path("results/extended/seed42/u1500/baseline.pt"),
    "M1a_constrained": Path("results/h7a/checkpoints/M1_fast_slow/baseline.pt"),
    "M1b_unconstrained": Path("results/h7a/checkpoints/M2_unconstrained/baseline.pt"),
}
COLORS = {
    "M0_original": "#334155",
    "M1a_constrained": "#2563eb",
    "M1b_unconstrained": "#d97706",
}


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
    cache_path: Path,
    model: torch.nn.Module,
    delay: int,
    contexts: int,
    seed: int,
    device: torch.device,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    if cache_path.exists():
        return torch.load(cache_path, map_location="cpu", weights_only=False)
    data = extract_query_features(model, make_retention_examples(delay, contexts, seed), device, batch_size)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, cache_path)
    return data


@torch.no_grad()
def masked_accuracy(
    model: torch.nn.Module,
    examples: tuple[list[list[int]], list[int], list[int]],
    mode: str | None,
    device: torch.device,
    batch_size: int,
) -> float:
    sequences, labels, _ = examples
    model.set_readout_mode_mask(mode)
    predictions = []
    try:
        for start in range(0, len(sequences), batch_size):
            inputs = torch.tensor(sequences[start : start + batch_size], dtype=torch.long, device=device)
            predictions.append(model(inputs)[:, -1].argmax(-1).cpu())
    finally:
        model.set_readout_mode_mask(None)
    return torch.cat(predictions).eq(torch.tensor(labels)).float().mean().item()


def plot_original_head(rows: list[dict], output: Path) -> None:
    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    positions = np.arange(len(DELAYS))
    for model in MODEL_SPECS:
        selected = [row for row in rows if row["model"] == model and row["measurement"] == "original_head"]
        axis.plot(
            positions,
            [100 * row["accuracy"] for row in selected],
            "o-",
            color=COLORS[model],
            label=model.replace("_", " "),
        )
    axis.axhline(25, color="#64748b", linestyle=":", label="Chance")
    axis.set(xlabel="Intervening NOOP steps", ylabel="Final QUERY accuracy (%)", ylim=(0, 105))
    axis.set_xticks(positions, [str(value) for value in DELAYS])
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output, dpi=200)
    plt.close(figure)


def plot_probe_accuracy(rows: list[dict], output: Path) -> None:
    figure, axes = plt.subplots(1, len(MODEL_SPECS), figsize=(14.5, 4.7), sharey=True)
    for axis, model in zip(axes, MODEL_SPECS):
        for feature, label, marker in (("fast_long_probe", "Group 1", "o-"), ("slow_long_probe", "Group 2", "s-"), ("full_long_probe", "Full state", "^-")):
            selected = [row for row in rows if row["model"] == model and row["measurement"] == feature]
            axis.plot([row["delay"] for row in selected], [100 * row["accuracy"] for row in selected], marker, label=label)
        axis.axhline(25, color="#64748b", linestyle=":")
        axis.set(title=model.replace("_", " "), xlabel="NOOP delay", xscale="symlog", ylim=(0, 105))
        axis.set_xticks((128, 256, 511), ("128", "256", "511"))
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Held-out long-trained probe accuracy (%)")
    axes[-1].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output, dpi=200)
    plt.close(figure)


def plot_decisive_summary(rows: list[dict], timescales: list[dict], output: Path) -> None:
    models = list(MODEL_SPECS)
    survival = []
    slow_probe = []
    original = []
    for model in models:
        survival.append(next(row["survival_511"] for row in timescales if row["model"] == model and row["event"] == "NOOP" and row["group"] == "slow"))
        slow_probe.append(next(row["accuracy"] for row in rows if row["model"] == model and row["measurement"] == "slow_long_probe" and row["delay"] == 511))
        original.append(next(row["accuracy"] for row in rows if row["model"] == model and row["measurement"] == "original_head" and row["delay"] == 511))
    figure, axes = plt.subplots(1, 3, figsize=(14.2, 4.5))
    labels = ("Group-2 state\nsurvival at 511", "Group-2 long-probe\naccuracy at 511", "Original-head\naccuracy at 511")
    values = (survival, slow_probe, original)
    for axis, label, metric in zip(axes, labels, values):
        positions = np.arange(len(models))
        axis.bar(positions, np.asarray(metric) * 100, color=[COLORS[name] for name in models])
        axis.set_xticks(positions, [name.split("_")[0] for name in models])
        axis.set_title(label)
        axis.set_ylim(0, 105)
        axis.grid(axis="y", alpha=0.2)
        for position, value in zip(positions, metric):
            axis.text(position, 100 * value + 2, f"{100 * value:.1f}%", ha="center")
    axes[0].set_ylabel("Percent")
    axes[1].axhline(25, color="#64748b", linestyle=":")
    axes[2].axhline(25, color="#64748b", linestyle=":")
    figure.tight_layout()
    figure.savefig(output, dpi=200)
    plt.close(figure)


def plot_masking(rows: list[dict], output: Path) -> None:
    figure, axes = plt.subplots(1, len(MODEL_SPECS), figsize=(14.5, 4.5), sharey=True)
    for axis, model in zip(axes, MODEL_SPECS):
        for mode, label, marker in (("none", "All modes", "o-"), ("fast_only", "Group-1 readout only", "s--"), ("slow_only", "Group-2 readout only", "^--")):
            selected = [row for row in rows if row["model"] == model and row["mask"] == mode]
            axis.plot([row["delay"] for row in selected], [100 * row["accuracy"] for row in selected], marker, label=label)
        axis.axhline(25, color="#64748b", linestyle=":")
        axis.set(title=model.replace("_", " "), xlabel="NOOP delay", ylim=(0, 105))
        axis.set_xticks((128, 256, 511))
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Original-head accuracy (%)")
    axes[-1].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output, dpi=200)
    plt.close(figure)


def plot_resources(rows: list[dict], output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
    labels = [row["model"].split("_")[0] for row in rows]
    colors = [COLORS[row["model"]] for row in rows]
    axes[0].bar(labels, [float(row["training_seconds"]) for row in rows], color=colors)
    axes[0].set(ylabel="Training time (seconds)", title="1,500 matched updates")
    axes[1].bar(labels, [float(row["peak_gpu_memory_bytes"]) / 2**20 for row in rows], color=colors)
    axes[1].set(ylabel="Peak allocated GPU memory (MiB)", title="Recorded allocation")
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output, dpi=200)
    plt.close(figure)


def plot_timescales(rows: list[dict], survival_output: Path, modes_output: Path) -> None:
    steps = np.arange(0, 512)
    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    for model in MODEL_SPECS:
        for group, display, linestyle in (("slow", "group 2", "-"), ("fast", "group 1", "--")):
            row = next(item for item in rows if item["model"] == model and item["event"] == "NOOP" and item["group"] == group)
            survival = np.exp(float(row["mean_log_retention"]) * steps)
            axis.plot(steps, survival, linestyle, color=COLORS[model], label=f"{model.split('_')[0]} {display}")
    axis.set(xlabel="Passive NOOP steps", ylabel="Estimated surviving contribution", yscale="log", ylim=(1e-12, 1.1))
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, ncol=2)
    figure.tight_layout()
    figure.savefig(survival_output, dpi=200)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9.0, 4.8))
    positions = np.arange(8)
    width = 0.19
    for offset, model in enumerate(MODEL_SPECS):
        values = [
            -float(next(item["mean_log_retention"] for item in rows if item["model"] == model and item["event"] == "NOOP" and item["group"] == f"mode_{mode}"))
            for mode in range(8)
        ]
        centered_offset = offset - (len(MODEL_SPECS) - 1) / 2
        axis.bar(positions + centered_offset * width, values, width, color=COLORS[model], label=model.split("_")[0])
    axis.axvline(3.5, color="#64748b", linewidth=1)
    axis.set(
        xlabel="SSM mode index (group 1: 0-3; group 2: 4-7)",
        ylabel="Effective decay magnitude: -mean log retention per NOOP",
        yscale="log",
    )
    axis.set_xticks(positions)
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(modes_output, dpi=200)
    plt.close(figure)


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = args.output_dir / "plots"
    cache_dir = args.output_dir / "probes" / "cache"
    plots_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    masking_rows: list[dict] = []
    timescale_rows: list[dict] = []
    predictions: dict[tuple[str, int], np.ndarray] = {}
    model_metadata: list[dict] = []

    for model_index, (model_name, checkpoint) in enumerate(MODEL_SPECS.items()):
        model, payload = load_checkpoint(checkpoint, device)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        training_record = checkpoint.parent / "baseline_validation.json"
        recorded = json.loads(training_record.read_text(encoding="utf-8")) if training_record.exists() else {}
        training_seconds = recorded.get("training", {}).get("elapsed_seconds", "")
        peak_memory = recorded.get("peak_gpu_memory_bytes", "")
        if model_name == "M0_original":
            baseline_records = (
                Path("results/a3/checkpoints/baseline_validation.json"),
                Path("results/extended/seed42/u600/baseline_validation.json"),
                Path("results/extended/seed42/u1000/baseline_validation.json"),
                Path("results/extended/seed42/u1500/baseline_validation.json"),
            )
            loaded_records = [json.loads(path.read_text(encoding="utf-8")) for path in baseline_records]
            training_seconds = sum(item["training"]["elapsed_seconds"] for item in loaded_records)
            peak_memory = max(item["peak_gpu_memory_bytes"] for item in loaded_records)
        model_metadata.append(
            {
                "model": model_name,
                "checkpoint": str(checkpoint),
                "parameters": parameter_count,
                "validation_accuracy": payload.get("validation", {}).get("query_accuracy", ""),
                "training_seconds": training_seconds,
                "peak_gpu_memory_bytes": peak_memory,
            }
        )

        audited = audit(checkpoint, args.data_dir / "val.jsonl.gz", device, args.audit_batches)
        for row in audited:
            timescale_rows.append({"model": model_name, **row})

        short_train = combine([
            cached_extract(cache_dir / f"{model_name}_short_train_d{delay}.pt", model, delay, args.short_train_contexts, 71042 + delay, device, args.batch_size)
            for delay in (10, 32, 64)
        ])
        short_validation = combine([
            cached_extract(cache_dir / f"{model_name}_short_val_d{delay}.pt", model, delay, args.validation_contexts, 72042 + delay, device, args.batch_size)
            for delay in (10, 32, 64)
        ])
        long_train = combine([
            cached_extract(cache_dir / f"{model_name}_long_train_d{delay}.pt", model, delay, args.long_train_contexts, 73042 + delay, device, args.batch_size)
            for delay in (128, 256, 511)
        ])
        long_validation = combine([
            cached_extract(cache_dir / f"{model_name}_long_val_d{delay}.pt", model, delay, args.validation_contexts, 74042 + delay, device, args.batch_size)
            for delay in (128, 256, 511)
        ])
        test = {
            delay: cached_extract(cache_dir / f"{model_name}_test_d{delay}.pt", model, delay, args.test_contexts, 75042 + delay, device, args.batch_size)
            for delay in DELAYS
        }

        probes: dict[str, dict[str, dict]] = {"short": {}, "long": {}}
        for feature_index, feature in enumerate(("fast", "slow", "full")):
            probes["short"][feature] = fit_probe(
                probe_inputs(short_train, feature), short_train["labels"],
                probe_inputs(short_validation, feature), short_validation["labels"],
                81042 + model_index * 20 + feature_index, device,
            )
            probes["long"][feature] = fit_probe(
                probe_inputs(long_train, feature), long_train["labels"],
                probe_inputs(long_validation, feature), long_validation["labels"],
                82042 + model_index * 20 + feature_index, device,
            )
        probes["long"]["final_feature"] = fit_probe(
            probe_inputs(long_train, "final_feature"), long_train["labels"],
            probe_inputs(long_validation, "final_feature"), long_validation["labels"],
            83042 + model_index, device,
        )

        for delay, data in test.items():
            original_correct = data["original_logits"].argmax(-1).eq(data["labels"])
            predictions[(model_name, delay)] = original_correct.numpy().astype(np.int8)
            per_class = [original_correct[data["labels"] == value].float().mean().item() for value in range(1, 5)]
            rows.append({
                "model": model_name, "delay": delay, "measurement": "original_head",
                "accuracy": original_correct.float().mean().item(), "balanced_accuracy": float(np.mean(per_class)),
                "probe_training_horizon": "joint_short_32_128",
            })
            for regime in ("short", "long"):
                for feature, probe in probes[regime].items():
                    if regime == "long" and delay < 128:
                        continue
                    result = evaluate_probe(probe, probe_inputs(data, feature), data["labels"], device)
                    rows.append({
                        "model": model_name, "delay": delay, "measurement": f"{feature}_{regime}_probe",
                        "accuracy": result["accuracy"], "balanced_accuracy": result["balanced_accuracy"],
                        "probe_training_horizon": "10_32_64" if regime == "short" else "128_256_511_diagnostic",
                    })

        for delay in (128, 256, 511):
            examples = make_retention_examples(delay, args.test_contexts, 75042 + delay)
            for mode in (None, "fast_only", "slow_only"):
                masking_rows.append({
                    "model": model_name,
                    "delay": delay,
                    "mask": "none" if mode is None else mode,
                    "accuracy": masked_accuracy(model, examples, mode, device, args.batch_size),
                })

    contrasts = []
    for delay in DELAYS:
        for first, second in (
            ("M1a_constrained", "M0_original"),
            ("M1a_constrained", "M1b_unconstrained"),
        ):
            effect, low, high = paired_bootstrap(predictions[(first, delay)], predictions[(second, delay)], 90042 + delay)
            contrasts.append({
                "delay": delay, "contrast": f"{first}_minus_{second}",
                "effect_pp": 100 * effect, "ci_low_pp": 100 * low, "ci_high_pp": 100 * high,
                "paired_examples": len(predictions[(first, delay)]),
            })

    write_csv(args.output_dir / "summary.csv", rows)
    write_csv(args.output_dir / "timescales" / "timescale_audit.csv", timescale_rows)
    write_csv(args.output_dir / "predictions" / "original_head_contrasts.csv", contrasts)
    write_csv(args.output_dir / "predictions" / "mode_masking.csv", masking_rows)
    write_csv(args.output_dir / "model_metadata.csv", model_metadata)
    np.savez_compressed(
        args.output_dir / "predictions" / "original_head_correctness.npz",
        **{f"{model}_delay_{delay}": values for (model, delay), values in predictions.items()},
    )
    plot_original_head(rows, plots_dir / "original_head_by_delay.png")
    plot_probe_accuracy(rows, plots_dir / "fast_slow_probe_by_delay.png")
    plot_decisive_summary(rows, timescale_rows, plots_dir / "decisive_pilot_summary.png")
    plot_masking(masking_rows, plots_dir / "mode_masking.png")
    plot_resources(model_metadata, plots_dir / "parameter_runtime_comparison.png")
    plot_timescales(
        timescale_rows,
        plots_dir / "effective_survival_by_step.png",
        plots_dir / "noop_log_retention_by_mode.png",
    )

    enforcement = next(
        row for row in contrasts
        if row["delay"] == 511 and row["contrast"] == "M1a_constrained_minus_M1b_unconstrained"
    )
    result = {
        "scope": "Seed-42 pilot; backbones trained only on lengths 32-128; long-trained probes are diagnostic only.",
        "models": model_metadata,
        "metrics": rows,
        "timescales": timescale_rows,
        "mode_masking": masking_rows,
        "contrasts": contrasts,
        "primary_contrast_511": enforcement,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_report(result, args.output_dir / "report.md")
    return result


def write_report(result: dict, path: Path) -> None:
    def metric(model: str, measurement: str, delay: int) -> float:
        return next(row["accuracy"] for row in result["metrics"] if row["model"] == model and row["measurement"] == measurement and row["delay"] == delay)

    def survival(model: str) -> float:
        return next(row["survival_511"] for row in result["timescales"] if row["model"] == model and row["event"] == "NOOP" and row["group"] == "slow")

    enforcement = result["primary_contrast_511"]
    m1a_head = metric("M1a_constrained", "original_head", 511)
    m1b_head = metric("M1b_unconstrained", "original_head", 511)
    m1a_probe = metric("M1a_constrained", "slow_long_probe", 511)
    m1b_probe = metric("M1b_unconstrained", "slow_long_probe", 511)
    supported = enforcement["ci_low_pp"] > 0
    verdict = "supported in this pilot" if supported else "not established by this one-seed pilot"
    text = f"""# H7a seed-42 pilot: fast/slow SSM mode decomposition

## Research question

Does enforcing a persistent four-fast/four-slow decomposition preserve useful long-horizon information beyond the same separated initialization without persistent constraints? The decisive comparison is M1a versus M1b.

## Controlled architecture

All models retain one Mamba block, 64 inner channels, eight modes per channel (512 recurrent scalars), the fixed width-four causal convolution, input-dependent delta/B/C, and the original jointly trained `Linear(32,5)` head. M1a hard-constrains fast rates above 1.0 and slow rates below 0.004. M1b starts from exactly the same fast/slow decay matrix but removes the constraints. Each model has 10,149 parameters.

The slow cap was fixed from the baseline short-validation delta audit (mean NOOP delta 0.345) before M1 long-horizon evaluation. No sequence longer than 128 entered backbone training or checkpoint selection.

## Three-result diagnosis at delay 511

| Model | Group-2 survival | Group-2 long-trained probe | Original short-trained head |
|---|---:|---:|---:|
| M0 | {100*survival('M0_original'):.3g}% | {100*metric('M0_original','slow_long_probe',511):.1f}% | {100*metric('M0_original','original_head',511):.1f}% |
| M1a constrained | {100*survival('M1a_constrained'):.1f}% | {100*m1a_probe:.1f}% | {100*m1a_head:.1f}% |
| M1b unconstrained | {100*survival('M1b_unconstrained'):.1f}% | {100*m1b_probe:.1f}% | {100*m1b_head:.1f}% |

M1a-M1b original-head effect at delay 511: {enforcement['effect_pp']:+.1f} pp (paired 95% bootstrap CI [{enforcement['ci_low_pp']:.1f}, {enforcement['ci_high_pp']:.1f}], n={enforcement['paired_examples']}). Persistent constraint enforcement is **{verdict}**. Long-trained probe accuracy is a diagnostic of accessible information, not strict length generalization.

"Group 2" means modes 4--7. It is the deliberately slow group in M1a/M1b and the corresponding mode slice in M0; it is not inherently a slow group in the original baseline.

## Interpretation boundaries

- High numerical survival alone does not demonstrate semantic memory.
- A long-trained probe can expose retained information but receives supervision unavailable to the original head.
- M1a versus M1b isolates persistent constraint enforcement beyond their identical separated initialization.
- This is one training seed. Even a positive paired example-level interval is not a multi-seed architecture conclusion.

## Artifacts

Raw metrics, paired contrasts, event-specific timescale audits, mode-masking results, cached frozen features, and publication figures are stored beside this report. The next decision is whether the evidence favors a dedicated slow-memory readout (information survives but the original head fails) or a selective replacement mechanism (retention improves but overwrite behavior degrades).
"""
    path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/generated"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/h7a"))
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--short-train-contexts", type=int, default=6)
    parser.add_argument("--long-train-contexts", type=int, default=6)
    parser.add_argument("--validation-contexts", type=int, default=2)
    parser.add_argument("--test-contexts", type=int, default=8)
    parser.add_argument("--audit-batches", type=int, default=32)
    return parser.parse_args()


if __name__ == "__main__":
    summary = run(parse_args())
    print(json.dumps({"primary_contrast_511": summary["primary_contrast_511"]}, indent=2))
