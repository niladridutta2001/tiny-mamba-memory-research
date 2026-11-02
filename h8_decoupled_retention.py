"""Evaluate H8 independent erase/write gates against a parameter-matched tied control."""

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

from h7a_utils import DELAYS, load_checkpoint, make_retention_examples, paired_bootstrap, trace_query
from h7b_identity_gate import drift_statistics, overwrite_diagnostic, probe_suite


MODEL_SPECS = {
    "M1a_constrained": Path("results/h7a/checkpoints/M1_fast_slow/baseline.pt"),
    "H8_independent": Path("results/h8/checkpoints/independent/baseline.pt"),
    "H8_tied": Path("results/h8/checkpoints/tied/baseline.pt"),
}
DISPLAY = {
    "M1a_constrained": "M1a constrained",
    "H8_independent": "H8 independent r/w",
    "H8_tied": "Tied-gate control",
}
COLORS = {"M1a_constrained": "#334155", "H8_independent": "#2563eb", "H8_tied": "#d97706"}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def gate_statistics(model_name: str, model: torch.nn.Module, device: torch.device, batch_size: int) -> list[dict]:
    sequences = make_retention_examples(128, 2, 206042)[0]
    values = {(gate, event): [] for gate in ("erase", "write") for event in (
        "NOOP", "UNRELATED_SET", "FOCUS_SET", "QUERY"
    )}
    for start in range(0, len(sequences), batch_size):
        inputs = torch.tensor(sequences[start:start + batch_size], dtype=torch.long, device=device)
        traced = trace_query(model, inputs)
        if traced["erase_gate"] is None:
            continue
        masks = {
            "NOOP": inputs == 0,
            "UNRELATED_SET": ((inputs >= 1) & (inputs <= 32)),
            "FOCUS_SET": torch.zeros_like(inputs, dtype=torch.bool),
            "QUERY": (inputs >= 33) & (inputs <= 40),
        }
        masks["FOCUS_SET"][:, 8] = True
        masks["UNRELATED_SET"][:, 8] = False
        for gate_name, gate in (("erase", traced["erase_gate"]), ("write", traced["write_gate"])):
            for event, mask in masks.items():
                selected = gate[mask]
                if selected.numel():
                    values[(gate_name, event)].append(selected.detach().cpu().reshape(-1))
    rows = []
    for (gate_name, event), pieces in values.items():
        if pieces:
            joined = torch.cat(pieces)
            rows.append({
                "model": model_name, "gate": gate_name, "event": event,
                "mean_gate": joined.mean().item(), "std_gate": joined.std().item(),
                "min_gate": joined.min().item(), "max_gate": joined.max().item(),
                "values": joined.numel(),
            })
    return rows


def make_plots(metrics: list[dict], gates: list[dict], drift: list[dict], overwrite: list[dict], metadata: list[dict], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.2, 4.7))
    for model in MODEL_SPECS:
        rows = [r for r in metrics if r["model"] == model and r["measurement"] == "original_head"]
        ax.plot(DELAYS, [100*r["accuracy"] for r in rows], "o-", color=COLORS[model], label=DISPLAY[model])
    ax.axhline(25, color="#64748b", linestyle=":", label="Chance")
    ax.set(xlabel="Intervening NOOP steps", ylabel="QUERY accuracy (%)", ylim=(0, 105), xscale="log")
    ax.set_xticks(DELAYS, [str(v) for v in DELAYS]); ax.grid(axis="y", alpha=.25); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(output / "retention_accuracy.png", dpi=200); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.7), sharey=True)
    events = ("NOOP", "UNRELATED_SET", "FOCUS_SET", "QUERY")
    labels = ("NOOP", "Other-key SET", "Tracked-key SET", "QUERY")
    x = np.arange(len(events)); width = .36
    for ax, gate_name in zip(axes, ("erase", "write")):
        for offset, model in enumerate(("H8_independent", "H8_tied")):
            means = [100*next(r["mean_gate"] for r in gates if r["model"] == model and r["gate"] == gate_name and r["event"] == e) for e in events]
            bars = ax.bar(x + (offset-.5)*width, means, width, color=COLORS[model], label=DISPLAY[model])
            ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=2)
        ax.set(title=f"{gate_name.capitalize()} gate", ylabel="Mean gate (%)", ylim=(0, 105))
        ax.set_xticks(x, labels, rotation=14); ax.grid(axis="y", alpha=.2)
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(output / "event_conditioned_gates.png", dpi=200); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.6), sharex=True)
    fields = (("retention_accuracy", "Pure retention (%)"), ("overwrite_accuracy", "Overwrite accuracy (%)"),
              ("stale_value_error_rate", "Stale-value errors (%)"))
    posts = (0, 1, 4, 16, 32)
    for ax, (field, label) in zip(axes, fields):
        for model in MODEL_SPECS:
            rows = [r for r in overwrite if r["model"] == model and r["pre_delay"] == 100]
            ax.plot(posts, [100*r[field] for r in rows], "o-", color=COLORS[model], label=DISPLAY[model])
        ax.set(xlabel="NOOPs after overwrite", ylabel=label, ylim=(0, 105)); ax.grid(axis="y", alpha=.2)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(output / "overwrite_tradeoff.png", dpi=200); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.4))
    for model in MODEL_SPECS:
        rows = [r for r in drift if r["model"] == model]
        axes[0].plot([r["delay"] for r in rows], [r["cumulative_relative_drift"] for r in rows], "o-", color=COLORS[model], label=DISPLAY[model])
    axes[0].set(xlabel="NOOP steps", ylabel="Cumulative relative slow-state drift"); axes[0].grid(axis="y", alpha=.2); axes[0].legend(frameon=False, fontsize=8)
    labels = [DISPLAY[r["model"]] for r in metadata]
    axes[1].bar(labels, [r["parameters"] for r in metadata], color=[COLORS[r["model"]] for r in metadata])
    axes[1].set(ylabel="Trainable parameters"); axes[1].tick_params(axis="x", rotation=15); axes[1].grid(axis="y", alpha=.2)
    fig.tight_layout(); fig.savefig(output / "drift_and_parameters.png", dpi=200); plt.close(fig)


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    models, metadata, metrics, gates, drift, correctness = {}, [], [], [], [], {}
    for index, (name, checkpoint) in enumerate(MODEL_SPECS.items()):
        model, payload = load_checkpoint(checkpoint, device)
        for parameter in model.parameters(): parameter.requires_grad_(False)
        models[name] = model
        validation_path = checkpoint.parent / "baseline_validation.json"
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        metadata.append({"model": name, "parameters": sum(p.numel() for p in model.parameters()),
                         "validation_accuracy": payload["validation"]["query_accuracy"],
                         "training_seconds": validation["training"]["elapsed_seconds"],
                         "peak_gpu_memory_bytes": validation["peak_gpu_memory_bytes"]})
        rows, correct = probe_suite(name, model, args.output_dir / "probes" / "cache", device, args.batch_size, index)
        metrics.extend(rows)
        for delay, values in correct.items(): correctness[(name, delay)] = values
        if name.startswith("H8"):
            gates.extend(gate_statistics(name, model, device, args.batch_size))
        for delay in (128, 256, 511): drift.append(drift_statistics(name, model, delay, device, args.batch_size))
    overwrite = overwrite_diagnostic(models, device, args.batch_size, args.overwrite_examples)
    contrasts = []
    for comparator in ("M1a_constrained", "H8_tied"):
        effect, low, high = paired_bootstrap(correctness[("H8_independent", 511)], correctness[(comparator, 511)], 209042+len(contrasts))
        contrasts.append({"contrast": f"H8_independent_minus_{comparator}", "delay": 511,
                          "effect_pp": 100*effect, "ci_low_pp": 100*low, "ci_high_pp": 100*high,
                          "examples": len(correctness[("H8_independent", 511)])})
    for path, rows in (("metrics.csv", metrics), ("gate_statistics.csv", gates), ("drift.csv", drift),
                       ("overwrite.csv", overwrite), ("contrasts.csv", contrasts), ("model_metadata.csv", metadata)):
        write_csv(args.output_dir / path, rows)
    make_plots(metrics, gates, drift, overwrite, metadata, args.output_dir / "plots")
    result = {"scope": "Seed-42 pilot; all backbones trained only at lengths 32-128.", "models": metadata,
              "metrics": metrics, "gate_statistics": gates, "drift": drift, "overwrite": overwrite,
              "contrasts": contrasts}
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_report(result, args.output_dir / "report.md")
    return result


def write_report(result: dict, path: Path) -> None:
    def metric(model: str, measurement: str) -> float:
        return next(r["accuracy"] for r in result["metrics"] if r["model"] == model and r["delay"] == 511 and r["measurement"] == measurement)
    def overwrite(model: str, post: int, field: str) -> float:
        return next(r[field] for r in result["overwrite"] if r["model"] == model and r["pre_delay"] == 100 and r["post_delay"] == post)
    lines = []
    for model in MODEL_SPECS:
        validation = next(r["validation_accuracy"] for r in result["models"] if r["model"] == model)
        lines.append(
            f"| {DISPLAY[model]} | {100*validation:.2f}% | {100*metric(model, 'original_head'):.2f}% | "
            f"{100*metric(model, 'slow_short_probe'):.2f}% | {100*metric(model, 'slow_long_probe'):.2f}% | "
            f"{100*overwrite(model, 0, 'overwrite_accuracy'):.2f}% | {100*overwrite(model, 32, 'overwrite_accuracy'):.2f}% | "
            f"{100*overwrite(model, 32, 'stale_value_error_rate'):.2f}% |"
        )
    first, second = result["contrasts"]
    text = f"""# H8 seed-42 pilot: decoupled retention and selective replacement

The four slow modes use separate channel-wise erase and write gates conditioned on the current convolution feature, just-updated fast state, and event embedding. The tied control retains the same parameters but enforces `r_t = w_t` during computation.

| Model | Validation | Delay-511 head | Slow short probe | Slow long probe | Immediate overwrite | Post-32 overwrite | Post-32 SVER |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(lines)}

Independent H8 minus M1a at delay 511: {first['effect_pp']:+.2f} pp (paired 95% bootstrap CI [{first['ci_low_pp']:.2f}, {first['ci_high_pp']:.2f}]). Independent H8 minus the parameter-matched tied control: {second['effect_pp']:+.2f} pp (CI [{second['ci_low_pp']:.2f}, {second['ci_high_pp']:.2f}], n={second['examples']}).

The seed-42 pilot falsifies H8 as proposed: independent erase/write gates improve overwrite behavior but reduce long-gap retention and decodability. The tied control is the strongest tested variant. Long-trained probes remain diagnostic, and a multi-seed experiment is required for an architecture-level conclusion.
"""
    path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/h8"))
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--overwrite-examples", type=int, default=192)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({"contrasts": result["contrasts"]}, indent=2))
