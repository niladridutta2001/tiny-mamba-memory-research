"""Train matched short-only models with horizon-geometry regularization."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from data.dataset import IGNORE_INDEX
from decay_only_retention_sweep import cluster_bootstrap_difference, extract, make_examples
from h5_retention_readout import original_metrics
from h5_short_horizon_control import balanced_known_value_subset, extract_query_features
from horizon_robust_readout import geometry_metrics
from task_model import StateTracker
from train import build_loader, evaluate, query_loss


VARIANTS = ("baseline", "class_conditional", "global_covariance")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def query_horizon_bins(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Return bins 0: <=8, 1: 9--32, 2: >32 for known-value queries."""
    _, length = inputs.shape
    positions = torch.arange(length, device=inputs.device).unsqueeze(0).expand_as(inputs)
    bins = torch.full_like(inputs, -1)
    for key in range(8):
        set_mask = torch.div((inputs - 1).clamp_min(0), 4, rounding_mode="floor").eq(key) & inputs.ge(1) & inputs.le(32)
        last_set = torch.cummax(torch.where(set_mask, positions, -1), dim=1).values
        query_mask = inputs.eq(33 + key) & targets.ge(1) & last_set.ge(0)
        gaps = positions - last_set
        bins = torch.where(query_mask & gaps.le(8), 0, bins)
        bins = torch.where(query_mask & gaps.gt(8) & gaps.le(32), 1, bins)
        bins = torch.where(query_mask & gaps.gt(32), 2, bins)
    return bins


def covariance(features: torch.Tensor) -> torch.Tensor:
    centered = features - features.mean(0)
    return centered.T @ centered / max(len(features) - 1, 1)


def geometry_loss(
    features: torch.Tensor,
    targets: torch.Tensor,
    bins: torch.Tensor,
    mode: str,
) -> tuple[torch.Tensor, int]:
    flat_x = features.reshape(-1, features.shape[-1])
    flat_y = targets.reshape(-1)
    flat_bins = bins.reshape(-1)
    losses: list[torch.Tensor] = []
    if mode == "class_conditional":
        for value in range(1, 5):
            class_mask = (flat_y == value) & (flat_bins >= 0)
            if int(class_mask.sum()) < 4:
                continue
            reference = flat_x[class_mask]
            reference_mean = reference.mean(0)
            reference_cov = covariance(reference)
            for horizon_bin in range(3):
                group = flat_x[class_mask & (flat_bins == horizon_bin)]
                if len(group) >= 2:
                    mean_penalty = (group.mean(0) - reference_mean).square().mean()
                    covariance_penalty = (covariance(group) - reference_cov).square().mean()
                    losses.append(mean_penalty + covariance_penalty)
    elif mode == "global_covariance":
        valid = flat_bins >= 0
        if int(valid.sum()) >= 4:
            reference_cov = covariance(flat_x[valid])
            for horizon_bin in range(3):
                group = flat_x[valid & (flat_bins == horizon_bin)]
                if len(group) >= 2:
                    losses.append((covariance(group) - reference_cov).square().mean())
    else:
        return features.new_zeros(()), 0
    return (torch.stack(losses).mean(), len(losses)) if losses else (features.new_zeros(()), 0)


@torch.no_grad()
def short_horizon_geometry(model: StateTracker, loader, device: torch.device) -> dict[str, float]:
    feature_parts, target_parts, bin_parts = [], [], []
    model.eval()
    for batch in loader:
        inputs = batch["input_ids"].to(device)
        targets = batch["targets"].to(device)
        hidden = model.hidden_states(inputs)
        bins = query_horizon_bins(inputs, targets)
        valid = (targets >= 1) & (bins >= 0)
        feature_parts.append(hidden[valid])
        target_parts.append(targets[valid])
        bin_parts.append(bins[valid])
    features = torch.cat(feature_parts)
    targets = torch.cat(target_parts)
    bins = torch.cat(bin_parts)
    class_loss, _ = geometry_loss(features, targets, bins, "class_conditional")
    global_loss, _ = geometry_loss(features, targets, bins, "global_covariance")
    return {"short_class_geometry_loss": float(class_loss), "short_global_covariance_loss": float(global_loss)}


def train_variant(
    model: StateTracker,
    variant: str,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    loader = build_loader(args.data_dir / "train.jsonl.gz", args.batch_size, True, args.seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    model.train()
    updates = 0
    ce_sum = geometry_sum = 0.0
    active_groups = 0
    started = time.perf_counter()
    while updates < args.max_updates:
        for batch in loader:
            inputs = batch["input_ids"].to(device)
            targets = batch["targets"].to(device)
            optimizer.zero_grad(set_to_none=True)
            hidden = model.hidden_states(inputs)
            logits = model.head(hidden)
            ce = query_loss(logits, targets)
            if variant == "baseline":
                regularizer, groups = hidden.new_zeros(()), 0
            else:
                bins = query_horizon_bins(inputs, targets)
                regularizer, groups = geometry_loss(hidden, targets, bins, variant)
            loss = ce + args.geometry_lambda * regularizer
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss for {variant} at update {updates + 1}")
            loss.backward()
            optimizer.step()
            ce_sum += float(ce.item())
            geometry_sum += float(regularizer.item())
            active_groups += groups
            updates += 1
            if updates >= args.max_updates:
                break
    return {
        "updates": updates,
        "mean_cross_entropy": ce_sum / updates,
        "mean_geometry_loss": geometry_sum / updates,
        "mean_active_geometry_groups": active_groups / updates,
        "elapsed_seconds": time.perf_counter() - started,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict], output: Path) -> None:
    names = [row["variant"] for row in rows]
    labels = ["Baseline", "Class-conditioned", "Global covariance"]
    positions = np.arange(len(rows))
    figure, axes = plt.subplots(1, 3, figsize=(13.8, 4.7))
    bars = axes[0].bar(positions, [100 * row["long_accuracy"] for row in rows])
    axes[0].bar_label(bars, fmt="%.1f", padding=3)
    axes[0].axhline(25, color="#6b7280", linestyle=":", label="Chance")
    axes[0].set_xticks(positions, labels, rotation=20, ha="right")
    axes[0].set(ylabel="Accuracy (%)", title="Untouched 511-step probe", ylim=(0, 105))
    axes[0].legend(frameon=False)
    width = 0.36
    axes[1].bar(positions - width / 2, [row["class_mean_shift_rms"] for row in rows], width, label="Class means")
    axes[1].bar(positions + width / 2, [row["within_class_covariance_gap"] for row in rows], width, label="Within-class covariance")
    axes[1].set_xticks(positions, labels, rotation=20, ha="right")
    axes[1].set(ylabel="Normalized mismatch", title="Short-to-long geometry")
    axes[1].legend(frameon=False)
    bars = axes[2].bar(positions, [row["mean_correct_class_margin"] for row in rows])
    axes[2].bar_label(bars, fmt="%.2f", padding=3)
    axes[2].axhline(0, color="#6b7280", linestyle=":")
    axes[2].set_xticks(positions, labels, rotation=20, ha="right")
    axes[2].set(ylabel="Mean margin", title="Correct-class margin")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Class-conditional horizon geometry regularization")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(output, dpi=180)
    plt.close(figure)


def run(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    initial = StateTracker("event_decay", d_conv=4)
    initial_state = copy.deepcopy(initial.state_dict())
    long_examples = make_examples(args.test_contexts_per_pair, args.test_seed)
    results: list[dict] = []
    training: dict[str, dict] = {}
    correct_by_variant: dict[str, np.ndarray] = {}
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for variant in VARIANTS:
        set_seed(args.seed)
        model = StateTracker("event_decay", d_conv=4).to(device)
        checkpoint_path = args.output_dir / f"{variant}.pt"
        if args.resume_existing and checkpoint_path.exists():
            payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(payload["model_state_dict"])
            training[variant] = payload["training"]
        else:
            model.load_state_dict(initial_state)
            training[variant] = train_variant(model, variant, args, device)
        validation_loader = build_loader(args.data_dir / "val.jsonl.gz", args.batch_size, False, args.seed)
        validation = evaluate(model, validation_loader, device)
        validation_geometry = short_horizon_geometry(model, validation_loader, device)
        long_data = extract(model.eval(), *long_examples[:3], 1.0, device, args.eval_batch_size)
        long_metrics = original_metrics(long_data)
        correct_by_variant[variant] = long_metrics["correct"]
        short_features = extract_query_features(model, args.data_dir / "val.jsonl.gz", device, args.eval_batch_size)
        short_balanced = balanced_known_value_subset(short_features, args.validation_per_value, args.sample_seed)
        geometry = geometry_metrics(short_balanced, long_data, model)
        results.append({
            "variant": variant,
            "validation_accuracy": validation["query_accuracy"],
            "long_accuracy": long_metrics["accuracy"],
            "long_nll": long_metrics["nll"],
            "class_mean_shift_rms": geometry["class_mean_shift_rms"],
            "within_class_covariance_gap": geometry["within_class_covariance_gap"],
            "mean_correct_class_margin": geometry["original_head_mean_margin"],
            "positive_margin_rate": geometry["original_head_positive_margin_rate"],
            **validation_geometry,
        })
        torch.save({"model_state_dict": model.state_dict(), "variant": variant, "training": training[variant]}, checkpoint_path)

    contrasts = []
    clusters = np.asarray(long_examples[3], dtype=np.int8)
    for index, variant in enumerate(VARIANTS[1:]):
        effect, low, high = cluster_bootstrap_difference(
            correct_by_variant[variant], correct_by_variant["baseline"], clusters, args.bootstrap_seed + index
        )
        contrasts.append({
            "comparison": f"{variant}_minus_baseline",
            "accuracy_effect_pp": 100 * effect,
            "ci_low_pp": 100 * low,
            "ci_high_pp": 100 * high,
            "cluster_count": 32,
        })
    write_csv(args.output_dir / "metrics.csv", results)
    write_csv(args.output_dir / "contrasts.csv", contrasts)
    np.savez_compressed(args.output_dir / "predictions.npz", labels=np.asarray(long_examples[1]), clusters=clusters, **correct_by_variant)
    plot(results, args.output_dir / "horizon_geometry_training.png")
    summary = {
        "scope": "Matched seed-42 event-decay models trained only on assignment sequences of length 32--128.",
        "objectives": {
            "baseline": "Query cross-entropy",
            "class_conditional": "Query cross-entropy plus class-conditional mean/covariance alignment across query-gap bins <=8, 9--32, >32",
            "global_covariance": "Query cross-entropy plus label-agnostic covariance alignment across the same bins",
        },
        "geometry_lambda": args.geometry_lambda,
        "training": training,
        "metrics": results,
        "contrasts": contrasts,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/generated"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/horizon_geometry_training/seed42"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-updates", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--geometry-lambda", type=float, default=1.0)
    parser.add_argument("--test-contexts-per-pair", type=int, default=8)
    parser.add_argument("--test-seed", type=int, default=53042)
    parser.add_argument("--validation-per-value", type=int, default=32)
    parser.add_argument("--sample-seed", type=int, default=56043)
    parser.add_argument("--bootstrap-seed", type=int, default=60042)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--resume-existing", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps(result, indent=2))
