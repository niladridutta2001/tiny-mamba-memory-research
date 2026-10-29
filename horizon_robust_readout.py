"""Diagnose feature shift and train short-only decay-consistent readouts."""

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

from decay_only_retention_sweep import extract, make_examples
from h5_retention_readout import ALPHAS, evaluate_readout, fit_readout, original_metrics
from h5_short_horizon_control import balanced_known_value_subset, extract_query_features
from hypothesis_pilot import load_model


SHORT_ALPHAS = (0.0, 0.1, 0.5, 1.0)
ROBUST_CONFIGS = (
    ("baseline", "none", 0.0),
    ("symmetric_0.1", "symmetric", 0.1),
    ("symmetric_1", "symmetric", 1.0),
    ("symmetric_5", "symmetric", 5.0),
    ("teacher_0.1", "teacher", 0.1),
    ("teacher_1", "teacher", 1.0),
    ("teacher_5", "teacher", 5.0),
)


def subset(data: dict[str, torch.Tensor], indices: torch.Tensor) -> dict[str, torch.Tensor]:
    return {name: values[indices] for name, values in data.items()}


def balanced_indices(labels: torch.Tensor, per_value: int, seed: int) -> torch.Tensor:
    marker = {"labels": labels, "index": torch.arange(len(labels))}
    return balanced_known_value_subset(marker, per_value, seed)["index"]


def fit_robust_readout(
    train_by_alpha: dict[float, dict[str, torch.Tensor]],
    mode: str,
    weight: float,
    seed: int,
    device: torch.device,
) -> dict:
    natural = train_by_alpha[1.0]["query_hidden"].to(device)
    labels = train_by_alpha[1.0]["labels"].to(device)
    mean = natural.mean(0)
    scale = natural.std(0).clamp_min(1e-5)
    standardized = {
        alpha: (data["query_hidden"].to(device) - mean) / scale for alpha, data in train_by_alpha.items()
    }
    torch.manual_seed(seed)
    head = torch.nn.Linear(32, 5).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.03, weight_decay=1e-4)
    for _ in range(500):
        optimizer.zero_grad(set_to_none=True)
        natural_logits = head(standardized[1.0])
        loss = F.cross_entropy(natural_logits, labels)
        if mode != "none":
            natural_logp = F.log_softmax(natural_logits, dim=-1)
            natural_p = natural_logp.exp()
            penalties = []
            for alpha in SHORT_ALPHAS[:-1]:
                perturbed_logp = F.log_softmax(head(standardized[alpha]), dim=-1)
                perturbed_p = perturbed_logp.exp()
                if mode == "teacher":
                    penalties.append(F.kl_div(perturbed_logp, natural_p.detach(), reduction="batchmean"))
                else:
                    forward = F.kl_div(perturbed_logp, natural_p, reduction="batchmean")
                    reverse = F.kl_div(natural_logp, perturbed_p, reduction="batchmean")
                    penalties.append(0.5 * (forward + reverse))
            loss = loss + weight * torch.stack(penalties).mean()
        loss.backward()
        optimizer.step()
    return {"head": head.eval(), "mean": mean, "scale": scale}


def covariance(features: torch.Tensor) -> torch.Tensor:
    centered = features - features.mean(0)
    return centered.T @ centered / max(len(features) - 1, 1)


def pooled_within_class_covariance(features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    residuals = torch.cat([features[labels == value] - features[labels == value].mean(0) for value in range(1, 5)])
    return covariance(residuals)


def matrix_power_psd(matrix: torch.Tensor, power: float) -> torch.Tensor:
    regularizer = 1e-4 * matrix.trace().clamp_min(1e-8) / matrix.shape[0]
    values, vectors = torch.linalg.eigh(matrix + regularizer * torch.eye(matrix.shape[0]))
    return (vectors * values.clamp_min(1e-8).pow(power)) @ vectors.T


def align_target(
    target: torch.Tensor,
    target_reference: torch.Tensor,
    source: torch.Tensor,
    method: str,
) -> torch.Tensor:
    source_mean, target_mean = source.mean(0), target_reference.mean(0)
    if method == "mean_std":
        return (target - target_mean) / target_reference.std(0).clamp_min(1e-5) * source.std(0).clamp_min(1e-5) + source_mean
    if method == "coral":
        return (target - target_mean) @ matrix_power_psd(covariance(target_reference), -0.5) @ matrix_power_psd(covariance(source), 0.5) + source_mean
    raise ValueError(method)


def fit_affine(source: torch.Tensor, target: torch.Tensor, ridge: float = 1e-3) -> torch.Tensor:
    augmented = torch.cat([source, torch.ones(len(source), 1)], dim=1)
    penalty = ridge * torch.eye(augmented.shape[1])
    penalty[-1, -1] = 0
    return torch.linalg.solve(augmented.T @ augmented + penalty, augmented.T @ target)


def apply_affine(features: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    return torch.cat([features, torch.ones(len(features), 1)], dim=1) @ transform


def replace_features(data: dict[str, torch.Tensor], features: torch.Tensor) -> dict[str, torch.Tensor]:
    return {**data, "query_hidden": features}


def geometry_metrics(source: dict[str, torch.Tensor], target: dict[str, torch.Tensor], model: torch.nn.Module) -> dict:
    source_x, target_x = source["query_hidden"], target["query_hidden"]
    source_y, target_y = source["labels"], target["labels"]
    scale = source_x.std(0).clamp_min(1e-5)
    shifts = []
    for value in range(1, 5):
        difference = (target_x[target_y == value].mean(0) - source_x[source_y == value].mean(0)) / scale
        shifts.append(difference.square().mean().sqrt())
    source_cov = pooled_within_class_covariance(source_x, source_y)
    target_cov = pooled_within_class_covariance(target_x, target_y)
    covariance_gap = torch.linalg.matrix_norm(target_cov - source_cov) / torch.linalg.matrix_norm(source_cov).clamp_min(1e-8)
    model_device = next(model.parameters()).device
    logits = model.head(target_x.to(model_device)).cpu()
    correct = logits.gather(1, target_y[:, None]).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, target_y[:, None], float("-inf"))
    margin = correct - masked.max(1).values
    return {
        "class_mean_shift_rms": float(torch.stack(shifts).mean()),
        "within_class_covariance_gap": float(covariance_gap),
        "original_head_mean_margin": float(margin.mean()),
        "original_head_positive_margin_rate": float((margin > 0).float().mean()),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_alignment(geometry: list[dict], accuracy: list[dict], output: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(14.8, 4.7))
    alphas = [row["eval_alpha"] for row in geometry]
    axes[0].plot(alphas, [row["class_mean_shift_rms"] for row in geometry], "o-", label="Class-mean shift")
    axes[0].plot(alphas, [row["within_class_covariance_gap"] for row in geometry], "s-", label="Covariance gap")
    axes[0].set(xlabel=r"Long-test decay $\alpha$", ylabel="Normalized shift", title="Short-to-long feature geometry")
    axes[0].legend(frameon=False)
    axes[1].bar(np.arange(len(alphas)), [row["original_head_mean_margin"] for row in geometry])
    axes[1].axhline(0, color="#6b7280", linestyle=":")
    axes[1].set_xticks(range(len(alphas)), [f"{alpha:g}" for alpha in alphas])
    axes[1].set(xlabel=r"Long-test decay $\alpha$", ylabel="Mean correct-class margin", title="Original classifier margin")
    methods = (
        ("raw_accuracy", "Raw"),
        ("short_stats_accuracy", "Short-only stats"),
        ("short_affine_accuracy", "Short-only affine"),
        ("target_mean_std_accuracy", "Long-calib mean/std*"),
        ("target_coral_accuracy", "Long-calib CORAL*"),
    )
    positions = np.arange(len(alphas)); width = 0.16
    for index, (field, label) in enumerate(methods):
        bars = axes[2].bar(positions + (index - 2) * width, [100 * row[field] for row in accuracy], width, label=label)
        if field in {"raw_accuracy", "target_coral_accuracy"}:
            axes[2].bar_label(bars, fmt="%.1f", fontsize=7, padding=2)
    axes[2].axhline(25, color="#6b7280", linestyle=":")
    axes[2].set_xticks(positions, [f"{alpha:g}" for alpha in alphas])
    axes[2].set(xlabel=r"Long-test decay $\alpha$", ylabel="Accuracy (%)", title="Unsupervised affine alignment", ylim=(0, 105))
    axes[2].legend(frameon=False)
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Feature alignment (* uses a separate unlabeled long-horizon calibration set)")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_robust(rows: list[dict], output: Path) -> None:
    names = [name for name, _, _ in ROBUST_CONFIGS]
    matrix = np.array([[100 * next(row["accuracy"] for row in rows if row["config"] == name and row["eval_alpha"] == alpha) for alpha in ALPHAS] for name in names])
    figure, axis = plt.subplots(figsize=(7.6, 5.4))
    image = axis.imshow(matrix, vmin=20, vmax=100, cmap="viridis", aspect="auto")
    axis.set_xticks(range(len(ALPHAS)), [f"{alpha:g}" for alpha in ALPHAS])
    axis.set_yticks(range(len(names)), [name.replace("_", " ") for name in names])
    axis.set(xlabel=r"511-step evaluation decay $\alpha$", ylabel="Short-only training objective", title="Horizon-robust short-only readouts: accuracy (%)")
    for row in range(len(names)):
        for column in range(len(ALPHAS)):
            axis.text(column, row, f"{matrix[row, column]:.1f}", ha="center", va="center", color="white" if matrix[row, column] < 62 else "black")
    figure.colorbar(image, ax=axis, fraction=0.045, pad=0.04)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_model(args.checkpoint, device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    raw_train = {alpha: extract_query_features(model, args.data_dir / "train.jsonl.gz", device, args.batch_size, alpha) for alpha in SHORT_ALPHAS}
    raw_validation = {alpha: extract_query_features(model, args.data_dir / "val.jsonl.gz", device, args.batch_size, alpha) for alpha in SHORT_ALPHAS}
    train_indices = balanced_indices(raw_train[1.0]["labels"], args.train_per_value, args.sample_seed)
    validation_indices = balanced_indices(raw_validation[1.0]["labels"], args.validation_per_value, args.sample_seed + 1)
    short_train = {alpha: subset(data, train_indices) for alpha, data in raw_train.items()}
    short_validation = {alpha: subset(data, validation_indices) for alpha, data in raw_validation.items()}

    long_test_examples = make_examples(args.test_contexts_per_pair, args.test_seed)
    long_calibration_examples = make_examples(args.calibration_contexts_per_pair, args.calibration_seed)
    long_test = {alpha: extract(model, *long_test_examples[:3], alpha, device, args.batch_size) for alpha in ALPHAS}
    long_calibration = {
        alpha: extract(model, *long_calibration_examples[:3], alpha, device, args.batch_size) for alpha in ALPHAS
    }
    source_head = fit_readout(short_train[1.0], args.readout_seed, device)
    short_affine = {
        alpha: fit_affine(short_train[alpha]["query_hidden"], short_train[1.0]["query_hidden"])
        for alpha in ALPHAS if alpha != 1.0
    }

    geometry_rows, alignment_rows = [], []
    for alpha in ALPHAS:
        geometry_rows.append({"eval_alpha": alpha, **geometry_metrics(short_validation[1.0], long_test[alpha], model)})
        raw = evaluate_readout(source_head, long_test[alpha], device)
        short_reference = short_train[alpha]["query_hidden"]
        natural_reference = short_train[1.0]["query_hidden"]
        short_stats_features = (long_test[alpha]["query_hidden"] - short_reference.mean(0)) / short_reference.std(0).clamp_min(1e-5) * natural_reference.std(0).clamp_min(1e-5) + natural_reference.mean(0)
        short_affine_features = long_test[alpha]["query_hidden"] if alpha == 1.0 else apply_affine(long_test[alpha]["query_hidden"], short_affine[alpha])
        short_stats = evaluate_readout(source_head, replace_features(long_test[alpha], short_stats_features), device)
        affine = evaluate_readout(source_head, replace_features(long_test[alpha], short_affine_features), device)
        calibration_features = long_calibration[alpha]["query_hidden"]
        target_mean_std = evaluate_readout(source_head, replace_features(long_test[alpha], align_target(long_test[alpha]["query_hidden"], calibration_features, natural_reference, "mean_std")), device)
        target_coral = evaluate_readout(source_head, replace_features(long_test[alpha], align_target(long_test[alpha]["query_hidden"], calibration_features, natural_reference, "coral")), device)
        alignment_rows.append({
            "eval_alpha": alpha,
            "raw_accuracy": raw["accuracy"],
            "short_stats_accuracy": short_stats["accuracy"],
            "short_affine_accuracy": affine["accuracy"],
            "target_mean_std_accuracy": target_mean_std["accuracy"],
            "target_coral_accuracy": target_coral["accuracy"],
            "raw_nll": raw["nll"],
            "short_stats_nll": short_stats["nll"],
            "short_affine_nll": affine["nll"],
            "target_mean_std_nll": target_mean_std["nll"],
            "target_coral_nll": target_coral["nll"],
        })

    robust_rows, validation_rows = [], []
    for name, mode, weight in ROBUST_CONFIGS:
        readout = fit_robust_readout(short_train, mode, weight, args.readout_seed, device)
        natural_validation = evaluate_readout(readout, short_validation[1.0], device)
        consistency = np.mean([evaluate_readout(readout, short_validation[alpha], device)["accuracy"] for alpha in SHORT_ALPHAS[:-1]])
        validation_rows.append({"config": name, "natural_accuracy": natural_validation["accuracy"], "natural_nll": natural_validation["nll"], "mean_perturbed_accuracy": consistency})
        for alpha in ALPHAS:
            result = evaluate_readout(readout, long_test[alpha], device)
            robust_rows.append({"config": name, "mode": mode, "lambda": weight, "eval_alpha": alpha, "accuracy": result["accuracy"], "nll": result["nll"]})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "feature_geometry.csv", geometry_rows)
    write_csv(args.output_dir / "affine_alignment.csv", alignment_rows)
    write_csv(args.output_dir / "robust_readout.csv", robust_rows)
    write_csv(args.output_dir / "short_validation.csv", validation_rows)
    plot_alignment(geometry_rows, alignment_rows, args.output_dir / "feature_alignment.png")
    plot_robust(robust_rows, args.output_dir / "horizon_robust_readout.png")
    summary = {
        "constraint": "No length-511 labels are used for training or model selection. Short-stats, short-affine, and robust heads see no long features. Long-calibration mean/std and CORAL use a separate unlabeled 511-step calibration set, never the labeled test batch.",
        "feature_geometry": geometry_rows,
        "affine_alignment": alignment_rows,
        "short_validation": validation_rows,
        "robust_readout": robust_rows,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("results/selective_retention/seed42/event_decay/u1500/baseline.pt"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/generated"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/horizon_robust_readout/seed42"))
    parser.add_argument("--train-per-value", type=int, default=96)
    parser.add_argument("--validation-per-value", type=int, default=32)
    parser.add_argument("--test-contexts-per-pair", type=int, default=8)
    parser.add_argument("--calibration-contexts-per-pair", type=int, default=8)
    parser.add_argument("--test-seed", type=int, default=53042)
    parser.add_argument("--calibration-seed", type=int, default=59042)
    parser.add_argument("--sample-seed", type=int, default=56042)
    parser.add_argument("--readout-seed", type=int, default=54042)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps(result, indent=2))
