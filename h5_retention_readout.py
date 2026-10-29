"""H5: retrain only the readout on frozen, decay-modified hidden states."""

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

from decay_only_retention_sweep import (
    cluster_bootstrap_difference,
    extract,
    make_examples,
)
from hypothesis_pilot import load_model


ALPHAS = (0.0, 0.1, 1.0)


def fit_readout(train: dict[str, torch.Tensor], seed: int, device: torch.device) -> dict:
    features = train["query_hidden"].to(device)
    targets = train["labels"].to(device)
    mean = features.mean(0)
    scale = features.std(0).clamp_min(1e-5)
    standardized = (features - mean) / scale
    torch.manual_seed(seed)
    head = torch.nn.Linear(32, 5).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.03, weight_decay=1e-4)
    for _ in range(500):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(head(standardized), targets)
        loss.backward()
        optimizer.step()
    return {"head": head.eval(), "mean": mean, "scale": scale}


@torch.no_grad()
def evaluate_readout(readout: dict, data: dict[str, torch.Tensor], device: torch.device) -> dict:
    features = data["query_hidden"].to(device)
    targets = data["labels"].to(device)
    logits = readout["head"]((features - readout["mean"]) / readout["scale"])
    predictions = logits.argmax(-1)
    correct = predictions.eq(targets)
    probabilities = logits.softmax(-1)
    target_probability = probabilities.gather(1, targets[:, None]).squeeze(1)
    return {
        "accuracy": float(correct.float().mean().item()),
        "nll": float(F.cross_entropy(logits, targets).item()),
        "mean_correct_class_probability": float(target_probability.mean().item()),
        "correct": correct.cpu().numpy().astype(np.int8),
        "predictions": predictions.cpu().numpy().astype(np.int8),
    }


def original_metrics(data: dict[str, torch.Tensor]) -> dict:
    logits = data["logits"]
    targets = data["labels"]
    predictions = logits.argmax(-1)
    correct = predictions.eq(targets)
    probabilities = logits.softmax(-1)
    return {
        "accuracy": float(correct.float().mean().item()),
        "nll": float(F.cross_entropy(logits, targets).item()),
        "mean_correct_class_probability": float(probabilities.gather(1, targets[:, None]).mean().item()),
        "correct": correct.numpy().astype(np.int8),
        "predictions": predictions.numpy().astype(np.int8),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(matched: list[dict], cross: list[dict], output: Path) -> None:
    alphas = [row["alpha"] for row in matched]
    positions = np.arange(len(alphas))
    width = 0.34
    figure, axes = plt.subplots(1, 3, figsize=(14.4, 4.8))
    axes[0].bar(positions - width / 2, [100 * row["original_accuracy"] for row in matched], width, label="Original head")
    axes[0].bar(positions + width / 2, [100 * row["retrained_accuracy"] for row in matched], width, label="Matched retrained head")
    axes[0].axhline(25, color="#6b7280", linestyle="--", linewidth=1.3, label="Chance")
    axes[0].set_xticks(positions, [str(alpha) for alpha in alphas])
    axes[0].set(xlabel=r"Evaluation decay $\alpha$", ylabel="Test accuracy (%)", title="Readout adaptation", ylim=(0, 105))
    axes[0].legend(frameon=False)

    axes[1].bar(positions - width / 2, [row["original_nll"] for row in matched], width, label="Original head")
    axes[1].bar(positions + width / 2, [row["retrained_nll"] for row in matched], width, label="Matched retrained head")
    axes[1].set_xticks(positions, [str(alpha) for alpha in alphas])
    axes[1].set(xlabel=r"Evaluation decay $\alpha$", ylabel="Test NLL", title="Confidence and calibration")
    axes[1].legend(frameon=False)

    matrix = np.empty((len(ALPHAS), len(ALPHAS)), dtype=np.float64)
    for row_index, train_alpha in enumerate(ALPHAS):
        for column_index, eval_alpha in enumerate(ALPHAS):
            matrix[row_index, column_index] = 100 * next(
                row["accuracy"] for row in cross if row["train_alpha"] == train_alpha and row["eval_alpha"] == eval_alpha
            )
    image = axes[2].imshow(matrix, vmin=25, vmax=100, cmap="viridis")
    axes[2].set_xticks(range(len(ALPHAS)), [str(alpha) for alpha in ALPHAS])
    axes[2].set_yticks(range(len(ALPHAS)), [str(alpha) for alpha in ALPHAS])
    axes[2].set(xlabel=r"Evaluation decay $\alpha$", ylabel=r"Readout training decay $\alpha$", title="Cross-decay readout accuracy (%)")
    for row in range(len(ALPHAS)):
        for column in range(len(ALPHAS)):
            color = "white" if matrix[row, column] < 62 else "black"
            axes[2].text(column, row, f"{matrix[row, column]:.1f}", ha="center", va="center", color=color)
    figure.colorbar(image, ax=axes[2], fraction=0.046, pad=0.04)
    for axis in axes[:2]:
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("H5: frozen backbone, decay-matched five-class readout")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(output, dpi=180)
    plt.close(figure)


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_model(args.checkpoint, device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    train_examples = make_examples(args.train_contexts_per_pair, args.train_seed)
    validation_examples = make_examples(args.validation_contexts_per_pair, args.validation_seed)
    test_examples = make_examples(args.test_contexts_per_pair, args.test_seed)
    train_ids = {tuple(sequence) for sequence in train_examples[0]}
    validation_ids = {tuple(sequence) for sequence in validation_examples[0]}
    test_ids = {tuple(sequence) for sequence in test_examples[0]}
    if train_ids & validation_ids or train_ids & test_ids or validation_ids & test_ids:
        raise RuntimeError("train, validation, and test sequences must be disjoint")

    datasets: dict[str, dict[float, dict[str, torch.Tensor]]] = {"train": {}, "validation": {}, "test": {}}
    for alpha in ALPHAS:
        datasets["train"][alpha] = extract(model, *train_examples[:3], alpha, device, args.batch_size)
        datasets["validation"][alpha] = extract(model, *validation_examples[:3], alpha, device, args.batch_size)
        datasets["test"][alpha] = extract(model, *test_examples[:3], alpha, device, args.batch_size)

    readouts = {alpha: fit_readout(datasets["train"][alpha], args.readout_seed, device) for alpha in ALPHAS}
    matched_rows: list[dict] = []
    cross_rows: list[dict] = []
    contrasts: list[dict] = []
    saved_arrays: dict[str, np.ndarray] = {
        "labels": np.asarray(test_examples[1], dtype=np.int8),
        "keys": np.asarray(test_examples[2], dtype=np.int8),
        "clusters": np.asarray(test_examples[3], dtype=np.int8),
    }
    clusters = saved_arrays["clusters"]

    for alpha_index, alpha in enumerate(ALPHAS):
        original = original_metrics(datasets["test"][alpha])
        retrained = evaluate_readout(readouts[alpha], datasets["test"][alpha], device)
        validation = evaluate_readout(readouts[alpha], datasets["validation"][alpha], device)
        matched_rows.append({
            "alpha": alpha,
            "original_accuracy": original["accuracy"],
            "retrained_accuracy": retrained["accuracy"],
            "accuracy_gain_pp": 100 * (retrained["accuracy"] - original["accuracy"]),
            "original_nll": original["nll"],
            "retrained_nll": retrained["nll"],
            "original_correct_class_probability": original["mean_correct_class_probability"],
            "retrained_correct_class_probability": retrained["mean_correct_class_probability"],
            "validation_accuracy": validation["accuracy"],
            "n_train": len(train_examples[0]),
            "n_validation": len(validation_examples[0]),
            "n_test": len(test_examples[0]),
        })
        effect, low, high = cluster_bootstrap_difference(
            retrained["correct"], original["correct"], clusters, args.bootstrap_seed + alpha_index
        )
        contrasts.append({
            "alpha": alpha,
            "comparison": "matched_retrained_vs_original_head",
            "accuracy_effect_pp": 100 * effect,
            "ci_low_pp": 100 * low,
            "ci_high_pp": 100 * high,
            "cluster_count": 32,
        })
        saved_arrays[f"original_alpha_{str(alpha).replace('.', '_')}"] = original["predictions"]
        saved_arrays[f"matched_alpha_{str(alpha).replace('.', '_')}"] = retrained["predictions"]

    for train_alpha in ALPHAS:
        for eval_alpha in ALPHAS:
            result = evaluate_readout(readouts[train_alpha], datasets["test"][eval_alpha], device)
            cross_rows.append({
                "train_alpha": train_alpha,
                "eval_alpha": eval_alpha,
                "accuracy": result["accuracy"],
                "nll": result["nll"],
                "mean_correct_class_probability": result["mean_correct_class_probability"],
            })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "matched_metrics.csv", matched_rows)
    write_csv(args.output_dir / "cross_decay_metrics.csv", cross_rows)
    write_csv(args.output_dir / "contrasts.csv", contrasts)
    np.savez_compressed(args.output_dir / "predictions.npz", **saved_arrays)
    summary = {
        "hypothesis": "Reducing attenuation improves decodability, but the original readout is mismatched to the modified hidden-state distribution.",
        "falsification": "A decay-matched retrained readout fails to improve over the frozen original head.",
        "scope": "Frozen seed-42 independently gated width-4 backbone; separate generated train/validation/test sequences.",
        "checkpoint": str(args.checkpoint),
        "readout": {"type": "Linear(32, 5)", "updates": 500, "optimizer": "AdamW", "learning_rate": 0.03},
        "matched_metrics": matched_rows,
        "cross_decay_metrics": cross_rows,
        "contrasts": contrasts,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot(matched_rows, cross_rows, args.output_dir / "h5_retention_readout.png")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("results/selective_retention/seed42/event_decay/u1500/baseline.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/h5_retention_readout/seed42"))
    parser.add_argument("--train-contexts-per-pair", type=int, default=12)
    parser.add_argument("--validation-contexts-per-pair", type=int, default=4)
    parser.add_argument("--test-contexts-per-pair", type=int, default=8)
    parser.add_argument("--train-seed", type=int, default=51042)
    parser.add_argument("--validation-seed", type=int, default=52042)
    parser.add_argument("--test-seed", type=int, default=53042)
    parser.add_argument("--readout-seed", type=int, default=54042)
    parser.add_argument("--bootstrap-seed", type=int, default=55042)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({"matched_metrics": result["matched_metrics"], "contrasts": result["contrasts"]}, indent=2))
