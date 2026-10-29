"""H5 control: separate frozen-backbone head trained on length-32--128 data."""

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
from torch.utils.data import DataLoader

from data.dataset import IGNORE_INDEX, StateTrackingDataset, collate_batch
from decay_only_retention_sweep import cluster_bootstrap_difference, extract, make_examples
from h5_retention_readout import ALPHAS, evaluate_readout, fit_readout, original_metrics
from hypothesis_pilot import load_model


@torch.no_grad()
def extract_query_features(
    model: torch.nn.Module,
    path: Path,
    device: torch.device,
    batch_size: int,
    decay_alpha: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Collect frozen alpha=1 final features at every real QUERY position."""
    loader = DataLoader(
        StateTrackingDataset(path),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=0,
    )
    feature_parts: list[torch.Tensor] = []
    label_parts: list[torch.Tensor] = []
    length_parts: list[torch.Tensor] = []
    for batch in loader:
        inputs = batch["input_ids"].to(device)
        targets = batch["targets"].to(device)
        decay_scale = model.event_retention_scale(inputs)
        if decay_scale is not None and decay_alpha != 1.0:
            decay_scale = decay_scale.clone()
            eligible = inputs == 0
            for offset in range(1, model.d_conv):
                preceded_by_noop = torch.zeros_like(eligible)
                preceded_by_noop[:, offset:] = inputs[:, :-offset] == 0
                eligible &= preceded_by_noop
            decay_scale[eligible] *= decay_alpha
        hidden = model.hidden_states(inputs, decay_scale=decay_scale)
        mask = targets != IGNORE_INDEX
        feature_parts.append(hidden[mask].cpu())
        label_parts.append(targets[mask].cpu())
        lengths = batch["lengths"].to(device).unsqueeze(1).expand_as(targets)
        length_parts.append(lengths[mask].cpu())
    query_hidden = torch.cat(feature_parts)
    return {
        "query_hidden": query_hidden,
        "labels": torch.cat(label_parts),
        "sequence_lengths": torch.cat(length_parts),
        "logits": model.head(query_hidden.to(device)).cpu(),
    }


def balanced_known_value_subset(data: dict[str, torch.Tensor], per_value: int, seed: int) -> dict[str, torch.Tensor]:
    """Match H5's sample count and balanced nonzero value-label distribution."""
    generator = torch.Generator().manual_seed(seed)
    selected: list[torch.Tensor] = []
    for value in range(1, 5):
        candidates = torch.nonzero(data["labels"] == value, as_tuple=False).flatten()
        if len(candidates) < per_value:
            raise ValueError(f"need {per_value} short-horizon examples for value {value}, found {len(candidates)}")
        selected.append(candidates[torch.randperm(len(candidates), generator=generator)[:per_value]])
    indices = torch.cat(selected)
    indices = indices[torch.randperm(len(indices), generator=generator)]
    return {name: values[indices] for name, values in data.items()}


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict], output: Path) -> None:
    positions = np.arange(len(rows))
    width = 0.25
    figure, axis = plt.subplots(figsize=(10.2, 5.0))
    series = (
        ("original_accuracy", "Original jointly trained head"),
        ("short_head_accuracy", "Frozen-backbone head: lengths 32–128"),
        ("long_head_accuracy", "Matched long-horizon head"),
    )
    for index, (field, label) in enumerate(series):
        values = [100 * row[field] for row in rows]
        bars = axis.bar(positions + (index - 1) * width, values, width, label=label)
        axis.bar_label(bars, fmt="%.1f", padding=2, fontsize=8)
    axis.axhline(25, color="#6b7280", linestyle=":", linewidth=1.3, label="Chance")
    axis.set_xticks(positions, [f"{row['eval_alpha']:g}" for row in rows])
    axis.set(
        xlabel=r"Evaluation decay multiplier $\alpha$",
        ylabel="Held-out 511-step accuracy (%)",
        title="H5 control: supervision horizon versus joint-training history",
        ylim=(0, 108),
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_model(args.checkpoint, device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    raw_short_train = extract_query_features(model, args.data_dir / "train.jsonl.gz", device, args.batch_size)
    raw_short_validation = extract_query_features(model, args.data_dir / "val.jsonl.gz", device, args.batch_size)
    short_train = balanced_known_value_subset(raw_short_train, args.train_per_value, args.short_sample_seed)
    short_validation = balanced_known_value_subset(
        raw_short_validation, args.validation_per_value, args.short_sample_seed + 1
    )
    short_readout = fit_readout(short_train, args.readout_seed, device)
    short_train_result = evaluate_readout(short_readout, short_train, device)
    short_validation_result = evaluate_readout(short_readout, short_validation, device)
    original_short_validation = original_metrics(short_validation)

    long_train_examples = make_examples(args.long_train_contexts_per_pair, args.long_train_seed)
    long_test_examples = make_examples(args.test_contexts_per_pair, args.test_seed)
    train_sets = {
        alpha: extract(model, *long_train_examples[:3], alpha, device, args.batch_size) for alpha in ALPHAS
    }
    test_sets = {
        alpha: extract(model, *long_test_examples[:3], alpha, device, args.batch_size) for alpha in ALPHAS
    }
    long_readouts = {
        alpha: fit_readout(train_sets[alpha], args.readout_seed, device) for alpha in ALPHAS
    }

    clusters = np.asarray(long_test_examples[3], dtype=np.int8)
    rows: list[dict] = []
    contrasts: list[dict] = []
    predictions: dict[str, np.ndarray] = {
        "labels": np.asarray(long_test_examples[1], dtype=np.int8),
        "clusters": clusters,
    }
    for alpha_index, alpha in enumerate(ALPHAS):
        original = original_metrics(test_sets[alpha])
        short = evaluate_readout(short_readout, test_sets[alpha], device)
        long = evaluate_readout(long_readouts[alpha], test_sets[alpha], device)
        rows.append({
            "eval_alpha": alpha,
            "original_accuracy": original["accuracy"],
            "short_head_accuracy": short["accuracy"],
            "long_head_accuracy": long["accuracy"],
            "original_nll": original["nll"],
            "short_head_nll": short["nll"],
            "long_head_nll": long["nll"],
            "n_test": len(long_test_examples[0]),
        })
        for comparison_index, (name, first, second) in enumerate((
            ("short_minus_original", short, original),
            ("long_minus_short", long, short),
        )):
            effect, low, high = cluster_bootstrap_difference(
                first["correct"], second["correct"], clusters,
                args.bootstrap_seed + alpha_index * 10 + comparison_index,
            )
            contrasts.append({
                "eval_alpha": alpha,
                "comparison": name,
                "accuracy_effect_pp": 100 * effect,
                "ci_low_pp": 100 * low,
                "ci_high_pp": 100 * high,
                "cluster_count": 32,
            })
        predictions[f"original_alpha_{str(alpha).replace('.', '_')}"] = original["predictions"]
        predictions[f"short_alpha_{str(alpha).replace('.', '_')}"] = short["predictions"]
        predictions[f"long_alpha_{str(alpha).replace('.', '_')}"] = long["predictions"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "head_comparison.csv", rows)
    write_csv(args.output_dir / "contrasts.csv", contrasts)
    np.savez_compressed(args.output_dir / "predictions.npz", **predictions)
    summary = {
        "question": "Does a separately optimized short-horizon head outperform the jointly trained original head on 511-step examples?",
        "control": {
            "backbone": "Frozen alpha=1 backbone during head training",
            "head": "Linear(32,5), same initialization seed and 500-update optimizer budget as H5 P3",
            "training_source": "Assignment train split, sequence lengths 32--128",
            "sampling": "Balanced labels 1--4; sample count matched to the long-horizon head",
            "n_train": len(short_train["labels"]),
            "n_validation": len(short_validation["labels"]),
            "min_sequence_length": int(short_train["sequence_lengths"].min()),
            "max_sequence_length": int(short_train["sequence_lengths"].max()),
            "short_train_accuracy": short_train_result["accuracy"],
            "short_validation_accuracy": short_validation_result["accuracy"],
            "short_validation_nll": short_validation_result["nll"],
            "original_head_short_validation_accuracy": original_short_validation["accuracy"],
            "original_head_short_validation_nll": original_short_validation["nll"],
        },
        "metrics": rows,
        "contrasts": contrasts,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot(rows, args.output_dir / "h5_short_horizon_control.png")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("results/selective_retention/seed42/event_decay/u1500/baseline.pt"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/generated"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/h5_short_horizon_control/seed42"))
    parser.add_argument("--train-per-value", type=int, default=96)
    parser.add_argument("--validation-per-value", type=int, default=32)
    parser.add_argument("--long-train-contexts-per-pair", type=int, default=12)
    parser.add_argument("--test-contexts-per-pair", type=int, default=8)
    parser.add_argument("--long-train-seed", type=int, default=51042)
    parser.add_argument("--test-seed", type=int, default=53042)
    parser.add_argument("--short-sample-seed", type=int, default=56042)
    parser.add_argument("--readout-seed", type=int, default=54042)
    parser.add_argument("--bootstrap-seed", type=int, default=57042)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps(result, indent=2))
