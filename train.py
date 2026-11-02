"""Train and validate the five-class Mamba state tracker."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from data.dataset import IGNORE_INDEX, StateTrackingDataset, collate_batch
from task_model import StateTracker


def deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    parent = config.pop("inherits", None)
    if parent is None:
        return config
    return deep_merge(load_config(path.parent / parent), config)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def query_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Cross-entropy at QUERY positions only."""
    return F.cross_entropy(logits.reshape(-1, 5), targets.reshape(-1), ignore_index=IGNORE_INDEX)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, float | int]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    query_count = 0
    batch_count = 0
    started = time.perf_counter()
    for batch_count, batch in enumerate(loader, start=1):
        inputs = batch["input_ids"].to(device)
        targets = batch["targets"].to(device)
        logits = model(inputs)
        mask = targets != IGNORE_INDEX
        queries = int(mask.sum().item())
        loss_sum += float(query_loss(logits, targets).item()) * queries
        correct += int((logits.argmax(-1)[mask] == targets[mask]).sum().item())
        query_count += queries
        if max_batches is not None and batch_count >= max_batches:
            break
    elapsed = time.perf_counter() - started
    return {
        "loss": loss_sum / max(query_count, 1),
        "query_accuracy": correct / max(query_count, 1),
        "correct_queries": correct,
        "query_count": query_count,
        "batches": batch_count,
        "elapsed_seconds": elapsed,
        "queries_per_second": query_count / max(elapsed, 1e-12),
    }


def train_updates(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_updates: int,
) -> dict[str, float | int]:
    model.train()
    loss_sum = 0.0
    query_count = 0
    updates = 0
    started = time.perf_counter()
    while updates < max_updates:
        for batch in loader:
            inputs = batch["input_ids"].to(device)
            targets = batch["targets"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = query_loss(logits, targets)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at update {updates + 1}: {loss.item()}")
            loss.backward()
            optimizer.step()
            queries = int((targets != IGNORE_INDEX).sum().item())
            loss_sum += float(loss.item()) * queries
            query_count += queries
            updates += 1
            if updates >= max_updates:
                break
    elapsed = time.perf_counter() - started
    return {
        "loss": loss_sum / max(query_count, 1),
        "query_count": query_count,
        "updates": updates,
        "elapsed_seconds": elapsed,
    }


def build_loader(path: Path, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        StateTrackingDataset(path),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_batch,
        generator=generator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def run(args: argparse.Namespace) -> dict:
    config = load_config(args.config)
    seed = args.seed if args.seed is not None else int(config["seed"])
    config["seed"] = seed
    set_seed(seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    batch_size = args.batch_size or int(config["training"]["batch_size"])
    max_updates = args.max_updates or int(config["training"]["max_updates"])
    learning_rate = float(config["training"]["learning_rate"])

    train_loader = build_loader(args.data_dir / "train.jsonl.gz", batch_size, True, seed)
    val_loader = build_loader(args.data_dir / "val.jsonl.gz", batch_size, False, seed)
    model_config = config.get("model", {})
    model = StateTracker(
        variant=str(model_config.get("variant", "baseline")),
        route_init=float(model_config.get("route_init", 0.95)),
        d_conv=int(model_config.get("d_conv", 4)),
        fast_min_rate=float(model_config.get("fast_min_rate", 1.0)),
        slow_max_rate=float(model_config.get("slow_max_rate", 0.004)),
        slow_gate_bias_init=float(model_config.get("slow_gate_bias_init", -4.0)),
        erase_gate_bias_init=float(model_config.get("erase_gate_bias_init", -4.0)),
        write_gate_bias_init=float(model_config.get("write_gate_bias_init", -2.0)),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    completed_updates = 0
    if args.resume is not None:
        payload = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(payload["model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        completed_updates = int(payload["updates"])
        if completed_updates >= max_updates:
            raise ValueError(
                f"resume checkpoint already has {completed_updates} updates; "
                f"target --max-updates must be larger than that"
            )

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    additional_updates = max_updates - completed_updates
    train_metrics = train_updates(model, train_loader, optimizer, device, additional_updates)
    train_metrics["starting_update"] = completed_updates
    train_metrics["ending_update"] = max_updates
    validation_metrics = evaluate(model, val_loader, device, args.max_validation_batches)
    peak_memory = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    result = {
        "seed": seed,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "parameter_count": parameter_count,
        "model_variant": model.variant,
        "training": train_metrics,
        "validation": validation_metrics,
        "peak_gpu_memory_bytes": peak_memory,
        "resumed_from": str(args.resume) if args.resume is not None else None,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "baseline.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "updates": max_updates,
            "validation": validation_metrics,
        },
        checkpoint_path,
    )
    (args.output_dir / "baseline_validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["checkpoint"] = str(checkpoint_path)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/generated"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/checkpoints"))
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--max-validation-batches", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))
