"""Audit empirical delta, effective retention, and implied timescales for H7a."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from h7a_utils import load_checkpoint
from train import build_loader


@torch.no_grad()
def audit(checkpoint: Path, data: Path, device: torch.device, max_batches: int = 32) -> list[dict]:
    model, _ = load_checkpoint(checkpoint, device)
    mixer = model.backbone.layers[0].mixer
    decay = mixer.decay_matrix().cpu()
    event_logs: dict[str, list[torch.Tensor]] = {name: [] for name in ("NOOP", "SET", "QUERY")}
    event_deltas: dict[str, list[torch.Tensor]] = {name: [] for name in event_logs}
    for batch_index, batch in enumerate(build_loader(data, 16, False, 42)):
        inputs = batch["input_ids"].to(device)
        embedded = model.backbone.embedding(inputs)
        block_input = model.backbone.layers[0].norm(embedded)
        u, _ = mixer.in_proj(block_input).chunk(2, dim=-1)
        u = F.silu(mixer.conv1d(u.transpose(1, 2))[:, :, : inputs.shape[1]].transpose(1, 2))
        raw = mixer.x_proj(u)
        delta = F.softplus(mixer.dt_proj(raw[..., : mixer.args.dt_rank])).cpu()
        log_retention = delta.unsqueeze(-1) * decay.view(1, 1, mixer.args.d_inner, mixer.args.d_state)
        masks = {
            "NOOP": inputs.cpu() == 0,
            "SET": (inputs.cpu() >= 1) & (inputs.cpu() <= 32),
            "QUERY": (inputs.cpu() >= 33) & (inputs.cpu() <= 40),
        }
        for event, mask in masks.items():
            event_deltas[event].append(delta[mask])
            event_logs[event].append(log_retention[mask])
        if batch_index + 1 >= max_batches:
            break
    rows: list[dict] = []
    for event in event_logs:
        deltas = torch.cat(event_deltas[event])
        logs = torch.cat(event_logs[event])
        groups = [("fast", slice(0, 4)), ("slow", slice(4, 8)), ("all", slice(0, 8))]
        groups.extend((f"mode_{index}", slice(index, index + 1)) for index in range(8))
        for group, mode_slice in groups:
            mean_log = logs[:, :, mode_slice].mean().item()
            row = {
                "event": event,
                "group": group,
                "token_count": int(logs.shape[0]),
                "delta_mean": deltas.mean().item(),
                "delta_q10": torch.quantile(deltas, 0.1).item(),
                "delta_q50": torch.quantile(deltas, 0.5).item(),
                "delta_q90": torch.quantile(deltas, 0.9).item(),
                "mean_log_retention": mean_log,
                "mean_step_retention": float(torch.exp(torch.tensor(mean_log)).item()),
                "e_folding_steps": float(-1.0 / mean_log),
            }
            for horizon in (32, 128, 256, 511):
                row[f"survival_{horizon}"] = float(torch.exp(torch.tensor(mean_log * horizon)).item())
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/generated/val.jsonl.gz"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--max-batches", type=int, default=32)
    args = parser.parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    rows = audit(args.checkpoint, args.data, device, args.max_batches)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        args.output.with_suffix(".json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
