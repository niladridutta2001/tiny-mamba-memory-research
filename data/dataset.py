"""PyTorch dataset adapter for the assignment-provided gzip JSONL files."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from generate_dataset import encode_event

IGNORE_INDEX = -100
NOOP_TOKEN = 0


class StateTrackingDataset(Dataset):
    def __init__(self, path: str | Path):
        self.path = Path(path)
        with gzip.open(self.path, "rt", encoding="utf-8") as source:
            self.records = [json.loads(line) for line in source]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        tokens = torch.tensor([encode_event(event) for event in record["events"]], dtype=torch.long)
        targets = torch.tensor(
            [IGNORE_INDEX if value is None else value for value in record["query_targets"]], dtype=torch.long
        )
        return {
            "input_ids": tokens,
            "targets": targets,
            "length": len(tokens),
            "sample_id": record["sample_id"],
            "probe_index": record["probe_index"],
            "probe_gap": record["probe_gap"],
            "overwrite_count": record["overwrite_count"],
        }


def collate_batch(samples: list[dict]) -> dict:
    width = max(sample["length"] for sample in samples)
    batch_size = len(samples)
    input_ids = torch.full((batch_size, width), NOOP_TOKEN, dtype=torch.long)
    targets = torch.full((batch_size, width), IGNORE_INDEX, dtype=torch.long)
    lengths = torch.empty(batch_size, dtype=torch.long)
    for row, sample in enumerate(samples):
        length = sample["length"]
        input_ids[row, :length] = sample["input_ids"]
        targets[row, :length] = sample["targets"]
        lengths[row] = length
    return {
        "input_ids": input_ids,
        "targets": targets,
        "lengths": lengths,
        "sample_ids": [sample["sample_id"] for sample in samples],
        "probe_indices": [sample["probe_index"] for sample in samples],
        "probe_gaps": [sample["probe_gap"] for sample in samples],
        "overwrite_counts": [sample["overwrite_count"] for sample in samples],
    }
