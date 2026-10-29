#!/usr/bin/env python3
"""Generate reproducible SET / QUERY / NOOP data for Tiny Mamba-1.

Copied from the assignment-provided generator. Output is gzip JSON Lines.
"""
import argparse
import gzip
import json
import random
from pathlib import Path

KEYS = "ABCDEFGH"
NUM_VALUES = 4
SPLITS = {
    "train": ((32, 128), 10601, "standard"),
    "val": ((32, 128), 10602, "standard"),
    "test_id": ((128, 128), 10603, "standard"),
    "test_ood_256": ((256, 256), 10604, "standard"),
    "test_ood_512": ((512, 512), 10605, "standard"),
    "stress_gap_512": ((512, 512), 10606, "gap"),
    "stress_overwrite_512": ((512, 512), 10607, "overwrite"),
}


def encode_event(event):
    op, key, val = event
    if op == "NOOP":
        return 0
    if op == "SET":
        return 1 + KEYS.index(key) * NUM_VALUES + (val - 1)
    if op == "QUERY":
        return 33 + KEYS.index(key)
    raise ValueError(f"Unknown event: {event!r}")


def random_event(rng, exclude_key=None):
    available = [key for key in KEYS if key != exclude_key]
    probability = rng.random()
    if probability < 0.32:
        return ["SET", rng.choice(available), rng.randint(1, NUM_VALUES)]
    if probability < 0.50:
        return ["QUERY", rng.choice(available), None]
    return ["NOOP", None, None]


def make_sequence(rng, length, mode, sample_index):
    focus = rng.choice(KEYS)
    probe_index = None
    overwrites = None
    if mode == "standard":
        events = [random_event(rng) for _ in range(length)]
        events[0] = ["QUERY", rng.choice(KEYS), None]
        events[-1] = ["QUERY", rng.choice(KEYS), None]
    elif mode == "gap":
        events = [["SET", focus, rng.randint(1, NUM_VALUES)]]
        events.extend(random_event(rng, exclude_key=focus) for _ in range(length - 2))
        events.append(["QUERY", focus, None])
        probe_index = length - 1
    elif mode == "overwrite":
        overwrites = (1, 2, 4, 8)[sample_index % 4]
        events = [random_event(rng, exclude_key=focus) for _ in range(length)]
        write_positions = [0] + [int(1 + j * (length - 3) / overwrites) for j in range(1, overwrites + 1)]
        previous_value = None
        for position in write_positions:
            choices = [value for value in range(1, NUM_VALUES + 1) if value != previous_value]
            previous_value = rng.choice(choices)
            events[position] = ["SET", focus, previous_value]
        events[-1] = ["QUERY", focus, None]
        probe_index = length - 1
    else:
        raise ValueError(mode)

    state, latest_update, labels = {}, {}, []
    probe_gap = None
    for timestep, (op, key, value) in enumerate(events):
        if op == "SET":
            state[key] = value
            latest_update[key] = timestep
            labels.append(None)
        elif op == "QUERY":
            labels.append(state.get(key, 0))
            if timestep == probe_index and key in latest_update:
                probe_gap = timestep - latest_update[key]
        else:
            labels.append(None)
    return {
        "events": events,
        "query_targets": labels,
        "probe_index": probe_index,
        "probe_gap": probe_gap,
        "overwrite_count": overwrites,
    }


def generate(output_dir, preset, seed_offset=0):
    sizes = {"small": (1000, 200), "assignment": (4000, 500), "full": (50000, 5000)}
    n_train, n_other = sizes[preset]
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "task": "causal_key_value_tracking",
        "preset": preset,
        "keys": KEYS,
        "values": [1, 2, 3, 4],
        "unset_class": 0,
        "input_vocabulary_size": 41,
        "event_encoding": "NOOP=0, SET(A,1..4)=1..4,...,SET(H,1..4)=29..32, QUERY(A..H)=33..40",
        "non_query_label": None,
        "seed_offset": seed_offset,
        "splits": {},
    }
    for split, (lengths, seed, mode) in SPLITS.items():
        count = n_train if split == "train" else n_other
        rng = random.Random(seed + seed_offset)
        path = output_dir / f"{split}.jsonl.gz"
        with gzip.open(path, "wt", encoding="utf-8", compresslevel=5) as output:
            for index in range(count):
                length = rng.randint(*lengths)
                item = make_sequence(rng, length, mode, index)
                item = {"sample_id": f"{split}_{index:06d}", "split": split, **item}
                output.write(json.dumps(item, separators=(",", ":")) + "\n")
        manifest["splits"][split] = {
            "count": count,
            "lengths": list(lengths),
            "seed": seed + seed_offset,
            "mode": mode,
            "filename": path.name,
        }
        print(f"{split:>21}: {count:>5} sequences -> {path.name}")
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as output:
        json.dump(manifest, output, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=["small", "assignment", "full"], default="assignment")
    parser.add_argument("--output-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--seed-offset", type=int, default=0)
    arguments = parser.parse_args()
    generate(arguments.output_dir, arguments.preset, arguments.seed_offset)
