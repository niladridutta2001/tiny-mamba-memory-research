import torch

from h5_short_horizon_control import balanced_known_value_subset


def test_balanced_known_value_subset_matches_requested_class_counts() -> None:
    data = {
        "query_hidden": torch.arange(80 * 32, dtype=torch.float32).reshape(80, 32),
        "labels": torch.tensor([value for value in range(1, 5) for _ in range(20)]),
        "sequence_lengths": torch.arange(80) % 97 + 32,
        "logits": torch.zeros(80, 5),
    }
    selected = balanced_known_value_subset(data, per_value=7, seed=123)
    assert len(selected["labels"]) == 28
    assert torch.bincount(selected["labels"], minlength=5).tolist() == [0, 7, 7, 7, 7]
    assert set(selected) == set(data)
