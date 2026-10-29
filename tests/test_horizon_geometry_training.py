import torch

from horizon_geometry_training import query_horizon_bins


def test_query_horizon_bins_track_latest_same_key_set() -> None:
    # SET(A,1), QUERY(A), 8 NOOPs, QUERY(A), 24 NOOPs, QUERY(A)
    inputs = torch.tensor([[1, 33, *([0] * 8), 33, *([0] * 24), 33]])
    targets = torch.full_like(inputs, -100)
    targets[0, 1] = targets[0, 10] = targets[0, 35] = 1
    bins = query_horizon_bins(inputs, targets)
    assert bins[0, 1].item() == 0
    assert bins[0, 10].item() == 1
    assert bins[0, 35].item() == 2
