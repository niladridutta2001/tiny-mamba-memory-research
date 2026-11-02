import torch

from h7a_utils import make_retention_examples, probe_inputs, trace_query
from task_model import StateTracker


def test_probe_examples_are_balanced_and_seed_disjoint() -> None:
    first = make_retention_examples(32, 2, 71042)
    second = make_retention_examples(32, 2, 72042)
    assert len(first[0]) == 8 * 4 * 2
    assert torch.bincount(torch.tensor(first[1]), minlength=5)[1:].tolist() == [16, 16, 16, 16]
    assert first[0] != second[0]


def test_trace_returns_exact_query_feature_shapes_and_key_conditioning() -> None:
    model = StateTracker("fast_slow").eval()
    inputs = torch.randint(0, 41, (3, 14))
    traced = trace_query(model, inputs)
    assert traced["fast"].shape == (3, 256)
    assert traced["slow"].shape == (3, 256)
    assert traced["full"].shape == (3, 512)
    assert traced["ssm_output"].shape == (3, 64)
    assert traced["final_feature"].shape == (3, 32)
    data = {"fast": traced["fast"], "keys": torch.tensor([0, 3, 7])}
    prepared = probe_inputs(data, "fast")
    assert prepared.shape == (3, 264)
    torch.testing.assert_close(prepared[:, -8:].sum(-1), torch.ones(3))
