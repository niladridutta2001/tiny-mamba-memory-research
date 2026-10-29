import torch

from h5b_internal_readout import trace_internal
from task_model import StateTracker


def test_internal_trace_matches_normal_backbone_and_shapes() -> None:
    torch.manual_seed(17)
    model = StateTracker("event_decay", d_conv=4).eval()
    inputs = torch.randint(0, 41, (2, 11))
    scale = model.event_retention_scale(inputs)
    traced = trace_internal(model, inputs, scale, query_index=8)
    expected = model.hidden_states(inputs)
    torch.testing.assert_close(traced["final_hidden_all"], expected, rtol=1e-5, atol=1e-6)
    assert traced["state"].shape == (2, 512)
    assert traced["ssm_output"].shape == (2, 64)
    assert traced["final_feature"].shape == (2, 32)
