import torch

from h7a_utils import trace_query
from task_model import StateTracker


def test_identity_gate_variants_are_parameter_matched() -> None:
    adaptive = StateTracker("slow_identity_gate")
    static = StateTracker("slow_identity_static")
    assert sum(p.numel() for p in adaptive.parameters()) == sum(p.numel() for p in static.parameters())
    assert sum(p.numel() for p in adaptive.parameters()) == 10_277


def test_gate_starts_near_identity_and_has_expected_shape() -> None:
    model = StateTracker("slow_identity_gate")
    mixer = model.backbone.layers[0].mixer
    gate = mixer.slow_update_gate(torch.randn(2, 11, 64))
    assert gate is not None and gate.shape == (2, 11, 64)
    torch.testing.assert_close(gate, torch.full_like(gate, torch.sigmoid(torch.tensor(-4.0))))


def test_adaptive_gate_depends_on_input_but_static_control_does_not() -> None:
    adaptive = StateTracker("slow_identity_gate")
    static = StateTracker("slow_identity_static")
    with torch.no_grad():
        adaptive.backbone.layers[0].mixer.slow_gate_weight.fill_(1.0)
        static.backbone.layers[0].mixer.slow_gate_weight.fill_(1.0)
    first = torch.zeros(2, 5, 64)
    second = torch.ones(2, 5, 64)
    adaptive_mixer = adaptive.backbone.layers[0].mixer
    static_mixer = static.backbone.layers[0].mixer
    assert not torch.equal(adaptive_mixer.slow_update_gate(first), adaptive_mixer.slow_update_gate(second))
    torch.testing.assert_close(static_mixer.slow_update_gate(first), static_mixer.slow_update_gate(second))


def test_manual_trace_matches_model_query_logits() -> None:
    for variant in ("slow_identity_gate", "slow_identity_static"):
        model = StateTracker(variant).eval()
        inputs = torch.randint(0, 41, (2, 19))
        with torch.no_grad():
            expected = model(inputs)[:, -1]
            traced = trace_query(model, inputs)
        torch.testing.assert_close(traced["original_logits"], expected, rtol=1e-5, atol=1e-6)
        assert traced["state_sequence"].shape == (2, 19, 64, 8)
