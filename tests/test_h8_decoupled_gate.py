import torch

from h7a_utils import trace_query
from task_model import StateTracker


def test_decoupled_and_tied_controls_are_parameter_matched() -> None:
    independent = StateTracker("decoupled_event_gate")
    tied = StateTracker("decoupled_tied_gate")
    assert sum(p.numel() for p in independent.parameters()) == sum(p.numel() for p in tied.parameters())


def test_decoupled_gates_have_expected_shape_and_different_initial_priors() -> None:
    model = StateTracker("decoupled_event_gate")
    mixer = model.backbone.layers[0].mixer
    u = torch.zeros(2, 64)
    fast = torch.zeros(2, 64, 4)
    erase, write = mixer.decoupled_slow_gates(u, fast, torch.zeros(2, dtype=torch.long))
    assert erase.shape == write.shape == (2, 64)
    assert erase.mean() < write.mean()


def test_tied_control_uses_one_gate_but_independent_variant_can_separate() -> None:
    tied = StateTracker("decoupled_tied_gate")
    independent = StateTracker("decoupled_event_gate")
    u = torch.randn(3, 64)
    fast = torch.randn(3, 64, 4)
    event_ids = torch.tensor([0, 1, 33])
    tied_r, tied_w = tied.backbone.layers[0].mixer.decoupled_slow_gates(u, fast, event_ids)
    independent_r, independent_w = independent.backbone.layers[0].mixer.decoupled_slow_gates(
        u, fast, event_ids
    )
    torch.testing.assert_close(tied_r, tied_w)
    assert not torch.equal(independent_r, independent_w)


def test_decoupled_manual_trace_matches_model_query_logits() -> None:
    for variant in ("decoupled_event_gate", "decoupled_tied_gate"):
        model = StateTracker(variant).eval()
        inputs = torch.randint(0, 41, (2, 19))
        with torch.no_grad():
            expected = model(inputs)[:, -1]
            traced = trace_query(model, inputs)
        torch.testing.assert_close(traced["original_logits"], expected, rtol=1e-5, atol=1e-6)
        assert traced["erase_gate"].shape == (2, 19, 64)
        assert traced["write_gate"].shape == (2, 19, 64)


def test_decoupled_recurrence_is_causal() -> None:
    model = StateTracker("decoupled_event_gate").eval()
    first = torch.randint(0, 41, (2, 20))
    second = first.clone()
    second[:, 11:] = torch.randint(0, 41, second[:, 11:].shape)
    with torch.no_grad():
        first_logits = model(first)
        second_logits = model(second)
    torch.testing.assert_close(first_logits[:, :11], second_logits[:, :11], rtol=1e-5, atol=1e-6)
