import torch

from task_model import StateTracker


def test_partition_indices_cover_state_once() -> None:
    fast = set(range(0, 4))
    slow = set(range(4, 8))
    assert fast.isdisjoint(slow)
    assert fast | slow == set(range(8))


def test_h7a_shapes_constraints_and_parameter_match() -> None:
    m1 = StateTracker("fast_slow", fast_min_rate=1.0, slow_max_rate=0.004)
    m2 = StateTracker("fast_slow_unconstrained", fast_min_rate=1.0, slow_max_rate=0.004)
    inputs = torch.randint(0, 41, (2, 17))
    assert m1.hidden_states(inputs).shape == (2, 17, 32)
    assert m1(inputs).shape == (2, 17, 5)
    assert m1.backbone.layers[0].mixer.decay_matrix().shape == (64, 8)
    assert sum(parameter.numel() for parameter in m1.parameters()) == sum(
        parameter.numel() for parameter in m2.parameters()
    )
    decay = m1.backbone.layers[0].mixer.decay_matrix()
    assert torch.all(decay < 0)
    assert torch.all(decay[:, 4:].abs() < 0.004)
    assert torch.all(decay[:, :4].abs() > 1.0)
    torch.testing.assert_close(
        m1.backbone.layers[0].mixer.decay_matrix(),
        m2.backbone.layers[0].mixer.decay_matrix(),
    )


def test_gradients_reach_both_mode_groups_and_shared_path() -> None:
    model = StateTracker("fast_slow")
    loss = model(torch.randint(0, 41, (2, 12))).square().mean()
    loss.backward()
    mixer = model.backbone.layers[0].mixer
    assert mixer.A_fast_theta.grad is not None and torch.isfinite(mixer.A_fast_theta.grad).all()
    assert mixer.A_slow_theta.grad is not None and torch.isfinite(mixer.A_slow_theta.grad).all()
    assert mixer.x_proj.weight.grad is not None
    assert model.head.weight.grad is not None


def test_readout_mask_does_not_change_recurrent_state() -> None:
    from h7a_utils import trace_query

    model = StateTracker("fast_slow").eval()
    inputs = torch.randint(0, 41, (2, 20))
    reference = trace_query(model, inputs)["full"]
    model.set_readout_mode_mask("fast_only")
    masked = trace_query(model, inputs)["full"]
    model.set_readout_mode_mask(None)
    torch.testing.assert_close(reference, masked)


def test_no_future_token_leakage() -> None:
    model = StateTracker("fast_slow").eval()
    first = torch.randint(0, 41, (2, 20))
    second = first.clone()
    second[:, 12:] = torch.randint(0, 41, (2, 8))
    with torch.no_grad():
        first_logits = model(first)
        second_logits = model(second)
    torch.testing.assert_close(first_logits[:, :12], second_logits[:, :12])


def test_baseline_decay_path_is_original_parameterization() -> None:
    model = StateTracker("baseline")
    mixer = model.backbone.layers[0].mixer
    torch.testing.assert_close(mixer.decay_matrix(), -torch.exp(mixer.A_log.float()))
    assert not hasattr(mixer, "A_fast_theta")
