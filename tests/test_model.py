import torch
import torch.nn.functional as F

from task_model import StateTracker


def test_shapes_outputs_and_gradients() -> None:
    model = StateTracker()
    inputs = torch.randint(0, 41, (2, 12))
    targets = torch.full((2, 12), -100, dtype=torch.long)
    targets[:, (2, 8)] = torch.randint(0, 5, (2, 2))
    logits = model(inputs)
    assert logits.shape == (2, 12, 5)
    assert torch.isfinite(logits).all()
    loss = F.cross_entropy(logits.reshape(-1, 5), targets.reshape(-1), ignore_index=-100)
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_causality_future_tokens_do_not_change_earlier_outputs() -> None:
    torch.manual_seed(1)
    model = StateTracker().eval()
    left = torch.randint(0, 41, (1, 16))
    right = left.clone()
    right[:, 9:] = torch.randint(0, 41, (1, 7))
    with torch.no_grad():
        left_logits = model(left)
        right_logits = model(right)
    torch.testing.assert_close(left_logits[:, :9], right_logits[:, :9], rtol=0, atol=0)


def test_task_head_has_five_classes_and_vocab_is_padded_to_48() -> None:
    model = StateTracker()
    assert model.head.out_features == 5
    assert model.backbone.args.vocab_size == 48


def test_hidden_state_interface_matches_forward_head() -> None:
    model = StateTracker().eval()
    inputs = torch.randint(0, 41, (2, 7))
    with torch.no_grad():
        hidden = model.hidden_states(inputs)
        logits = model(inputs)
    assert hidden.shape == (2, 7, 32)
    torch.testing.assert_close(logits, model.head(hidden))


def test_delta_scale_one_preserves_baseline_and_resets_state() -> None:
    model = StateTracker().eval()
    inputs = torch.randint(0, 41, (2, 9))
    with torch.no_grad():
        baseline = model(inputs)
        scaled = model.scaled_logits(inputs, 1.0, "noop_only")
    torch.testing.assert_close(baseline, scaled)
    assert model.backbone.layers[0].mixer.delta_scale_by_position is None


def test_routing_variants_are_parameter_matched() -> None:
    baseline = StateTracker()
    routed = StateTracker("key_routing")
    tied = StateTracker("tied_routing")
    baseline_count = sum(parameter.numel() for parameter in baseline.parameters())
    assert sum(parameter.numel() for parameter in routed.parameters()) == baseline_count + 8 * 64
    assert sum(parameter.numel() for parameter in tied.parameters()) == baseline_count + 8 * 64


def test_key_routes_only_set_tokens_and_tied_control_removes_key_selectivity() -> None:
    inputs = torch.tensor([[0, 1, 5, 33, 32]])  # NOOP, SET A, SET B, QUERY A, SET H
    routed = StateTracker("key_routing")
    tied = StateTracker("tied_routing")
    with torch.no_grad():
        routed.route_logits[0].fill_(-2.0)
        routed.route_logits[1].fill_(2.0)
        tied.route_logits[0].fill_(-2.0)
        tied.route_logits[1].fill_(2.0)
    routed_values = routed.write_routes(inputs)
    tied_values = tied.write_routes(inputs)
    assert routed_values is not None and tied_values is not None
    torch.testing.assert_close(routed_values[:, 0], torch.ones_like(routed_values[:, 0]))
    torch.testing.assert_close(routed_values[:, 3], torch.ones_like(routed_values[:, 3]))
    assert not torch.allclose(routed_values[:, 1], routed_values[:, 2])
    torch.testing.assert_close(tied_values[:, 1], tied_values[:, 2])


def test_routing_receives_gradients_and_transient_mixer_state_resets() -> None:
    model = StateTracker("key_routing")
    inputs = torch.tensor([[1, 0, 33, 5, 0, 34]])
    model(inputs).sum().backward()
    assert model.route_logits.grad is not None
    assert torch.isfinite(model.route_logits.grad).all()
    assert model.backbone.layers[0].mixer.write_route_by_position is None


def test_separate_write_and_decay_identity_interventions_preserve_baseline() -> None:
    model = StateTracker().eval()
    inputs = torch.randint(0, 41, (2, 9))
    write_route = torch.ones(2, 9, 64)
    decay_scale = torch.ones(2, 9)
    with torch.no_grad():
        baseline = model(inputs)
        write_identity = model.head(model.hidden_states(inputs, write_route_override=write_route))
        decay_identity = model.head(model.hidden_states(inputs, decay_scale=decay_scale))
    torch.testing.assert_close(baseline, write_identity)
    torch.testing.assert_close(baseline, decay_identity)
    mixer = model.backbone.layers[0].mixer
    assert mixer.write_route_by_position is None
    assert mixer.decay_scale_by_position is None


def test_event_gate_variants_start_at_baseline_and_are_parameter_matched() -> None:
    torch.manual_seed(11)
    baseline = StateTracker().eval()
    torch.manual_seed(11)
    decay = StateTracker("event_decay").eval()
    torch.manual_seed(11)
    coupled = StateTracker("coupled_event_gate").eval()
    inputs = torch.tensor([[0, 1, 33, 5, 40, 0]])
    with torch.no_grad():
        torch.testing.assert_close(baseline(inputs), decay(inputs))
        torch.testing.assert_close(baseline(inputs), coupled(inputs))
    assert decay.event_gate.shape == (3, 64)
    assert coupled.event_gate.shape == (3, 64)
    assert sum(p.numel() for p in decay.parameters()) == sum(p.numel() for p in coupled.parameters())


def test_decoupled_event_gate_scales_only_decay_and_receives_gradients() -> None:
    model = StateTracker("event_decay")
    inputs = torch.tensor([[0, 1, 33, 5, 0, 34]])
    scale = model.event_retention_scale(inputs)
    assert scale is not None and scale.shape == (1, 6, 64)
    torch.testing.assert_close(scale, torch.ones_like(scale))
    model(inputs).sum().backward()
    assert model.event_gate.grad is not None
    assert torch.isfinite(model.event_gate.grad).all()
    mixer = model.backbone.layers[0].mixer
    assert mixer.decay_scale_by_position is None
    assert mixer.delta_scale_by_position is None


def test_convolution_width_is_configurable() -> None:
    narrow = StateTracker("event_decay", d_conv=1)
    wide = StateTracker("event_decay", d_conv=8)
    assert narrow.backbone.args.d_conv == 1
    assert wide.backbone.args.d_conv == 8
    assert narrow.backbone.layers[0].mixer.conv1d.kernel_size == (1,)
    assert wide.backbone.layers[0].mixer.conv1d.kernel_size == (8,)
    inputs = torch.randint(0, 41, (2, 12))
    assert narrow(inputs).shape == wide(inputs).shape == (2, 12, 5)


def test_convolution_history_reset_is_transient_and_causal() -> None:
    model = StateTracker("event_decay", d_conv=4).eval()
    inputs = torch.randint(0, 41, (2, 12))
    reset = torch.zeros_like(inputs, dtype=torch.bool)
    reset[:, 6] = True
    with torch.no_grad():
        baseline = model(inputs)
        intervened = model.head(model.hidden_states(inputs, conv_reset=reset))
    torch.testing.assert_close(baseline[:, :6], intervened[:, :6])
    assert not torch.allclose(baseline[:, 6:], intervened[:, 6:])
    assert model.backbone.layers[0].mixer.conv_reset_by_position is None


def test_adaptive_and_static_kernel_models_are_parameter_matched_and_start_at_width_four() -> None:
    adaptive = StateTracker("adaptive_kernel", d_conv=8).eval()
    static = StateTracker("static_kernel_mix", d_conv=8).eval()
    assert sum(p.numel() for p in adaptive.parameters()) == sum(p.numel() for p in static.parameters())
    inputs = torch.tensor([[0, 1, 33, 5, 40, 0]])
    for model in (adaptive, static):
        mixer = model.backbone.layers[0].mixer
        mixer.capture_conv_alpha = True
        with torch.no_grad():
            logits = model(inputs)
        assert logits.shape == (1, 6, 5)
        assert mixer.captured_conv_alpha is not None
        assert torch.all(mixer.captured_conv_alpha[..., 2] > 0.94)
        mixer.capture_conv_alpha = False
    torch.testing.assert_close(
        static.backbone.layers[0].mixer.captured_conv_alpha[:, :1].expand(-1, 5, -1),
        static.backbone.layers[0].mixer.captured_conv_alpha[:, 1:],
    )


def test_adaptive_kernel_router_receives_gradients() -> None:
    model = StateTracker("adaptive_kernel", d_conv=8)
    inputs = torch.randint(0, 41, (2, 12))
    model(inputs).sum().backward()
    mixer = model.backbone.layers[0].mixer
    assert mixer.conv_router.weight.grad is not None
    assert torch.isfinite(mixer.conv_router.weight.grad).all()


def test_kernel_only_variants_are_parameter_matched_and_have_no_retention_gate() -> None:
    adaptive = StateTracker("adaptive_kernel_only", d_conv=8)
    static = StateTracker("static_kernel_mix_only", d_conv=8)
    assert not hasattr(adaptive, "event_gate")
    assert not hasattr(static, "event_gate")
    assert adaptive.event_retention_scale(torch.zeros(1, 3, dtype=torch.long)) is None
    assert static.event_retention_scale(torch.zeros(1, 3, dtype=torch.long)) is None
    assert sum(parameter.numel() for parameter in adaptive.parameters()) == sum(
        parameter.numel() for parameter in static.parameters()
    )
    inputs = torch.randint(0, 41, (2, 12))
    adaptive(inputs).sum().backward()
    router = adaptive.backbone.layers[0].mixer.conv_router
    assert router.weight.grad is not None
    assert torch.isfinite(router.weight.grad).all()
