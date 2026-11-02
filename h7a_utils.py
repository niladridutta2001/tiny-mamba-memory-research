"""Shared utilities for the H7a fast/slow SSM pilot."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from einops import einsum

from generate_dataset import KEYS, NUM_VALUES, encode_event, random_event
from task_model import StateTracker


MODEL_NAMES = ("M0_original", "M1a_constrained", "M1b_unconstrained")
DELAYS = (10, 32, 64, 128, 256, 511)


def model_from_config(config: dict) -> StateTracker:
    model_config = config.get("model", {})
    return StateTracker(
        variant=str(model_config.get("variant", "baseline")),
        route_init=float(model_config.get("route_init", 0.95)),
        d_conv=int(model_config.get("d_conv", 4)),
        fast_min_rate=float(model_config.get("fast_min_rate", 1.0)),
        slow_max_rate=float(model_config.get("slow_max_rate", 0.004)),
        slow_gate_bias_init=float(model_config.get("slow_gate_bias_init", -4.0)),
        erase_gate_bias_init=float(model_config.get("erase_gate_bias_init", -4.0)),
        write_gate_bias_init=float(model_config.get("write_gate_bias_init", -2.0)),
    )


def load_checkpoint(path: Path, device: torch.device) -> tuple[StateTracker, dict]:
    payload = torch.load(path, map_location=device, weights_only=False)
    model = model_from_config(payload.get("config", {})).to(device)
    model.load_state_dict(payload["model_state_dict"])
    return model.eval(), payload


def make_retention_examples(delay: int, contexts_per_pair: int, seed: int) -> tuple[list[list[int]], list[int], list[int]]:
    """Balanced SET/NOOP/QUERY probes with unrelated pre-SET burn-in."""
    rng = random.Random(seed)
    noop = encode_event(["NOOP", None, None])
    sequences: list[list[int]] = []
    labels: list[int] = []
    keys: list[int] = []
    for key_index, key in enumerate(KEYS):
        for value in range(1, NUM_VALUES + 1):
            for _ in range(contexts_per_pair):
                burn_in = [encode_event(random_event(rng, exclude_key=key)) for _ in range(8)]
                sequences.append(
                    burn_in
                    + [encode_event(["SET", key, value])]
                    + [noop] * delay
                    + [encode_event(["QUERY", key, None])]
                )
                labels.append(value)
                keys.append(key_index)
    return sequences, labels, keys


@torch.no_grad()
def trace_query(model: StateTracker, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
    """Run the actual one-layer recurrence and expose query-time internal features."""
    layer = model.backbone.layers[0]
    mixer = layer.mixer
    if mixer.args.conv_mode != "fixed":
        raise ValueError("H7a trace expects the unchanged fixed causal convolution")
    embedded = model.backbone.embedding(inputs)
    block_input = layer.norm(embedded)
    projected = mixer.in_proj(block_input)
    u, residual_gate = projected.split([mixer.args.d_inner, mixer.args.d_inner], dim=-1)
    length = inputs.shape[1]
    u = mixer.conv1d(u.transpose(1, 2))[:, :, :length].transpose(1, 2)
    u = F.silu(u)
    raw = mixer.x_proj(u)
    delta_raw, b_vector, c_vector = raw.split(
        [mixer.args.dt_rank, mixer.args.d_state, mixer.args.d_state], dim=-1
    )
    delta = F.softplus(mixer.dt_proj(delta_raw))
    decay = mixer.decay_matrix()
    delta_a = torch.exp(einsum(delta, decay, "b l d, d n -> b l d n"))
    delta_b_u = einsum(delta, b_vector, u, "b l d, b l n, b l d -> b l d n")
    state = torch.zeros(
        inputs.shape[0], mixer.args.d_inner, mixer.args.d_state, device=inputs.device, dtype=delta.dtype
    )
    recurrent_outputs = []
    state_outputs = []
    slow_gate = mixer.slow_update_gate(u)
    erase_gates = []
    write_gates = []
    decoupled = mixer.args.slow_gate_mode.startswith("decoupled")
    for timestep in range(length):
        candidate = delta_a[:, timestep] * state + delta_b_u[:, timestep]
        if decoupled:
            fast = candidate[:, :, :4]
            erase_gate, write_gate = mixer.decoupled_slow_gates(
                u[:, timestep], fast, inputs[:, timestep]
            )
            slow = (
                (1.0 - erase_gate.unsqueeze(-1)) * state[:, :, 4:]
                + write_gate.unsqueeze(-1) * delta_b_u[:, timestep, :, 4:]
            )
            state = torch.cat((fast, slow), dim=-1)
            erase_gates.append(erase_gate)
            write_gates.append(write_gate)
        elif slow_gate is None:
            state = candidate
        else:
            gate = slow_gate[:, timestep].unsqueeze(-1)
            state = torch.cat(
                (candidate[:, :, :4], (1.0 - gate) * state[:, :, 4:] + gate * candidate[:, :, 4:]),
                dim=-1,
            )
        state_outputs.append(state)
        recurrent_outputs.append(einsum(state, c_vector[:, timestep], "b d n, b n -> b d"))
    recurrent = torch.stack(recurrent_outputs, dim=1)
    y = recurrent + u * mixer.D.float()
    gated = y * F.silu(residual_gate)
    mixed = mixer.out_proj(gated)
    final_hidden = model.backbone.norm_f(mixed + embedded)
    return {
        "fast": state[:, :, :4].flatten(1),
        "slow": state[:, :, 4:].flatten(1),
        "full": state.flatten(1),
        "ssm_output": recurrent[:, -1],
        "final_feature": final_hidden[:, -1],
        "original_logits": model.head(final_hidden[:, -1]),
        "delta": delta,
        "log_retention": einsum(delta, decay, "b l d, d n -> b l d n"),
        "slow_gate": slow_gate,
        "erase_gate": torch.stack(erase_gates, dim=1) if erase_gates else None,
        "write_gate": torch.stack(write_gates, dim=1) if write_gates else None,
        "state_sequence": torch.stack(state_outputs, dim=1),
    }


@torch.no_grad()
def extract_query_features(
    model: StateTracker,
    examples: tuple[list[list[int]], list[int], list[int]],
    device: torch.device,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    sequences, labels, keys = examples
    pieces = {name: [] for name in ("fast", "slow", "full", "ssm_output", "final_feature", "original_logits")}
    for start in range(0, len(sequences), batch_size):
        inputs = torch.tensor(sequences[start : start + batch_size], dtype=torch.long, device=device)
        traced = trace_query(model, inputs)
        for name in pieces:
            pieces[name].append(traced[name].cpu())
    return {
        **{name: torch.cat(values) for name, values in pieces.items()},
        "labels": torch.tensor(labels, dtype=torch.long),
        "keys": torch.tensor(keys, dtype=torch.long),
    }


def probe_inputs(data: dict[str, torch.Tensor], feature: str) -> torch.Tensor:
    values = data[feature].float()
    if feature in {"fast", "slow", "full"}:
        values = torch.cat((values, F.one_hot(data["keys"], num_classes=8).float()), dim=-1)
    return values


def fit_probe(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    validation_x: torch.Tensor,
    validation_y: torch.Tensor,
    seed: int,
    device: torch.device,
) -> dict:
    mean = train_x.mean(0)
    scale = train_x.std(0).clamp_min(1e-5)
    train_x = train_x.to(device)
    train_y = (train_y - 1).to(device)
    validation_x = validation_x.to(device)
    validation_y = (validation_y - 1).to(device)
    mean = mean.to(device)
    scale = scale.to(device)
    torch.manual_seed(seed)
    head = torch.nn.Linear(train_x.shape[1], 4).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.03, weight_decay=1e-4)
    best_state = None
    best_accuracy = -1.0
    for _ in range(500):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(head((train_x - mean) / scale), train_y)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            accuracy = head((validation_x - mean) / scale).argmax(-1).eq(validation_y).float().mean().item()
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_state = {name: value.detach().cpu().clone() for name, value in head.state_dict().items()}
    assert best_state is not None
    head.load_state_dict(best_state)
    return {"head": head.eval(), "mean": mean, "scale": scale, "validation_accuracy": best_accuracy}


@torch.no_grad()
def evaluate_probe(probe: dict, features: torch.Tensor, labels: torch.Tensor, device: torch.device) -> dict:
    logits = probe["head"]((features.to(device) - probe["mean"]) / probe["scale"])
    predictions = logits.argmax(-1).cpu() + 1
    correct = predictions.eq(labels)
    per_class = [correct[labels == value].float().mean().item() for value in range(1, 5)]
    return {
        "accuracy": correct.float().mean().item(),
        "balanced_accuracy": float(np.mean(per_class)),
        "correct": correct.numpy().astype(np.int8),
    }


def paired_bootstrap(first: np.ndarray, second: np.ndarray, seed: int, iterations: int = 5000) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    differences = first.astype(np.float32) - second.astype(np.float32)
    effects = np.empty(iterations, dtype=np.float32)
    for index in range(iterations):
        sampled = rng.integers(0, len(differences), len(differences))
        effects[index] = differences[sampled].mean()
    return float(differences.mean()), float(np.quantile(effects, 0.025)), float(np.quantile(effects, 0.975))
