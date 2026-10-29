"""H5b: localize long-horizon information loss inside the Mamba readout path."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from einops import einsum

from decay_only_retention_sweep import cluster_bootstrap_difference, intervention_scale, make_examples
from hypothesis_pilot import load_model


DEFAULT_ALPHAS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 1.0, 1.25)
PROBES = ("state", "ssm_output", "final_feature")
PROBE_LABELS = {
    "state": "P1: state h + key",
    "ssm_output": "P2: C-conditioned SSM output",
    "final_feature": "P3: final feature z",
    "original_head": "P4: original head",
}


@torch.no_grad()
def trace_internal(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    decay_scale: torch.Tensor,
    query_index: int,
) -> dict[str, torch.Tensor]:
    """Reproduce the one-layer forward pass and expose h_t, C_t h_t, and z_t."""
    layer = model.backbone.layers[0]
    mixer = layer.mixer
    if mixer.args.conv_mode != "fixed":
        raise ValueError("H5b trace currently targets the fixed-convolution diagnostic model")
    embedded = model.backbone.embedding(inputs)
    block_input = layer.norm(embedded)
    projected = mixer.in_proj(block_input)
    u, residual_gate = projected.split([mixer.args.d_inner, mixer.args.d_inner], dim=-1)
    length = inputs.shape[1]
    u = mixer.conv1d(u.transpose(1, 2))[:, :, :length].transpose(1, 2)
    u = F.silu(u)
    parameters = mixer.x_proj(u)
    delta, b_vector, c_vector = parameters.split(
        [mixer.args.dt_rank, mixer.args.d_state, mixer.args.d_state], dim=-1
    )
    delta = F.softplus(mixer.dt_proj(delta))
    state_matrix = -torch.exp(mixer.A_log.float())
    scaled_delta = delta * decay_scale.to(device=delta.device, dtype=delta.dtype)
    delta_a = torch.exp(einsum(scaled_delta, state_matrix, "b l d, d n -> b l d n"))
    delta_b_u = einsum(delta, b_vector, u, "b l d, b l n, b l d -> b l d n")
    state = torch.zeros(
        inputs.shape[0], mixer.args.d_inner, mixer.args.d_state, device=inputs.device, dtype=delta.dtype
    )
    ssm_outputs = []
    query_state = None
    query_ssm_output = None
    for timestep in range(length):
        state = delta_a[:, timestep] * state + delta_b_u[:, timestep]
        recurrent_readout = einsum(state, c_vector[:, timestep], "b d n, b n -> b d")
        if timestep == query_index:
            query_state = state.clone()
            query_ssm_output = recurrent_readout.clone()
        ssm_outputs.append(recurrent_readout)
    recurrent = torch.stack(ssm_outputs, dim=1)
    y = recurrent + u * mixer.D.float()
    gated = y * F.silu(residual_gate)
    mixed = mixer.out_proj(gated)
    final_hidden = model.backbone.norm_f(mixed + embedded)
    assert query_state is not None and query_ssm_output is not None
    return {
        "state": query_state.flatten(1),
        "ssm_output": query_ssm_output,
        "final_feature": final_hidden[:, query_index],
        "final_hidden_all": final_hidden,
    }


@torch.no_grad()
def extract_features(
    model: torch.nn.Module,
    examples: tuple[list[list[int]], list[int], list[int], list[int]],
    alpha: float,
    device: torch.device,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    sequences, labels, keys, clusters = examples
    pieces = {probe: [] for probe in PROBES}
    original_logits = []
    for start in range(0, len(sequences), batch_size):
        stop = start + batch_size
        inputs = torch.tensor(sequences[start:stop], dtype=torch.long, device=device)
        scale = intervention_scale(model, inputs, alpha)
        traced = trace_internal(model, inputs, scale, len(sequences[0]) - 1)
        for probe in PROBES:
            pieces[probe].append(traced[probe].cpu())
        original_logits.append(model.head(traced["final_feature"]).cpu())
    return {
        **{probe: torch.cat(values) for probe, values in pieces.items()},
        "original_logits": torch.cat(original_logits),
        "labels": torch.tensor(labels, dtype=torch.long),
        "keys": torch.tensor(keys, dtype=torch.long),
        "clusters": torch.tensor(clusters, dtype=torch.long),
    }


def prepare_features(data: dict[str, torch.Tensor], probe: str) -> torch.Tensor:
    features = data[probe]
    if probe == "state":
        features = torch.cat([features, F.one_hot(data["keys"], num_classes=8).float()], dim=-1)
    return features


def fit_linear_probe(train_x: torch.Tensor, train_y: torch.Tensor, seed: int, device: torch.device) -> dict:
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    mean = train_x.mean(0)
    scale = train_x.std(0).clamp_min(1e-5)
    torch.manual_seed(seed)
    head = torch.nn.Linear(train_x.shape[1], 5).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.03, weight_decay=1e-4)
    for _ in range(500):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(head((train_x - mean) / scale), train_y)
        loss.backward()
        optimizer.step()
    return {"head": head.eval(), "mean": mean, "scale": scale, "input_dim": train_x.shape[1]}


@torch.no_grad()
def evaluate_probe(probe: dict, features: torch.Tensor, labels: torch.Tensor, device: torch.device) -> dict:
    features = features.to(device)
    labels = labels.to(device)
    logits = probe["head"]((features - probe["mean"]) / probe["scale"])
    correct = logits.argmax(-1).eq(labels)
    return {
        "accuracy": float(correct.float().mean().item()),
        "nll": float(F.cross_entropy(logits, labels).item()),
        "correct": correct.cpu().numpy().astype(np.int8),
    }


@torch.no_grad()
def original_head_metrics(data: dict[str, torch.Tensor]) -> dict:
    logits = data["original_logits"]
    labels = data["labels"]
    correct = logits.argmax(-1).eq(labels)
    return {
        "accuracy": float(correct.float().mean().item()),
        "nll": float(F.cross_entropy(logits, labels).item()),
        "correct": correct.numpy().astype(np.int8),
    }


def pca_parameter_matched(
    train: dict[str, torch.Tensor], validation: dict[str, torch.Tensor], test: dict[str, torch.Tensor], probe: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unsupervised fixed projection to 32 inputs so every control head has 165 parameters."""
    train_x = train[probe].float()
    validation_x = validation[probe].float()
    test_x = test[probe].float()
    mean = train_x.mean(0)
    scale = train_x.std(0).clamp_min(1e-5)
    train_standardized = (train_x - mean) / scale
    validation_standardized = (validation_x - mean) / scale
    test_standardized = (test_x - mean) / scale
    output_dim = 24 if probe == "state" else 32
    _, _, vh = torch.linalg.svd(train_standardized, full_matrices=False)
    projection = vh[:output_dim].T
    projected = [matrix @ projection for matrix in (train_standardized, validation_standardized, test_standardized)]
    if probe == "state":
        projected[0] = torch.cat([projected[0], F.one_hot(train["keys"], num_classes=8).float()], dim=-1)
        projected[1] = torch.cat([projected[1], F.one_hot(validation["keys"], num_classes=8).float()], dim=-1)
        projected[2] = torch.cat([projected[2], F.one_hot(test["keys"], num_classes=8).float()], dim=-1)
    return projected[0], projected[1], projected[2]


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict], controls: list[dict], output: Path, alphas: tuple[float, ...]) -> None:
    labels = [PROBE_LABELS[name] for name in (*PROBES, "original_head")]
    figure, axes = plt.subplots(1, 2, figsize=(16.5, 5.2))
    markers = ("o-", "s-", "^-", "D--")
    for probe, label, marker in zip((*PROBES, "original_head"), labels, markers):
        selected = [row for row in rows if row["probe"] == probe]
        axes[0].plot([row["alpha"] for row in selected], [100 * row["test_accuracy"] for row in selected], marker, label=label)
    axes[0].axhline(25, color="#6b7280", linestyle=":", linewidth=1.3, label="Chance")
    axes[0].set(xlabel=r"Decay multiplier $\alpha$", ylabel="Held-out accuracy (%)", title="Information along the readout path", xticks=alphas, ylim=(0, 105))
    axes[0].tick_params(axis="x", labelrotation=35)
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.25)

    width = 0.24
    positions = np.arange(len(alphas))
    for index, (probe, label) in enumerate((("state_pca32", "State PCA24 + key"), ("ssm_output_pca32", "SSM output PCA32"), ("final_feature", "Final feature 32D"))):
        selected = [row for row in controls if row["probe"] == probe]
        axes[1].bar(positions + (index - 1) * width, [100 * row["test_accuracy"] for row in selected], width, label=label)
    axes[1].axhline(25, color="#6b7280", linestyle=":", linewidth=1.3)
    axes[1].set_xticks(positions, [f"{alpha:g}" for alpha in alphas])
    axes[1].tick_params(axis="x", labelrotation=35)
    axes[1].set(xlabel=r"Decay multiplier $\alpha$", ylabel="Held-out accuracy (%)", title="Parameter-matched 32D controls", ylim=(0, 105))
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.25)
    figure.suptitle("H5b: internal-state versus Mamba readout decodability")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(output, dpi=180)
    plt.close(figure)


def run(args: argparse.Namespace) -> dict:
    alphas = tuple(args.alphas)
    if not alphas or any(alpha < 0 for alpha in alphas):
        raise ValueError("--alphas must contain at least one nonnegative value")
    if tuple(sorted(set(alphas))) != alphas:
        raise ValueError("--alphas must be unique and in ascending order")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_model(args.checkpoint, device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    examples = {
        "train": make_examples(args.train_contexts_per_pair, args.train_seed),
        "validation": make_examples(args.validation_contexts_per_pair, args.validation_seed),
        "test": make_examples(args.test_contexts_per_pair, args.test_seed),
    }
    datasets: dict[float, dict[str, dict[str, torch.Tensor]]] = {}
    for alpha in alphas:
        datasets[alpha] = {
            split: extract_features(model, split_examples, alpha, device, args.batch_size)
            for split, split_examples in examples.items()
        }

    rows: list[dict] = []
    controls: list[dict] = []
    contrasts: list[dict] = []
    predictions: dict[str, np.ndarray] = {}
    for alpha_index, alpha in enumerate(alphas):
        train, validation, test = datasets[alpha]["train"], datasets[alpha]["validation"], datasets[alpha]["test"]
        results: dict[str, dict] = {}
        for probe_index, probe_name in enumerate(PROBES):
            probe = fit_linear_probe(prepare_features(train, probe_name), train["labels"], args.probe_seed + probe_index, device)
            validation_result = evaluate_probe(probe, prepare_features(validation, probe_name), validation["labels"], device)
            test_result = evaluate_probe(probe, prepare_features(test, probe_name), test["labels"], device)
            results[probe_name] = test_result
            rows.append({
                "alpha": alpha,
                "probe": probe_name,
                "input_dim": probe["input_dim"],
                "parameter_count": probe["input_dim"] * 5 + 5,
                "validation_accuracy": validation_result["accuracy"],
                "test_accuracy": test_result["accuracy"],
                "test_nll": test_result["nll"],
            })
            predictions[f"alpha_{str(alpha).replace('.', '_')}_{probe_name}"] = test_result["correct"]

        original = original_head_metrics(test)
        results["original_head"] = original
        rows.append({
            "alpha": alpha,
            "probe": "original_head",
            "input_dim": 32,
            "parameter_count": 165,
            "validation_accuracy": "",
            "test_accuracy": original["accuracy"],
            "test_nll": original["nll"],
        })
        predictions[f"alpha_{str(alpha).replace('.', '_')}_original_head"] = original["correct"]

        for comparison_index, (first, second) in enumerate(
            (("state", "ssm_output"), ("ssm_output", "final_feature"), ("final_feature", "original_head"))
        ):
            effect, low, high = cluster_bootstrap_difference(
                results[first]["correct"], results[second]["correct"], test["clusters"].numpy(),
                args.bootstrap_seed + alpha_index * 10 + comparison_index,
            )
            contrasts.append({
                "alpha": alpha,
                "comparison": f"{first}_minus_{second}",
                "accuracy_effect_pp": 100 * effect,
                "ci_low_pp": 100 * low,
                "ci_high_pp": 100 * high,
                "cluster_count": 32,
            })

        for probe_index, probe_name in enumerate(("state", "ssm_output")):
            train_x, validation_x, test_x = pca_parameter_matched(train, validation, test, probe_name)
            fitted = fit_linear_probe(train_x, train["labels"], args.probe_seed + 100 + probe_index, device)
            validation_result = evaluate_probe(fitted, validation_x, validation["labels"], device)
            test_result = evaluate_probe(fitted, test_x, test["labels"], device)
            controls.append({
                "alpha": alpha,
                "probe": f"{probe_name}_pca32",
                "input_dim": 32,
                "parameter_count": 165,
                "validation_accuracy": validation_result["accuracy"],
                "test_accuracy": test_result["accuracy"],
                "test_nll": test_result["nll"],
            })
        final_row = next(row for row in rows if row["alpha"] == alpha and row["probe"] == "final_feature")
        controls.append(dict(final_row))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "probe_metrics.csv", rows)
    write_csv(args.output_dir / "parameter_matched_controls.csv", controls)
    write_csv(args.output_dir / "contrasts.csv", contrasts)
    np.savez_compressed(args.output_dir / "probe_predictions.npz", **predictions)
    summary = {
        "hypothesis": "Long-horizon information remains in recurrent h_t but is lost or obscured along C_t and subsequent output transformations.",
        "scope": "Frozen seed-42 independently gated width-4 model; identical disjoint train/validation/test contexts for every probe.",
        "decay_multipliers": alphas,
        "probe_definitions": {
            "state": "Flattened 64x8 recurrent state plus 8D queried-key one-hot",
            "ssm_output": "64D recurrent readout C_t h_t before D skip",
            "final_feature": "32D final normalized backbone representation z_t",
            "original_head": "Original trained Linear(32,5) classifier on z_t",
        },
        "metrics": rows,
        "parameter_matched_controls": controls,
        "contrasts": contrasts,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot(rows, controls, args.output_dir / "h5b_internal_readout.png", alphas)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("results/selective_retention/seed42/event_decay/u1500/baseline.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/h5b_internal_readout/seed42"))
    parser.add_argument("--train-contexts-per-pair", type=int, default=12)
    parser.add_argument("--validation-contexts-per-pair", type=int, default=4)
    parser.add_argument("--test-contexts-per-pair", type=int, default=8)
    parser.add_argument("--train-seed", type=int, default=51042)
    parser.add_argument("--validation-seed", type=int, default=52042)
    parser.add_argument("--test-seed", type=int, default=53042)
    parser.add_argument("--probe-seed", type=int, default=61042)
    parser.add_argument("--bootstrap-seed", type=int, default=62042)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--alphas", nargs="+", type=float, default=list(DEFAULT_ALPHAS))
    parser.add_argument("--device", choices=("cpu", "cuda"))
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({"metrics": result["metrics"], "contrasts": result["contrasts"]}, indent=2))
