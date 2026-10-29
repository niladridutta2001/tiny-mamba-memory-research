from pathlib import Path

import torch

from train import load_config, query_loss


def test_query_loss_ignores_non_query_positions() -> None:
    torch.manual_seed(9)
    logits = torch.randn(1, 4, 5)
    targets = torch.tensor([[-100, 2, -100, 4]])
    original = query_loss(logits, targets)
    changed = logits.clone()
    changed[:, (0, 2)] = 1_000
    torch.testing.assert_close(original, query_loss(changed, targets))


def test_query_loss_backpropagates_only_at_queries() -> None:
    logits = torch.randn(1, 3, 5, requires_grad=True)
    targets = torch.tensor([[-100, 3, -100]])
    query_loss(logits, targets).backward()
    assert torch.count_nonzero(logits.grad[:, 0]).item() == 0
    assert torch.count_nonzero(logits.grad[:, 1]).item() > 0
    assert torch.count_nonzero(logits.grad[:, 2]).item() == 0


def test_inherited_variant_configs_keep_baseline_training_settings() -> None:
    proposed = load_config(Path("configs/proposed.yaml"))
    ablation = load_config(Path("configs/ablation.yaml"))
    assert proposed["training"]["batch_size"] == 16
    assert proposed["model"]["variant"] == "key_routing"
    assert ablation["model"]["variant"] == "tied_routing"
    assert ablation["model"]["route_init"] == 0.95
