import torch

from horizon_robust_readout import apply_affine, fit_affine


def test_fitted_affine_recovers_known_mapping() -> None:
    torch.manual_seed(9)
    source = torch.randn(100, 4)
    weight = torch.randn(4, 3)
    bias = torch.randn(3)
    target = source @ weight + bias
    fitted = fit_affine(source, target, ridge=1e-8)
    torch.testing.assert_close(apply_affine(source, fitted), target, rtol=1e-4, atol=1e-5)
