import torch

from h5_retention_readout import evaluate_readout, fit_readout


def test_h5_readout_has_five_classes_and_does_not_mutate_features() -> None:
    generator = torch.Generator().manual_seed(7)
    data = {
        "query_hidden": torch.randn(40, 32, generator=generator),
        "labels": torch.tensor([1, 2, 3, 4] * 10),
    }
    original = data["query_hidden"].clone()
    readout = fit_readout(data, 11, torch.device("cpu"))
    result = evaluate_readout(readout, data, torch.device("cpu"))
    assert readout["head"].out_features == 5
    assert result["predictions"].shape == (40,)
    torch.testing.assert_close(data["query_hidden"], original)
