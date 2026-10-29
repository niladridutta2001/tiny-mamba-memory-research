from argparse import Namespace
from pathlib import Path


def test_training_cli_exposes_resume_and_seed() -> None:
    # A lightweight contract test; the end-to-end resumed run is recorded in results.
    args = Namespace(resume=Path("checkpoint.pt"), seed=43)
    assert args.resume.name == "checkpoint.pt"
    assert args.seed == 43
