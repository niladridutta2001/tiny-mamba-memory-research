import torch

from decay_only_retention_sweep import NOOP_START, QUERY_INDEX, intervention_scale, make_examples
from task_model import StateTracker


def test_decay_only_examples_have_exact_511_step_gap() -> None:
    sequences, labels, keys, clusters = make_examples(1, 123)
    assert len(sequences) == len(labels) == len(keys) == len(clusters) == 32
    assert all(len(sequence) == QUERY_INDEX + 1 for sequence in sequences)
    assert all(sequence[NOOP_START:QUERY_INDEX] == [0] * (QUERY_INDEX - NOOP_START) for sequence in sequences)


def test_decay_intervention_begins_after_convolutional_washout() -> None:
    model = StateTracker("event_decay", d_conv=4)
    inputs = torch.zeros(2, QUERY_INDEX + 1, dtype=torch.long)
    natural = model.event_retention_scale(inputs)
    scale = intervention_scale(model, inputs, 0.25)
    washout_start = NOOP_START + model.d_conv - 1
    torch.testing.assert_close(scale[:, :washout_start], natural[:, :washout_start])
    torch.testing.assert_close(scale[:, washout_start:QUERY_INDEX], natural[:, washout_start:QUERY_INDEX] * 0.25)
    torch.testing.assert_close(scale[:, QUERY_INDEX:], natural[:, QUERY_INDEX:])
