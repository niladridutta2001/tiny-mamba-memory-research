from generate_dataset import KEYS
from h2_multiseed import make_design, sequences_for


def test_h2_zero_density_controls_are_identical_and_fixed_length() -> None:
    delay = 32
    designs = make_design(delay, n=32, seed=7)
    set_sequences, set_targets, _ = sequences_for(designs, delay, 0.0, "set")
    query_sequences, query_targets, _ = sequences_for(designs, delay, 0.0, "query")

    assert set_sequences == query_sequences
    assert set_targets.tolist() == query_targets.tolist()
    assert all(len(sequence) == 32 + 1 + delay + 1 for sequence in set_sequences)


def test_h2_full_density_never_sets_the_tracked_key() -> None:
    delay = 10
    designs = make_design(delay, n=32, seed=11)
    sequences, _, _ = sequences_for(designs, delay, 1.0, "set")

    for design, sequence in zip(designs, sequences):
        tracked_index = KEYS.index(design["tracked_key"])
        tracked_set_tokens = set(range(1 + 4 * tracked_index, 1 + 4 * tracked_index + 4))
        intervening = sequence[33 : 33 + delay]
        assert tracked_set_tokens.isdisjoint(intervening)
        assert all(1 <= token <= 32 for token in intervening)
