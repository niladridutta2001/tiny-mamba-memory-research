import random

from generate_dataset import encode_event, make_sequence


def test_oracle_handles_unset_overwrite_and_noop() -> None:
    item = make_sequence(random.Random(4), 64, "standard", 0)
    state = {}
    for event, target in zip(item["events"], item["query_targets"]):
        op, key, value = event
        if op == "SET":
            state[key] = value
            assert target is None
        elif op == "QUERY":
            assert target == state.get(key, 0)
        else:
            assert target is None


def test_targets_exist_only_at_query_positions() -> None:
    item = make_sequence(random.Random(7), 128, "standard", 0)
    assert all((event[0] == "QUERY") == (target is not None) for event, target in zip(item["events"], item["query_targets"]))


def test_no_target_leakage_in_query_tokens() -> None:
    assert {encode_event(["QUERY", "A", None]) for _ in range(5)} == {33}
    assert set(range(5)) == {0, 1, 2, 3, 4}


def test_generation_is_deterministic() -> None:
    left = make_sequence(random.Random(123), 64, "standard", 0)
    right = make_sequence(random.Random(123), 64, "standard", 0)
    assert left == right


def test_exact_41_token_encoding() -> None:
    tokens = {encode_event(["NOOP", None, None])}
    tokens.update(encode_event(["SET", key, value]) for key in "ABCDEFGH" for value in range(1, 5))
    tokens.update(encode_event(["QUERY", key, None]) for key in "ABCDEFGH")
    assert tokens == set(range(41))
