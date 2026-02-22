import pandas as pd
from rusket.prefixspan import prefixspan, sequences_from_event_log


def test_prefixspan_basic():
    # Sequence database
    # S1: A -> B -> C (1, 2, 3)
    # S2: A -> C (1, 3)
    # S3: B -> C (2, 3)
    # S4: A -> B (1, 2)

    sequences = [
        [1, 2, 3],
        [1, 3],
        [2, 3],
        [1, 2],
    ]

    # A appears in 3
    # B appears in 3
    # C appears in 3
    # A -> B appears in 2 (S1, S4)
    # A -> C appears in 2 (S1, S2)
    # B -> C appears in 2 (S1, S3)
    # A -> B -> C appears in 1 (S1)

    df = prefixspan(sequences, min_support=2)

    # Should find length 1: [1], [2], [3]
    # Length 2: [1, 2], [1, 3], [2, 3]
    assert len(df) == 6

    # Let's check support of [1, 2]
    # Convert sequences in dataframe to tuples for easy matching
    df["sequence_tuple"] = df["sequence"].apply(tuple)

    assert df[df["sequence_tuple"] == (1, 2)].iloc[0]["support"] == 2
    assert df[df["sequence_tuple"] == (1,)].iloc[0]["support"] == 3


def test_sequences_from_event_log():
    data = {
        "user": ["u1", "u1", "u1", "u2", "u2", "u3", "u3"],
        "time": [10, 20, 30, 15, 25, 5, 35],
        "item": ["A", "B", "C", "A", "C", "B", "C"],
    }
    df = pd.DataFrame(data)

    seqs, mapping = sequences_from_event_log(df, "user", "time", "item")

    assert len(seqs) == 3

    # mapping should map int -> item labels A, B, C
    assert list(mapping.values()) == ["A", "B", "C"] or len(mapping) == 3


def test_sequences_from_event_log_polars():
    try:
        import polars as pl
    except ImportError:
        import pytest
        pytest.skip("Polars not installed")

    data = {
        "user": ["u1", "u1", "u1", "u2", "u2", "u3", "u3"],
        "time": [10, 20, 30, 15, 25, 5, 35],
        "item": ["A", "B", "C", "A", "C", "B", "C"],
    }
    df = pl.DataFrame(data)

    seqs, mapping = sequences_from_event_log(df, "user", "time", "item")

    # Three unique users
    assert len(seqs) == 3
    # Mapped integer indices must match lengths 3, 2, 2
    assert len(seqs[0]) == 3
    assert len(seqs[1]) == 2
    assert len(seqs[2]) == 2

    assert len(mapping) == 3
