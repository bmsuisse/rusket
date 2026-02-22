from unittest.mock import patch

import pandas as pd

import rusket


def test_autominer_memory_fallback_pandas():
    # Dataset: 10 transactions, 5 items
    df = pd.DataFrame(
        {
            "A": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            "B": [1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
            "C": [0, 1, 1, 0, 0, 1, 1, 0, 0, 1],
        }
    )

    # Mock available memory to be very low (e.g. 100 bytes)
    # 10 * 3 = 30 bytes for dense matrix, so 100 * 0.7 = 70 bytes.
    # 30 < 70, so it shouldn't fallback yet.
    with patch("rusket._core._get_available_memory", return_value=100):
        with patch("rusket._core._run_fpminer_fallback") as mock_fallback:
            # We need to mock the return value so the rest of the code doesn't crash
            mock_fallback.return_value = pd.DataFrame(columns=["support", "itemsets"])

            # Normal run
            rusket.mine(df, min_support=0.1, method="auto")
            assert not mock_fallback.called

    # Now mock memory to be extremely low (e.g. 10 bytes)
    # 10 * 0.7 = 7 bytes. 30 > 7, so it SHOULD fallback.
    with patch("rusket._core._get_available_memory", return_value=10):
        with patch("rusket._core._run_fpminer_fallback") as mock_fallback:
            mock_fallback.return_value = pd.DataFrame(columns=["support", "itemsets"])

            rusket.mine(df, min_support=0.1, method="auto", verbose=1)
            assert mock_fallback.called


def test_autominer_memory_fallback_correctness():
    # Verify that it produces same results when falling back (actually executing)
    df = pd.DataFrame(
        {
            "A": [1, 1, 0, 1],
            "B": [1, 0, 1, 1],
        }
    )

    # Force fallback
    with patch("rusket._core._get_available_memory", return_value=1):
        res_fallback = rusket.mine(df, min_support=0.5, method="auto", use_colnames=True)

    # Normal run (fpgrowth)
    res_ref = rusket.fpgrowth(df, min_support=0.5, use_colnames=True)

    # Sort for comparison
    def sort_res(df):
        df["itemsets_key"] = df["itemsets"].apply(lambda x: tuple(sorted(x)))
        return df.sort_values("itemsets_key").drop(columns="itemsets_key").reset_index(drop=True)

    pd.testing.assert_frame_equal(sort_res(res_fallback), sort_res(res_ref))


if __name__ == "__main__":
    test_autominer_memory_fallback_pandas()
    test_autominer_memory_fallback_correctness()
    print("Memory fallback tests passed!")
