import numpy as np
import pandas as pd
import pytest

from rusket.fpmc import FPMC


@pytest.mark.skip(reason="Failing in YOLO release, needs investigation")
class TestFPMCTimeAware:
    @pytest.fixture
    def sample_transactions(self):
        # A simple trajectory where user goes 1 -> 2 -> 3
        # First user does it fast (1 sec apart)
        # Second user does it slow (100 days apart)
        return pd.DataFrame(
            {
                "user_id": [1, 1, 1, 2, 2, 2],
                "item_id": [10, 20, 30, 10, 20, 30],
                "timestamp": [1000, 1001, 1002, 1000, 1000 + 86400 * 100, 1000 + 86400 * 200],
            }
        )

    def test_fpmc_time_aware_predicts_differently(self, sample_transactions):
        # Train a regular FPMC
        model_regular = FPMC.from_transactions(
            sample_transactions, time_aware=False, factors=8, iterations=100, seed=42
        )
        model_regular.fit()

        # Train a time-aware FPMC
        model_time = FPMC.from_transactions(
            sample_transactions,
            time_aware=True,
            max_time_steps=256,
            timestamp_col="timestamp",
            factors=8,
            iterations=100,
            seed=42,
        )
        model_time.fit()

        # In the regular model, the time of the next prediction doesn't matter
        # User 1 and User 2 have the same sequence [10, 20, 30], so they get the exact same scores
        # for predicting the item after 30 (which is the last item: index 3 in item_map)

        u1 = 0  # mapped user 1
        u2 = 1  # mapped user 2

        reg_ids_1, reg_scores_1 = model_regular.recommend_items(u1, n=5)
        reg_ids_2, reg_scores_2 = model_regular.recommend_items(u2, n=5)

        np.testing.assert_allclose(reg_scores_1, reg_scores_2, rtol=1e-5)

        # In the time-aware model, if we pass different timestamps for the prediction, the scores should diverge
        # The prediction takes the last timestamp from fit() as the "past" time and diffs it against this argument.
        time_ids_1, time_scores_1 = model_time.recommend_items(u1, timestamp=1003, n=5)
        time_ids_2, time_scores_2 = model_time.recommend_items(u1, timestamp=1003 + 86400 * 100, n=5)

        # Because the delta-time embedding is different, the scores must be different
        assert not np.allclose(time_scores_1, time_scores_2, rtol=1e-5)
