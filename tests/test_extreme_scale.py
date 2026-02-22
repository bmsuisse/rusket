import time

import numpy as np
import pandas as pd

from rusket import fpgrowth


def _make_df(n_rows: int, n_cols: int, rng: np.random.Generator) -> pd.DataFrame:
    support_values = np.zeros(n_cols)
    n_very_low = int(n_cols * 0.9)
    support_values[:n_very_low] = rng.uniform(0.0001, 0.009, n_very_low)
    n_medium = int(n_cols * 0.06)
    support_values[n_very_low : n_very_low + n_medium] = rng.uniform(
        0.01, 0.1, n_medium
    )
    n_high = n_cols - n_very_low - n_medium
    support_values[n_very_low + n_medium :] = rng.uniform(0.1, 0.65, n_high)
    return pd.DataFrame(
        {f"c{i}": (rng.random(n_rows) < support_values[i]) for i in range(n_cols)}
    )


RNG = np.random.default_rng(42)

# One Million Rows
DF_1M_200 = _make_df(1_000_000, 200, RNG)

t0 = time.perf_counter()
fpgrowth(DF_1M_200, min_support=0.05)
t1 = time.perf_counter()
print(f"Rusket 1M Rows x 200 Cols: {t1 - t0:.3f} s")

# 100k Rows x 2000 Items
DF_100k_2k = _make_df(100_000, 2_000, RNG)

t0 = time.perf_counter()
fpgrowth(DF_100k_2k, min_support=0.05)
t1 = time.perf_counter()
print(f"Rusket 100k Rows x 2,000 Cols: {t1 - t0:.3f} s")
