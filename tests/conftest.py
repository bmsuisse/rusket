"""pytest configuration and shared fixtures."""

from __future__ import annotations

import os
import sys
import zipfile
from pathlib import Path

import pytest

# Ensure tests/ dir is on path so test_fpbase imports work
sys.path.insert(0, os.path.dirname(__file__))

# ---------------------------------------------------------------------------
# Cache directory for downloaded test datasets
# ---------------------------------------------------------------------------

CACHE_DIR = Path(__file__).parent / ".dataset_cache"


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "integration: real-world dataset tests (may download data on first run)",
    )
    config.addinivalue_line(
        "markers",
        "kaggle: requires a Kaggle API token (~/.kaggle/kaggle.json)",
    )


# ---------------------------------------------------------------------------
# UCI Online Retail II  — no authentication required
# Cached to tests/.dataset_cache/online_retail_II.parquet after first download
# ---------------------------------------------------------------------------

UCI_URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
UCI_PARQUET = CACHE_DIR / "online_retail_II.parquet"
UCI_SAMPLE_PARQUET = CACHE_DIR / "online_retail_II_sample.parquet"  # 10k rows
UCI_SAMPLE_ROWS = 10_000


@pytest.fixture(scope="session")
def online_retail_df():  # type: ignore[return]
    """Return a 10k-row Pandas DataFrame from the UCI Online Retail II dataset.

    On first run the dataset is downloaded from UCI and cached to
    ``tests/.dataset_cache/online_retail_II_sample.parquet``.  Subsequent
    runs load the parquet cache (< 1 s).

    Skips automatically if the download fails (e.g. in offline CI).
    """
    import pandas as pd

    CACHE_DIR.mkdir(exist_ok=True)

    if UCI_SAMPLE_PARQUET.exists():
        return pd.read_parquet(UCI_SAMPLE_PARQUET)

    # Try to download
    try:
        import urllib.request

        zip_path = CACHE_DIR / "online_retail_II.zip"
        if not zip_path.exists():
            urllib.request.urlretrieve(UCI_URL, zip_path)

        xl_path = CACHE_DIR / "online_retail_II.xlsx"
        if not xl_path.exists():
            with zipfile.ZipFile(zip_path, "r") as z:
                xlsx_names = [n for n in z.namelist() if n.lower().endswith(".xlsx")]
                if not xlsx_names:
                    pytest.skip("No .xlsx found in UCI zip")
                z.extract(xlsx_names[0], CACHE_DIR)
                (CACHE_DIR / xlsx_names[0]).rename(xl_path)

        # Read only sheet 0 (Year 2009-2010) and take a reproducible sample
        df_raw = pd.read_excel(xl_path, sheet_name=0, engine="openpyxl")
        df_raw.columns = [c.strip().replace(" ", "_") for c in df_raw.columns]
        df = df_raw[
            (df_raw["Country"] == "United Kingdom")
            & (df_raw["Quantity"] > 0)
            & (df_raw["Price"] > 0)
            & (~df_raw["Invoice"].astype(str).str.startswith("C"))
            & df_raw["Customer_ID"].notna()
            & df_raw["StockCode"].astype(str).str.match(r"^\d{5}")
        ].copy()
        df["Description"] = df["Description"].astype(str).str.strip()  # type: ignore[operator]
        df["Revenue"] = (df["Quantity"] * df["Price"]).round(2)
        # Ensure uniform dtypes for Parquet serialisation (Excel reads mixed types)
        df["StockCode"] = df["StockCode"].astype(str)
        df["Invoice"] = df["Invoice"].astype(str)
        df["Customer_ID"] = df["Customer_ID"].astype(str)
        df["Description"] = df["Description"].astype(str)

        # Reproducible sample of invoices to preserve basket integrity!
        import numpy as np

        all_invoices = df["Invoice"].unique()  # type: ignore[union-attr]
        rng = np.random.default_rng(42)
        sample_invoices = rng.choice(all_invoices, size=min(1000, len(all_invoices)), replace=False)  # type: ignore[arg-type]
        sample = df[df["Invoice"].isin(sample_invoices)].reset_index(drop=True)  # type: ignore[union-attr]

        sample.to_parquet(UCI_SAMPLE_PARQUET, index=False)
        return sample

    except Exception as exc:
        pytest.skip(f"Could not download UCI Online Retail II dataset: {exc}")


# ---------------------------------------------------------------------------
# Instacart Market Basket 2017 — requires Kaggle token
# ---------------------------------------------------------------------------

INSTACART_DIR = CACHE_DIR / "instacart"
_INSTACART_PRIOR = INSTACART_DIR / "order_products__prior.csv"
_INSTACART_ORDERS = INSTACART_DIR / "orders.csv"
_INSTACART_PRODUCTS = INSTACART_DIR / "products.csv"


@pytest.fixture(scope="session")
def instacart_df():  # type: ignore[return]
    """Return a 50k-row Pandas DataFrame from the Instacart Market Basket dataset.

    Requires the Kaggle CLI and a valid ``~/.kaggle/kaggle.json`` token.
    Skips automatically if unavailable.

    Download command (done once):
        kaggle competitions download -c instacart-market-basket-analysis
    then extract CSVs into tests/.dataset_cache/instacart/.
    """
    import pandas as pd

    if not (_INSTACART_PRIOR.exists() and _INSTACART_ORDERS.exists()):
        # Try to auto-download via Kaggle API
        try:
            import subprocess  # noqa: S404 (controlled command)

            INSTACART_DIR.mkdir(parents=True, exist_ok=True)
            zip_dest = CACHE_DIR / "instacart.zip"
            subprocess.run(
                [
                    "kaggle",
                    "competitions",
                    "download",
                    "-c",
                    "instacart-market-basket-analysis",
                    "-p",
                    str(CACHE_DIR),
                ],
                check=True,
                capture_output=True,
                timeout=120,
            )
            with zipfile.ZipFile(zip_dest, "r") as z:
                z.extractall(INSTACART_DIR)
        except Exception as exc:
            pytest.skip(f"Instacart data not available and auto-download failed: {exc}")

    prior = pd.read_csv(_INSTACART_PRIOR, usecols=["order_id", "product_id"])  # type: ignore
    orders = pd.read_csv(_INSTACART_ORDERS, usecols=["order_id", "user_id"])  # type: ignore
    products = pd.read_csv(_INSTACART_PRODUCTS)

    df = (
        prior.merge(orders, on="order_id")
        .merge(products[["product_id", "product_name"]], on="product_id")
        .sample(n=min(50_000, len(prior)), random_state=42)
        .reset_index(drop=True)
    )
    return df
