import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
from tabulate import tabulate

from rusket import mine

DATA_DIR = Path("data")


def get_ml100k():
    """Download ML-100k dataset and format it for frequent itemset mining."""
    DATA_DIR.mkdir(exist_ok=True)
    zip_path = DATA_DIR / "ml-100k.zip"
    unzip_dir = DATA_DIR / "ml-100k"

    if not zip_path.exists():
        print("Downloading ML-100k...")
        urllib.request.urlretrieve("http://files.grouplens.org/datasets/movielens/ml-100k.zip", zip_path)

    if not unzip_dir.exists():
        print("Unzipping ML-100k...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(DATA_DIR)

    base_path = unzip_dir / "u.data"
    df = pd.read_csv(
        base_path,
        sep=r"\t",
        names=["user_id", "item_id", "rating", "timestamp"],
        engine="python",
    )

    # Convert to transactions (user_id -> list of movies highly rated)
    # Filter for ratings >= 4 to make it sparser or more meaningful
    df_high = df[df["rating"] >= 4]
    
    transactions = df_high.groupby("user_id")["item_id"].apply(list).tolist()
    return transactions


def bench_algorithms():
    txns = get_ml100k()
    print(f"Loaded {len(txns)} transactions from ML-100k")

    from rusket.transactions import from_transactions
    df = from_transactions(txns)

    algs = ["fpgrowth", "eclat", "fin", "lcm", "negfin"]
    supports = [0.05, 0.1, 0.2]

    results = []

    for sup in supports:
        print(f"\n--- Support = {sup} ---")
        for alg in algs:
            t0 = time.perf_counter()
            try:
                out = mine(df, min_support=sup, method=alg)
                t1 = time.perf_counter()
                results.append({"Support": sup, "Algorithm": alg, "Time (s)": t1 - t0, "Itemsets": len(out)})
                print(f"{alg:10} | {t1 - t0:.4f}s | {len(out)} itemsets")
            except Exception as e:
                print(f"{alg:10} | FAILED | {str(e)}")

    res_df = pd.DataFrame(results)
    print("\n\nSummary Table:")
    print(tabulate(res_df, headers="keys", tablefmt="psql"))


if __name__ == "__main__":
    bench_algorithms()
