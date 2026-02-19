import time
import random
import pandas as pd
import numpy as np
from rusket import fpgrowth, association_rules
from efficient_apriori import apriori
import argparse

def generate_data(n_transactions=1000, avg_items=5, n_unique_items=100):
    """Generates synthetic market basket data."""
    dataset = []
    items = [f"item_{i}" for i in range(n_unique_items)]
    for _ in range(n_transactions):
        n_items = max(1, int(random.gauss(avg_items, 1)))
        transaction = random.sample(items, min(n_items, n_unique_items))
        dataset.append(transaction)
    return dataset

def benchmark_rusket(dataset, min_support, min_confidence):
    from mlxtend.preprocessing import TransactionEncoder
    import pandas as pd
    
    start_total = time.perf_counter()
    
    # Data transformation (one-hot encoding)
    start_trans = time.perf_counter()
    te = TransactionEncoder()
    te_ary = te.fit(dataset).transform(dataset)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    end_trans = time.perf_counter()
    
    # FP-Growth
    start_fpg = time.perf_counter()
    df_freq = fpgrowth(df, min_support=min_support, use_colnames=True)
    end_fpg = time.perf_counter()
    
    # Association Rules
    start_rules = time.perf_counter()
    df_rules = association_rules(df_freq, num_itemsets=len(df), min_threshold=min_confidence)
    end_rules = time.perf_counter()
    
    end_total = time.perf_counter()
    
    return {
        "trans_time": end_trans - start_trans,
        "fpgrowth_time": end_fpg - start_fpg,
        "rules_time": end_rules - start_rules,
        "total_time": end_total - start_total,
        "n_itemsets": len(df_freq),
        "n_rules": len(df_rules)
    }

def benchmark_efficient_apriori(dataset, min_support, min_confidence):
    start = time.perf_counter()
    itemsets, rules = apriori(dataset, min_support=min_support, min_confidence=min_confidence)
    end = time.perf_counter()
    
    # efficient-apriori returns itemsets as {size: {tuple: count}}
    n_itemsets = sum(len(v) for v in itemsets.values())
    
    return {
        "total_time": end - start,
        "n_itemsets": n_itemsets,
        "n_rules": len(rules)
    }

def run_benchmarks():
    parser = argparse.ArgumentParser(description="Benchmark Rusket against Efficient-Apriori")
    parser.add_argument("--transactions", type=int, default=5000)
    parser.add_argument("--items", type=int, default=10)
    parser.add_argument("--unique", type=int, default=200)
    parser.add_argument("--support", type=float, default=0.01)
    parser.add_argument("--confidence", type=float, default=0.1)
    args = parser.parse_args()

    print(f"Generating data: {args.transactions} transactions, {args.items} avg items, {args.unique} unique items...")
    dataset = generate_data(args.transactions, args.items, args.unique)
    
    print(f"\nBenchmarks (min_support={args.support}, min_confidence={args.confidence}):")
    
    print("-" * 50)
    print("Running Rusket (FP-Growth)...")
    res_rusket = benchmark_rusket(dataset, args.support, args.confidence)
    print(f"  One-hot:   {res_rusket['trans_time']:.4f}s")
    print(f"  FP-Growth: {res_rusket['fpgrowth_time']:.4f}s")
    print(f"  Rules:     {res_rusket['rules_time']:.4f}s")
    print(f"  Total:     {res_rusket['total_time']:.4f}s")
    print(f"  Found:     {res_rusket['n_itemsets']} itemsets, {res_rusket['n_rules']} rules")

    print("-" * 50)
    print("Running Efficient-Apriori...")
    res_ea = benchmark_efficient_apriori(dataset, args.support, args.confidence)
    print(f"  Total:     {res_ea['total_time']:.4f}s")
    print(f"  Found:     {res_ea['n_itemsets']} itemsets, {res_ea['n_rules']} rules")
    
    print("-" * 50)
    speedup = res_ea['total_time'] / res_rusket['total_time']
    print(f"Rusket Speedup: {speedup:.2f}x faster")
    
    if res_rusket['n_itemsets'] != res_ea['n_itemsets']:
        print(f"WARNING: Itemset count mismatch! Rusket={res_rusket['n_itemsets']}, EA={res_ea['n_itemsets']}")
    
    if res_rusket['n_rules'] != res_ea['n_rules']:
        print(f"WARNING: Rule count mismatch! Rusket={res_rusket['n_rules']}, EA={res_ea['n_rules']}")

if __name__ == "__main__":
    run_benchmarks()
