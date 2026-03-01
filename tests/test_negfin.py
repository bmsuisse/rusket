import numpy as np
import pandas as pd
from scipy import sparse

from rusket.mine import mine
from rusket.negfin import NegFIN

# Simple dataset 1
TRANSACTIONS = [
    ["bread", "milk"],
    ["bread", "diaper", "beer", "eggs"],
    ["milk", "diaper", "beer", "cola"],
    ["bread", "milk", "diaper", "beer"],
    ["bread", "milk", "diaper", "cola"],
]

# Simple dataset 2 (creates dense overlapping patterns)
DENSE_TRANSACTIONS = [
    ["A", "B", "C", "D"],
    ["A", "B", "C"],
    ["A", "B", "D"],
    ["B", "C", "D"],
    ["A", "B", "C", "D", "E"],
]


def test_negfin_basic():
    """Test basic negFIN mining with Pandas DataFrame input."""
    model = NegFIN.from_transactions(TRANSACTIONS)
    res = model.mine(min_support=0.5)

    assert len(res) > 0
    assert "support" in res.columns
    assert "itemsets" in res.columns

    # 'bread' appears in 4/5 transactions
    sup_bread = res[res["itemsets"].apply(lambda x: tuple(x) == ("bread",))]["support"].values[0]
    assert sup_bread == 0.8


def test_negfin_sparse():
    """Test negFIN with CSR matrix input."""
    # Convert TRANSACTIONS to CSR
    items = sorted({item for txn in TRANSACTIONS for item in txn})
    item_to_idx = {item: i for i, item in enumerate(items)}

    row_ind = []
    col_ind = []
    for i, txn in enumerate(TRANSACTIONS):
        for item in txn:
            row_ind.append(i)
            col_ind.append(item_to_idx[item])

    data = np.ones(len(row_ind), dtype=np.uint8)
    csr = sparse.csr_matrix((data, (row_ind, col_ind)), shape=(len(TRANSACTIONS), len(items)))

    model = NegFIN(data=csr, item_names=items)
    res = model.mine(min_support=0.6)

    assert len(res) > 0
    assert "support" in res.columns
    assert "itemsets" in res.columns

    sup_diaper = res[res["itemsets"].apply(lambda x: tuple(x) == ("diaper",))]["support"].values[0]
    assert np.isclose(sup_diaper, 0.8)


def test_negfin_compare_dense():
    """Ensure NegFIN results match ECLAT results perfectly for dense datasets."""
    from rusket.transactions import from_transactions

    df = from_transactions(DENSE_TRANSACTIONS)

    res_negfin = mine(df, min_support=0.2, method="negfin")
    res_eclat = mine(df, min_support=0.2, method="eclat")

    # Sort items within each tuple and sort the dataframe to compare
    res_negfin["itemsets"] = res_negfin["itemsets"].apply(lambda x: tuple(sorted(x)))
    res_eclat["itemsets"] = res_eclat["itemsets"].apply(lambda x: tuple(sorted(x)))

    res_negfin = res_negfin.sort_values(by=["support", "itemsets"]).reset_index(drop=True)
    res_eclat = res_eclat.sort_values(by=["support", "itemsets"]).reset_index(drop=True)

    pd.testing.assert_frame_equal(res_negfin, res_eclat)
