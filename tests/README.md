# Rusket Testing Guide & Discoveries

This document captures important edge cases and discoveries found while writing and fixing tests for the `rusket` test suite.

## 1. PySpark and `JAVA_GATEWAY_EXITED`
When running PySpark tests natively via `pytest` (e.g. testing `to_spark` or `SparkSession` initializations), the Java Gateway can occasionally crash or be unavailable on the host machine. 
**Best Practice:** Always wrap Spark setup in a fixture with a `try/except` block and gracefully `pytest.skip` if the gateway fails, rather than letting the entire test suite fail.

```python
@pytest.fixture(scope="module")
def spark_session():
    try:
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.master("local[1]").getOrCreate()
        yield spark
        spark.stop()
    except Exception as e:
        pytest.skip(f"PySpark init failed (often JAVA_GATEWAY_EXITED): {e}")
```

## 2. Pandas Sparse DataFrames (`pd.SparseDtype`)
Pandas `DataFrame.sparse.from_spmatrix()` constructs sparse dataframes, but in newer versions of Pandas (>2.0), this can raise `FutureWarning: Allowing arbitrary scalar fill_value in SparseDtype is deprecated` when the underlying sparse matrix is a type (like bool/int) but the `fill_value` defaults to `NaN` or mismatching scalars.
**Best Practice:** Explicitly cast to the proper `SparseDtype` with the correct `fill_value`.
```python
sdf = pd.DataFrame.sparse.from_spmatrix(sparse_ary)
if hasattr(pd, "SparseDtype"):
    sdf = sdf.astype(pd.SparseDtype(bool, False))
```

## 3. Deprecated Mining Functions
Legacy functions like `fit_transactions()`, `mine_hupm()`, and `prefixspan()` have been deprecated in favor of the OOP architecture.
**Best Practice:** Always use the `.from_transactions()` classmethod chain:
```python
# Instead of: prefixspan(seqs)
PrefixSpan(seqs).mine()

# Instead of: model.fit_transactions(df)
ALS.from_transactions(df, ...)
```

## 4. `sequences_from_event_log` Returns Flat CSR Arrays
The `sequences_from_event_log` utility converts DataFrames into prefixspan-compatible sequences. To ensure blazing fast zero-copy transfers to Rust, it does **not** return a `list[list[int]]` of sequences. Instead, it returns a 2-tuple `(indptr, indices)` reflecting a flat CSR structure.
**Best Practice:** When writing tests asserting the number of sequences returned, do not use `len(seqs)`. Use `len(seqs[0]) - 1` (where `seqs[0]` is the `indptr` array).
```python
seqs, mapping = sequences_from_event_log(df, ...)
num_sequences = len(seqs[0]) - 1
```
