# rusket

**Blazing-fast FP-Growth and Association Rules for Python — pure Rust via PyO3.**

[![PyPI](https://img.shields.io/pypi/v/rusket.svg)](https://pypi.org/project/rusket/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://pypi.org/project/rusket/)
[![License: BSD-3](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

---

## What is rusket?

`rusket` is a **drop-in replacement** for `mlxtend.frequent_patterns.fpgrowth` and `mlxtend.frequent_patterns.association_rules` — identical API, significantly faster, dramatically lower memory footprint.

The core algorithm is implemented in **Rust** via [PyO3](https://pyo3.rs) and [maturin](https://github.com/PyO3/maturin), with three optimised dispatch paths exposed to Python:

| Input | Rust path | Notes |
|---|---|---|
| Dense pandas DataFrame | `fpgrowth_from_dense` | Flat `uint8` buffer — zero-copy |
| Sparse pandas DataFrame | `fpgrowth_from_csr` | Raw CSR arrays — zero-copy |
| Polars DataFrame | `fpgrowth_from_dense` | Arrow-backed `numpy` buffer |

## Why rusket?

| Feature | rusket | mlxtend |
|---|---|---|
| Speed (medium dataset) | **~0.4 s** | ~4 s |
| Memory (large dataset) | ~3 s | OOM |
| Polars support | ✅ | ❌ |
| Sparse DataFrame support | ✅ | ⚠️ limited |
| Zero Python dependencies | ✅ (`numpy`, `pandas`) | ❌ (many) |
| 12 association metrics | ✅ | ✅ |

## Quick Example

```python
import pandas as pd
from rusket import fpgrowth, association_rules

df = pd.DataFrame({
    "milk":  [1, 1, 0, 1],
    "bread": [1, 0, 1, 1],
    "eggs":  [0, 1, 1, 1],
})

freq = fpgrowth(df, min_support=0.5, use_colnames=True)
rules = association_rules(freq, num_itemsets=len(df), metric="confidence", min_threshold=0.6)
print(rules[["antecedents", "consequents", "confidence"]])
```

[Get Started](quickstart.md){ .md-button .md-button--primary }
[API Reference](api-reference.md){ .md-button }
[View on GitHub](https://github.com/bmsuisse/rusket){ .md-button }
