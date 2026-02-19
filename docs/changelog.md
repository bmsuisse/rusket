# Changelog

All notable changes are documented here.  
This project follows [Semantic Versioning](https://semver.org/).

---

## [0.1.0] — 2026-02-19

### Added

- **`fpgrowth()`** — FP-Growth frequent itemset mining backed by Rust + PyO3
    - Dense pandas path: flat `uint8` buffer via `fpgrowth_from_dense` (zero-copy)
    - Sparse pandas path: CSR arrays via `fpgrowth_from_csr` (zero-copy)
    - Polars path: Arrow-backed NumPy buffer via `fpgrowth_from_dense`
    - Parallel mining via Rayon
- **`association_rules()`** — Association rule generation with 12 metrics:
    `confidence`, `lift`, `support`, `leverage`, `conviction`,
    `zhangs_metric`, `jaccard`, `certainty`, `kulczynski`,
    `representativity`, `antecedent support`, `consequent support`
- Drop-in API compatibility with `mlxtend.frequent_patterns`
- `max_len` parameter to cap itemset size
- `support_only` flag for fast support-only mode
- `return_metrics` selector to include only desired metric columns
- Full test suite mirroring mlxtend behaviour

### Performance (vs mlxtend)

| Dataset | Speedup | Memory |
|---|---|---|
| Small (5 × 11) | ~10× | – |
| Medium (10k × 400) | ~5–8× | ~8× less |
| Large (100k × 1000) | N/A (OOM) | handles it |
