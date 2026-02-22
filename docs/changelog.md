# Changelog

All notable changes are documented here.
This project follows [Semantic Versioning](https://semver.org/).

---

### Benchmark

- add script comparing dEclat vs ECLAT

### Style

- run ruff format

### 📖 Documentation

- expose llm.txt in docs root and fix test_real_world.py sampling
- migrate to Mintlify

### 📦 Miscellaneous

- include Mintlify config and generated MDX docs

### 🚀 Features

- Add ultra-fast Sparse ItemKNN algorithm using BM25 and Rust Rayon

### Style

- apply ruff formatting and fixes
- Update logo colors from purple to orange.
- refine logos with orange theme, update mkdocs palette and extra.css

### 🐛 Bug Fixes

- resolve PySpark ChunkedArray fallback warning and implement BPR fit_transactions
- fix pyright errors reported on ci

### 📖 Documentation

- add Polars/PySpark PrefixSpan tests and cookbook examples
- improve API documentation, update marketing copy, and setup PySpark skips
- enhance PrefixSpan and HUPM cookbook sections with clearer descriptions, business scenarios, and updated Python code examples.

### 📦 Miscellaneous

- commit remaining unstaged files from previous sessions
- bump version to 0.1.21
- bump version to 0.1.22
- bump version to 0.1.23

### 🔧 Refactoring

- simplify BaseModel and remove implicit recommender duplication
- update logo SVG basket elements to use curved paths and refined wire details.

### 🚀 Features

- core algorithms via Faer, HUPM, Arrow Streams, and Hybrid Recommender
- complete PySpark and Polars integration for PrefixSpan via native PyArrow sequences
- implement recommend_items for association rule models
- Introduce new documentation notebooks, update PySpark integration documentation, and add a notebook conversion workflow.
- automated doc sync scripts (changelog, API ref, llm.txt)
- enhance recommender system documentation and examples, update core logic, and refresh logos.
- merge feature/fpgrowth-mlxtend-api

### ⚡ Performance

- Boost FPGrowth performance with a new architecture, update benchmarks and documentation, add new logos, and remove temporary test files."

### 🐛 Bug Fixes

- skip mlxtend comparison at >1M rows to prevent CI timeout

### 📖 Documentation

- add genai and lancedb integration examples to cookbook
- add cookbook examples for ALS PCA visualization and Spark MLlib translation
- conquer 1 billion row challenge architecture and bump v0.1.20

### 🔄 CI/CD

- trigger Deploy Docs on benchmarks/** changes too

### 🔧 Refactoring

- clean Python layer — remove stale timing vars, dead code, AI-slop comments

### 🐛 Bug Fixes

- Loosen numerical tolerance for parallel Hogwild! BPR test to fix CI

### 📖 Documentation

- use relative path for logo in README

### 📖 Documentation

- Comprehensive Interactive Cookbook with Real-World Datasets

### Bench

- add Cholesky to ALS benchmark script and fix pyright

### 📖 Documentation

- feature rusket.mine as the primary public api endpoint across mkdocs and readme
- append comprehensive cookbook examples for prefixspan, hupm, bpr, similarity, and recommender modules

### 📦 Miscellaneous

- safe checkpoint

### 🚀 Features

- add method='auto' routing to dynamically select eclat or fpgrowth based on dataset density

### 🚀 Features

- YOLO release v0.1.16

### ⚡ Performance

- implement rayon multi-threading for FPMiner chunk ingestion
- revert SmallVec regression, clean HashMap FPMiner + scale to 1B benchmark
- item pre-filter + with_capacity hint in FPMiner
- fix freq-sort to ascending (Eclat-optimal: least-frequent items first)

### 🐛 Bug Fixes

- pyright unbound variables correctly initialized
- pyright complaints about unbound variables and missing als_fit_implicit argument
- benchmark now uses 8GB in-memory limit instead of disk-spilling at scale
- streaming.py cleanup + als_fit_implicit cg_iters stub + psutil available RAM strategy
- batched mining at 250M rows per batch to avoid OOM at 800M+
- SCALE_TARGETS scoping + launch 1B Eclat scale-up
- restore SEP in benchmark f-strings

### 📖 Documentation

- add FPMiner out-of-core streaming section and 300M benchmark
- add ALS feature and market basket analysis to README

### 🚀 Features

- add verbose mode to fpgrowth, eclat, and FPMiner for large-scale feedback
- implement hybrid memory/disk out-of-core FPMiner with dynamic RAM limit
- add verbose iteration timing + out-of-core 1B support
- comprehensive cookbook + ALS speed improvements
- HashMap FPMiner + creative benchmark (method × chunk-size × scale)
- frequency-sorted remap + mine_auto + hint_n_transactions (Borgelt 2003)
- Anderson Acceleration for ALS outer loop (anderson_m param)

### 🚀 Features

- FPMiner streaming accumulator v0.1.14

### 🚀 Features

- direct scipy CSR support in fpgrowth/eclat + pd.factorize + scale benchmarks

### 🚀 Features

- automated scale benchmark with Plotly chart (1M-500M rows)

### 🚀 Features

- sparse CSR from_transactions + million-scale benchmarks (66× faster)

### Bench

- add real-world dataset benchmark (auto-downloads, with timeouts)

### 📖 Documentation

- add Eclat API, real-world benchmarks, and usage examples

### 🚀 Features

- add from_transactions, from_pandas, from_polars, from_spark helpers

### Test

- add dedicated test_eclat.py for standalone eclat() function

### ⚡ Performance

- arena-based FPNode with flat children storage (7.8x speedup)

### 🐛 Bug Fixes

- add readme and license to pyproject.toml for PyPI, bump to 0.1.9

### 🚀 Features

- add Eclat algorithm (method='eclat') with 2.4-2.8x speedup on sparse data
- make eclat the default method (faster in all benchmarks)
- expose eclat() as standalone public function

### 🐛 Bug Fixes

- remove orphaned FPGrowth import after FP-TDA removal

### 📦 Miscellaneous

- remove FP-TDA implementation
- add MIT license
- add dependabot.yml to match httprx structure

### 🚀 Features

- implement zero-copy slice algorithm for FP-TDA

### 📦 Miscellaneous

- remove tracked __pycache__ / .pyc files

### 🐛 Bug Fixes

- remove target-cpu=native from .cargo/config.toml to fix CI SIGILL crashes
- exclude test_benchmark.py from regular pytest run to prevent mlxtend timeouts
- increase CI timeout to 45min for slow free-threaded Python builds
- benchmark CI - conditional baseline compare + PyPI trusted publishing (OIDC)
- fptda iterative mining to avoid stack overflow on sparse data

### 📖 Documentation

- compact logo, remove fast pattern mining subtitle

### 📦 Miscellaneous

- merge feat/regression-benchmarks into main
- bump version to 0.1.5

### 🔧 Refactoring

- extract FPBase, add FPTda class, FP-TDA in benchmarks

### 🚀 Features

- regression benchmark tests + fix warnings
- add FP-TDA algorithm (IJISRT25NOV1256)\n\nImplements the Frequent-Pattern Two-Dimensional Array algorithm as a\ndrop-in alternative to FP-Growth. Uses right-to-left column projection\non sorted transaction lists instead of conditional subtree construction.\n\n- src/fptda.rs: Rust core (fptda_from_dense / fptda_from_csr)\n- rusket/fptda.py: Python wrapper, identical API to fpgrowth()\n- rusket/__init__.py: export rusket.fptda\n- tests/test_fptda.py: 22 tests (mix-ins + cross-check vs fpgrowth)\n- src/fpgrowth.rs: made process_item_counts/flatten_results pub(crate)\n- src/lib.rs: register new pyfunction bindings

### Style

- apply ruff format and fix lint errors

### 🐛 Bug Fixes

- remove tracked site/ dir, rename fpgrowth-pyo3→rusket, fix docs workflow

### 📖 Documentation

- add CI/CD workflow guidance to AGENTS.md
- publish real benchmark numbers with Plotly interactive chart
- add GitHub Pages enable step to AGENTS.md
- replace cookbook notebook with clean markdown, simplify docs workflow
- add YOLO section to AGENTS.md; merge feat/regression-benchmarks

### 🚀 Features

- add benchmark against efficient-apriori
- Bump version to 0.1.3, refine FPGrowth Arrow data type handling, update dependencies, and refactor test and project files.

### 🐛 Bug Fixes

- add mkdocs-jupyter dependency for github pages

### 📦 Miscellaneous

- fix docs deployment and format readme

### ⚡ Performance

- zero-copy pyarrow backend implementation

### 🐛 Bug Fixes

- resolve SIGABRT panic in fpgrowth.rs and restore missing validation checks in python port

### 📖 Documentation

- add comprehensive Jupyter cookbook with Plotly graphs and benchmark results
- add pyarrow zero-copy dataframe slicing examples

### 📦 Miscellaneous

- add pytest-timeout to dev dependencies
- bump version to 0.1.1

### 📖 Documentation

- emphasize ultimate blazing speed in README

### 📦 Miscellaneous

- add maturin and pyright to dev dependencies for CI

### 🔄 CI/CD

- configure automated pypi release and github tags workflow

### 🚀 Features

- optimised FP-Growth (mimalloc + SmallVec + PAR_ITEMS_CUTOFF=4 + parallel freq count + dedup)

