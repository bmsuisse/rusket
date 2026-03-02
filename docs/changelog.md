# Changelog

All notable changes are documented here.
This project follows [Semantic Versioning](https://semver.org/).

---

### 📦 Miscellaneous

- auto-format code with Ruff [skip ci]
- release v0.1.79
- bump version to 0.1.80 and add sasrec benchmark

### 🚀 Features

- Add enable_pruning option to optuna_optimize

### 📦 Miscellaneous

- remove AI slop from try/except and unused variables; bump version to 0.1.78

### 📦 Miscellaneous

- auto-format code with Ruff [skip ci]
- fixes typing errors for YOLO release
- release v0.1.77

### 📖 Documentation

- comprehensive MLOps & Optuna documentation

### 📦 Miscellaneous

- auto-format code with Ruff [skip ci]
- auto-format code with Ruff [skip ci]
- YOLO release v0.1.76

### 🚀 Features

- Add SASRec and LightGCN notebooks, refine existing documentation, and enhance model selection and MLflow integration.

### 📦 Miscellaneous

- auto-format code with Ruff [skip ci]
- ignore mlflow.db
- release v0.1.73

### 📦 Miscellaneous

- release v0.1.72 with MLflow, Metrics, and Splitters

### 📦 Miscellaneous

- auto-format code with Ruff [skip ci]
- Bump `rusket` package version to 0.1.70 and delete `test_nxr.py`.
- bump version to 0.1.70 and YOLO release optimizations

### 🚀 Features

- wip release 0.1.71

### 📖 Documentation

- add Hyperparameter Tuning section to README

### 📦 Miscellaneous

- auto-format code with Ruff [skip ci]

### 🚀 Features

- parallelize Python CV path with ThreadPoolExecutor
- generic Rust rayon-parallel CV for all factor models
- Rust CV for BPR/SVD/LightGCN, MLflow nested runs fix, bump v0.1.70

### 📦 Miscellaneous

- auto-format code with Ruff [skip ci]

### 🚀 Features

- add MLflow tracking + callbacks to optuna_optimize, docs, bump v0.1.69

### 📦 Miscellaneous

- auto-format code with Ruff [skip ci]

### 🚀 Features

- Rust-native parallel cross-validation + Optuna Bayesian HPO

### 🚀 Features

- add cross_validate grid-search for ALS/eALS hyperparameter tuning, bump to 0.1.67

### 📦 Miscellaneous

- auto-format code with Ruff [skip ci]

### 🚀 Features

- add eALS wrapper class, document eALS, fix networkxr pyright, bump to 0.1.66

### 🔧 Refactoring

- Replace `AutoMiner` with `FPGrowth` in documentation and examples, update ALS benchmarks to include eALS, and refine auto-mining density heuristics.

### 🚀 Features

- Introduce Incremental PCA and Approximate Nearest Neighbors, add ALS sparse matrix support, and remove AutoMiner functionality.
- implement PaCMAP and NN-Descent DR algorithms

### 📖 Documentation

- sync generated documentation

### 🚀 Features

- Implement optimized element-wise coordinate descent ALS (eALS)

### 🚀 Features

- add generic load_model function and lancedb serving examples
- negFIN algorithm and FPGrowth/Eclat SIMD optimizations

### 📦 Miscellaneous

- auto-format code with Ruff [skip ci]
- resolve lock conflicts, yolo merge, release v0.1.62

### 🚀 Features

- Introduce FIN and LCM frequent itemset miners, EASE recommender, PCA, model selection utilities, and NetworkX visualization integrations.

### 🐛 Bug Fixes

- remove .cargo/config.toml from repo (caused SIGILL in CI)

### Bench

- add pytest-benchmark suite for Pipeline API

### ⚡ Performance

- BLAS-accelerated pipeline batch scoring (faer matmul)
- optimize FPGrowth tree building — flat branch buffer, skip HashMap dedup, direct CSR insert

### 📦 Miscellaneous

- auto-format code with Ruff [skip ci]

### 📦 Miscellaneous

- auto-format code with Ruff [skip ci]
- bump version to 0.1.59 for Pipeline API release

### 🚀 Features

- add multi-stage Pipeline API (retrieve → rerank → filter)

### 🐛 Bug Fixes

- preserve original types for _item_labels instead of forcing str()

### 📦 Miscellaneous

- auto-format code with Ruff [skip ci]

### ⚡ Performance

- 2x faster Cholesky + 1.7x faster CG solver

### 🐛 Bug Fixes

- add missing svd_solver param to pca_fit type stub

### 📦 Miscellaneous

- auto-format code with Ruff [skip ci]

### 🐛 Bug Fixes

- evaluate() label-to-index mapping for real-world IDs

### 🚀 Features

- deterministic SVD sign-flip for PCA (matches Spark MLlib / scikit-learn)

### 📦 Miscellaneous

- bump to v0.1.53, drop 3.13t from CI

### 🐛 Bug Fixes

- add strict=True to zip() in test_lcm.py (ruff B905)

### 🐛 Bug Fixes

- make BLAS backend cross-platform (Accelerate on macOS, faer on Linux)

### 📦 Miscellaneous

- auto-format code with Ruff [skip ci]

### 🚀 Features

- add save/load, from_arrow, expand tests, update README for all 15 algorithms
- Add Rust-backed PCA implementation with a scikit-learn compatible Python API.
- PCA with Apple Accelerate BLAS + Gram matrix trick

### 📦 Miscellaneous

- auto-format code with Ruff [skip ci]
- update README and docs/index.md

### 🚀 Features

- Add first-class `pyarrow.Table` support for data processing and model results, leveraging zero-copy conversions and introducing `from_arrow`.

### Merge

- spark PyArrow-native UDFs + SIMD optimizations

### Test

- expand benchmark suite with 4-tier dataset matrix

### ⚡ Performance

- SIMD-optimise dot products (8-wide), eliminate atomic CAS in LightGCN, remove clone overhead in FIN/ECLAT
- Hogwild parallel SVD++ (user-grouped rayon par_iter)
- optimize scoring with SIMD GEMM and enable native CPU target

### 🐛 Bug Fixes

- call .fit() in test_spark_als after from_transactions() API refactor

### 📦 Miscellaneous

- auto-format code with Ruff [skip ci]
- auto-format code with Ruff [skip ci]
- auto-format code with Ruff [skip ci]

### 🔧 Refactoring

- use PyArrow natively in applyInArrow UDFs, drop Polars hop

### 🚀 Features

- SIMD optimizations for SVD++, BPR, Eclat, ALS-CG hot paths

### 🐛 Bug Fixes

- replace deprecated recommend_items with recommend_for_cart in tests

### 📦 Miscellaneous

- auto-format code with Ruff [skip ci]
- auto-format code with Ruff [skip ci]

### 🚀 Features

- add Kosarak regression test, Rust unit tests for association_rules, and benchmark vs arm-rs
- Introduce SVD model, standardize `verbose` and `seed` parameters, and optimize imports across modules.
- Refactor benchmarks, add new comparison benchmarks, enhance SVD API with type hints and `fitted` property, and expand documentation and test coverage.
- add SVD model, LibRecommender benchmarks, zero-dep messaging

### 🐛 Bug Fixes

- als_model references in Recommender docs and test setups
- add tabulate to dev dependencies for mkdocs
- keep _orig_type resolution before to_dataframe data coercion
- lazy evaluation of num_itemsets to support PySpark dfs
- preserve spark dataframe column order after inner join in from_transactions

### 📦 Miscellaneous

- auto-format code with Ruff [skip ci]
- bump version to 0.1.46

### 📦 Miscellaneous

- auto-format code with Ruff [skip ci]

### Style

- apply ruff formatting

### 📦 Miscellaneous

- add remaining unstaged files from main

### 🔄 CI/CD

- enforce regression and benchmark job pass before PyPI release

### Test

- extract benchmark and regression tests into a separate workflow

### 📦 Miscellaneous

- auto-format code with Ruff [skip ci]

### 🐛 Bug Fixes

- resolve E0432 unresolved import and E0308 type mismatch in FPGrowth and ALS

### 📖 Documentation

- promote OOP API, update Ferris logo color to blue, fix pyright errors

### 📦 Miscellaneous

- auto-format code with Ruff [skip ci]
- auto-format code with Ruff [skip ci]
- updating docs and testing
- auto-format code with Ruff [skip ci]

### 🚀 Features

- handle miner kwargs and preserve DataFrame return types
- use labels by default in mining algorithms
- Enhance ALS and FPGrowth algorithms, update ALS benchmarks, add Databricks cookbook, refresh project logos, and remove `test_colnames.py`.

### 📦 Miscellaneous

- auto-format code with Ruff [skip ci]

### 🔄 CI/CD

- skip git-auto-commit on tags to prevent race conditions and bump to v0.1.39

### Optimizer

- enhance PySpark toArrow to utilize pandas ArrowDtype

### 📦 Miscellaneous

- auto-format code with Ruff [skip ci]
- resolve u.vlock conflict

### 🔄 CI/CD

- fix detached head for auto-commit and bump to v0.1.37
- remove experimental python 3.14 to fix pipeline hang and bump to v0.1.38

### 🔄 CI/CD

- add git-auto-commit for ruff format and bump to v0.1.36

### 📦 Miscellaneous

- fix ruff format issues from type ignores and bump to v0.1.35

### 📦 Miscellaneous

- bypass pandas-stubs typing issues for older Python versions and bump to v0.1.34

### 📦 Miscellaneous

- fix Ruff trailing whitespace formatting error in test_fpbase.py, bump to v0.1.33

### 📦 Miscellaneous

- fix PrefixSpan KeyError in test_spark_prefixspan and bump to v0.1.32

### 📦 Miscellaneous

- fix pandas FutureWarning in tests and bump to v0.1.31

### 📦 Miscellaneous

- fix PySpark assertion and Pytest deprecation warnings, bump v0.1.30

### Benchmarks

- add comprehensive benchmark scripts and final report against Python libraries
- fix missing imports and numpy compatibility, fix ruff lints

### Mining

- optimize prefixspan removing hashmaps and pyo3 object lists for 1.15x speedup
- optimize prefixspan with zero-copy numpy ffi over pyo3 for 2.05x speedup

### 📦 Miscellaneous

- untrack benchmarking, profile, and recbole test artifacts
- fix pytest warnings/pyright errors and bump version to v0.1.29

### 📦 Miscellaneous

- bump to v0.1.28, fix typing issues in tests

### Style

- run ruff format and fix lints
- Auto-format with Ruff

### 🐛 Bug Fixes

- ensure Spark input is handled before Polars coercion in from_transactions

### 📖 Documentation

- update logo asset path to `logo_single.svg` in documentation and configuration.

### 📦 Miscellaneous

- update uv.lock for 0.1.26
- exclude non-essential files from sdist
- exclude dev/docs from sdist and wheels

### 🚀 Features

- Add strict UI typings (SupportsItemFactors), classes API filtering, and generated Schema
- add natively rust-backed evaluation metrics and model selection splitters

### 📖 Documentation

- sync changelog and api reference for 0.1.26

### 📦 Miscellaneous

- bump version to 0.1.26

### 🚀 Features

- `from_transactions` now preserves input DataFrame type for Pandas, Polars, and Spark with updated type hints and tests.

### Benchmark

- add script comparing eALS vs iALS

### Debug

- re-raise exception in als_grouped worker to reveal root cause

### Merge

- feature/fin-lcm-miner into main (FIN/LCM algorithms, FM/FPMC)

### ⚡ Performance

- SIMD unrolling for dot and axpy hot-loops in ALS solver

### 🐛 Bug Fixes

- auto-coerce 0/1 pandas DataFrames to bool in dispatch, silence non-bool DeprecationWarning
- add criterion dev-dependency for bench targets
- validate DataFrame before coercing to bool so invalid values (e.g. 2) raise ValueError
- add fitted property to ItemKNN
- suppress DeprecationWarning in als_grouped Spark worker
- use internal model indices in als_grouped worker to correctly map user_labels
- resolve all pyright errors and ruff format/lint failures for CI
- resolve all ruff format/lint and pyright CI failures

### 📖 Documentation

- fix MDX parsing errors for Mintlify
- add business-oriented LightGCN and SASRec example notebooks
- migrate to Zensical for GitHub Pages deployment

### 📦 Miscellaneous

- Remove Python profiling, benchmarking, and RecBole testing scripts, and update Cargo.toml and .gitignore.
- untrack generated artifacts (tensorboard logs, dSYM, recbole_data, saved)
- untrack ai slop benches/ directory
- bump version to 0.1.25

### 🚀 Features

- implement FIN and LCM algorithms with fast bitset operations
- wip RecBole benchmarking and FM/FPMC algorithms
- add LightGCN and SASRec recommendation models

### Bench

- fix unfair benchmark timing and optimize EASE with Cholesky

### Benchmark

- add script comparing dEclat vs ECLAT
- add script comparing eALS vs iALS

### Style

- run ruff format

### ⚡ Performance

- SIMD unrolling for dot and axpy hot-loops in ALS solver

### 📖 Documentation

- expose llm.txt in docs root and fix test_real_world.py sampling
- migrate to Mintlify
- auto-update API reference, changelog, and llm.txt
- fix MDX parsing errors for Mintlify
- auto-update API reference, changelog, and llm.txt
- add als 25m benchmark sweep chart
- update changelog for YOLO release

### 📦 Miscellaneous

- include Mintlify config and generated MDX docs

### 🚀 Features

- Add ultra-fast Sparse ItemKNN algorithm using BM25 and Rust Rayon
- implement FIN and LCM algorithms with fast bitset operations
- wip RecBole benchmarking and FM/FPMC algorithms
- Add grouped PySpark support for ALS

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

