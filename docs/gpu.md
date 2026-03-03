# GPU Acceleration

`rusket` supports optional GPU acceleration via **CuPy** or **PyTorch CUDA** for models that benefit from large-scale matrix operations (Gramians, Cholesky solves, batch scoring, attention layers).

## Installation

```bash
# CuPy — recommended, fastest
pip install cupy-cuda12x

# Or PyTorch
pip install torch
```

## Global Enable/Disable

Instead of passing `use_gpu=True` to every model, enable CUDA once:

```python
import rusket

rusket.enable_gpu()   # all models now default to GPU
rusket.disable_gpu()  # back to CPU (the default)
rusket.is_gpu_enabled()  # check current state
```

## Per-Model Override

An explicit `use_gpu` on a model always wins over the global setting:

```python
rusket.enable_gpu()

als  = rusket.ALS(factors=128)              # → GPU (from global)
ease = rusket.EASE(use_gpu=False)           # → CPU (explicit override)
bpr  = rusket.BPR(use_gpu=True)            # → GPU (explicit)

rusket.disable_gpu()
svd  = rusket.SVD()                         # → CPU (from global)
knn  = rusket.ItemKNN(use_gpu=True)        # → GPU (explicit override)
```

## Supported Models

All 12 recommender models respect the global GPU flag:

| Model | GPU-accelerated operations |
|-------|--------------------------|
| **ALS / eALS** | Gramian computation, Cholesky factor solve, batch recommendation scoring |
| **BPR** | SGD factor updates, batch recommend |
| **SVD** | Factor updates, batch scoring |
| **EASE** | Gram matrix inversion (Cholesky) |
| **ItemKNN / UserKNN** | Similarity computation and scoring |
| **LightGCN** | Graph convolution layers, scoring |
| **FM** | Prediction scores |
| **FPMC** | Factor updates |
| **SASRec / BERT4Rec** | Self-attention forward pass, scoring |
| **NMF** | Multiplicative update rules |

## Checking Availability

```python
import rusket

if rusket.check_gpu_available():
    rusket.enable_gpu()
    print("GPU backend active")
else:
    print("No GPU — using Rust CPU backend (still fast!)")
```

> **No GPU? No problem.** The Rust CPU backend (Rayon multi-threading, SIMD) is already highly optimised. GPU acceleration helps most for very large factor dimensions (≥256) or batch scoring millions of users simultaneously.
