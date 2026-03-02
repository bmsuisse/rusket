# 6. Optuna Hyperparameter Tuning

`rusket` comes with built-in Bayesian hyperparameter optimization via [Optuna](https://optuna.org). Instead of doing a manual grid search, you can use `rusket.optuna_optimize()` to intelligently and efficiently explore the parameter space. It leverages the Rust-native cross-validation engine, making each trial evaluation extremely fast.

## Basic Usage

The easiest way to start is to use the default search space provided by `rusket` for your model class (e.g., `ALS`, `eALS`, or `BPR`).

```python
import pandas as pd
import rusket

# Load your interaction data
df = pd.DataFrame({
    "user_id": [1, 1, 2, 2, 3],
    "item_id": [10, 20, 10, 30, 20],
    "clicks": [5, 1, 3, 2, 1]
})

# Run the optimization
result = rusket.optuna_optimize(
    rusket.ALS,
    df,
    user_col="user_id",
    item_col="item_id",
    rating_col="clicks",  # optional
    n_trials=30,          # number of parameter combinations to try
    n_folds=3,            # 3-fold cross validation per trial
    metric="ndcg",        # metric to optimize (e.g., ndcg, precision, recall)
    k=10                  # cut-off for the ranking metric
)

print(f"Best score (NDCG@10): {result.best_score:.4f}")
print(f"Best parameters: {result.best_params}")
```

## Customizing the Search Space

If you want to fine-tune the parameter boundaries, you can define a custom `search_space` using `OptunaSearchSpace`:

```python
from rusket import eALS, OptunaSearchSpace, optuna_optimize

search_space = [
    OptunaSearchSpace.int("factors", 16, 256, log=True),
    OptunaSearchSpace.float("alpha", 1.0, 100.0, log=True),
    OptunaSearchSpace.float("regularization", 1e-4, 1.0, log=True),
    OptunaSearchSpace.int("iterations", 5, 30),
]

result = optuna_optimize(
    eALS,
    df,
    user_col="user_id",
    item_col="item_id",
    search_space=search_space,
    n_trials=50,
    metric="precision",
    refit_best=True  # Automatically retrains the best model on the full dataset!
)

# Because we used refit_best=True, we can use the best model immediately:
items, scores = result.best_model.recommend_items(user_id=1, n=5)
```

## MLflow Integration

You can easily log all Optuna trials directly into MLflow:

```python
import mlflow
import rusket

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("rusket-tuning")

# Setting mlflow_tracking=True takes care of everything automatically
result = rusket.optuna_optimize(
    rusket.ALS, 
    df,
    user_col="user_id", 
    item_col="item_id",
    n_trials=50, 
    mlflow_tracking=True
)
```
