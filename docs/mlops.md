# MLOps: Hyperparameter Tuning & MLflow

Rusket provides native integration with modern MLOps tools to make experimentation, tracking, and deployment seamless.

## MLflow Integration

Rusket has built-in support for [MLflow](https://mlflow.org/) experiment tracking and model packaging. You can easily track hyperparameters, training duration, and package your models as `mlflow.pyfunc` artifacts for deployment on standard deployment endpoints (such as SageMaker, Azure ML, or Databricks Model Serving).

### Autologging

Simply call `autolog()` anywhere in your script before fitting your models. Rusket will automatically log:
- Model hyperparameters (e.g., `factors`, `iterations`, `learning_rate`, `alpha`, `min_support`)
- Training duration in seconds (metric: `training_duration_seconds`)
- Any model attributes matching standard naming conventions

```python
import pandas as pd
import mlflow
import rusket
import rusket.mlflow

# Enable automatic logging
rusket.mlflow.autolog()

df = pd.DataFrame({"user_id": [1, 1, 2], "item_id": [10, 20, 10]})

mlflow.set_experiment("rusket-als")
with mlflow.start_run():
    # Model parameters and training time will be automatically logged to MLflow
    model = rusket.ALS(factors=64, iterations=15).from_transactions(df).fit()
```

### Saving and Loading Models

Rusket provides a custom MLflow Model Flavor wrapper stringently adhering to PyFunc behaviour. You can save and load `rusket` models directly as PyFunc models. This allows you to deploy them to standard inference servers and use them via `.predict()` natively.

```python
import rusket.mlflow
import mlflow.pyfunc

# Save native model to an MLflow directory
rusket.mlflow.save_model(model, "my_als_model")

# Load it back through the standard pyfunc API
loaded_model = mlflow.pyfunc.load_model("my_als_model")

# Use standard pandas predict() interface. 
# It expects a DataFrame with a 'user' or 'user_id' column, returning top-10 items
input_df = pd.DataFrame({"user_id": [1, 2]})
predictions = loaded_model.predict(input_df)

print(predictions)
# Output: DataFrame with columns ['user', 'items', 'scores']
```

## Hyperparameter Tuning with Optuna

Rusket provides blazing fast Bayesian hyperparameter optimization using [Optuna](https://optuna.org/)'s TPE sampler. Under the hood, trials are scored using the native Rust multi-threaded cross-validation logic (`CrossValidationResult`), meaning the memory stays in Rust across CV evaluation folds, greatly supercharging speed.

### Quick Search

The `optuna_optimize` function provides sensible default search ranges for all implicit recommenders (`ALS`, `BPR`, `LightGCN`, `ItemKNN`, `SVD`, `EASE`, etc).

```python
import rusket

result = rusket.optuna_optimize(
    rusket.ALS,
    df,
    user_col="user_id",
    item_col="item_id",
    n_trials=50,
    metric="ndcg",
    k=10,
)

print(f"Best ndcg@10: {result.best_score:.4f}")
print(f"Best params:  {result.best_params}")
```

### Custom Search Spaces & MLflow Tracking

You can provide explicit search spaces (log-uniform floats, categorical options, ints) and opt to automatically log all trials to an active MLflow experiment. Cross-validation averages out the model accuracy for a safer approximation to out-of-core test data.

```python
import mlflow
from rusket import OptunaSearchSpace

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("als-tuning")

result = rusket.optuna_optimize(
    rusket.eALS,
    df,
    user_col="user_id",
    item_col="item_id",
    search_space=[
        OptunaSearchSpace.int("factors", 16, 256, log=True),
        OptunaSearchSpace.float("alpha", 1.0, 100.0, log=True),
        OptunaSearchSpace.float("regularization", 1e-4, 1.0, log=True),
        OptunaSearchSpace.int("iterations", 5, 30),
    ],
    n_trials=100,
    n_folds=3,            # 3-Fold Cross-Validation (KFold logic in Rust)
    metric="precision",   # Optimize for Precision
    refit_best=True,      # Automatically train the best model on the full dataset
    mlflow_tracking=True, # Every trial is logged to MLflow
)

# Since we used refit_best=True, we can extract the best trained model directly
best_model = result.best_model
items, scores = best_model.recommend_items(user_id=42, n=10)
```
