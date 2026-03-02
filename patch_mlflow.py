with open("rusket/mlflow.py", "r") as f:
    lines = f.readlines()

new_lines = []
skip = False
for line in lines:
    if line.startswith("if HAS_MLFLOW:"):
        break
    new_lines.append(line)

new_code = """if HAS_MLFLOW:
    class _RusketWrapper(mlflow.pyfunc.PythonModel):  # type: ignore
        \"""PyFunc wrapper for rusket models.\"""

        def load_context(self, context: Any) -> None:
            from .model import load_model

            model_path = context.artifacts["model_path"]
            self.model = load_model(model_path)

        def predict(self, context: Any, model_input: pd.DataFrame) -> pd.DataFrame:
            \"""Predict recommendations for a dataframe of users.

            Input dataframe should have a 'user' column (or user inputs directly).
            \"""
            import pandas as pd

            if isinstance(model_input, pd.DataFrame):
                if "user" in model_input.columns:
                    users = model_input["user"].tolist()
                elif "user_id" in model_input.columns:
                    users = model_input["user_id"].tolist()
                else:
                    users = model_input.iloc[:, 0].tolist()
            else:
                users = list(model_input)

            results = []
            for u in users:
                try:
                    items, scores = self.model.recommend_items(u, n=10, exclude_seen=True)
                    results.append({"user": u, "items": items.tolist(), "scores": scores.scores.tolist()})
                except Exception:
                    results.append({"user": u, "items": [], "scores": []})

            return pd.DataFrame(results)
else:
    _RusketWrapper = None  # type: ignore
"""
new_lines.append(new_code)

for line in lines[lines.index("def save_model(model: Any, path: str, **kwargs: Any) -> None:\n"):]:
    new_lines.append(line)

with open("rusket/mlflow.py", "w") as f:
    f.writelines(new_lines)
