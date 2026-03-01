from pathlib import Path

import rusket


def test_load_model(tmp_path: Path):
    import pandas as pd

    df = pd.DataFrame({"user_id": [1, 1, 2], "item_id": [1, 2, 2]})

    # Train and save a model
    model = rusket.ALS(factors=2, iterations=1)
    model.fit(df)

    model_path = tmp_path / "model.pkl"
    model.save(model_path)

    # Load back using the generic function
    loaded = rusket.load_model(model_path)

    assert isinstance(loaded, rusket.ALS)
    assert loaded.factors == 2
    assert loaded.iterations == 1
    assert loaded.user_factors is not None
