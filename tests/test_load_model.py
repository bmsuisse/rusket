from pathlib import Path

import rusket


def test_load_model(tmp_path: Path):
    import numpy as np
    from scipy import sparse as sp

    data = np.array([1, 1, 1], dtype=np.float32)
    row = np.array([0, 0, 1])
    col = np.array([0, 1, 1])
    csr = sp.csr_matrix((data, (row, col)), shape=(2, 2))

    # Train and save a model
    model = rusket.ALS(factors=2, iterations=1)
    model.fit(csr)

    model_path = tmp_path / "model.pkl"
    model.save(model_path)

    # Load back using the generic function
    loaded = rusket.load_model(model_path)

    assert isinstance(loaded, rusket.ALS)
    assert loaded.factors == 2
    assert loaded.iterations == 1
    assert loaded.user_factors is not None
