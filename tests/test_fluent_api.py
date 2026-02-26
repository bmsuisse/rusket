import pandas as pd
import pytest

import rusket


def test_als_fluent_api():
    pytest.importorskip("plotly")

    # 1. Create dummy data (items: 1, 2, 3, 4 -> length 4)
    data = pd.DataFrame({"user_id": [0, 0, 1, 1, 2, 3], "item_id": [1, 2, 2, 3, 1, 4]})

    # 2. Test fluent API: model.fit().pca().plot()
    model = rusket.ALS(factors=8, iterations=5).from_pandas(data, "user_id", "item_id")

    # Check that pca() returns a ProjectedSpace object via direct import
    from rusket.pca import ProjectedSpace

    proj = model.fit().pca(n_components=2)
    assert isinstance(proj, ProjectedSpace)
    assert proj.data.shape == (4, 2)

    # Check plot executes without error
    fig = proj.plot(title="Test Fluent API")
    assert fig is not None


def test_fpmc_fluent_api():
    pytest.importorskip("plotly")

    # 1. Create dummy sequence data (items: 1, 2, 3, 4 -> length 5 counting 0th)
    sequences = [[1, 2], [2, 3], [2, 3], [3, 4], [1, 4], [4, 1]]

    # 2. Test fluent API: model.fit().pca().plot()
    model = rusket.FPMC(factors=8, iterations=5)

    # Check that pca() returns a ProjectedSpace object
    from rusket.pca import ProjectedSpace

    proj = model.fit(sequences).pca(n_components=2)
    assert isinstance(proj, ProjectedSpace)
    assert proj.data.shape == (5, 2)
