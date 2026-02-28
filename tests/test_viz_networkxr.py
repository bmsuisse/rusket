import pytest
from hypothesis import given
from hypothesis import strategies as st

import rusket


@given(
    st.lists(
        st.fixed_dictionaries(
            {
                "antecedents": st.lists(st.text(min_size=1, max_size=5), min_size=1, max_size=3).map(tuple),
                "consequents": st.lists(st.text(min_size=1, max_size=5), min_size=1, max_size=3).map(tuple),
                "lift": st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
                "confidence": st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            }
        ),
        min_size=1,
        max_size=10,
    )
)
def test_to_networkxr_hypothesis(rules):
    pytest.importorskip("networkxr")
    import networkxr as nxr

    # Test valid conversion without pandas
    G = rusket.viz.to_networkxr(rules, edge_attr="lift")
    assert isinstance(G, nxr.DiGraph)

    # Verify all edges exist
    for row in rules:
        for ant in row["antecedents"]:
            for con in row["consequents"]:
                assert G.has_edge(ant, con)

    # Fallback missing target column
    with pytest.raises(ValueError):
        rusket.viz.to_networkxr(rules, target_col="missing_col")


def test_to_networkxr_empty():
    pytest.importorskip("networkxr")
    import networkxr as nxr

    G = rusket.viz.to_networkxr([])
    assert isinstance(G, nxr.DiGraph)
    assert not list(G.nodes)
