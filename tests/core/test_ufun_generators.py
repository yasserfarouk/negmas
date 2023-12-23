from hypothesis import example, given
from hypothesis import strategies as st

from negmas.preferences.generators import make_piecewise_linear_pareto


@given(
    n_pareto=st.integers(2, 40),
    n_segments_min=st.integers(1, 10),
    n_segments_range=st.integers(0, 10),
)
@example(
    n_pareto=2,
    n_segments_min=2,
    n_segments_range=1,
)
def test_make_piecewise_pareto(n_pareto, n_segments_min, n_segments_range):
    make_piecewise_linear_pareto(
        n_pareto,
        n_segments=(n_segments_min, n_segments_min + n_segments_range)
        if n_segments_range
        else n_segments_min,
    )
