from __future__ import annotations

from negmas.helpers.prob import ScipyDistribution
from negmas.preferences import IPUtilityFunction


def test_construction():
    outcomes = [("o1",), ("o2",)]
    f = IPUtilityFunction(
        outcomes=outcomes,
        distributions=[
            ScipyDistribution(type="uniform", loc=0.0, scale=0.5),
            ScipyDistribution(type="uniform", loc=0.1, scale=0.5),
        ],
    )
    assert str(f(("o1",))) == "U(0.0, 0.5)"
