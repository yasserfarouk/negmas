from __future__ import annotations

import time

# seed(0)
import pytest

from negmas import (
    HyperRectangleUtilityFunction,
    LimitedOutcomesNegotiator,
    SAOMechanism,
)


def test_a_session():
    time.perf_counter()
    n = 50
    p = SAOMechanism(outcomes=n, n_steps=50)
    for _ in range(4):
        p.add(
            LimitedOutcomesNegotiator(p_ending=0.01, name=f"agent {_}"),
            preferences=HyperRectangleUtilityFunction(
                [None], [lambda x: x[0]], outcomes=((_,) for _ in range(n))
            ),
        )
    p.run()
    # print(f'{len(p.negotiators)} negotiators')
    assert len(p.history) > 0
    # print(f'Took {time.perf_counter()-start}')


if __name__ == "__main__":
    pytest.main(args=[__file__])
