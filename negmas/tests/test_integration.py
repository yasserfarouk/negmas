import time

# seed(0)
import pytest

from negmas import (
    HyperRectangleUtilityFunction,
    LimitedOutcomesNegotiator,
    SAOMechanism,
)

start = time.monotonic()

print(f"Import took {time.monotonic()-start}")


def test_a_session():
    start = time.monotonic()
    p = SAOMechanism(outcomes=50, n_steps=50)
    for _ in range(4):
        p.add(
            LimitedOutcomesNegotiator(p_ending=0.01, name=f"agent {_}"),
            ufun=HyperRectangleUtilityFunction([None], [lambda x: x[0]]),
        )
    p.run()
    # print(f'{len(p.negotiators)} negotiators')
    assert len(p.history) > 0
    # print(f'Took {time.monotonic()-start}')


if __name__ == "__main__":
    pytest.main(args=[__file__])
