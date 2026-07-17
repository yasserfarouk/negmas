"""Tests for ``negmas.outcomes.common.os_or_none``.

Regression coverage for non-sized iterables (e.g. generators) passed as
``outcomes`` — see the ``len(outcomes)`` breakage introduced in commit
``18f5013c`` and reverted thereafter.
"""

from __future__ import annotations

import pytest

from negmas.outcomes.common import os_or_none


class TestOsOrNone:
    def test_none_returns_none(self):
        assert os_or_none(None, None, None) is None

    def test_empty_list_returns_none(self):
        assert os_or_none(None, None, []) is None

    def test_empty_tuple_returns_none(self):
        assert os_or_none(None, None, ()) is None

    def test_outcome_space_takes_precedence(self):
        sentinel = object()
        assert os_or_none(sentinel, ["ignored"], [1, 2]) is sentinel

    def test_list_of_outcomes(self):
        os = os_or_none(None, None, [(0,), (1,), (2,)])
        assert os is not None
        assert len(list(os.enumerate_or_sample())) == 3

    @pytest.mark.parametrize(
        "iterable_factory",
        [
            lambda n: ((_,) for _ in range(n)),  # generator
            lambda n: iter([(_,) for _ in range(n)]),  # list iterator
            lambda n: map(lambda _: (_,), range(n)),  # map object
        ],
    )
    def test_non_sized_iterables_supported(self, iterable_factory):
        n = 5
        os = os_or_none(None, None, iterable_factory(n))
        assert os is not None
        assert len(list(os.enumerate_or_sample())) == n


if __name__ == "__main__":
    pytest.main(args=[__file__, "-v"])
