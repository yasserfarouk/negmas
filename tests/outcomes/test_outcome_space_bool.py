"""Tests for ``DiscreteCartesianOutcomeSpace.__bool__``.

Regression coverage for the ``OverflowError`` that plain truthiness tests
(``if outcome_space`` / ``not outcome_space``) used to raise on outcome spaces
whose cardinality exceeds ``Py_ssize_t``. ``__bool__`` is defined so Python
uses it instead of ``__len__`` for truthiness.
"""

from __future__ import annotations

import pytest

from negmas.outcomes import ContiguousIssue
from negmas.outcomes.outcome_space import DiscreteCartesianOutcomeSpace


def _huge_os() -> DiscreteCartesianOutcomeSpace:
    # 3 issues each with ~10^9 values -> cardinality ~10^27 > Py_ssize_t max (~9.2e18)
    return DiscreteCartesianOutcomeSpace(
        [ContiguousIssue((0, 10**9), name="x") for _ in range(3)]
    )


class TestOutcomeSpaceBool:
    def test_len_overflows_for_huge_space(self):
        os = _huge_os()
        assert os.cardinality > 2**63
        with pytest.raises(OverflowError):
            len(os)

    def test_bool_true_for_huge_space(self):
        assert bool(_huge_os()) is True

    def test_not_false_for_huge_space(self):
        assert _huge_os() is not False

    def test_if_branch_safe_for_huge_space(self):
        os = _huge_os()
        # Must not raise; space is non-empty so the truthy branch is taken.
        assert (True if os else False) is True

    def test_bool_true_for_normal_space(self):
        os = DiscreteCartesianOutcomeSpace(
            [ContiguousIssue((0, 4), name="x"), ContiguousIssue((0, 4), name="y")]
        )
        assert bool(os) is True
        assert len(os) == 25


if __name__ == "__main__":
    pytest.main(args=[__file__, "-v"])
