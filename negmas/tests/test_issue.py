from __future__ import annotations

import pytest

from negmas.outcomes import generate_issues
from negmas.outcomes.outcome_space import CartesianOutcomeSpace

from .fixtures import bissue, cissue, dissue, hamlet, uissue


def test_type(hamlet, cissue, dissue):
    assert cissue.type == "continuous", "Continuous type passes"
    assert dissue.type == "contiguous", "int passes"
    assert hamlet.type == "categorical", "string list passes"


def test_is_continuous(cissue, dissue, hamlet):
    assert cissue.is_continuous() is True, "Continuous type passes"
    assert dissue.is_continuous() is False, "string list passes"
    assert hamlet.is_continuous() is False, "int passes"


def test_is_discrete(cissue, dissue, hamlet):
    assert cissue.is_discrete() is False, "Continuous type passes"
    assert dissue.is_discrete() is True, "string list passes"
    assert hamlet.is_discrete() is True, "int passes"


def test_string_conversion(uissue, hamlet):
    assert str(uissue).endswith(": ['val 0', 'val 1', 'val 2', 'val 3', 'val 4']")
    assert str(hamlet) == "THE problem: ['val 0', 'val 1', 'val 2', 'val 3', 'val 4']"


def test_cartinatlity(cissue, dissue, hamlet):
    assert cissue.cardinality == float("inf")
    assert hamlet.cardinality == 5
    assert dissue.cardinality == 10


def test_n_outcomes(cissue, dissue, hamlet):
    assert CartesianOutcomeSpace([cissue, dissue, hamlet]).cardinality == float("inf")
    assert CartesianOutcomeSpace([dissue, hamlet]).cardinality == 50
    assert CartesianOutcomeSpace([dissue]).cardinality == 10
    assert CartesianOutcomeSpace([]).cardinality == 1


def test_rand(cissue, dissue, hamlet, bissue):
    for _ in range(100):
        assert 0.0 <= cissue.rand() < 1.0
        assert 0 <= dissue.rand() < 10
        assert hamlet.rand() in hamlet.values
        assert bissue.rand() in bissue.values


def test_rand_invalid(cissue, dissue, hamlet, bissue):
    for _ in range(100):
        assert not 0.0 <= cissue.rand_invalid() < 1.0
        assert not 0 <= dissue.rand_invalid() < 10
        assert hamlet.rand_invalid() not in hamlet.values
        assert bissue.rand_invalid() not in bissue.values


def test_possibilities(cissue, dissue, hamlet, bissue):
    with pytest.raises(ValueError):
        _ = list(cissue.all)
    assert len(list(dissue.all)) == 10
    assert len(list(hamlet.all)) == 5
    assert len(list(bissue.all)) == 2


def test_issue_generation_defaults():
    options = ["a", "b", "c"]
    issues = generate_issues([(0.0, 1.0), options, 5])
    assert len(issues) == 3
    assert (
        issues[0].is_continuous()
        and issues[0]._values[0] == 0.0
        and issues[0]._values[1] == 1.0
    )
    for i, o in enumerate(options):
        assert issues[1]._values[i] == o
    assert issues[2]._values == (0, 4)
    for i, issue in enumerate(issues):
        assert str(i) == issue.name


def test_issue_generation_multiples():
    issues_ = generate_issues([5], [10])
    assert len(issues_) == 10
    for i, issue in enumerate(issues_):
        assert issue.name.startswith(str(i))
        assert issue._values == (0, 4)


if __name__ == "__main__":
    pytest.main(args=[__file__])
