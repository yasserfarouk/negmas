import pytest
import random

from .fixtures import *
from negmas import Issues, outcome_is_valid, outcome_in_range


def test_dict_outcomes(issues, valid_outcome_dict, invalid_outcome_dict):
    assert outcome_is_valid(valid_outcome_dict, issues)
    assert not outcome_is_valid(invalid_outcome_dict, issues)


def test_list_outcomes(int_issues, valid_outcome_list, invalid_outcome_list):
    assert outcome_is_valid(valid_outcome_list, int_issues)
    assert not outcome_is_valid(invalid_outcome_list, int_issues)


def test_outcome_in_verious_ranges():
    or1 = {"price": (0.0, 2.0), "distance": [0.3, 0.4], "type": ["a", "b"], "area": 3}
    or2 = {"price": [(0.0, 1.0), (1.5, 2.0)], "area": [(3, 4), (7, 9)]}

    assert outcome_in_range({"date": "2018.10.4"}, or1)
    assert not outcome_in_range({"date": "2018.10.4"}, or1, strict=True)
    assert outcome_in_range({"area": 3}, or1)
    assert not outcome_in_range({"type": "c"}, or1)
    assert outcome_in_range({"type": "a"}, or1)
    assert not outcome_in_range({"date": "2018.10.4"}, or2, strict=True)
    assert not outcome_in_range({"area": 3}, or2)
    assert outcome_in_range({"area": 3.0001}, or2)
    assert not outcome_in_range({"area": 5}, or2)
    assert outcome_in_range({"price": 0.4}, or2)
    assert not outcome_in_range({"price": 1.2}, or2)
    assert not outcome_in_range({"price": 0.4, "area": 4}, or2)
    assert not outcome_in_range({"price": 0.4, "area": 10}, or2)
    assert not outcome_in_range({"price": 1.2, "area": 10}, or2)
    assert not outcome_in_range({"price": 1.2, "area": 4}, or2)
    assert outcome_in_range({"type": "a"}, or2)
    or1 = {"price": 10}
    assert outcome_in_range({"price": 10}, or1)
    assert not outcome_in_range({"price": 11}, or1)


def test_issues_construction():
    issues = Issues(price=(0.0, 1.0), cost=[1, 2, 3], delivery=["yes", "no"])
    assert len(issues.issues) == 3
    assert str(issues) == "price: (0.0, 1.0)\ncost: [1, 2, 3]\ndelivery: ['yes', 'no']"
    assert issues.is_infinite()
    assert not issues.is_finite()
    assert all(
        a == b for a, b in zip(issues.types, ["continuous", "discrete", "discrete"])
    )

    issues = Issues.from_single_issue(Issue(10, "issue"))
    assert len(issues.issues) == 1
    assert str(issues) == "issue: 10"

    issues = Issues(price=[2, 3], cost=[1, 2, 3], delivery=["yes", "no"])
    assert issues.is_finite()
    assert not issues.is_infinite()
    assert issues.cardinality() == issues.n_outcomes() == 2 * 3 * 2

    valid = issues.rand_valid()
    invalid = issues.rand_invalid()
    assert outcome_in_range(valid, issues.outcome_range)
    assert outcome_in_range(invalid, issues.outcome_range)


def test_outcome_constraint():
    r1 = {
        "price": [(0.0, 1.0), (3.0, 4.0)],
        "quantity": [1, 2, 4, 5],
        "delivery": "yes",
    }

    notr1 = {"price": [(3.01, 3.99), (4.01, 10.0)], "quantity": 3, "delivery": "no"}

    r2 = {"price": (3.01, 10.0), "quantity": [2, 3], "delivery": ["yes", "no"]}

    # r = OutcomeSpace(r1)


if __name__ == "__main__":
    pytest.main(args=[__file__])
