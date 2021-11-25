import pytest

from negmas import (
    Issue,
    enumerate_issues,
    issues_from_outcomes,
    outcome_in_range,
    outcome_is_valid,
)
from negmas.tests.fixtures import *


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


def test_from_outcomes():

    issues = [
        Issue([2, 3], "price"),
        Issue([1, 2, 3], "cost"),
        Issue(["yes", "no"], "delivery"),
    ]
    found = issues_from_outcomes(
        enumerate_issues(issues), issue_names=["price", "cost", "delivery"]
    )
    for i, f in zip(issues, found):
        assert i.name == f.name
        assert all(a == b for a, b in zip(sorted(i.values), f._values))

    issues = [
        Issue((1, 7), "price"),
        Issue((0, 5), "cost"),
        Issue(["yes", "no"], "delivery"),
    ]
    found = issues_from_outcomes(
        enumerate_issues(issues, max_n_outcomes=1000),
        numeric_as_ranges=True,
        issue_names=["price", "cost", "delivery"],
    )
    for i, f in zip(issues, found):
        v = sorted(i.values)
        assert i.name == f.name
        assert f._values[0] >= v[0] and f._values[1] <= v[1]


if __name__ == "__main__":
    pytest.main(args=[__file__])
