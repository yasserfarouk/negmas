from __future__ import annotations

import pytest

from negmas import enumerate_issues, issues_from_outcomes, make_issue, outcome_is_valid

from .fixtures import (
    cissue,
    dissue,
    int_issues,
    invalid_outcome_dict,
    invalid_outcome_list,
    issues,
    sissue,
    valid_outcome_dict,
    valid_outcome_list,
)


def test_dict_outcomes(issues, valid_outcome_dict, invalid_outcome_dict):
    assert outcome_is_valid(valid_outcome_dict, issues)
    assert not outcome_is_valid(invalid_outcome_dict, issues)


def test_list_outcomes(int_issues, valid_outcome_list, invalid_outcome_list):
    assert outcome_is_valid(valid_outcome_list, int_issues)
    assert not outcome_is_valid(invalid_outcome_list, int_issues)


def test_from_outcomes():

    issues = [
        make_issue([2, 3], "price"),
        make_issue([1, 2, 3], "cost"),
        make_issue(["yes", "no"], "delivery"),
    ]
    found = issues_from_outcomes(
        enumerate_issues(issues), issue_names=["price", "cost", "delivery"]
    )
    for i, f in zip(issues, found):
        assert i.name == f.name
        for a, b in zip(sorted(i.all), sorted(f.all)):
            assert a == b

    issues = [
        make_issue((1, 7), "price"),
        make_issue((0, 5), "cost"),
        make_issue(["yes", "no"], "delivery"),
    ]
    found = issues_from_outcomes(
        enumerate_issues(issues, max_cardinality=1000),
        numeric_as_ranges=True,
        issue_names=["price", "cost", "delivery"],
    )
    for i, f in zip(issues, found):
        v = sorted(i.values)
        assert i.name == f.name
        assert f._values[0] >= v[0] and f._values[1] <= v[1]


if __name__ == "__main__":
    pytest.main(args=[__file__])
