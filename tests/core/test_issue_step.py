"""Tests for the `step` parameter of range (contiguous) issues."""

from __future__ import annotations

import copy

import pytest

from negmas.outcomes import (
    ContiguousIssue,
    discretize_and_enumerate_issues,
    enumerate_issues,
    issues_from_xml_str,
    issues_to_xml_str,
    make_issue,
    make_os,
)
from negmas.preferences import LinearAdditiveUtilityFunction
from negmas.sao import AspirationNegotiator, SAOMechanism
from negmas.serialization import deserialize, serialize


def test_step_even_values_example():
    issue = make_issue((0, 10), step=2, name="q")
    assert list(issue.all) == [0, 2, 4, 6, 8, 10]
    assert issue.cardinality == 6
    assert issue.min_value == 0
    assert issue.max_value == 10
    assert issue.step == 2


def test_step_as_third_tuple_value():
    a = make_issue((0, 10, 2), name="q")
    b = make_issue((0, 10), step=2, name="q")
    assert a == b
    assert list(a.all) == [0, 2, 4, 6, 8, 10]
    assert a.step == 2


def test_step_third_value_must_fit_in_range():
    # 20 > 10 - 0 so it cannot be a step
    with pytest.raises(ValueError):
        make_issue((0, 10, 20))
    # a step exactly equal to the range is fine (two values)
    assert list(make_issue((0, 10, 10)).all) == [0, 10]


def test_step_cannot_be_given_twice():
    with pytest.raises(ValueError):
        make_issue((0, 10, 2), step=2)


def test_step_clamps_max_to_last_attainable_value():
    issue = make_issue((0, 10, 3))
    assert list(issue.all) == [0, 3, 6, 9]
    assert issue.cardinality == 4
    assert issue.max_value == 9
    # clamping is idempotent
    assert make_issue(issue.values, name=issue.name) == issue


def test_step_nonzero_min():
    issue = make_issue((3, 12, 4))
    assert list(issue.all) == [3, 7, 11]
    assert issue.min_value == 3
    assert issue.max_value == 11


def test_step_is_valid():
    issue = make_issue((0, 10), step=2)
    assert issue.is_valid(0)
    assert issue.is_valid(4)
    assert issue.is_valid(10)
    assert not issue.is_valid(5)
    assert not issue.is_valid(-2)
    assert not issue.is_valid(12)
    assert 4 in issue
    assert 5 not in issue


def test_step_value_at_matches_all():
    issue = make_issue((-4, 9, 3))
    values = list(issue.all)
    assert [issue.value_at(i) for i in range(issue.cardinality)] == values
    with pytest.raises(IndexError):
        issue.value_at(issue.cardinality)


def test_step_generators_stay_on_grid():
    issue = make_issue((0, 20), step=5)
    valid = set(issue.all)
    assert set(issue.value_generator(3)) <= valid
    assert set(issue.ordered_value_generator(9)) <= valid
    assert list(issue.ordered_value_generator()) == [0, 5, 10, 15, 20]
    assert list(issue) == [0, 5, 10, 15, 20]


def test_step_random_values_stay_on_grid():
    issue = make_issue((0, 20), step=5)
    valid = set(issue.all)
    assert {issue.rand() for _ in range(200)} <= valid
    assert set(issue.rand_outcomes(3)) <= valid
    assert len(issue.rand_outcomes(3)) == 3
    assert sorted(issue.rand_outcomes(100)) == sorted(valid)
    assert set(issue.rand_outcomes(50, with_replacement=True)) <= valid
    assert not issue.is_valid(issue.rand_invalid())


def test_step_invalid_values_rejected():
    with pytest.raises(ValueError):
        make_issue((0, 10), step=0)
    with pytest.raises(ValueError):
        make_issue((0, 10), step=-2)
    with pytest.raises(ValueError):
        make_issue((0, 10), step=2.5)  # type: ignore
    with pytest.raises(ValueError):
        make_issue((0.0, 10.0), step=2)
    with pytest.raises(ValueError):
        make_issue([1, 2, 3], step=2)
    with pytest.raises(ValueError):
        make_issue((0, float("inf")), step=2)


def test_step_copy_and_serialization_roundtrip():
    issue = make_issue((0, 10), step=2, name="q")
    assert copy.copy(issue) == issue
    assert copy.deepcopy(issue) == issue
    assert copy.deepcopy(issue).step == 2
    restored = deserialize(serialize(issue))
    assert restored == issue
    assert restored.step == 2  # type: ignore
    assert ContiguousIssue.from_dict(issue.to_dict()) == issue
    assert ContiguousIssue.from_dict(issue.to_dict()).step == 2


def test_step_differences_break_equality_and_hash():
    a = make_issue((0, 10), name="q")
    b = make_issue((0, 10), step=2, name="q")
    assert a != b
    assert hash(a) != hash(b)
    assert len({a, b}) == 2


def test_step_contains():
    coarse = make_issue((0, 10), step=2, name="q")
    fine = make_issue((0, 10), name="q")
    assert not coarse.contains(fine)
    assert fine.contains(coarse)
    assert coarse.contains(make_issue((0, 8), step=2, name="q"))
    assert coarse.contains(make_issue((0, 8), step=4, name="q"))
    assert not coarse.contains(make_issue((1, 9), step=2, name="q"))
    # non-contiguous discrete issues go through the generic (enumerating) path
    assert coarse.contains(make_issue([0, 4, 8], name="q"))
    assert not coarse.contains(make_issue([0, 3], name="q"))


def test_step_to_discrete():
    issue = make_issue((0, 20), step=5, name="q")
    assert issue.to_discrete(10) is issue
    reduced = issue.to_discrete(3, compact=True)
    assert reduced.cardinality == 3
    assert set(reduced.all) <= set(issue.all)
    assert reduced.step == 5


def test_step_intersect():
    a = make_issue((0, 10), step=2, name="q")
    b = make_issue((4, 20), step=2, name="q")
    assert list(a.intersect(b).all) == [4, 6, 8, 10]
    c = make_issue((1, 11), step=2, name="q")
    with pytest.raises(ValueError):  # disjoint grids
        a.intersect(c)
    d = make_issue((0, 10), step=3, name="q")
    assert list(a.intersect(d).all) == [0, 6]


def test_step_xml_roundtrip():
    issue = make_issue((0, 10), step=2, name="q")
    restored = issues_from_xml_str(issues_to_xml_str([issue]))[0]
    assert restored is not None
    # Genius XML has no notion of a step so the issue is written (and read back) as
    # an enumerated discrete issue whose values are strings (as for any discrete issue)
    assert list(restored[0].all) == [str(_) for _ in issue.all]


def test_step_outcome_space_and_enumeration():
    os = make_os([make_issue((0, 10), step=2, name="q"), make_issue(3, name="p")])
    assert os.cardinality == 18
    outcomes = list(os.enumerate())
    assert len(outcomes) == 18
    assert all(os.is_valid(o) for o in outcomes)
    assert not os.is_valid((5, 0))
    assert len(enumerate_issues(list(os.issues))) == 18
    assert len(discretize_and_enumerate_issues(list(os.issues))) == 18
    assert all(o in os for o in os.sample(20, with_replacement=True))


def test_step_negotiation_runs():
    issues = [make_issue((0, 10), step=2, name="q"), make_issue((0, 20, 5), name="p")]
    os = make_os(issues)
    session = SAOMechanism(outcome_space=os, n_steps=30)
    for i in range(2):
        ufun = LinearAdditiveUtilityFunction.random(outcome_space=os, normalized=True)
        session.add(AspirationNegotiator(name=f"a{i}"), ufun=ufun)
    session.run()
    for state in session.history:
        if state.current_offer is not None:
            assert os.is_valid(state.current_offer), state.current_offer
    if session.agreement is not None:
        assert os.is_valid(session.agreement)


def test_step_one_is_unchanged():
    """A step of 1 (the default) must behave exactly as before."""
    for values in ((0, 10), (-3, 4), 7, (0, 10, 1)):
        a = make_issue(values, name="q")
        b = make_issue((0, 10) if values == (0, 10, 1) else values, name="q")
        assert a == b
        assert a.values == b.values
        assert str(a) == str(b)
        assert repr(a) == repr(b)
        assert a.to_dict() == b.to_dict()
        assert a._to_xml_str(0) == b._to_xml_str(0)
        assert list(a.all) == list(b.all)
        assert a.step == 1


def test_step_of_other_issue_types_is_one():
    assert make_issue((0.0, 1.0), name="q").step == 1
    assert make_issue((0, 10), name="q").step == 1
