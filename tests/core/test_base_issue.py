from __future__ import annotations

import numpy as np
import pytest

from negmas.outcomes import ContinuousInfiniteIssue, CountableInfiniteIssue, make_os
from negmas.outcomes.base_issue import make_issue
from negmas.outcomes.callable_issue import CallableIssue
from negmas.outcomes.cardinal_issue import DiscreteCardinalIssue
from negmas.outcomes.categorical_issue import CategoricalIssue
from negmas.outcomes.contiguous_issue import ContiguousIssue
from negmas.outcomes.continuous_issue import ContinuousIssue
from negmas.outcomes.optional_issue import OptionalIssue
from negmas.outcomes.ordinal_issue import DiscreteOrdinalIssue, generate_values


def test_make_issue_generation():
    assert isinstance(make_issue((0, 5)), ContiguousIssue)
    assert isinstance(make_issue(10), ContiguousIssue)
    assert isinstance(make_issue([1, 2, 3, 5]), DiscreteCardinalIssue)
    assert isinstance(make_issue(["a", "b", "c"]), CategoricalIssue)
    assert isinstance(make_issue(lambda: 1), CallableIssue)
    assert isinstance(make_issue((0.0, 5.0)), ContinuousIssue)
    assert isinstance(make_issue((0, float("inf"))), CountableInfiniteIssue)
    assert isinstance(make_issue((float("-inf"), 0)), CountableInfiniteIssue)
    assert isinstance(make_issue((0.0, float("inf"))), ContinuousInfiniteIssue)
    assert isinstance(
        make_issue((float("-inf"), float("inf"))), ContinuousInfiniteIssue
    )
    assert isinstance(make_issue((float("-inf"), 5.0)), ContinuousInfiniteIssue)
    issue = make_issue((0.0, float("inf")), optional=True)
    assert isinstance(issue, OptionalIssue) and isinstance(
        issue.base, ContinuousInfiniteIssue
    )
    issue = make_issue((0, 5), optional=True)
    assert isinstance(issue, OptionalIssue) and isinstance(issue.base, ContiguousIssue)
    issue = make_issue(10, optional=True)
    assert isinstance(issue, OptionalIssue) and isinstance(issue.base, ContiguousIssue)
    issue = make_issue([1, 2, 3, 5], optional=True)
    assert isinstance(issue, OptionalIssue) and isinstance(
        issue.base, DiscreteCardinalIssue
    )
    issue = make_issue(["a", "b", "c"], optional=True)
    assert isinstance(issue, OptionalIssue) and isinstance(issue.base, CategoricalIssue)
    issue = make_issue(lambda: 1, optional=True)
    assert isinstance(issue, OptionalIssue) and isinstance(issue.base, CallableIssue)
    issue = make_issue((0.0, 5.0), optional=True)
    assert isinstance(issue, OptionalIssue) and isinstance(issue.base, ContinuousIssue)
    issue = make_issue((0, float("inf")), optional=True)
    assert isinstance(issue, OptionalIssue) and isinstance(
        issue.base, CountableInfiniteIssue
    )
    issue = make_issue((float("-inf"), 0), optional=True)
    assert isinstance(issue, OptionalIssue) and isinstance(
        issue.base, CountableInfiniteIssue
    )
    issue = make_issue((0.0, float("inf")), optional=True)
    assert isinstance(issue, OptionalIssue) and isinstance(
        issue.base, ContinuousInfiniteIssue
    )
    issue = make_issue((float("-inf"), float("inf")), optional=True)
    assert isinstance(issue, OptionalIssue) and isinstance(
        issue.base, ContinuousInfiniteIssue
    )
    issue = make_issue((float("-inf"), 5.0), optional=True)
    assert isinstance(issue, OptionalIssue) and isinstance(
        issue.base, ContinuousInfiniteIssue
    )


def test_value_generation_ordinal():
    for i in (5, 12, 343, 6445, 22345, 344323):
        lst = generate_values(i)
        assert len(lst) == i
        assert all(isinstance(_, str) for _ in lst)
        assert all(int(_) == v for _, v in zip(lst, range(len(lst))))
        assert all(a > b for a, b in zip(lst[1:], lst[:-1]))


def test_can_create_different_types():
    assert isinstance(make_issue((0, 5)), ContiguousIssue)
    assert isinstance(make_issue(10), ContiguousIssue)
    assert isinstance(make_issue([1, 2, 3, 4, 5]), DiscreteCardinalIssue)
    assert isinstance(make_issue(["a", "b", "c"]), CategoricalIssue)
    assert isinstance(make_issue(lambda: 1), CallableIssue)
    assert isinstance(make_issue((0.0, 5.0)), ContinuousIssue)


def test_contains_discrete_list():
    issue = make_issue([0, 2, 10])
    assert 0 in issue
    assert 11 not in issue
    assert 0.0 in issue
    assert "abc" not in issue


def test_contains_discrete():
    for v in (10, (0, 9)):
        issue = make_issue(v)
        assert 0 in issue
        assert 11 not in issue
        assert 0.0 in issue
        assert "abc" not in issue


def test_contains_continuous():
    issue = make_issue((0, 5.0))
    assert 0 in issue
    assert 11 not in issue
    assert 4.56227344383 in issue
    assert "abc" not in issue


@pytest.mark.parametrize(
    "v", (10, (-1, 5), [1, 3, 4, 5], ["a", "b", "c", 5], [(0, 1), (8, 6)])
)
def test_can_loop_over_discrete_issue(v):
    issue = make_issue(v)
    assert issue.is_discrete()
    assert len(issue) > 0  # type: ignore I know that this is a discrete issue and len is applicable to it
    for _ in issue:
        pass


def test_can_loop_over_continuous_issue():
    issue = make_issue((0.0, 1.0))
    assert not issue.is_discrete()
    with pytest.raises(TypeError):
        assert len(issue) > 0  # type: ignore I know that this is a discrete issue and len is applicable to it
    for _ in issue:
        pass


def test_values_contained_in_issues_contiguous():
    i1, i2 = make_issue(10), make_issue(5)
    i3 = make_issue([1, 3, 4, 7])
    i4 = make_issue(["1", "2"])
    i5 = make_issue((0, 10))
    for v in i1:
        assert v in i1

    assert i2 in i1
    assert not i1 in i2
    assert i3 in i1
    assert not i1 in i3
    assert not i2 in i3
    for ix in (i1, i2, i3):
        assert not i4 in ix
        assert not ix in i4

    for ix in (i1, i2, i3):
        assert ix in i5
        assert i5 not in ix

    assert i4 not in i5
    assert i5 not in i4

    for v in i2:
        assert v in i1
    for v in i3:
        assert v in i1

    for x in (i1, i2, i3, i4, i5):
        assert x in x


def test_values_contained_in_outcome_spaces():
    a = make_os((make_issue(10), make_issue(list("abcdefg")), make_issue((1.0, 4.0))))
    b = make_os((make_issue(10), make_issue(list("adefg")), make_issue((1.0, 3.0))))
    c = make_os((make_issue(10), make_issue(list("adefgz")), make_issue((1.0, 3.0))))
    assert (4, "d", 3.5) in a
    assert (4, "d", 13.5) not in a

    for x in (a, b, c):
        assert x in x
    assert b in a
    assert not a in b
    assert c not in a

    assert not make_issue(10) in a
    assert make_issue(10, name=a.issues[0].name) in a
    for i in a.issues:
        assert i in a
        assert not i in b
        assert i not in c


def test_ordinal_issue_with_multiple_int_types():
    make_issue([0, 1, np.asarray([10])[0]])
    with pytest.raises(TypeError):
        make_issue([0, 1.0, np.asarray([10])[0]])
    with pytest.raises(TypeError):
        DiscreteOrdinalIssue([0, "1", np.asarray([10])[0]])


def test_ordinal_issue_with_multiple_float_types():
    make_issue([0.0, 1.0, np.asarray([10.0])[0]])
    with pytest.raises(TypeError):
        DiscreteOrdinalIssue([0.0, "1.0", np.asarray([10.0])[0]])
    with pytest.raises(TypeError):
        make_issue([0, 1.0, np.asarray([10.0])[0]])
