import pytest

from negmas.outcomes import ContinuousInfiniteIssue, CountableInfiniteIssue
from negmas.outcomes.base_issue import make_issue
from negmas.outcomes.callable_issue import CallableIssue
from negmas.outcomes.cardinal_issue import DiscreteCardinalIssue
from negmas.outcomes.categorical_issue import CategoricalIssue
from negmas.outcomes.contiguous_issue import ContiguousIssue
from negmas.outcomes.continuous_issue import ContinuousIssue
from negmas.outcomes.ordinal_issue import generate_values


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


def test_value_generation_ordinal():
    for i, w in zip((5, 12, 343, 6445, 22345, 344323), range(1, 6)):
        lst = list(generate_values(i))
        assert len(lst) == i
        assert all(isinstance(_, str) for _ in lst)
        assert all(len(_) == w for _ in lst)
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
        with pytest.raises(TypeError):
            assert "abc" not in issue


def test_contains_continuous():
    issue = make_issue((0, 5.0))
    assert 0 in issue
    assert 11 not in issue
    assert 4.56227344383 in issue
    with pytest.raises(TypeError):
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
