from negmas.outcomes import ContinuousInfiniteIssue, CountableInfiniteIssue
from negmas.outcomes.base_issue import make_issue
from negmas.outcomes.callable_issue import CallableIssue
from negmas.outcomes.cardinal_issue import CardinalIssue
from negmas.outcomes.categorical_issue import CategoricalIssue
from negmas.outcomes.contiguous_issue import ContiguousIssue
from negmas.outcomes.continuous_issue import ContinuousIssue
from negmas.outcomes.ordinal_issue import generate_values


def test_make_issue_generation():
    assert isinstance(make_issue((0, 5)), ContiguousIssue)
    assert isinstance(make_issue(10), ContiguousIssue)
    assert isinstance(make_issue([1, 2, 3, 5]), CardinalIssue)
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
        lst = generate_values(i)
        assert len(lst) == i
        assert all(isinstance(_, str) for _ in lst)
        assert all(len(_) == w for _ in lst)
        assert all(int(_) == v for _, v in zip(lst, range(len(lst))))
        assert all(a > b for a, b in zip(lst[1:], lst[:-1]))


def test_can_create_different_types():
    assert isinstance(make_issue((0, 5)), ContiguousIssue)
    assert isinstance(make_issue(10), ContiguousIssue)
    assert isinstance(make_issue([1, 2, 3, 4, 5]), CardinalIssue)
    assert isinstance(make_issue(["a", "b", "c"]), CategoricalIssue)
    assert isinstance(make_issue(lambda: 1), CallableIssue)
    assert isinstance(make_issue((0.0, 5.0)), ContinuousIssue)
