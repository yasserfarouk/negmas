from negmas.outcomes.base_issue import Issue
from negmas.outcomes.callable_issue import CallableIssue
from negmas.outcomes.categorical_issue import CategoricalIssue
from negmas.outcomes.contiguous_issue import ContiguousIssue
from negmas.outcomes.continuous_issue import ContinuousIssue
from negmas.outcomes.ordinal_issue import OrdinalIssue


def test_can_create_different_types():
    assert isinstance(Issue((0, 5)), ContiguousIssue)
    assert isinstance(Issue(10), ContiguousIssue)
    assert isinstance(Issue([1, 2, 3, 4, 5]), OrdinalIssue)
    assert isinstance(Issue(["a", "b", "c"]), CategoricalIssue)
    assert isinstance(Issue(lambda: 1), CallableIssue)
    assert isinstance(Issue((0.0, 5.0)), ContinuousIssue)
