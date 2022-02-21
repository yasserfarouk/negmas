from __future__ import annotations

from negmas.outcomes.ordinal_issue import DiscreteOrdinalIssue, OrdinalIssue

__all__ = ["CardinalIssue"]


class CardinalIssue(OrdinalIssue):
    """
    An `Issue` for which differences between values are meaningful.
    """

    def __init__(self, values, name=None) -> None:
        super().__init__(values, name)
        try:
            for _ in range(10):
                self.rand_valid() - self.rand_valid()  # type: ignore
        except Exception as e:
            raise ValueError(
                f"Cardinal issues should support subtraction between issue values. {self.value_type} does not ({e})"
            )


class DiscreteCardinalIssue(DiscreteOrdinalIssue, CardinalIssue):
    """
    An issue that has an ordering and for which differences between values is defined (i.e. subtraction)
    """
