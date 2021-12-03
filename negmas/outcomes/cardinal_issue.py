from negmas.outcomes.ordinal_issue import OrdinalIssue

__all__ = ["CardinalIssue"]


class CardinalIssue(OrdinalIssue):
    """
    An issue that has an ordering and for which differences between values is defined (i.e. subtraction)

    """

    def __init__(self, values, name=None, id=None) -> None:
        super().__init__(values, name, id)
        try:
            for _ in range(10):
                self.rand_valid() - self.rand_valid()  # type: ignore
        except:
            raise ValueError(
                f"Cardinal issues should support subtraction between issue values. {self.value_type} does not"
            )
