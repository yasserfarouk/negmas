"""Callable issue implementation that generates values via a callable."""

from __future__ import annotations

from typing import Any, Generator

from negmas.outcomes.base_issue import Issue

__all__ = ["CallableIssue"]


class CallableIssue(Issue):
    """
    An `Issue` with a callable for generating values. This is a very limited issue type and most operations are not supported on it.
    """

    def __init__(self, values, name=None) -> None:
        """Initialize the instance.

        Args:
            values: Values.
            name: Name.
        """
        super().__init__(values, name)
        self._value_type = object

    def _to_xml_str(self, indx):
        raise NotImplementedError("CallableIssue cannot be saved to xml")

    @property
    def type(self) -> str:
        """Type.

        Returns:
            str: The result.
        """
        return "uncountable"

    def is_uncountable(self) -> bool:
        """Check if uncountable.

        Returns:
            bool: The result.
        """
        return True

    def is_continuous(self) -> bool:
        """Check if continuous.

        Returns:
            bool: The result.
        """
        return False

    def ordered_value_generator(
        self, n: int | float | None = 10, grid=True, compact=False, endpoints=True
    ) -> Generator[Any, None, None]:
        """Ordered value generator.

        Args:
            n: Number of items.
            grid: Grid.
            compact: Compact.
            endpoints: Endpoints.

        Returns:
            Generator[Any, None, None]: The result.
        """
        raise NotImplementedError(
            "Cannot generate values in order from a Callable issue"
        )

    def value_generator(
        self, n: int | float | None = 10, grid=True, compact=False, endpoints=True
    ) -> Generator[Any, None, None]:
        """Value generator.

        Args:
            n: Number of items.
            grid: Grid.
            compact: Compact.
            endpoints: Endpoints.

        Returns:
            Generator[Any, None, None]: The result.
        """
        if n is None or isinstance(n, float):
            raise ValueError("Real valued issue with no discretization value")
        yield from (self._values() for _ in range(n))

    def rand(self):
        """Picks a random valid value."""
        return self._values()

    def rand_outcomes(
        self, n: int, with_replacement=False, fail_if_not_enough=False
    ) -> list:
        """Rand outcomes.

        Args:
            n: Number of items.
            with_replacement: With replacement.
            fail_if_not_enough: Fail if not enough.

        Returns:
            list: The result.
        """
        if not with_replacement:
            raise ValueError(
                "'values' is specified as a callable for this issue. Cannot "
                "sample from it without replacement"
            )

        return [self._values() for _ in range(n)]

    def rand_invalid(self):
        """Pick a random *invalid* value"""

        raise ValueError(
            "Cannot generate invalid outcomes because values is given as a callable"
        )

    def is_valid(self):
        """Check if valid."""
        raise ValueError("Cannot check the validity of callable issues")

    def value_at(self, index: int):
        """Value at.

        Args:
            index: Index.
        """
        raise ValueError("Cannot index a callable issue")
