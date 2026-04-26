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
        """Initializes the instance."""
        super().__init__(values, name)
        self._value_type = object

    def _to_xml_str(self, indx):
        raise NotImplementedError("CallableIssue cannot be saved to xml")

    @property
    def type(self) -> str:
        """Return the issue type identifier (always 'uncountable' for callables)."""
        return "uncountable"

    def is_uncountable(self) -> bool:
        """Check if the issue has uncountably many values (always True for callables)."""
        return True

    def is_continuous(self) -> bool:
        """Check if this issue has continuous values (always False for callables)."""
        return False

    def ordered_value_generator(
        self, n: int | float | None = 10, grid=True, compact=False, endpoints=True
    ) -> Generator[Any, None, None]:
        """Not supported for callable issues.

        Args:
            n: Number of values to generate.
            grid: Ignored - ordering not supported.
            compact: Ignored - ordering not supported.
            endpoints: Ignored - ordering not supported.

        Raises:
            NotImplementedError: Always raised since ordering is undefined for callables.
        """
        raise NotImplementedError(
            "Cannot generate values in order from a Callable issue"
        )

    def value_generator(
        self, n: int | float | None = 10, grid=True, compact=False, endpoints=True
    ) -> Generator[Any, None, None]:
        """Generate n values by calling the underlying callable.

        Args:
            n: Number of values to generate (must be a finite integer).
            grid: Ignored - values come from callable.
            compact: Ignored - values come from callable.
            endpoints: Ignored - values come from callable.

        Yields:
            Values produced by calling the stored callable n times.

        Raises:
            ValueError: If n is None or a float.
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
        """Generate n random values by calling the underlying callable.

        Args:
            n: Number of random values to generate.
            with_replacement: Must be True; sampling without replacement is not supported.
            fail_if_not_enough: Ignored for callable issues.

        Returns:
            A list of n values produced by calling the stored callable.

        Raises:
            ValueError: If with_replacement is False.
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
        """Validity checking is not supported for callable issues.

        Raises:
            ValueError: Always raised since validity cannot be determined.
        """
        raise ValueError("Cannot check the validity of callable issues")

    def value_at(self, index: int):
        """Indexing is not supported for callable issues.

        Args:
            index: Ignored - indexing not supported.

        Raises:
            ValueError: Always raised since callables cannot be indexed.
        """
        raise ValueError("Cannot index a callable issue")
