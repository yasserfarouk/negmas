"""
Base Constraints on offering constaints
"""

from __future__ import annotations


from abc import ABC, abstractmethod

from attrs import define

from negmas.gb.common import GBState, ThreadState

__all__ = [
    "OfferingConstraint",
    "LocalOfferingConstraint",
    "AnyOfferingConstraint",
    "AllOfferingConstraints",
]


@define
class OfferingConstraint(ABC):
    @abstractmethod
    def __call__(
        self, state: GBState | ThreadState, history: list[GBState | ThreadState]
    ) -> bool:
        """
        Base class for all offering constaints

        Args:
            offer (Outcome): Outcome to be checked for validity
            history (list[GBState]): History of past states

        Returns:
            bool
        """
        ...

    def __and__(self, other: OfferingConstraint) -> OfferingConstraint:
        """and  .

        Args:
            other: Other.

        Returns:
            OfferingConstraint: The result.
        """
        return AllOfferingConstraints([self, other])

    def __or__(self, other: OfferingConstraint) -> OfferingConstraint:
        """or  .

        Args:
            other: Other.

        Returns:
            OfferingConstraint: The result.
        """
        return AnyOfferingConstraint([self, other])

    def __not__(self) -> OfferingConstraint:
        """not  .

        Returns:
            OfferingConstraint: The result.
        """
        return InverseOfferingConstraint(self)


@define
class LocalOfferingConstraint(OfferingConstraint, ABC):
    @abstractmethod
    def __call__(self, state: ThreadState, history: list[ThreadState]) -> bool:
        """Make instance callable.

        Args:
            state: Current state.
            history: History.

        Returns:
            bool: The result.
        """
        ...  # type: ignore

    def eval_globally(self, source: str, state: GBState, history: list[GBState]):
        """Eval globally.

        Args:
            source: Source identifier.
            state: Current state.
            history: History.
        """
        return self(state.threads[source], [_.threads[source] for _ in history])


@define
class AnyOfferingConstraint(OfferingConstraint):
    """AnyOfferingConstraint implementation."""

    constraints: list[OfferingConstraint]

    def __call__(self, *args, **kwargs) -> bool:
        """Make instance callable.

        Args:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            bool: The result.
        """
        return any([_(*args, **kwargs) for _ in self.constraints])


@define
class AllOfferingConstraints(OfferingConstraint):
    """AllOfferingConstraints implementation."""

    constaints: list[OfferingConstraint]

    def __call__(self, *args, **kwargs) -> bool:
        """Make instance callable.

        Args:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            bool: The result.
        """
        return all([_(*args, **kwargs) for _ in self.constaints])


@define
class InverseOfferingConstraint(OfferingConstraint):
    """InverseOfferingConstraint implementation."""

    constraint: OfferingConstraint

    def __call__(self, *args, **kwargs) -> bool:
        """Make instance callable.

        Args:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            bool: The result.
        """
        return not self.constraint(*args, **kwargs)
