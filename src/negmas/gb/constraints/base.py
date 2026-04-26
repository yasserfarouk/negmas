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
        """Combine two constraints with logical AND.

        Args:
            other: The constraint to combine with this one.

        Returns:
            A new constraint that is satisfied only when both constraints are satisfied.
        """
        return AllOfferingConstraints([self, other])

    def __or__(self, other: OfferingConstraint) -> OfferingConstraint:
        """Combine two constraints with logical OR.

        Args:
            other: The constraint to combine with this one.

        Returns:
            A new constraint that is satisfied when either constraint is satisfied.
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
            history: List of previous thread states for context.

        Returns:
            True if the constraint is satisfied, False otherwise.
        """
        ...  # type: ignore

    def eval_globally(self, source: str, state: GBState, history: list[GBState]):
        """Evaluate constraint in global context by extracting the relevant thread.

        Args:
            source: Identifier of the negotiation thread to evaluate.
            state: Current global negotiation state.
            history: List of previous global negotiation states.

        Returns:
            True if the constraint is satisfied for the specified thread.
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
