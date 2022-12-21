"""
Base Constraints on offering constaints
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from attr import define

from negmas.gb.common import GBState, ThreadState
from negmas.outcomes.common import Outcome

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
        self,
        state: GBState,
        history: list[GBState],
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
        return AllOfferingConstraints([self, other])

    def __or__(self, other: OfferingConstraint) -> OfferingConstraint:
        return AnyOfferingConstraint([self, other])

    def __not__(self) -> OfferingConstraint:
        return InverseOfferingConstraint(self)


@define
class LocalOfferingConstraint(OfferingConstraint, ABC):
    @abstractmethod
    def __call__(
        self,
        state: ThreadState,
        history: list[ThreadState],
    ) -> bool:
        ...

    def eval_globally(self, source: str, state: GBState, history: list[GBState]):
        return self(state.threads[source], [_.threads[source] for _ in history])


@define
class AnyOfferingConstraint(OfferingConstraint):
    constraints: list[OfferingConstraint]

    def __call__(self, *args, **kwargs) -> bool:
        return any([_(*args, **kwargs) for _ in self.constraints])


@define
class AllOfferingConstraints(OfferingConstraint):
    constaints: list[OfferingConstraint]

    def __call__(self, *args, **kwargs) -> bool:
        return all([_(*args, **kwargs) for _ in self.constaints])


@define
class InverseOfferingConstraint(OfferingConstraint):
    constraint: OfferingConstraint

    def __call__(self, *args, **kwargs) -> bool:
        return not self.constraint(*args, **kwargs)
