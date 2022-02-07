from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from yaml import warnings

from negmas.helpers.prob import (
    Distribution,
    Real,
    UniformDistribution,
    make_distribution,
)
from negmas.outcomes.common import Outcome

from .base import SAOComponent, SAODoNothingComponent

if TYPE_CHECKING:
    from negmas.sao import SAONegotiator, SAOState


class ProbConcessionEstimator(SAODoNothingComponent):
    """
    Estimates the partner's concession
    """

    @abstractmethod
    def __call__(self, state: SAOState) -> Distribution:
        """
        Returns current estimate of the partner's concession
        """


class CrispConcessionEstimator(SAODoNothingComponent):
    """
    Estimates the partner's concession
    """

    @abstractmethod
    def __call__(self, state: SAOState) -> float:
        """
        Returns current estimate of the partner's concession
        """


@dataclass
class ProbSelfProjectionConcessionEstimator(ProbConcessionEstimator):
    _last_util: Distribution = field(init=False, default=UniformDistribution())
    _concession: Distribution = field(init=False, default=UniformDistribution())
    _n_received: int = field(init=False, default=0)

    def __call__(self, state: SAOState) -> Distribution:
        return self._concession

    def set_negotiator(self, negotiator) -> None:
        self._negotiator = negotiator
        self._n_received = 0

    def on_offer_received(
        self,
        partner: str,
        offer: Outcome | None,
        state_received: SAOState,
        state_offered: SAOState = None,
    ):
        if not self._negotiator:
            warnings.warn(
                f"{self.__class__.__name__}: Not assigned to any negotiator (make sure that `set_negotiator` is called by the parent negotiator)!! Will assume no concession"
            )
            return
        if not self._negotiator.ufun:
            warnings.warn(
                f"{self.__class__.__name__}: {self._negotiator.name} [id {self._negotiator.id}] do not have a ufun!! Will assume no concession"
            )
            return
        self._n_received += 1
        u = self._negotiator.ufun(offer)
        if u is not None and self._n_received > 1:
            ud = make_distribution(u)
            self._concession = ud - self._last_util
            self.last_ufil = ud

    def on_offer_accepted(
        self,
        partner: str,
        offer: Outcome | None,
        state_offered: SAOState,
        state_response: SAOState = None,
    ):
        ...

    def on_offer_rejected(
        self,
        partner: str,
        offer: Outcome | None,
        state_offered: SAOState,
        state_response: SAOState = None,
    ):
        ...


class ConcessionRecommender(SAOComponent, Protocol):
    """
    Decides the level of concession to use
    """

    @abstractmethod
    def __call__(self, partner_concession: Distribution, state: SAOState) -> float:
        """
        Returns a recommended concession in the range [0, 1]
        """


@dataclass
class NaiveConcessionRecommender(ConcessionRecommender):
    kindness = 0.001
    punish = False
    initial_concession: float = 0.05
    _did_my_kindness: bool = field(init=False, default=False)

    def __call__(self, partner_concession: Distribution, state: SAOState) -> float:
        """
        Returns current estimate of the partner's concession
        """
        pc = float(partner_concession)
        concession = 0.0 if pc < 0 and not self.punish else pc
        if not self._did_my_kindness:
            self._did_my_kindness = True
            concession += self.initial_concession
        return concession + self.kindness

    def set_negotiator(self, negotiator: SAONegotiator) -> None:
        self._negotiator = negotiator
        self._did_my_kindness = False

    def on_offer_received(
        self,
        partner: str,
        offer: Outcome | None,
        state_received: SAOState,
        state_offered: SAOState = None,
    ):
        ...

    def on_offer_accepted(
        self,
        partner: str,
        offer: Outcome | None,
        state_offered: SAOState,
        state_response: SAOState = None,
    ):
        ...

    def on_offer_rejected(
        self,
        partner: str,
        offer: Outcome | None,
        state_offered: SAOState,
        state_response: SAOState = None,
    ):
        ...
