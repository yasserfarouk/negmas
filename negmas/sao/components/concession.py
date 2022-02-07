from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from yaml import warnings

from negmas.common import PreferencesChange, Value
from negmas.helpers.prob import Distribution, UniformDistribution, make_distribution
from negmas.outcomes.common import Outcome
from negmas.preferences import BaseUtilityFunction, RankOnlyUtilityFunction

from .base import SAOComponent, SAODoNothingComponent

if TYPE_CHECKING:
    from negmas.sao import SAONegotiator, SAOState

    from .inverter import UtilityInverter

__all__ = [
    "ConcessionEstimator",
    "ConcessionRecommender",
    "CrispSelfProjectionConcessionEstimator",
    "ProbSelfProjectionConcessionEstimator",
    "KindConcessionRecommender",
]


@dataclass
class ConcessionEstimator(SAODoNothingComponent):
    """
    Estimates the partner's concession
    """

    rank_only: bool = False
    total_concession: bool = False
    _ufun: BaseUtilityFunction | None = field(init=False, default=None)

    @abstractmethod
    def __call__(self, state: SAOState) -> Value:
        """
        Returns current estimate of the partner's concession
        """

    def on_preferences_changed(self, changes: list[PreferencesChange]):
        if not self._negotiator or not self._negotiator.ufun:
            raise ValueError("Negotiator or ufun are not known")
        self._ufun = (
            self._negotiator.ufun
            if not self.rank_only
            else RankOnlyUtilityFunction(self._negotiator.ufun)
        )

    @property
    def ufun(self) -> BaseUtilityFunction:
        if self._ufun is None:
            raise ValueError("Utility fun is not known")
        return self._ufun


class ConcessionRecommender(SAOComponent, Protocol):
    """
    Decides the level of concession to use
    """

    @abstractmethod
    def __call__(self, partner_concession: Value, state: SAOState) -> float:
        """
        Returns a recommended concession in the range [0, 1]
        """


@dataclass
class CrispSelfProjectionConcessionEstimator(ConcessionEstimator):
    """
    A concession estimator that uses projection of the offers from the partner into the agents own ufun space.

    Remarks:
        - It is crisp in the sense that it recommends a concession value in the range [0 - 1]
    """

    _pivot_util: float = field(init=False, default=0.0)
    _concession: float = field(init=False, default=0.0)
    _n_received: int = field(init=False, default=0)

    def __call__(self, state: SAOState) -> float:
        return self._concession

    def set_negotiator(self, negotiator) -> None:
        self._negotiator = negotiator
        self._n_received = 0

    def before_responding(self, state: SAOState, offer: Outcome):
        if not self._negotiator:
            warnings.warn(
                f"{self.__class__.__name__}: Not assigned to any negotiator (make sure that `set_negotiator` is called by the parent negotiator)!! Will assume no concession"
            )
            return
        if not self._ufun:
            warnings.warn(
                f"{self.__class__.__name__}: {self._negotiator.name} [id {self._negotiator.id}] do not have a ufun!! Will assume no concession"
            )
            return
        self._n_received += 1
        u = self._ufun(offer)
        if u is None:
            return
        ud = float(u)
        if self._n_received < 2:
            self._pivot_util = ud
            return
        self._concession = ud - self._pivot_util
        if not self.total_concession:
            self._pivot_util = ud


@dataclass
class ProbSelfProjectionConcessionEstimator(ConcessionEstimator):
    """
    A concession estimator that uses projection of the offers from the partner into the agents own ufun space.

    Remarks:
        - It is probabilitistic in the sense that it recommends a distribution of concession values over the range [0-1]
    """

    _pivot_util: Distribution = field(init=False, default=UniformDistribution())
    _concession: Distribution = field(init=False, default=UniformDistribution())
    _n_received: int = field(init=False, default=0)

    def __call__(self, state: SAOState) -> Distribution:
        return self._concession

    def set_negotiator(self, negotiator) -> None:
        self._negotiator = negotiator
        self._n_received = 0

    def before_responding(self, state: SAOState, offer: Outcome):
        if not self._negotiator:
            warnings.warn(
                f"{self.__class__.__name__}: Not assigned to any negotiator (make sure that `set_negotiator` is called by the parent negotiator)!! Will assume no concession"
            )
            return
        if not self._ufun:
            warnings.warn(
                f"{self.__class__.__name__}: {self._negotiator.name} [id {self._negotiator.id}] do not have a ufun!! Will assume no concession"
            )
            return
        self._n_received += 1
        u = self._ufun(offer)
        if u is None:
            return
        ud = make_distribution(u)
        if self._n_received < 1:
            self._pivot_util = ud
            return
        self._concession = ud - self._pivot_util
        if not self.total_concession:
            self._pivot_util = ud


@dataclass
class KindConcessionRecommender(ConcessionRecommender):
    """
    A simple recommender that does one small concession first then a tit-for-tat response

    Args:
        kindness: A fraction of the utility range to concede everytime no matter what.
        punish: If True, the partner will be punished by pushing our lower utility limit up if the concession (or its expectation) was negative
        initial_concession: The amount of concession to do in the first step
        must_concede: If `True` the agent is guaranteed to concede in the first step
        inverter: Used only if `must_concede` is `True` to determine the lowest level of concession possible
    """

    kindness: float = 0.01
    punish: float = False
    initial_concession: float = 0.05
    must_concede: bool = True
    inverter: UtilityInverter | None = None
    _n_calls: int = field(init=False, default=0)

    def set_inverter(self, inverter: UtilityInverter | None) -> None:
        self.inverter = inverter

    def __call__(self, partner_concession: Value, state: SAOState) -> float:
        """
        Returns current estimate of the partner's concession
        """
        pc = float(partner_concession)
        concession = 0.0 if pc < 0 and not self.punish else pc
        self._n_calls += 1
        if self._n_calls > 2:
            return concession + self.kindness
        if self._n_calls == 1 or not self.must_concede:
            return concession + self.initial_concession
        concession += self.initial_concession + self.kindness
        if not self.inverter:
            return concession
        inv = self.inverter.recommender._inv
        increment = (
            0.5 * self.initial_concession if self.initial_concession > 1e-2 else 0.01
        )
        i = 0
        while (
            concession < 1.0
            and inv is not None
            and len(inv.some((1.0 - concession, 1.0))) < 2
        ):
            concession += increment * (1 + i / 20)
            i += 1
        return concession

    def set_negotiator(self, negotiator: SAONegotiator) -> None:
        self._negotiator = negotiator
        self._did_my_kindness = False
