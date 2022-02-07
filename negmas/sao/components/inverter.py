from __future__ import annotations

from abc import ABC, abstractmethod
from random import choice
from typing import TYPE_CHECKING, Callable, Literal, Sequence

from negmas.common import PreferencesChange
from negmas.preferences import (
    BaseUtilityFunction,
    InverseUFun,
    PresortingInverseUtilityFunction,
    RankOnlyUtilityFunction,
    SamplingInverseUtilityFunction,
)
from negmas.sao.components import SAODoNothingComponent
from negmas.warnings import NegmasUnexpectedValueWarning, warn

__all__ = [
    "UtilityInverter",
    "UtilityBasedOutcomeSetRecommender",
    "OutcomeSetRecommender",
]

if TYPE_CHECKING:
    from negmas.outcomes import Outcome
    from negmas.sao import SAONegotiator, SAOState

    from .selectors import OfferSelectorProtocol


def make_inverter(
    ufun: BaseUtilityFunction,
    ufun_inverter: Callable[[BaseUtilityFunction], InverseUFun] | None = None,
    rank_only: bool = False,
    max_cardinality: int = 10_000,
) -> InverseUFun:
    """
    Creates an `InverseUFun` object from the given ufun with appropriate type if the type is not given

    Args:
        ufun (BaseUtilityFunction): The ufun to invert
        rank_only (bool): If True, only the relative ranks of outcomes will be used in inversion not the values themselves.
        ufun_inverter (Callable[[BaseUtilityFunction], InverseUFun] | None): An optional factory to generate a `InverseUFun` from a `BaseUtilityFunction` .
        max_cardinality (int): The maximum cardinality at which we switch to using a `SamplingInverseUtilityFunction`
    """
    if rank_only:
        ufun = RankOnlyUtilityFunction(ufun, randomize_equal=False, name=ufun.name)
    if ufun_inverter:
        return ufun_inverter(ufun)
    return (
        SamplingInverseUtilityFunction(ufun)
        if ufun.outcome_space is None
        or ufun.outcome_space.cardinality >= max_cardinality
        else PresortingInverseUtilityFunction(ufun)
    )


class OutcomeSetRecommender(SAODoNothingComponent):
    """
    Recommends outcomes for the negotiator
    """

    def __init__(
        self,
        type: Literal["min"]
        | Literal["max"]
        | Literal["one"]
        | Literal["some"]
        | Literal["all"] = "some",
    ):
        self._type = type

    @abstractmethod
    def __call__(self, state: SAOState) -> Sequence[Outcome]:
        ...


class UtilityBasedOutcomeSetRecommender(SAODoNothingComponent):
    """
    Recommends a set of outcome appropriate for proposal

    """

    def __init__(
        self,
        rank_only: bool = False,
        ufun_inverter: Callable[[BaseUtilityFunction], InverseUFun] | None = None,
        max_cardinality: int = 10_000,
        eps: float = 0.0001,
        inversion_method: Literal["min"]
        | Literal["max"]
        | Literal["one"]
        | Literal["some"]
        | Literal["all"] = "some",
    ):
        self._rank_only = rank_only
        self._max_cartinality = max_cardinality
        self._inverter_factory = ufun_inverter
        self._inv_method = None
        self._type = inversion_method
        self._eps = eps
        self.set_negotiator(None)  # type: ignore (It is OK. We do not really need to pass this at all here.)

    def set_negotiator(self, negotiator: SAONegotiator) -> None:
        super().set_negotiator(negotiator)
        self._inv: InverseUFun | None = None
        self._min = self._max = self._best = None
        self._inv_method = None

    def on_preferences_changed(self, changes: list[PreferencesChange]):
        if self._negotiator is None:
            raise ValueError("Unknown negotiator in a component")
        ufun = self._negotiator.ufun

        if not ufun:
            self._inv = None
            self._min = self._max = self._best = None
            return
        self._inv = make_inverter(
            ufun, self._inverter_factory, self._rank_only, self._max_cartinality
        )
        if self._type == "one":
            self._inv_method = self._inv.one_in
            self._single_inv_return = True
        elif self._type == "min":
            self._inv_method = self._inv.worst_in
            self._single_inv_return = True
        elif self._type == "max":
            self._inv_method = self._inv.best_in
            self._single_inv_return = True
        elif self._type == "some":
            self._inv_method = self._inv.some
            self._single_inv_return = False
        elif self._type == "all":
            self._inv_method = self._inv.all  # type: ignore
            self._single_inv_return = False
        else:
            raise ValueError(f"Unknown selectortype: {self._type}")

        _worst, self._best = ufun.extreme_outcomes()
        self._min, self._max = float(ufun(_worst)), float(ufun(self._best))
        if self._min < ufun.reserved_value:
            self._min = ufun.reserved_value

    def before_proposing(self, state: SAOState):
        if self._negotiator is None:
            raise ValueError("Unknown negotiator in a component")
        if self._inv is None or self._max is None or self._min is None:
            warn(
                f"It seems that on_prefrences_changed() was not called until propose for a ufun of type {self._negotiator.ufun.__class__.__name__}"
                f" for negotiator {self._negotiator.id} of type {self.__class__.__name__}",
                NegmasUnexpectedValueWarning,
            )
            self.on_preferences_changed([PreferencesChange.General])
        if self._inv is None or self._max is None or self._min is None:
            raise ValueError(
                "Failed to find an invertor, a selector, or exreme outputs"
            )
        if not self._inv.initialized:
            self._inv.init()

    def scale_utilities(self, urange: tuple[float, ...]) -> tuple[float, ...]:
        """
        Scales given utilities to the range of the ufun.

        Remarks:

            - Assumes that the input utilities are in the range [0-1] no matter what
              is the range of the ufun.
            - Subtracts the `tolerance` from the first and adds it to the last utility value
              which slightly enlarges the range to account for small rounding errors
        """
        if self._negotiator is None:
            raise ValueError("Unknown negotiator in a component")
        if self._max is None or self._min is None or self._best is None:
            warn(
                f"It seems that on_prefrences_changed() was not called until propose for a ufun of type {self._negotiator.ufun.__class__.__name__}"
                f" for negotiator {self._negotiator.id} of type {self.__class__.__name__}",
                NegmasUnexpectedValueWarning,
            )
            self.on_preferences_changed([PreferencesChange.General])
        if self._max is None or self._min is None:
            raise ValueError("Cannot find extreme outcomes.")
        adjusted = [(self._max - self._min) * _ + self._min for _ in urange]
        if adjusted:
            adjusted[0] -= self._eps
            adjusted[-1] += self._eps
        return tuple(adjusted)

    def __call__(
        self, urange: tuple[float, float], state: SAOState
    ) -> Sequence[Outcome]:
        """
        Receives a normalized [0-> 1] utility range and returns a utility range relative to the ufun taking the tolerance _eps into account

        Remarks:

            - This method calls `scale_utilities` on the input range
        """
        if self._negotiator is None:
            raise ValueError("Unknown negotiator in a component")
        urange = self.scale_utilities(urange)
        outcomes = self._inv_method(urange)  # type: ignore
        if outcomes is None:
            return []
        if self._single_inv_return:
            return [outcomes]  # type: ignore
        return outcomes

    @property
    def tolerance(self):
        return self._eps

    @property
    def ufun_max(self):
        return self._max

    @property
    def ufun_min(self):
        return self._min


class UtilityInverter(SAODoNothingComponent):
    """
    A component that can recommend an outcome based on utility
    """

    def __init__(
        self,
        *args,
        offer_selector: OfferSelectorProtocol
        | Literal["min"]
        | Literal["max"]
        | None = None,
        **kwargs,
    ):
        if offer_selector is None:
            type_ = "one"
        elif isinstance(offer_selector, Callable):
            type_ = "some"
        else:
            type_ = offer_selector
        self._recommender = UtilityBasedOutcomeSetRecommender(
            *args, inversion_method=type_, **kwargs
        )
        self._selector: Callable[
            [Sequence[Outcome], SAOState], Outcome | None
        ] | None = (
            None if not isinstance(offer_selector, Callable) else offer_selector
        )
        self.set_negotiator(None)  # type: ignore (It is OK. We do not really need to pass this at all here.)

    @property
    def recommender(self) -> UtilityBasedOutcomeSetRecommender:
        return self._recommender

    def set_negotiator(self, negotiator: SAONegotiator) -> None:
        self._recommender.set_negotiator(negotiator)

    def on_preferences_changed(self, changes: list[PreferencesChange]):
        self._recommender.on_preferences_changed(changes)

    def before_proposing(self, state: SAOState):
        self._recommender.before_proposing(state)

    def __call__(self, urange: tuple[float, float], state: SAOState) -> Outcome | None:
        """
        Receives a normalized [0-> 1] utility range and returns a utility range relative to the ufun taking the tolerance _eps into account

        Remarks:

            - This method calls `scale_utilities` on the input range
        """
        outcomes = self._recommender(urange, state)
        if not outcomes:
            return None
        if len(outcomes) == 1:
            return outcomes[0]
        if self._selector is None:
            return choice(outcomes)
        outcome = self._selector(outcomes, state)
        if not outcome:
            return self._recommender._best
        return outcome

    def scale_utilities(self, urange):
        return self._recommender.scale_utilities(urange)

    @property
    def tolerance(self):
        return self._recommender._eps

    @property
    def ufun_max(self):
        return self._recommender._max

    @property
    def ufun_min(self):
        return self._recommender._min
