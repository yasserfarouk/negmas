from __future__ import annotations

from random import choice
from typing import TYPE_CHECKING, Callable, Literal, Sequence

from negmas.common import PreferencesChange
from negmas.gb.components import GBComponent
from negmas.preferences import (
    BaseUtilityFunction,
    InverseUFun,
    PresortingInverseUtilityFunction,
    RankOnlyUtilityFunction,
)
from negmas.warnings import NegmasUnexpectedValueWarning, warn

__all__ = [
    "UtilityInverter",
    "UtilityBasedOutcomeSetRecommender",
]

if TYPE_CHECKING:
    from negmas.gb import GBNegotiator, GBState
    from negmas.outcomes import Outcome

    from .selectors import OfferSelectorProtocol


def make_inverter(
    ufun: BaseUtilityFunction,
    ufun_inverter: Callable[[BaseUtilityFunction], InverseUFun] | None = None,
    rank_only: bool = False,
    max_cardinality: int | float = float("inf"),
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
    return PresortingInverseUtilityFunction(ufun)
    # return (
    #     SamplingInverseUtilityFunction(ufun)
    #     if ufun.outcome_space is None
    #     or ufun.outcome_space.is_discrete() and ufun.outcome_space.cardinality >= max_cardinality
    #     else PresortingInverseUtilityFunction(ufun)
    # )


class UtilityBasedOutcomeSetRecommender(GBComponent):
    """
    Recommends a set of outcome appropriate for proposal

    """

    def __init__(
        self,
        rank_only: bool = False,
        ufun_inverter: Callable[[BaseUtilityFunction], InverseUFun] | None = None,
        max_cardinality: int | float = float("inf"),
        eps: float = 0.0001,
        inversion_method: Literal["min"]
        | Literal["max"]
        | Literal["one"]
        | Literal["some"]
        | Literal["all"] = "some",
    ):
        super().__init__()
        self._rank_only = rank_only
        self._max_cardinality = max_cardinality
        self._ufun_inverter = ufun_inverter
        self._inv_method = None
        self._inversion_method = inversion_method
        self.eps = eps
        self.inv: InverseUFun | None = None
        self.min = self.max = self.best = None
        self._inv_method = None

    def set_negotiator(self, negotiator: GBNegotiator) -> None:
        super().set_negotiator(negotiator)
        self.inv: InverseUFun | None = None
        self.min = self.max = self.best = None
        self._inv_method = None

    def on_preferences_changed(self, changes: list[PreferencesChange]):
        if self._negotiator is None:
            raise ValueError("Unknown negotiator in a component")
        ufun = self._negotiator.ufun

        if not ufun:
            self.inv = None
            self.min = self.max = self.best = None
            return
        self.inv = make_inverter(
            ufun, self._ufun_inverter, self._rank_only, self._max_cardinality
        )
        if self._inversion_method == "one":
            self._inv_method = self.inv.one_in
            self._single_inv_return = True
        elif self._inversion_method == "min":
            self._inv_method = self.inv.worst_in
            self._single_inv_return = True
        elif self._inversion_method == "max":
            self._inv_method = self.inv.best_in
            self._single_inv_return = True
        elif self._inversion_method == "some":
            self._inv_method = self.inv.some
            self._single_inv_return = False
        elif self._inversion_method == "all":
            self._inv_method = self.inv.all  # type: ignore
            self._single_inv_return = False
        else:
            raise ValueError(f"Unknown selectortype: {self._inversion_method}")

        _worst, self.best = ufun.extreme_outcomes()
        self.min, self.max = float(ufun(_worst)), float(ufun(self.best))
        if self.min < ufun.reserved_value:
            self.min = ufun.reserved_value

    def before_proposing(self, state: GBState):
        if self._negotiator is None:
            raise ValueError("Unknown negotiator in a component")
        if self.inv is None or self.max is None or self.min is None:
            warn(
                f"It seems that on_prefrences_changed() was not called until propose for a ufun of type {self._negotiator.ufun.__class__.__name__}"
                f" for negotiator {self._negotiator.id} of type {self.__class__.__name__}",
                NegmasUnexpectedValueWarning,
            )
            self.on_preferences_changed([PreferencesChange()])
        if self.inv is None or self.max is None or self.min is None:
            raise ValueError(
                "Failed to find an invertor, a selector, or exreme outputs"
            )
        if not self.inv.initialized:
            self.inv.init()

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
        if self.max is None or self.min is None or self.best is None:
            warn(
                f"It seems that on_prefrences_changed() was not called until propose for a ufun of type {self._negotiator.ufun.__class__.__name__}"
                f" for negotiator {self._negotiator.id} of type {self.__class__.__name__}",
                NegmasUnexpectedValueWarning,
            )
            self.on_preferences_changed([PreferencesChange()])
        if self.max is None or self.min is None:
            raise ValueError("Cannot find extreme outcomes.")
        adjusted = [(self.max - self.min) * _ + self.min for _ in urange]
        if adjusted:
            adjusted[0] -= self.eps
            adjusted[-1] += self.eps
        return tuple(adjusted)

    @property
    def tolerance(self):
        return self.eps

    @property
    def ufun_max(self):
        return self.max

    @property
    def ufun_min(self):
        return self.min

    def __call__(
        self, urange: tuple[float, float], state: GBState
    ) -> Sequence[Outcome]:
        """
        Receives a normalized [0-> 1] utility range and returns a utility range relative to the ufun taking the tolerance _eps into account

        Remarks:

            - This method calls `scale_utilities` on the input range
        """
        if self._negotiator is None:
            raise ValueError("Unknown negotiator in a component")
        urange = self.scale_utilities(urange)
        outcomes = self._inv_method(urange, normalized=False)  # type: ignore
        if outcomes is None:
            return []
        if self._single_inv_return:
            return [outcomes]  # type: ignore
        return outcomes


class UtilityInverter(GBComponent):
    """
    A component that can recommend an outcome based on utility
    """

    def set_negotiator(self, negotiator: GBNegotiator) -> None:
        super().set_negotiator(negotiator)
        self.recommender.set_negotiator(negotiator)

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
        self.recommender = UtilityBasedOutcomeSetRecommender(
            *args, inversion_method=type_, **kwargs
        )
        self.selector: Callable[[Sequence[Outcome], GBState], Outcome | None] | None
        self.selector = (
            None if not isinstance(offer_selector, Callable) else offer_selector
        )
        self.set_negotiator(None)  # type: ignore (It is OK. We do not really need to pass this at all here.)

    def on_preferences_changed(self, changes: list[PreferencesChange]):
        self.recommender.on_preferences_changed(changes)

    def before_proposing(self, state: GBState):
        self.recommender.before_proposing(state)

    def scale_utilities(self, urange):
        return self.recommender.scale_utilities(urange)

    @property
    def tolerance(self):
        return self.recommender.eps

    @property
    def ufun_max(self):
        return self.recommender.max

    @property
    def ufun_min(self):
        return self.recommender.min

    def __call__(self, urange: tuple[float, float], state: GBState) -> Outcome | None:
        """
        Receives a normalized [0-> 1] utility range and returns a utility range relative to the ufun taking the tolerance _eps into account

        Remarks:

            - This method calls `scale_utilities` on the input range
        """
        outcomes = self.recommender(urange, state)
        if not outcomes:
            return None
        if len(outcomes) == 1:
            return outcomes[0]
        if self.selector is None:
            return choice(outcomes)
        outcome = self.selector(outcomes, state)
        if not outcome:
            return self.recommender.best
        return outcome
