from __future__ import annotations

from random import choice
from typing import TYPE_CHECKING, Callable, Iterable, Literal, Sequence

from yaml import warnings

from negmas.preferences import (
    BaseUtilityFunction,
    InverseUFun,
    PresortingInverseUtilityFunction,
    RankOnlyUtilityFunction,
    SamplingInverseUtilityFunction,
)
from negmas.sao.components import SAODoNothingComponent
from negmas.warnings import NegmasUnexpectedValueWarning

__all__ = ["UtilityInverter"]

if TYPE_CHECKING:
    from negmas.common import PreferencesChange
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


class UtilityInverter(SAODoNothingComponent):
    """
    Represents a time-based negotiation strategy that is independent of the offers received during the negotiation.

    Args:
        name: The agent name
        preferences:  The utility function to attache with the agent
        owner: The `Agent` that owns the negotiator.
        parent: The parent which should be an `SAOController`
        rank_only (bool): rank_only
        max_cardinality (int): The maximum outcome space cardinality at which to use a `SamplingInverseUtilityFunction`. Only used if `ufun_inverter` is `None` .
        can_propose (bool): If `False` the agent cannot propose (can only accept/reject)
        eps: A fraction of maximum utility to use as slack when checking if an outcome's utility lies within the current range

    """

    def __init__(
        self,
        rank_only: bool = False,
        ufun_inverter: Callable[[BaseUtilityFunction], InverseUFun] | None = None,
        max_cardinality: int = 10_000,
        eps: float = 0.0001,
        offer_selector: OfferSelectorProtocol
        | Literal["min"]
        | Literal["max"]
        | None = None,
    ):
        self._rank_only = rank_only
        self._max_cartinality = max_cardinality
        self._inverter_factory = ufun_inverter
        self._selector_type = offer_selector
        self._selector: Callable[
            [Sequence[Outcome], SAOState], Outcome | None
        ] | None = None
        self._inv_method = None
        self._single_inv_return = False
        self._eps = eps
        self.set_negotiator(None)  # type: ignore (It is OK. We do not really need to pass this at all here.)

    def set_negotiator(self, negotiator: SAONegotiator) -> None:
        super().set_negotiator(negotiator)
        self._inv: InverseUFun | None = None
        self._min = self._max = self._best = None
        self._selector = None
        self._inv_method = None

    def on_preferences_changed(self, changes: list[PreferencesChange]):
        if self._negotiator is None:
            raise ValueError("Unknown negotiator in a component")
        ufun = self._negotiator.ufun

        if not ufun:
            self._inv, self._selector = None, None
            self._min = self._max = self._best = None
            return
        self._inv = make_inverter(
            ufun, self._inverter_factory, self._rank_only, self._max_cartinality
        )
        if self._selector_type is None:
            self._inv_method = self._inv.one_in
            self._single_inv_return = True
            self._selector = None
        elif isinstance(self._selector_type, Callable):
            self._selector, self._inv_method = self._selector_type, self._inv.some
            self._single_inv_return = False
        elif self._selector_type == "min":
            self._inv_method = self._inv.worst_in
            self._single_inv_return = True
            self._selector = None
        elif self._selector_type == "max":
            self._inv_method = self._inv.best_in
            self._single_inv_return = True
            self._selector = None
        else:
            raise ValueError(f"Unknown selectortype: {self._selector_type}")

        _worst, self._best = ufun.extreme_outcomes()
        self._min, self._max = float(ufun(_worst)), float(ufun(self._best))
        if self._min < ufun.reserved_value:
            self._min = ufun.reserved_value

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
            warnings.warn(
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

    def propose(self, urange: tuple[float, float], state: SAOState) -> Outcome | None:
        """
        Receives a normalized [0-> 1] utility range and returns a utility range relative to the ufun taking the tolerance _eps into account

        Remarks:

            - This method calls `scale_utilities` on the input range
        """
        if self._negotiator is None:
            raise ValueError("Unknown negotiator in a component")
        urange = self.scale_utilities(urange)
        outcomes = self._inv_method(urange)  # type: ignore
        if outcomes is None or self._single_inv_return:
            return outcomes  # type: ignore I know that this is a single outcome
        if not self._selector:
            return choice(outcomes)
        outcome = self._selector(outcomes, state)
        if not outcome:
            return self._best
        return outcome

    def before_proposing(self, state: SAOState):
        if self._negotiator is None:
            raise ValueError("Unknown negotiator in a component")
        if self._inv is None or self._max is None or self._min is None:
            warnings.warn(
                f"It seems that on_prefrences_changed() was not called until propose for a ufun of type {self._negotiator.ufun.__class__.__name__}"
                f" for negotiator {self._negotiator.id} of type {self.__class__.__name__}"
            )
            self.on_preferences_changed([PreferencesChange.General])
        if self._inv is None or self._max is None or self._min is None:
            raise ValueError(
                "Failed to find an invertor, a selector, or exreme outputs"
            )
        if not self._inv.initialized:
            self._inv.init()

    @property
    def tolerance(self):
        return self._eps

    @property
    def ufun_max(self):
        return self._max

    @property
    def ufun_min(self):
        return self._min
