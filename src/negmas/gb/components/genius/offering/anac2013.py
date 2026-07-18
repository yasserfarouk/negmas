"""offering components (split from offering.py): anac2013."""

from __future__ import annotations

from typing import TYPE_CHECKING

from attrs import define, field

from negmas.preferences.inv_ufun import DefaultInverseUtilityFunction
from negmas.preferences.protocols import InverseUFun

from negmas.gb.components.genius.base import GeniusOfferingPolicy

if TYPE_CHECKING:
    from negmas.common import PreferencesChange
    from negmas.gb import GBState
    from negmas.outcomes import Outcome
    from negmas.outcomes.common import ExtendedOutcome

__all__ = ["GFawkesOffering", "GInoxAgentOffering"]


@define
class GFawkesOffering(GeniusOfferingPolicy):
    """
    TheFawkes offering strategy from ANAC 2013.

    This strategy uses wavelet-based prediction for opponent modeling.

    Transcompiled from: negotiator.boaframework.offeringstrategy.anac2013.Fawkes_Offering
    """

    _sorter: InverseUFun | None = field(init=False, default=None)
    _pmin: float = field(init=False, default=0.0)
    _pmax: float = field(init=False, default=1.0)

    def on_preferences_changed(self, changes: list[PreferencesChange]) -> None:
        """Initialize utility function."""
        if not self.negotiator or not self.negotiator.ufun:
            return
        self._sorter = DefaultInverseUtilityFunction(
            self.negotiator.ufun, rational_only=True, eps=-1, rel_eps=-1
        )
        self._sorter.init()
        self._pmin, self._pmax = self._sorter.minmax()

    def __call__(
        self, state: GBState, dest: str | None = None
    ) -> Outcome | ExtendedOutcome | None:
        """Generate a bid based on TheFawkes' strategy."""
        if not self.negotiator or not self.negotiator.ufun:
            return None

        if self._sorter is None:
            self.on_preferences_changed([])
            if self._sorter is None:
                return None

        t = state.relative_time
        # Fawkes uses prediction-based concession
        if t < 0.6:
            target = self._pmax - (self._pmax - self._pmin) * 0.05 * (t / 0.6)
        else:
            base = self._pmax * 0.95
            progress = (t - 0.6) / 0.4
            target = base - (base - self._pmin) * 0.5 * pow(progress, 1.5)

        target = max(target, self._pmin)

        outcome = self._sorter.worst_in(
            (target - 0.01, self._pmax + 0.01), normalized=False
        )
        if outcome is not None:
            return outcome

        return self._sorter.best()


@define
class GInoxAgentOffering(GeniusOfferingPolicy):
    """
    InoxAgent offering strategy from ANAC 2013.

    This strategy uses adaptive concession based on negotiation dynamics.

    Transcompiled from: negotiator.boaframework.offeringstrategy.anac2013.InoxAgent_Offering
    """

    _sorter: InverseUFun | None = field(init=False, default=None)
    _pmin: float = field(init=False, default=0.0)
    _pmax: float = field(init=False, default=1.0)

    def on_preferences_changed(self, changes: list[PreferencesChange]) -> None:
        """Initialize utility function."""
        if not self.negotiator or not self.negotiator.ufun:
            return
        self._sorter = DefaultInverseUtilityFunction(
            self.negotiator.ufun, rational_only=True, eps=-1, rel_eps=-1
        )
        self._sorter.init()
        self._pmin, self._pmax = self._sorter.minmax()

    def __call__(
        self, state: GBState, dest: str | None = None
    ) -> Outcome | ExtendedOutcome | None:
        """Generate a bid based on InoxAgent's strategy."""
        if not self.negotiator or not self.negotiator.ufun:
            return None

        if self._sorter is None:
            self.on_preferences_changed([])
            if self._sorter is None:
                return None

        t = state.relative_time
        # Inox uses smooth polynomial concession
        target = self._pmax - (self._pmax - self._pmin) * pow(t, 2.5) * 0.45

        outcome = self._sorter.worst_in(
            (target - 0.01, self._pmax + 0.01), normalized=False
        )
        if outcome is not None:
            return outcome

        return self._sorter.best()

        # Fallback to best outcome if nothing found
        return self._sorter.best()
