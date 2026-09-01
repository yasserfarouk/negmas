"""offering components (split from offering.py): anac2013."""

from __future__ import annotations

from typing import TYPE_CHECKING

from attrs import define, field

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

    phase_end: float = 0.6
    """Relative time separating the early phase from the late one."""
    early_drop: float = 0.05
    """Fraction of the utility range conceded during the early phase."""
    late_base_factor: float = 0.95
    """Fraction of the maximum utility the late phase starts from."""
    late_drop: float = 0.5
    """Fraction of the remaining range conceded during the late phase."""
    late_exponent: float = 1.5
    """Exponent applied to the late-phase progress (larger concedes later)."""
    _sorter: InverseUFun | None = field(init=False, default=None)
    _pmin: float = field(init=False, default=0.0)
    _pmax: float = field(init=False, default=1.0)

    def on_preferences_changed(self, changes: list[PreferencesChange]) -> None:
        """Initialize utility function."""
        if not self.negotiator or not self.negotiator.ufun:
            return
        self._sorter = self.negotiator.ufun.invert(
            rational_only=True, eps=-1, rel_eps=-1
        )
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
        if t < self.phase_end:
            target = self._pmax - (self._pmax - self._pmin) * self.early_drop * (
                t / self.phase_end
            )
        else:
            base = self._pmax * self.late_base_factor
            progress = (t - self.phase_end) / (1.0 - self.phase_end)
            target = base - (base - self._pmin) * self.late_drop * pow(
                progress, self.late_exponent
            )

        target = max(target, self._pmin)

        outcome = self._sorter.worst_in(
            (
                target - self.utility_band_tolerance,
                self._pmax + self.utility_band_tolerance,
            ),
            normalized=False,
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

    concession_exponent: float = 2.5
    """Exponent applied to relative time (larger concedes later)."""
    max_concession: float = 0.45
    """Fraction of the utility range given away by the deadline."""
    _sorter: InverseUFun | None = field(init=False, default=None)
    _pmin: float = field(init=False, default=0.0)
    _pmax: float = field(init=False, default=1.0)

    def on_preferences_changed(self, changes: list[PreferencesChange]) -> None:
        """Initialize utility function."""
        if not self.negotiator or not self.negotiator.ufun:
            return
        self._sorter = self.negotiator.ufun.invert(
            rational_only=True, eps=-1, rel_eps=-1
        )
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
        target = (
            self._pmax
            - (self._pmax - self._pmin)
            * pow(t, self.concession_exponent)
            * self.max_concession
        )

        outcome = self._sorter.worst_in(
            (
                target - self.utility_band_tolerance,
                self._pmax + self.utility_band_tolerance,
            ),
            normalized=False,
        )
        if outcome is not None:
            return outcome

        return self._sorter.best()

        # Fallback to best outcome if nothing found
        return self._sorter.best()
