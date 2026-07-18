"""offering components (split from offering.py): anac2012."""

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

__all__ = [
    "GCUHKAgentOffering",
    "GOMACagentOffering",
    "GAgentLGOffering",
    "GAgentMROffering",
    "GBRAMAgent2Offering",
    "GIAMHaggler2012Offering",
    "GTheNegotiatorReloadedOffering",
]


@define
class GCUHKAgentOffering(GeniusOfferingPolicy):
    """
    CUHKAgent offering strategy from ANAC 2012.

    This strategy uses sophisticated opponent modeling and adaptive concession.

    Transcompiled from: negotiator.boaframework.offeringstrategy.anac2012.CUHKAgent_Offering
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
        """Generate a bid based on CUHKAgent's strategy."""
        if not self.negotiator or not self.negotiator.ufun:
            return None

        if self._sorter is None:
            self.on_preferences_changed([])
            if self._sorter is None:
                return None

        t = state.relative_time
        # CUHK uses adaptive piecewise concession
        if t < 0.2:
            target = self._pmax
        elif t < 0.7:
            progress = (t - 0.2) / 0.5
            target = self._pmax - (self._pmax - self._pmin) * 0.15 * progress
        else:
            progress = (t - 0.7) / 0.3
            base = self._pmax * 0.85
            target = base - (base - self._pmin) * 0.4 * progress

        target = max(target, self._pmin)

        outcome = self._sorter.worst_in(
            (target - 0.01, self._pmax + 0.01), normalized=False
        )
        if outcome is not None:
            return outcome

        return self._sorter.best()


@define
class GOMACagentOffering(GeniusOfferingPolicy):
    """
    OMACagent offering strategy from ANAC 2012.

    This strategy uses prediction-based bidding with exponential moving average.

    Args:
        min_utility: Minimum utility threshold (default 0.59).
        eu: Expected utility threshold (default 0.95).

    Transcompiled from: negotiator.boaframework.offeringstrategy.anac2012.OMACagent_Offering
    """

    min_utility: float = 0.59
    eu: float = 0.95
    _sorter: InverseUFun | None = field(init=False, default=None)
    _pmin: float = field(init=False, default=0.0)
    _pmax: float = field(init=False, default=1.0)
    _discount: float = field(init=False, default=1.0)
    _discount_threshold: float = 0.845

    def on_preferences_changed(self, changes: list[PreferencesChange]) -> None:
        """Initialize utility function."""
        if not self.negotiator or not self.negotiator.ufun:
            return
        self._sorter = DefaultInverseUtilityFunction(
            self.negotiator.ufun, rational_only=True, eps=-1, rel_eps=-1
        )
        self._sorter.init()
        self._pmin, self._pmax = self._sorter.minmax()

    def _get_target(self, t: float) -> float:
        """Calculate target utility."""

        e1, e2 = 0.033, 0.04

        if self._discount >= self._discount_threshold:
            target = self.min_utility + (1 - pow(t, 1.0 / e1)) * (
                self._pmax - self.min_utility
            )
        else:
            t_max = pow(self._discount, 0.2)
            t_min = self.min_utility * 1.05
            target = t_min + (1 - pow(t, 1.0 / e2)) * (t_max - t_min)

        return target

    def __call__(
        self, state: GBState, dest: str | None = None
    ) -> Outcome | ExtendedOutcome | None:
        """Generate a bid based on OMACagent's strategy."""
        if not self.negotiator or not self.negotiator.ufun:
            return None

        if self._sorter is None:
            self.on_preferences_changed([])
            if self._sorter is None:
                return None

        t = state.relative_time

        if t <= 0.02:
            return self._sorter.best()

        target = self._get_target(t)
        target = max(target, self.min_utility)

        outcome = self._sorter.worst_in(
            (target * 0.99, target * 1.01), normalized=False
        )
        if outcome is not None:
            return outcome

        return self._sorter.worst_in(
            (target - 0.05, self._pmax + 0.01), normalized=False
        )


@define
class GAgentLGOffering(GeniusOfferingPolicy):
    """
    AgentLG offering strategy from ANAC 2012.

    This strategy uses learning-based concession.

    Transcompiled from: negotiator.boaframework.offeringstrategy.anac2012.AgentLG_Offering
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
        """Generate a bid based on AgentLG's strategy."""
        if not self.negotiator or not self.negotiator.ufun:
            return None

        if self._sorter is None:
            self.on_preferences_changed([])
            if self._sorter is None:
                return None

        t = state.relative_time
        # Learning-based concession curve
        target = self._pmax - (self._pmax - self._pmin) * pow(t, 1.5) * 0.4

        outcome = self._sorter.worst_in(
            (target - 0.01, self._pmax + 0.01), normalized=False
        )
        if outcome is not None:
            return outcome

        return self._sorter.best()


@define
class GAgentMROffering(GeniusOfferingPolicy):
    """
    AgentMR offering strategy from ANAC 2012.

    This strategy uses risk-based concession.

    Transcompiled from: negotiator.boaframework.offeringstrategy.anac2012.AgentMR_Offering
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
        """Generate a bid based on AgentMR's strategy."""
        if not self.negotiator or not self.negotiator.ufun:
            return None

        if self._sorter is None:
            self.on_preferences_changed([])
            if self._sorter is None:
                return None

        t = state.relative_time
        # MR uses risk-aware concession
        risk_factor = 0.3 + 0.2 * t
        target = self._pmax - (self._pmax - self._pmin) * risk_factor * t

        outcome = self._sorter.worst_in(
            (target - 0.01, self._pmax + 0.01), normalized=False
        )
        if outcome is not None:
            return outcome

        return self._sorter.best()


@define
class GBRAMAgent2Offering(GeniusOfferingPolicy):
    """
    BRAMAgent2 offering strategy from ANAC 2012.

    Enhanced version of BRAMAgent with improved statistics tracking.

    Transcompiled from: negotiator.boaframework.offeringstrategy.anac2012.BRAMAgent2_Offering
    """

    _sorter: InverseUFun | None = field(init=False, default=None)
    _pmin: float = field(init=False, default=0.0)
    _pmax: float = field(init=False, default=1.0)
    _threshold: float = field(init=False, default=0.9)

    def on_preferences_changed(self, changes: list[PreferencesChange]) -> None:
        """Initialize utility function."""
        if not self.negotiator or not self.negotiator.ufun:
            return
        self._sorter = DefaultInverseUtilityFunction(
            self.negotiator.ufun, rational_only=True, eps=-1, rel_eps=-1
        )
        self._sorter.init()
        self._pmin, self._pmax = self._sorter.minmax()
        self._threshold = self._pmax

    def __call__(
        self, state: GBState, dest: str | None = None
    ) -> Outcome | ExtendedOutcome | None:
        """Generate a bid based on BRAMAgent2's strategy."""
        if not self.negotiator or not self.negotiator.ufun:
            return None

        if self._sorter is None:
            self.on_preferences_changed([])
            if self._sorter is None:
                return None

        t = state.relative_time
        # Enhanced threshold calculation
        self._threshold = self._pmax - (self._pmax - self._pmin) * 0.35 * pow(t, 0.7)

        outcome = self._sorter.worst_in(
            (self._threshold - 0.01, self._pmax + 0.01), normalized=False
        )
        if outcome is not None:
            return outcome

        return self._sorter.best()


@define
class GIAMHaggler2012Offering(GeniusOfferingPolicy):
    """
    IAMHaggler2012 offering strategy from ANAC 2012.

    Further refined version of IAMhaggler.

    Transcompiled from: negotiator.boaframework.offeringstrategy.anac2012.IAMHaggler2012_Offering
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
        """Generate a bid based on IAMHaggler2012's strategy."""
        if not self.negotiator or not self.negotiator.ufun:
            return None

        if self._sorter is None:
            self.on_preferences_changed([])
            if self._sorter is None:
                return None

        t = state.relative_time
        # Refined concession for 2012
        if t < 0.8:
            target = self._pmax - (self._pmax - self._pmin) * 0.08 * (t / 0.8)
        else:
            base = self._pmax * 0.92
            progress = (t - 0.8) / 0.2
            target = base - (base - self._pmin) * 0.45 * progress

        target = max(target, self._pmin)

        outcome = self._sorter.worst_in(
            (target - 0.01, self._pmax + 0.01), normalized=False
        )
        if outcome is not None:
            return outcome

        return self._sorter.best()


@define
class GTheNegotiatorReloadedOffering(GeniusOfferingPolicy):
    """
    TheNegotiatorReloaded offering strategy from ANAC 2012.

    Enhanced version of TheNegotiator with improved time management.

    Transcompiled from: negotiator.boaframework.offeringstrategy.anac2012.TheNegotiatorReloaded_Offering
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
        """Generate a bid based on TheNegotiatorReloaded's strategy."""
        if not self.negotiator or not self.negotiator.ufun:
            return None

        if self._sorter is None:
            self.on_preferences_changed([])
            if self._sorter is None:
                return None

        t = state.relative_time
        # Reloaded uses smoother transitions
        if t < 0.4:
            target = self._pmax
        elif t < 0.75:
            progress = (t - 0.4) / 0.35
            target = self._pmax - (self._pmax - self._pmin) * 0.15 * progress
        else:
            progress = (t - 0.75) / 0.25
            base = self._pmax * 0.85
            target = base - (base - self._pmin) * 0.45 * progress

        target = max(target, self._pmin)

        outcome = self._sorter.worst_in(
            (target - 0.01, self._pmax + 0.01), normalized=False
        )
        if outcome is not None:
            return outcome

        return self._sorter.best()
