"""offering components (split from offering.py): anac2010."""

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
    "GIAMCrazyHagglerOffering",
    "GAgentKOffering",
    "GAgentFSEGAOffering",
    "GAgentSmithOffering",
    "GNozomiOffering",
    "GYushuOffering",
    "GIAMhaggler2010Offering",
]


@define
class GIAMCrazyHagglerOffering(GeniusOfferingPolicy):
    """
    IAMCrazyHaggler offering strategy from ANAC 2010.

    This strategy generates random bids with utility above a breakoff threshold.
    It's a simple but effective hardliner strategy that never concedes below
    a minimum utility level.

    Args:
        breakoff: Minimum utility threshold (default 0.9).

    Transcompiled from: negotiator.boaframework.offeringstrategy.anac2010.IAMCrazyHaggler_Offering
    """

    breakoff: float = 0.9
    _sorter: InverseUFun | None = field(init=False, default=None)

    def on_preferences_changed(self, changes: list[PreferencesChange]) -> None:
        """Initialize the inverse utility function."""
        if not self.negotiator or not self.negotiator.ufun:
            return
        self._sorter = DefaultInverseUtilityFunction(
            self.negotiator.ufun, rational_only=True, eps=-1, rel_eps=-1
        )
        self._sorter.init()

    def __call__(
        self, state: GBState, dest: str | None = None
    ) -> Outcome | ExtendedOutcome | None:
        """Generate a random bid above the breakoff threshold."""
        if not self.negotiator or not self.negotiator.nmi:
            return None

        if self._sorter is None:
            self.on_preferences_changed([])
            if self._sorter is None:
                return None

        # Try to find a random outcome above breakoff
        for _ in range(100):
            outcome = self.negotiator.nmi.random_outcome()
            if outcome is not None and self.negotiator.ufun:
                util = float(self.negotiator.ufun(outcome))
                if util > self.breakoff:
                    return outcome

        # Fallback: return best outcome in range
        return self._sorter.worst_in((self.breakoff, 1.1), normalized=False)


@define
class GAgentKOffering(GeniusOfferingPolicy):
    """
    AgentK offering strategy from ANAC 2010.

    This strategy uses a time-dependent target utility that adapts based on
    the opponent's behavior. It maintains a map of offered bids and selects
    bids above a dynamic target threshold.

    Transcompiled from: negotiator.boaframework.offeringstrategy.anac2010.AgentK_Offering
    """

    _sorter: InverseUFun | None = field(init=False, default=None)
    _target: float = field(init=False, default=0.9)
    _bid_target: float = field(init=False, default=0.9)
    _pmax: float = field(init=False, default=1.0)
    _pmin: float = field(init=False, default=0.0)

    def on_preferences_changed(self, changes: list[PreferencesChange]) -> None:
        """Initialize utility function and parameters."""
        if not self.negotiator or not self.negotiator.ufun:
            return
        self._sorter = DefaultInverseUtilityFunction(
            self.negotiator.ufun, rational_only=True, eps=-1, rel_eps=-1
        )
        self._sorter.init()
        self._pmin, self._pmax = self._sorter.minmax()
        self._target = self._pmax
        self._bid_target = self._pmax

    def _calculate_target(self, t: float) -> float:
        """Calculate target utility based on time."""
        # AgentK uses a concession curve
        return self._pmin + (self._pmax - self._pmin) * (1.0 - t)

    def __call__(
        self, state: GBState, dest: str | None = None
    ) -> Outcome | ExtendedOutcome | None:
        """Generate a bid based on AgentK's strategy."""
        if not self.negotiator or not self.negotiator.ufun:
            return None

        if self._sorter is None:
            self.on_preferences_changed([])
            if self._sorter is None:
                return None

        t = state.relative_time
        self._target = self._calculate_target(t)

        # Find a bid above the target
        outcome = self._sorter.worst_in(
            (self._target - 0.01, self._pmax + 0.01), normalized=False
        )
        if outcome is not None:
            return outcome

        return self._sorter.best()


@define
class GAgentFSEGAOffering(GeniusOfferingPolicy):
    """
    AgentFSEGA offering strategy from ANAC 2010.

    This strategy uses a time-dependent utility threshold that decreases
    exponentially over time. It selects bids that maximize opponent utility
    while staying above the threshold.

    Transcompiled from: negotiator.boaframework.offeringstrategy.anac2010.AgentFSEGA_Offering
    """

    min_utility: float = 0.5
    sigma: float = 0.01
    _sorter: InverseUFun | None = field(init=False, default=None)
    _pmax: float = field(init=False, default=1.0)

    def on_preferences_changed(self, changes: list[PreferencesChange]) -> None:
        """Initialize utility function."""
        if not self.negotiator or not self.negotiator.ufun:
            return
        self._sorter = DefaultInverseUtilityFunction(
            self.negotiator.ufun, rational_only=True, eps=-1, rel_eps=-1
        )
        self._sorter.init()
        _, self._pmax = self._sorter.minmax()

    def _get_min_allowed(self, t: float) -> float:
        """Calculate minimum allowed utility at time t."""
        import math

        return max(0.98 * math.exp(math.log(0.52) * t), self.min_utility)

    def __call__(
        self, state: GBState, dest: str | None = None
    ) -> Outcome | ExtendedOutcome | None:
        """Generate a bid based on AgentFSEGA's strategy."""
        if not self.negotiator or not self.negotiator.ufun:
            return None

        if self._sorter is None:
            self.on_preferences_changed([])
            if self._sorter is None:
                return None

        t = state.relative_time
        min_allowed = self._get_min_allowed(t)

        # Find a bid above minimum allowed utility
        outcome = self._sorter.worst_in(
            (min_allowed - self.sigma, self._pmax + 0.01), normalized=False
        )
        if outcome is not None:
            return outcome

        return self._sorter.best()


@define
class GAgentSmithOffering(GeniusOfferingPolicy):
    """
    AgentSmith offering strategy from ANAC 2010.

    This strategy offers bids based on a time-dependent concession,
    similar to Boulware but with specific parameters.

    Transcompiled from: negotiator.boaframework.offeringstrategy.anac2010.AgentSmith_Offering
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
        """Generate a bid based on AgentSmith's strategy."""
        if not self.negotiator or not self.negotiator.ufun:
            return None

        if self._sorter is None:
            self.on_preferences_changed([])
            if self._sorter is None:
                return None

        t = state.relative_time
        # Boulware-like concession
        target = self._pmin + (self._pmax - self._pmin) * (1.0 - pow(t, 0.2))

        outcome = self._sorter.worst_in(
            (target - 0.01, self._pmax + 0.01), normalized=False
        )
        if outcome is not None:
            return outcome

        return self._sorter.best()


@define
class GNozomiOffering(GeniusOfferingPolicy):
    """
    Nozomi offering strategy from ANAC 2010.

    This strategy uses adaptive concession based on opponent behavior
    and time pressure.

    Transcompiled from: negotiator.boaframework.offeringstrategy.anac2010.Nozomi_Offering
    """

    _sorter: InverseUFun | None = field(init=False, default=None)
    _pmin: float = field(init=False, default=0.0)
    _pmax: float = field(init=False, default=1.0)
    _last_target: float = field(init=False, default=1.0)

    def on_preferences_changed(self, changes: list[PreferencesChange]) -> None:
        """Initialize utility function."""
        if not self.negotiator or not self.negotiator.ufun:
            return
        self._sorter = DefaultInverseUtilityFunction(
            self.negotiator.ufun, rational_only=True, eps=-1, rel_eps=-1
        )
        self._sorter.init()
        self._pmin, self._pmax = self._sorter.minmax()
        self._last_target = self._pmax

    def __call__(
        self, state: GBState, dest: str | None = None
    ) -> Outcome | ExtendedOutcome | None:
        """Generate a bid based on Nozomi's strategy."""
        if not self.negotiator or not self.negotiator.ufun:
            return None

        if self._sorter is None:
            self.on_preferences_changed([])
            if self._sorter is None:
                return None

        t = state.relative_time

        # Nozomi uses slower concession early, faster late
        if t < 0.8:
            target = self._pmax - (self._pmax - self._pmin) * 0.1 * (t / 0.8)
        else:
            remaining = (t - 0.8) / 0.2
            target = self._pmax * 0.9 - (self._pmax * 0.9 - self._pmin) * remaining

        target = max(target, self._pmin)
        self._last_target = target

        outcome = self._sorter.worst_in(
            (target - 0.01, self._pmax + 0.01), normalized=False
        )
        if outcome is not None:
            return outcome

        return self._sorter.best()


@define
class GYushuOffering(GeniusOfferingPolicy):
    """
    Yushu offering strategy from ANAC 2010.

    This strategy uses a sigmoid-like concession curve.

    Transcompiled from: negotiator.boaframework.offeringstrategy.anac2010.Yushu_Offering
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
        """Generate a bid based on Yushu's strategy."""
        if not self.negotiator or not self.negotiator.ufun:
            return None

        if self._sorter is None:
            self.on_preferences_changed([])
            if self._sorter is None:
                return None

        import math

        t = state.relative_time
        # Sigmoid-like concession
        sigmoid = 1.0 / (1.0 + math.exp(-12.0 * (t - 0.7)))
        target = self._pmax - (self._pmax - self._pmin) * sigmoid * 0.5

        outcome = self._sorter.worst_in(
            (target - 0.01, self._pmax + 0.01), normalized=False
        )
        if outcome is not None:
            return outcome

        return self._sorter.best()


@define
class GIAMhaggler2010Offering(GeniusOfferingPolicy):
    """
    IAMhaggler2010 offering strategy from ANAC 2010.

    This strategy uses sophisticated time-dependent concession with
    opponent modeling considerations.

    Transcompiled from: negotiator.boaframework.offeringstrategy.anac2010.IAMhaggler2010_Offering
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
        """Generate a bid based on IAMhaggler2010's strategy."""
        if not self.negotiator or not self.negotiator.ufun:
            return None

        if self._sorter is None:
            self.on_preferences_changed([])
            if self._sorter is None:
                return None

        t = state.relative_time
        # Conservative concession with acceleration near end
        if t < 0.9:
            target = self._pmax - (self._pmax - self._pmin) * 0.15 * (t / 0.9)
        else:
            base = self._pmax * 0.85
            remaining = (t - 0.9) / 0.1
            target = base - (base - self._pmin) * remaining * 0.5

        target = max(target, self._pmin)

        outcome = self._sorter.worst_in(
            (target - 0.01, self._pmax + 0.01), normalized=False
        )
        if outcome is not None:
            return outcome

        return self._sorter.best()
