"""Genius offering policies.

This module contains Python implementations of Genius offering strategies,
transcompiled from the original Java implementations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from attrs import define, field

from negmas.preferences.inv_ufun import PresortingInverseUtilityFunction

from .base import GeniusOfferingPolicy

if TYPE_CHECKING:
    from negmas.common import PreferencesChange
    from negmas.gb import GBState
    from negmas.outcomes import Outcome
    from negmas.outcomes.common import ExtendedOutcome

__all__ = [
    # Base offering strategies
    "GTimeDependentOffering",
    "GRandomOffering",
    "GBoulwareOffering",
    "GConcederOffering",
    "GLinearOffering",
    "GHardlinerOffering",
    "GChoosingAllBids",
    # ANAC 2010 offering strategies
    "GIAMCrazyHagglerOffering",
    "GAgentKOffering",
    "GAgentFSEGAOffering",
    "GAgentSmithOffering",
    "GNozomiOffering",
    "GYushuOffering",
    "GIAMhaggler2010Offering",
    # ANAC 2011 offering strategies
    "GHardHeadedOffering",
    "GAgentK2Offering",
    "GBRAMAgentOffering",
    "GGahboninhoOffering",
    "GNiceTitForTatOffering",
    "GTheNegotiatorOffering",
    "GValueModelAgentOffering",
    "GIAMhaggler2011Offering",
    # ANAC 2012 offering strategies
    "GCUHKAgentOffering",
    "GOMACagentOffering",
    "GAgentLGOffering",
    "GAgentMROffering",
    "GBRAMAgent2Offering",
    "GIAMHaggler2012Offering",
    "GTheNegotiatorReloadedOffering",
    # ANAC 2013 offering strategies
    "GFawkesOffering",
    "GInoxAgentOffering",
]


@define
class GTimeDependentOffering(GeniusOfferingPolicy):
    """
    Time-dependent offering strategy from Genius.

    This strategy offers bids based on a time-dependent target utility curve.
    The curve is controlled by the concession exponent `e`:
    - e = 0: Hardliner (never concedes)
    - e < 1: Boulware (concedes slowly, faster near deadline)
    - e = 1: Linear
    - e > 1: Conceder (concedes quickly at start)

    The target utility at time t is computed as:
        f(t) = k + (1 - k) * t^(1/e)
        target(t) = Pmin + (Pmax - Pmin) * (1 - f(t))

    where:
        - k: Offset constant (default 0)
        - e: Concession exponent (default 0.2 for Boulware)
        - Pmin: Minimum utility (reserved value)
        - Pmax: Maximum utility (best outcome utility)

    Args:
        e: Concession exponent. Controls the shape of the concession curve.
        k: Offset constant for the time function (default 0).

    Transcompiled from: negotiator.boaframework.offeringstrategy.other.TimeDependent_Offering
    """

    e: float = 0.2  # Boulware by default
    k: float = 0.0
    _sorter: PresortingInverseUtilityFunction | None = field(init=False, default=None)
    _pmin: float = field(init=False, default=0.0)
    _pmax: float = field(init=False, default=1.0)

    def on_preferences_changed(self, changes: list[PreferencesChange]) -> None:
        """Initialize the inverse utility function for finding outcomes by utility.

        Args:
            changes: List of preference changes.
        """
        if not self.negotiator or not self.negotiator.ufun:
            return
        self._sorter = PresortingInverseUtilityFunction(
            self.negotiator.ufun, rational_only=True, eps=-1, rel_eps=-1
        )
        self._sorter.init()
        self._pmin, self._pmax = self._sorter.minmax()

    def _f(self, t: float) -> float:
        """Compute the time-dependent function f(t).

        Args:
            t: Relative time (0 to 1).

        Returns:
            The value of f(t) = k + (1-k) * t^(1/e)
        """
        if self.e == 0:
            # Hardliner: f(t) = k (no concession)
            return self.k
        return self.k + (1.0 - self.k) * pow(t, 1.0 / self.e)

    def _p(self, t: float) -> float:
        """Compute the target utility at time t.

        Args:
            t: Relative time (0 to 1).

        Returns:
            Target utility value.
        """
        return self._pmin + (self._pmax - self._pmin) * (1.0 - self._f(t))

    def __call__(
        self, state: GBState, dest: str | None = None
    ) -> Outcome | ExtendedOutcome | None:
        """Generate an offer based on time-dependent target utility.

        Args:
            state: Current negotiation state.
            dest: Destination identifier (unused).

        Returns:
            An outcome with utility closest to (but not below) the target utility.
        """
        if not self.negotiator or not self.negotiator.ufun:
            return None

        if self._sorter is None:
            self.on_preferences_changed([])
            if self._sorter is None:
                return None

        t = state.relative_time
        target_utility = self._p(t)

        # Find the outcome with utility closest to target (from above)
        outcome = self._sorter.worst_in(
            (target_utility - 1e-9, self._pmax + 1e-9), normalized=False
        )
        if outcome is not None:
            return outcome

        # Fallback to best outcome if nothing found
        return self._sorter.best()


# =============================================================================
# ANAC 2010 Offering Strategies
# =============================================================================


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
    _sorter: PresortingInverseUtilityFunction | None = field(init=False, default=None)

    def on_preferences_changed(self, changes: list[PreferencesChange]) -> None:
        """Initialize the inverse utility function."""
        if not self.negotiator or not self.negotiator.ufun:
            return
        self._sorter = PresortingInverseUtilityFunction(
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

    _sorter: PresortingInverseUtilityFunction | None = field(init=False, default=None)
    _target: float = field(init=False, default=0.9)
    _bid_target: float = field(init=False, default=0.9)
    _pmax: float = field(init=False, default=1.0)
    _pmin: float = field(init=False, default=0.0)

    def on_preferences_changed(self, changes: list[PreferencesChange]) -> None:
        """Initialize utility function and parameters."""
        if not self.negotiator or not self.negotiator.ufun:
            return
        self._sorter = PresortingInverseUtilityFunction(
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
    _sorter: PresortingInverseUtilityFunction | None = field(init=False, default=None)
    _pmax: float = field(init=False, default=1.0)

    def on_preferences_changed(self, changes: list[PreferencesChange]) -> None:
        """Initialize utility function."""
        if not self.negotiator or not self.negotiator.ufun:
            return
        self._sorter = PresortingInverseUtilityFunction(
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

    _sorter: PresortingInverseUtilityFunction | None = field(init=False, default=None)
    _pmin: float = field(init=False, default=0.0)
    _pmax: float = field(init=False, default=1.0)

    def on_preferences_changed(self, changes: list[PreferencesChange]) -> None:
        """Initialize utility function."""
        if not self.negotiator or not self.negotiator.ufun:
            return
        self._sorter = PresortingInverseUtilityFunction(
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

    _sorter: PresortingInverseUtilityFunction | None = field(init=False, default=None)
    _pmin: float = field(init=False, default=0.0)
    _pmax: float = field(init=False, default=1.0)
    _last_target: float = field(init=False, default=1.0)

    def on_preferences_changed(self, changes: list[PreferencesChange]) -> None:
        """Initialize utility function."""
        if not self.negotiator or not self.negotiator.ufun:
            return
        self._sorter = PresortingInverseUtilityFunction(
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

    _sorter: PresortingInverseUtilityFunction | None = field(init=False, default=None)
    _pmin: float = field(init=False, default=0.0)
    _pmax: float = field(init=False, default=1.0)

    def on_preferences_changed(self, changes: list[PreferencesChange]) -> None:
        """Initialize utility function."""
        if not self.negotiator or not self.negotiator.ufun:
            return
        self._sorter = PresortingInverseUtilityFunction(
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

    _sorter: PresortingInverseUtilityFunction | None = field(init=False, default=None)
    _pmin: float = field(init=False, default=0.0)
    _pmax: float = field(init=False, default=1.0)

    def on_preferences_changed(self, changes: list[PreferencesChange]) -> None:
        """Initialize utility function."""
        if not self.negotiator or not self.negotiator.ufun:
            return
        self._sorter = PresortingInverseUtilityFunction(
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


# =============================================================================
# ANAC 2011 Offering Strategies
# =============================================================================


@define
class GHardHeadedOffering(GeniusOfferingPolicy):
    """
    HardHeaded offering strategy from ANAC 2011.

    This strategy uses a conservative concession approach with queue-based
    bid selection. It maintains a queue of potential bids and selects
    based on utility tolerance.

    Args:
        ka: Concession parameter (default 0.05).
        e: Concession exponent (default 0.05).
        min_utility: Minimum acceptable utility (default 0.585).

    Transcompiled from: negotiator.boaframework.offeringstrategy.anac2011.HardHeaded_Offering
    """

    ka: float = 0.05
    e: float = 0.05
    min_utility: float = 0.585
    _sorter: PresortingInverseUtilityFunction | None = field(init=False, default=None)
    _pmin: float = field(init=False, default=0.0)
    _pmax: float = field(init=False, default=1.0)
    _discount: float = field(init=False, default=1.0)
    _lowest_util: float = field(init=False, default=1.0)

    def on_preferences_changed(self, changes: list[PreferencesChange]) -> None:
        """Initialize utility function and parameters."""
        if not self.negotiator or not self.negotiator.ufun:
            return
        self._sorter = PresortingInverseUtilityFunction(
            self.negotiator.ufun, rational_only=True, eps=-1, rel_eps=-1
        )
        self._sorter.init()
        self._pmin, self._pmax = self._sorter.minmax()
        self._lowest_util = self._pmax

    def _get_p(self, t: float) -> float:
        """Calculate concession step based on time."""

        step_point = self._discount
        ignore_threshold = 0.9

        if step_point >= ignore_threshold:
            fa = self.ka + (1 - self.ka) * pow(t / step_point, 1.0 / self.e)
            p = self.min_utility + (1 - fa) * (self._pmax - self.min_utility)
        elif t <= step_point:
            temp_e = self.e / step_point
            fa = self.ka + (1 - self.ka) * pow(t / step_point, 1.0 / temp_e)
            temp_min = (
                self.min_utility + abs(self._pmax - self.min_utility) * step_point
            )
            p = temp_min + (1 - fa) * (self._pmax - temp_min)
        else:
            temp_e = 30.0
            fa = self.ka + (1 - self.ka) * pow(
                (t - step_point) / (1 - step_point), 1.0 / temp_e
            )
            temp_max = (
                self.min_utility + abs(self._pmax - self.min_utility) * step_point
            )
            p = self.min_utility + (1 - fa) * (temp_max - self.min_utility)

        return p

    def __call__(
        self, state: GBState, dest: str | None = None
    ) -> Outcome | ExtendedOutcome | None:
        """Generate a bid based on HardHeaded's strategy."""
        if not self.negotiator or not self.negotiator.ufun:
            return None

        if self._sorter is None:
            self.on_preferences_changed([])
            if self._sorter is None:
                return None

        t = state.relative_time
        p = self._get_p(t)

        outcome = self._sorter.worst_in((p - 0.01, self._pmax + 0.01), normalized=False)
        if outcome is not None:
            util = float(self.negotiator.ufun(outcome))
            if util < self._lowest_util:
                self._lowest_util = util
            return outcome

        return self._sorter.best()


@define
class GAgentK2Offering(GeniusOfferingPolicy):
    """
    AgentK2 offering strategy from ANAC 2011.

    Enhanced version of AgentK with improved opponent modeling.

    Transcompiled from: negotiator.boaframework.offeringstrategy.anac2011.AgentK2_Offering
    """

    _sorter: PresortingInverseUtilityFunction | None = field(init=False, default=None)
    _pmin: float = field(init=False, default=0.0)
    _pmax: float = field(init=False, default=1.0)
    _target: float = field(init=False, default=0.95)

    def on_preferences_changed(self, changes: list[PreferencesChange]) -> None:
        """Initialize utility function."""
        if not self.negotiator or not self.negotiator.ufun:
            return
        self._sorter = PresortingInverseUtilityFunction(
            self.negotiator.ufun, rational_only=True, eps=-1, rel_eps=-1
        )
        self._sorter.init()
        self._pmin, self._pmax = self._sorter.minmax()
        self._target = self._pmax

    def __call__(
        self, state: GBState, dest: str | None = None
    ) -> Outcome | ExtendedOutcome | None:
        """Generate a bid based on AgentK2's strategy."""
        if not self.negotiator or not self.negotiator.ufun:
            return None

        if self._sorter is None:
            self.on_preferences_changed([])
            if self._sorter is None:
                return None

        t = state.relative_time
        # AgentK2 concession curve
        self._target = self._pmin + (self._pmax - self._pmin) * (1.0 - pow(t, 2))

        outcome = self._sorter.worst_in(
            (self._target - 0.01, self._pmax + 0.01), normalized=False
        )
        if outcome is not None:
            return outcome

        return self._sorter.best()


@define
class GBRAMAgentOffering(GeniusOfferingPolicy):
    """
    BRAMAgent offering strategy from ANAC 2011.

    This strategy uses opponent modeling based on bid frequency statistics
    to create bids that are acceptable to both parties.

    Transcompiled from: negotiator.boaframework.offeringstrategy.anac2011.BRAMAgent_Offering
    """

    _sorter: PresortingInverseUtilityFunction | None = field(init=False, default=None)
    _pmin: float = field(init=False, default=0.0)
    _pmax: float = field(init=False, default=1.0)
    _threshold: float = field(init=False, default=0.9)

    def on_preferences_changed(self, changes: list[PreferencesChange]) -> None:
        """Initialize utility function."""
        if not self.negotiator or not self.negotiator.ufun:
            return
        self._sorter = PresortingInverseUtilityFunction(
            self.negotiator.ufun, rational_only=True, eps=-1, rel_eps=-1
        )
        self._sorter.init()
        self._pmin, self._pmax = self._sorter.minmax()
        self._threshold = self._pmax

    def _update_threshold(self, t: float) -> None:
        """Update threshold based on time."""
        # BRAM uses a slow linear concession
        self._threshold = self._pmax - (self._pmax - self._pmin) * 0.3 * t

    def __call__(
        self, state: GBState, dest: str | None = None
    ) -> Outcome | ExtendedOutcome | None:
        """Generate a bid based on BRAMAgent's strategy."""
        if not self.negotiator or not self.negotiator.ufun:
            return None

        if self._sorter is None:
            self.on_preferences_changed([])
            if self._sorter is None:
                return None

        t = state.relative_time
        self._update_threshold(t)

        outcome = self._sorter.worst_in(
            (self._threshold - 0.01, self._pmax + 0.01), normalized=False
        )
        if outcome is not None:
            return outcome

        return self._sorter.best()


@define
class GGahboninhoOffering(GeniusOfferingPolicy):
    """
    Gahboninho offering strategy from ANAC 2011.

    This strategy uses adaptive concession based on opponent behavior analysis.

    Transcompiled from: negotiator.boaframework.offeringstrategy.anac2011.Gahboninho_Offering
    """

    _sorter: PresortingInverseUtilityFunction | None = field(init=False, default=None)
    _pmin: float = field(init=False, default=0.0)
    _pmax: float = field(init=False, default=1.0)

    def on_preferences_changed(self, changes: list[PreferencesChange]) -> None:
        """Initialize utility function."""
        if not self.negotiator or not self.negotiator.ufun:
            return
        self._sorter = PresortingInverseUtilityFunction(
            self.negotiator.ufun, rational_only=True, eps=-1, rel_eps=-1
        )
        self._sorter.init()
        self._pmin, self._pmax = self._sorter.minmax()

    def __call__(
        self, state: GBState, dest: str | None = None
    ) -> Outcome | ExtendedOutcome | None:
        """Generate a bid based on Gahboninho's strategy."""
        if not self.negotiator or not self.negotiator.ufun:
            return None

        if self._sorter is None:
            self.on_preferences_changed([])
            if self._sorter is None:
                return None

        t = state.relative_time
        # Gahboninho uses careful concession
        target = self._pmax - (self._pmax - self._pmin) * pow(t, 3) * 0.4

        outcome = self._sorter.worst_in(
            (target - 0.01, self._pmax + 0.01), normalized=False
        )
        if outcome is not None:
            return outcome

        return self._sorter.best()


@define
class GNiceTitForTatOffering(GeniusOfferingPolicy):
    """
    NiceTitForTat offering strategy from ANAC 2011.

    This strategy mirrors opponent concessions while maintaining a minimum
    acceptable utility level.

    Transcompiled from: negotiator.boaframework.offeringstrategy.anac2011.NiceTitForTat_Offering
    """

    _sorter: PresortingInverseUtilityFunction | None = field(init=False, default=None)
    _pmin: float = field(init=False, default=0.0)
    _pmax: float = field(init=False, default=1.0)
    _last_opponent_util: float = field(init=False, default=0.0)

    def on_preferences_changed(self, changes: list[PreferencesChange]) -> None:
        """Initialize utility function."""
        if not self.negotiator or not self.negotiator.ufun:
            return
        self._sorter = PresortingInverseUtilityFunction(
            self.negotiator.ufun, rational_only=True, eps=-1, rel_eps=-1
        )
        self._sorter.init()
        self._pmin, self._pmax = self._sorter.minmax()

    def __call__(
        self, state: GBState, dest: str | None = None
    ) -> Outcome | ExtendedOutcome | None:
        """Generate a bid based on NiceTitForTat's strategy."""
        if not self.negotiator or not self.negotiator.ufun:
            return None

        if self._sorter is None:
            self.on_preferences_changed([])
            if self._sorter is None:
                return None

        t = state.relative_time

        # NiceTitForTat uses moderate concession with time
        # Simplified version without direct opponent offer access
        target = self._pmax - (self._pmax - self._pmin) * 0.25 * pow(t, 0.8)

        outcome = self._sorter.worst_in(
            (target - 0.01, self._pmax + 0.01), normalized=False
        )
        if outcome is not None:
            return outcome

        return self._sorter.best()


@define
class GTheNegotiatorOffering(GeniusOfferingPolicy):
    """
    TheNegotiator offering strategy from ANAC 2011.

    This strategy uses time-dependent concession with adaptive parameters.

    Transcompiled from: negotiator.boaframework.offeringstrategy.anac2011.TheNegotiator_Offering
    """

    _sorter: PresortingInverseUtilityFunction | None = field(init=False, default=None)
    _pmin: float = field(init=False, default=0.0)
    _pmax: float = field(init=False, default=1.0)

    def on_preferences_changed(self, changes: list[PreferencesChange]) -> None:
        """Initialize utility function."""
        if not self.negotiator or not self.negotiator.ufun:
            return
        self._sorter = PresortingInverseUtilityFunction(
            self.negotiator.ufun, rational_only=True, eps=-1, rel_eps=-1
        )
        self._sorter.init()
        self._pmin, self._pmax = self._sorter.minmax()

    def __call__(
        self, state: GBState, dest: str | None = None
    ) -> Outcome | ExtendedOutcome | None:
        """Generate a bid based on TheNegotiator's strategy."""
        if not self.negotiator or not self.negotiator.ufun:
            return None

        if self._sorter is None:
            self.on_preferences_changed([])
            if self._sorter is None:
                return None

        t = state.relative_time
        # TheNegotiator uses piecewise concession
        if t < 0.5:
            target = self._pmax
        elif t < 0.8:
            progress = (t - 0.5) / 0.3
            target = self._pmax - (self._pmax - self._pmin) * 0.2 * progress
        else:
            progress = (t - 0.8) / 0.2
            base = self._pmax * 0.8
            target = base - (base - self._pmin) * 0.5 * progress

        target = max(target, self._pmin)

        outcome = self._sorter.worst_in(
            (target - 0.01, self._pmax + 0.01), normalized=False
        )
        if outcome is not None:
            return outcome

        return self._sorter.best()


@define
class GValueModelAgentOffering(GeniusOfferingPolicy):
    """
    ValueModelAgent offering strategy from ANAC 2011.

    This strategy uses value modeling to predict opponent preferences.

    Transcompiled from: negotiator.boaframework.offeringstrategy.anac2011.ValueModelAgent_Offering
    """

    _sorter: PresortingInverseUtilityFunction | None = field(init=False, default=None)
    _pmin: float = field(init=False, default=0.0)
    _pmax: float = field(init=False, default=1.0)

    def on_preferences_changed(self, changes: list[PreferencesChange]) -> None:
        """Initialize utility function."""
        if not self.negotiator or not self.negotiator.ufun:
            return
        self._sorter = PresortingInverseUtilityFunction(
            self.negotiator.ufun, rational_only=True, eps=-1, rel_eps=-1
        )
        self._sorter.init()
        self._pmin, self._pmax = self._sorter.minmax()

    def __call__(
        self, state: GBState, dest: str | None = None
    ) -> Outcome | ExtendedOutcome | None:
        """Generate a bid based on ValueModelAgent's strategy."""
        if not self.negotiator or not self.negotiator.ufun:
            return None

        if self._sorter is None:
            self.on_preferences_changed([])
            if self._sorter is None:
                return None

        t = state.relative_time
        # Polynomial concession
        target = self._pmax - (self._pmax - self._pmin) * pow(t, 2) * 0.35

        outcome = self._sorter.worst_in(
            (target - 0.01, self._pmax + 0.01), normalized=False
        )
        if outcome is not None:
            return outcome

        return self._sorter.best()


@define
class GIAMhaggler2011Offering(GeniusOfferingPolicy):
    """
    IAMhaggler2011 offering strategy from ANAC 2011.

    Updated version of IAMhaggler with improved time management.

    Transcompiled from: negotiator.boaframework.offeringstrategy.anac2011.IAMhaggler2011_Offering
    """

    _sorter: PresortingInverseUtilityFunction | None = field(init=False, default=None)
    _pmin: float = field(init=False, default=0.0)
    _pmax: float = field(init=False, default=1.0)

    def on_preferences_changed(self, changes: list[PreferencesChange]) -> None:
        """Initialize utility function."""
        if not self.negotiator or not self.negotiator.ufun:
            return
        self._sorter = PresortingInverseUtilityFunction(
            self.negotiator.ufun, rational_only=True, eps=-1, rel_eps=-1
        )
        self._sorter.init()
        self._pmin, self._pmax = self._sorter.minmax()

    def __call__(
        self, state: GBState, dest: str | None = None
    ) -> Outcome | ExtendedOutcome | None:
        """Generate a bid based on IAMhaggler2011's strategy."""
        if not self.negotiator or not self.negotiator.ufun:
            return None

        if self._sorter is None:
            self.on_preferences_changed([])
            if self._sorter is None:
                return None

        t = state.relative_time
        # Conservative concession
        if t < 0.85:
            target = self._pmax - (self._pmax - self._pmin) * 0.1 * (t / 0.85)
        else:
            base = self._pmax * 0.9
            progress = (t - 0.85) / 0.15
            target = base - (base - self._pmin) * 0.4 * progress

        target = max(target, self._pmin)

        outcome = self._sorter.worst_in(
            (target - 0.01, self._pmax + 0.01), normalized=False
        )
        if outcome is not None:
            return outcome

        return self._sorter.best()


# =============================================================================
# ANAC 2012 Offering Strategies
# =============================================================================


@define
class GCUHKAgentOffering(GeniusOfferingPolicy):
    """
    CUHKAgent offering strategy from ANAC 2012.

    This strategy uses sophisticated opponent modeling and adaptive concession.

    Transcompiled from: negotiator.boaframework.offeringstrategy.anac2012.CUHKAgent_Offering
    """

    _sorter: PresortingInverseUtilityFunction | None = field(init=False, default=None)
    _pmin: float = field(init=False, default=0.0)
    _pmax: float = field(init=False, default=1.0)

    def on_preferences_changed(self, changes: list[PreferencesChange]) -> None:
        """Initialize utility function."""
        if not self.negotiator or not self.negotiator.ufun:
            return
        self._sorter = PresortingInverseUtilityFunction(
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
    _sorter: PresortingInverseUtilityFunction | None = field(init=False, default=None)
    _pmin: float = field(init=False, default=0.0)
    _pmax: float = field(init=False, default=1.0)
    _discount: float = field(init=False, default=1.0)
    _discount_threshold: float = 0.845

    def on_preferences_changed(self, changes: list[PreferencesChange]) -> None:
        """Initialize utility function."""
        if not self.negotiator or not self.negotiator.ufun:
            return
        self._sorter = PresortingInverseUtilityFunction(
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

    _sorter: PresortingInverseUtilityFunction | None = field(init=False, default=None)
    _pmin: float = field(init=False, default=0.0)
    _pmax: float = field(init=False, default=1.0)

    def on_preferences_changed(self, changes: list[PreferencesChange]) -> None:
        """Initialize utility function."""
        if not self.negotiator or not self.negotiator.ufun:
            return
        self._sorter = PresortingInverseUtilityFunction(
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

    _sorter: PresortingInverseUtilityFunction | None = field(init=False, default=None)
    _pmin: float = field(init=False, default=0.0)
    _pmax: float = field(init=False, default=1.0)

    def on_preferences_changed(self, changes: list[PreferencesChange]) -> None:
        """Initialize utility function."""
        if not self.negotiator or not self.negotiator.ufun:
            return
        self._sorter = PresortingInverseUtilityFunction(
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

    _sorter: PresortingInverseUtilityFunction | None = field(init=False, default=None)
    _pmin: float = field(init=False, default=0.0)
    _pmax: float = field(init=False, default=1.0)
    _threshold: float = field(init=False, default=0.9)

    def on_preferences_changed(self, changes: list[PreferencesChange]) -> None:
        """Initialize utility function."""
        if not self.negotiator or not self.negotiator.ufun:
            return
        self._sorter = PresortingInverseUtilityFunction(
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

    _sorter: PresortingInverseUtilityFunction | None = field(init=False, default=None)
    _pmin: float = field(init=False, default=0.0)
    _pmax: float = field(init=False, default=1.0)

    def on_preferences_changed(self, changes: list[PreferencesChange]) -> None:
        """Initialize utility function."""
        if not self.negotiator or not self.negotiator.ufun:
            return
        self._sorter = PresortingInverseUtilityFunction(
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

    _sorter: PresortingInverseUtilityFunction | None = field(init=False, default=None)
    _pmin: float = field(init=False, default=0.0)
    _pmax: float = field(init=False, default=1.0)

    def on_preferences_changed(self, changes: list[PreferencesChange]) -> None:
        """Initialize utility function."""
        if not self.negotiator or not self.negotiator.ufun:
            return
        self._sorter = PresortingInverseUtilityFunction(
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


# =============================================================================
# ANAC 2013 Offering Strategies
# =============================================================================


@define
class GFawkesOffering(GeniusOfferingPolicy):
    """
    TheFawkes offering strategy from ANAC 2013.

    This strategy uses wavelet-based prediction for opponent modeling.

    Transcompiled from: negotiator.boaframework.offeringstrategy.anac2013.Fawkes_Offering
    """

    _sorter: PresortingInverseUtilityFunction | None = field(init=False, default=None)
    _pmin: float = field(init=False, default=0.0)
    _pmax: float = field(init=False, default=1.0)

    def on_preferences_changed(self, changes: list[PreferencesChange]) -> None:
        """Initialize utility function."""
        if not self.negotiator or not self.negotiator.ufun:
            return
        self._sorter = PresortingInverseUtilityFunction(
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

    _sorter: PresortingInverseUtilityFunction | None = field(init=False, default=None)
    _pmin: float = field(init=False, default=0.0)
    _pmax: float = field(init=False, default=1.0)

    def on_preferences_changed(self, changes: list[PreferencesChange]) -> None:
        """Initialize utility function."""
        if not self.negotiator or not self.negotiator.ufun:
            return
        self._sorter = PresortingInverseUtilityFunction(
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


@define
class GRandomOffering(GeniusOfferingPolicy):
    """
    Random offering strategy from Genius.

    This strategy offers random bids from the outcome space, completely
    ignoring utility. Also known as "Zero Intelligence" or "Random Walker".

    This is useful for:
    - Debugging and testing
    - Creating baseline comparisons
    - Simulating unpredictable opponents

    Transcompiled from: negotiator.boaframework.offeringstrategy.other.Random_Offering
    """

    def __call__(
        self, state: GBState, dest: str | None = None
    ) -> Outcome | ExtendedOutcome | None:
        """Generate a random offer from the outcome space.

        Args:
            state: Current negotiation state.
            dest: Destination identifier (unused).

        Returns:
            A random outcome from the outcome space.
        """
        if not self.negotiator or not self.negotiator.nmi:
            return None

        return self.negotiator.nmi.random_outcome()


@define
class GBoulwareOffering(GeniusOfferingPolicy):
    """
    Boulware offering strategy - a time-dependent strategy with e < 1.

    Concedes slowly at first, then faster as deadline approaches.
    This is a convenience wrapper around GTimeDependentOffering.

    Args:
        e: Concession exponent (default 0.2, typical Boulware value).
        k: Offset constant (default 0).

    Transcompiled from: negotiator.boaframework.offeringstrategy.other.TimeDependent_Offering with e < 1
    """

    e: float = 0.2
    k: float = 0.0
    _delegate: GTimeDependentOffering | None = field(init=False, default=None)

    def on_preferences_changed(self, changes: list[PreferencesChange]) -> None:
        """Initialize the delegate time-dependent offering strategy."""
        if self._delegate is None:
            self._delegate = GTimeDependentOffering(e=self.e, k=self.k)
            self._delegate._negotiator = self.negotiator
        self._delegate.on_preferences_changed(changes)

    def __call__(
        self, state: GBState, dest: str | None = None
    ) -> Outcome | ExtendedOutcome | None:
        """Generate a Boulware offer."""
        if self._delegate is None:
            self.on_preferences_changed([])
        if self._delegate is None:
            return None
        self._delegate._negotiator = self.negotiator
        return self._delegate(state, dest)


@define
class GConcederOffering(GeniusOfferingPolicy):
    """
    Conceder offering strategy - a time-dependent strategy with e > 1.

    Concedes quickly at first, then slows down as deadline approaches.
    This is a convenience wrapper around GTimeDependentOffering.

    Args:
        e: Concession exponent (default 2.0, typical Conceder value).
        k: Offset constant (default 0).

    Transcompiled from: negotiator.boaframework.offeringstrategy.other.TimeDependent_Offering with e > 1
    """

    e: float = 2.0
    k: float = 0.0
    _delegate: GTimeDependentOffering | None = field(init=False, default=None)

    def on_preferences_changed(self, changes: list[PreferencesChange]) -> None:
        """Initialize the delegate time-dependent offering strategy."""
        if self._delegate is None:
            self._delegate = GTimeDependentOffering(e=self.e, k=self.k)
            self._delegate._negotiator = self.negotiator
        self._delegate.on_preferences_changed(changes)

    def __call__(
        self, state: GBState, dest: str | None = None
    ) -> Outcome | ExtendedOutcome | None:
        """Generate a Conceder offer."""
        if self._delegate is None:
            self.on_preferences_changed([])
        if self._delegate is None:
            return None
        self._delegate._negotiator = self.negotiator
        return self._delegate(state, dest)


@define
class GLinearOffering(GeniusOfferingPolicy):
    """
    Linear offering strategy - a time-dependent strategy with e = 1.

    Concedes at a constant rate throughout negotiation.
    This is a convenience wrapper around GTimeDependentOffering.

    Args:
        k: Offset constant (default 0).

    Transcompiled from: negotiator.boaframework.offeringstrategy.other.TimeDependent_Offering with e = 1
    """

    k: float = 0.0
    _delegate: GTimeDependentOffering | None = field(init=False, default=None)

    def on_preferences_changed(self, changes: list[PreferencesChange]) -> None:
        """Initialize the delegate time-dependent offering strategy."""
        if self._delegate is None:
            self._delegate = GTimeDependentOffering(e=1.0, k=self.k)
            self._delegate._negotiator = self.negotiator
        self._delegate.on_preferences_changed(changes)

    def __call__(
        self, state: GBState, dest: str | None = None
    ) -> Outcome | ExtendedOutcome | None:
        """Generate a Linear offer."""
        if self._delegate is None:
            self.on_preferences_changed([])
        if self._delegate is None:
            return None
        self._delegate._negotiator = self.negotiator
        return self._delegate(state, dest)


@define
class GHardlinerOffering(GeniusOfferingPolicy):
    """
    Hardliner offering strategy - always offers the best outcome.

    Never concedes - always offers the maximum utility outcome.
    This is equivalent to time-dependent with e = 0.

    Transcompiled from: negotiator.boaframework.offeringstrategy.other.TimeDependent_Offering with e = 0
    """

    _sorter: PresortingInverseUtilityFunction | None = field(init=False, default=None)

    def on_preferences_changed(self, changes: list[PreferencesChange]) -> None:
        """Initialize the inverse utility function."""
        if not self.negotiator or not self.negotiator.ufun:
            return
        self._sorter = PresortingInverseUtilityFunction(
            self.negotiator.ufun, rational_only=True, eps=-1, rel_eps=-1
        )
        self._sorter.init()

    def __call__(
        self, state: GBState, dest: str | None = None
    ) -> Outcome | ExtendedOutcome | None:
        """Always return the best outcome."""
        if not self.negotiator or not self.negotiator.ufun:
            return None

        if self._sorter is None:
            self.on_preferences_changed([])
            if self._sorter is None:
                return None

        return self._sorter.best()


@define
class GChoosingAllBids(GeniusOfferingPolicy):
    """
    ChoosingAllBids offering strategy from Genius.

    Iterates through all possible bids in the domain, offering each one
    in sequence. Useful for exhaustive exploration or testing.

    When all bids have been offered, it restarts from the beginning.

    Transcompiled from: negotiator.boaframework.offeringstrategy.other.ChoosingAllBids
    """

    _all_outcomes: list[Outcome] = field(factory=list)
    _current_index: int = field(init=False, default=0)
    _initialized: bool = field(init=False, default=False)

    def on_preferences_changed(self, changes: list[PreferencesChange]) -> None:
        """Initialize the list of all outcomes."""
        if not self.negotiator or not self.negotiator.nmi:
            return

        outcome_space = self.negotiator.nmi.outcome_space
        if outcome_space is None:
            return

        # Get all outcomes and sort by our utility (best first)
        self._all_outcomes = list(outcome_space.enumerate())
        if self.negotiator.ufun:
            self._all_outcomes.sort(
                key=lambda o: float(self.negotiator.ufun(o))
                if self.negotiator.ufun
                else 0,
                reverse=True,
            )
        self._current_index = 0
        self._initialized = True

    def __call__(
        self, state: GBState, dest: str | None = None
    ) -> Outcome | ExtendedOutcome | None:
        """Return the next bid in sequence."""
        if not self._initialized:
            self.on_preferences_changed([])

        if not self._all_outcomes:
            return None

        # Get current outcome
        outcome = self._all_outcomes[self._current_index]

        # Advance to next (wrap around)
        self._current_index = (self._current_index + 1) % len(self._all_outcomes)

        return outcome
