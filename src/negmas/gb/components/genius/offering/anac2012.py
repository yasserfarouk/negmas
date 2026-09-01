"""offering components (split from offering.py): anac2012."""

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

    phase1_end: float = 0.2
    """Relative time until which the maximum is offered unchanged."""
    phase2_end: float = 0.7
    """Relative time ending the moderate-concession phase."""
    phase2_drop: float = 0.15
    """Fraction of the utility range conceded during phase 2."""
    phase3_base_factor: float = 0.85
    """Fraction of the maximum utility phase 3 starts from."""
    phase3_drop: float = 0.4
    """Fraction of the remaining range conceded during phase 3."""
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
        """Generate a bid based on CUHKAgent's strategy."""
        if not self.negotiator or not self.negotiator.ufun:
            return None

        if self._sorter is None:
            self.on_preferences_changed([])
            if self._sorter is None:
                return None

        t = state.relative_time
        # CUHK uses adaptive piecewise concession
        if t < self.phase1_end:
            target = self._pmax
        elif t < self.phase2_end:
            progress = (t - self.phase1_end) / (self.phase2_end - self.phase1_end)
            target = (
                self._pmax - (self._pmax - self._pmin) * self.phase2_drop * progress
            )
        else:
            progress = (t - self.phase2_end) / (1.0 - self.phase2_end)
            base = self._pmax * self.phase3_base_factor
            target = base - (base - self._pmin) * self.phase3_drop * progress

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
class GOMACagentOffering(GeniusOfferingPolicy):
    """
    OMACagent offering strategy from ANAC 2012.

    This strategy uses prediction-based bidding with exponential moving average.

    Args:
        min_utility: Minimum utility threshold (default 0.59).
        eu: Expected utility threshold (default 0.95).
        discount_threshold: Discount below which the low-discount target curve is
            used instead of the normal one (default 0.845).
        e_high_discount: Concession exponent parameter used when the discount is
            at or above ``discount_threshold`` (default 0.033).
        e_low_discount: Concession exponent parameter used below that threshold
            (default 0.04).
        discount_power: Exponent applied to the discount when deriving the upper
            target bound in the low-discount branch (default 0.2).
        min_utility_margin: Multiplier applied to ``min_utility`` for the lower
            target bound in the low-discount branch (default 1.05).
        opening_time: Relative time before which the best outcome is offered
            unconditionally (default 0.02).
        narrow_band: Fractional half-width of the first (narrow) utility band
            searched around the target, i.e. ``[target * (1 - narrow_band),
            target * (1 + narrow_band)]`` (default 0.01).
        wide_band_tolerance: Absolute slack used for the fallback (wider) band
            search when the narrow band is empty (default 0.05).

    Transcompiled from: negotiator.boaframework.offeringstrategy.anac2012.OMACagent_Offering
    """

    min_utility: float = 0.59
    eu: float = 0.95
    discount_threshold: float = 0.845
    e_high_discount: float = 0.033
    e_low_discount: float = 0.04
    discount_power: float = 0.2
    min_utility_margin: float = 1.05
    opening_time: float = 0.02
    narrow_band: float = 0.01
    wide_band_tolerance: float = 0.05
    _sorter: InverseUFun | None = field(init=False, default=None)
    _pmin: float = field(init=False, default=0.0)
    _pmax: float = field(init=False, default=1.0)
    _discount: float = field(init=False, default=1.0)

    def on_preferences_changed(self, changes: list[PreferencesChange]) -> None:
        """Initialize utility function."""
        if not self.negotiator or not self.negotiator.ufun:
            return
        self._sorter = self.negotiator.ufun.invert(
            rational_only=True, eps=-1, rel_eps=-1
        )
        self._pmin, self._pmax = self._sorter.minmax()

    def _get_target(self, t: float) -> float:
        """Calculate target utility."""

        e1, e2 = self.e_high_discount, self.e_low_discount

        if self._discount >= self.discount_threshold:
            target = self.min_utility + (1 - pow(t, 1.0 / e1)) * (
                self._pmax - self.min_utility
            )
        else:
            t_max = pow(self._discount, self.discount_power)
            t_min = self.min_utility * self.min_utility_margin
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

        if t <= self.opening_time:
            return self._sorter.best()

        target = self._get_target(t)
        target = max(target, self.min_utility)

        outcome = self._sorter.worst_in(
            (target * (1 - self.narrow_band), target * (1 + self.narrow_band)),
            normalized=False,
        )
        if outcome is not None:
            return outcome

        return self._sorter.worst_in(
            (
                target - self.wide_band_tolerance,
                self._pmax + self.utility_band_tolerance,
            ),
            normalized=False,
        )


@define
class GAgentLGOffering(GeniusOfferingPolicy):
    """
    AgentLG offering strategy from ANAC 2012.

    This strategy uses learning-based concession.

    Transcompiled from: negotiator.boaframework.offeringstrategy.anac2012.AgentLG_Offering
    """

    concession_exponent: float = 1.5
    """Exponent applied to relative time (larger concedes later)."""
    max_concession: float = 0.4
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
        """Generate a bid based on AgentLG's strategy."""
        if not self.negotiator or not self.negotiator.ufun:
            return None

        if self._sorter is None:
            self.on_preferences_changed([])
            if self._sorter is None:
                return None

        t = state.relative_time
        # Learning-based concession curve
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


@define
class GAgentMROffering(GeniusOfferingPolicy):
    """
    AgentMR offering strategy from ANAC 2012.

    This strategy uses risk-based concession.

    Transcompiled from: negotiator.boaframework.offeringstrategy.anac2012.AgentMR_Offering
    """

    base_risk: float = 0.3
    """Risk factor at ``t=0`` in the risk-aware concession rate."""
    risk_growth: float = 0.2
    """How much the risk factor grows over the negotiation."""
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
        """Generate a bid based on AgentMR's strategy."""
        if not self.negotiator or not self.negotiator.ufun:
            return None

        if self._sorter is None:
            self.on_preferences_changed([])
            if self._sorter is None:
                return None

        t = state.relative_time
        # MR uses risk-aware concession
        risk_factor = self.base_risk + self.risk_growth * t
        target = self._pmax - (self._pmax - self._pmin) * risk_factor * t

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
class GBRAMAgent2Offering(GeniusOfferingPolicy):
    """
    BRAMAgent2 offering strategy from ANAC 2012.

    Enhanced version of BRAMAgent with improved statistics tracking.

    Transcompiled from: negotiator.boaframework.offeringstrategy.anac2012.BRAMAgent2_Offering
    """

    concession_exponent: float = 0.7
    """Exponent applied to relative time (larger concedes later)."""
    max_concession: float = 0.35
    """Fraction of the utility range given away by the deadline."""
    _sorter: InverseUFun | None = field(init=False, default=None)
    _pmin: float = field(init=False, default=0.0)
    _pmax: float = field(init=False, default=1.0)
    _threshold: float = field(init=False, default=0.9)

    def on_preferences_changed(self, changes: list[PreferencesChange]) -> None:
        """Initialize utility function."""
        if not self.negotiator or not self.negotiator.ufun:
            return
        self._sorter = self.negotiator.ufun.invert(
            rational_only=True, eps=-1, rel_eps=-1
        )
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
        self._threshold = self._pmax - (
            self._pmax - self._pmin
        ) * self.max_concession * pow(t, self.concession_exponent)

        outcome = self._sorter.worst_in(
            (
                self._threshold - self.utility_band_tolerance,
                self._pmax + self.utility_band_tolerance,
            ),
            normalized=False,
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

    phase_end: float = 0.8
    """Relative time separating the early phase from the late one."""
    early_drop: float = 0.08
    """Fraction of the utility range conceded during the early phase."""
    late_base_factor: float = 0.92
    """Fraction of the maximum utility the late phase starts from."""
    late_drop: float = 0.45
    """Fraction of the remaining range conceded during the late phase."""
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
        """Generate a bid based on IAMHaggler2012's strategy."""
        if not self.negotiator or not self.negotiator.ufun:
            return None

        if self._sorter is None:
            self.on_preferences_changed([])
            if self._sorter is None:
                return None

        t = state.relative_time
        # Refined concession for 2012
        if t < self.phase_end:
            target = self._pmax - (self._pmax - self._pmin) * self.early_drop * (
                t / self.phase_end
            )
        else:
            base = self._pmax * self.late_base_factor
            progress = (t - self.phase_end) / (1.0 - self.phase_end)
            target = base - (base - self._pmin) * self.late_drop * progress

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
class GTheNegotiatorReloadedOffering(GeniusOfferingPolicy):
    """
    TheNegotiatorReloaded offering strategy from ANAC 2012.

    Enhanced version of TheNegotiator with improved time management.

    Transcompiled from: negotiator.boaframework.offeringstrategy.anac2012.TheNegotiatorReloaded_Offering
    """

    phase1_end: float = 0.4
    """Relative time until which the maximum is offered unchanged."""
    phase2_end: float = 0.75
    """Relative time ending the moderate-concession phase."""
    phase2_drop: float = 0.15
    """Fraction of the utility range conceded during phase 2."""
    phase3_base_factor: float = 0.85
    """Fraction of the maximum utility phase 3 starts from."""
    phase3_drop: float = 0.45
    """Fraction of the remaining range conceded during phase 3."""
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
        """Generate a bid based on TheNegotiatorReloaded's strategy."""
        if not self.negotiator or not self.negotiator.ufun:
            return None

        if self._sorter is None:
            self.on_preferences_changed([])
            if self._sorter is None:
                return None

        t = state.relative_time
        # Reloaded uses smoother transitions
        if t < self.phase1_end:
            target = self._pmax
        elif t < self.phase2_end:
            progress = (t - self.phase1_end) / (self.phase2_end - self.phase1_end)
            target = (
                self._pmax - (self._pmax - self._pmin) * self.phase2_drop * progress
            )
        else:
            progress = (t - self.phase2_end) / (1.0 - self.phase2_end)
            base = self._pmax * self.phase3_base_factor
            target = base - (base - self._pmin) * self.phase3_drop * progress

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
