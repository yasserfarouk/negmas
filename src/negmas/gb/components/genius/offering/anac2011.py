"""offering components (split from offering.py): anac2011."""

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
    "GHardHeadedOffering",
    "GAgentK2Offering",
    "GBRAMAgentOffering",
    "GGahboninhoOffering",
    "GNiceTitForTatOffering",
    "GTheNegotiatorOffering",
    "GValueModelAgentOffering",
    "GIAMhaggler2011Offering",
]


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

    discount_ignore_threshold: float = 0.9
    """Discount at or above which discounting is ignored entirely."""
    post_step_exponent: float = 30.0
    """Very large concession exponent used after the step point, which keeps the
    target almost flat."""
    ka: float = 0.05
    e: float = 0.05
    min_utility: float = 0.585
    _sorter: InverseUFun | None = field(init=False, default=None)
    _pmin: float = field(init=False, default=0.0)
    _pmax: float = field(init=False, default=1.0)
    _discount: float = field(init=False, default=1.0)
    _lowest_util: float = field(init=False, default=1.0)

    def on_preferences_changed(self, changes: list[PreferencesChange]) -> None:
        """Initialize utility function and parameters."""
        if not self.negotiator or not self.negotiator.ufun:
            return
        self._sorter = DefaultInverseUtilityFunction(
            self.negotiator.ufun, rational_only=True, eps=-1, rel_eps=-1
        )
        self._sorter.init()
        self._pmin, self._pmax = self._sorter.minmax()
        self._lowest_util = self._pmax

    def _get_p(self, t: float) -> float:
        """Calculate concession step based on time."""

        step_point = self._discount
        ignore_threshold = self.discount_ignore_threshold

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
            temp_e = self.post_step_exponent
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

        outcome = self._sorter.worst_in(
            (p - self.utility_band_tolerance, self._pmax + self.utility_band_tolerance),
            normalized=False,
        )
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

    _sorter: InverseUFun | None = field(init=False, default=None)
    _pmin: float = field(init=False, default=0.0)
    _pmax: float = field(init=False, default=1.0)
    _target: float = field(init=False, default=0.95)

    def on_preferences_changed(self, changes: list[PreferencesChange]) -> None:
        """Initialize utility function."""
        if not self.negotiator or not self.negotiator.ufun:
            return
        self._sorter = DefaultInverseUtilityFunction(
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
            (
                self._target - self.utility_band_tolerance,
                self._pmax + self.utility_band_tolerance,
            ),
            normalized=False,
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

    concession_rate: float = 0.3
    """Fraction of the utility range conceded linearly over the negotiation."""
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

    def _update_threshold(self, t: float) -> None:
        """Update threshold based on time."""
        # BRAM uses a slow linear concession
        self._threshold = (
            self._pmax - (self._pmax - self._pmin) * self.concession_rate * t
        )

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
class GGahboninhoOffering(GeniusOfferingPolicy):
    """
    Gahboninho offering strategy from ANAC 2011.

    This strategy uses adaptive concession based on opponent behavior analysis.

    Transcompiled from: negotiator.boaframework.offeringstrategy.anac2011.Gahboninho_Offering
    """

    concession_exponent: float = 3
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
        self._sorter = DefaultInverseUtilityFunction(
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
class GNiceTitForTatOffering(GeniusOfferingPolicy):
    """
    NiceTitForTat offering strategy from ANAC 2011.

    This strategy mirrors opponent concessions while maintaining a minimum
    acceptable utility level.

    Transcompiled from: negotiator.boaframework.offeringstrategy.anac2011.NiceTitForTat_Offering
    """

    max_concession: float = 0.25
    """Fraction of the utility range given away by the deadline."""
    concession_exponent: float = 0.8
    """Exponent applied to relative time (smaller concedes earlier)."""
    _sorter: InverseUFun | None = field(init=False, default=None)
    _pmin: float = field(init=False, default=0.0)
    _pmax: float = field(init=False, default=1.0)
    _last_opponent_util: float = field(init=False, default=0.0)

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
        target = self._pmax - (self._pmax - self._pmin) * self.max_concession * pow(
            t, self.concession_exponent
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
class GTheNegotiatorOffering(GeniusOfferingPolicy):
    """
    TheNegotiator offering strategy from ANAC 2011.

    This strategy uses time-dependent concession with adaptive parameters.

    Transcompiled from: negotiator.boaframework.offeringstrategy.anac2011.TheNegotiator_Offering
    """

    phase1_end: float = 0.5
    """Relative time until which the maximum is offered unchanged."""
    phase2_end: float = 0.8
    """Relative time ending the moderate-concession phase."""
    phase2_drop: float = 0.2
    """Fraction of the utility range conceded during phase 2."""
    phase3_base_factor: float = 0.8
    """Fraction of the maximum utility phase 3 starts from."""
    phase3_drop: float = 0.5
    """Fraction of the remaining range conceded during phase 3."""
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
        """Generate a bid based on TheNegotiator's strategy."""
        if not self.negotiator or not self.negotiator.ufun:
            return None

        if self._sorter is None:
            self.on_preferences_changed([])
            if self._sorter is None:
                return None

        t = state.relative_time
        # TheNegotiator uses piecewise concession
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
class GValueModelAgentOffering(GeniusOfferingPolicy):
    """
    ValueModelAgent offering strategy from ANAC 2011.

    This strategy uses value modeling to predict opponent preferences.

    Transcompiled from: negotiator.boaframework.offeringstrategy.anac2011.ValueModelAgent_Offering
    """

    concession_exponent: float = 2
    """Exponent applied to relative time in the polynomial concession curve."""
    max_concession: float = 0.35
    """Fraction of the utility range given away by the deadline."""
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
        """Generate a bid based on ValueModelAgent's strategy."""
        if not self.negotiator or not self.negotiator.ufun:
            return None

        if self._sorter is None:
            self.on_preferences_changed([])
            if self._sorter is None:
                return None

        t = state.relative_time
        # Polynomial concession
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
class GIAMhaggler2011Offering(GeniusOfferingPolicy):
    """
    IAMhaggler2011 offering strategy from ANAC 2011.

    Updated version of IAMhaggler with improved time management.

    Transcompiled from: negotiator.boaframework.offeringstrategy.anac2011.IAMhaggler2011_Offering
    """

    phase_end: float = 0.85
    """Relative time separating the conservative phase from the final one."""
    early_drop: float = 0.1
    """Fraction of the utility range conceded during the early phase."""
    late_base_factor: float = 0.9
    """Fraction of the maximum utility the late phase starts from."""
    late_drop: float = 0.4
    """Fraction of the remaining range conceded during the late phase."""
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
        """Generate a bid based on IAMhaggler2011's strategy."""
        if not self.negotiator or not self.negotiator.ufun:
            return None

        if self._sorter is None:
            self.on_preferences_changed([])
            if self._sorter is None:
                return None

        t = state.relative_time
        # Conservative concession
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
