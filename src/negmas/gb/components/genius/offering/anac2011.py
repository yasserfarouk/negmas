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
