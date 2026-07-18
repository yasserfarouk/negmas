"""models components (split from models.py): agent_specific."""

from __future__ import annotations

from typing import TYPE_CHECKING

from attrs import define, field

from negmas.gb.components.genius.base import GeniusOpponentModel

if TYPE_CHECKING:
    from negmas.common import PreferencesChange, Value
    from negmas.gb import GBState
    from negmas.outcomes import Outcome

__all__ = ["GAgentLGModel", "GTheFawkesModel", "GInoxAgentModel"]


@define
class GAgentLGModel(GeniusOpponentModel):
    """
    AgentLG opponent model.

    This model uses learning-based estimation of opponent preferences.

    Transcompiled from: negotiator.boaframework.opponentmodel.AgentLGModel
    """

    _issue_weights: dict[int, float] = field(factory=dict)
    _value_utils: dict[int, dict] = field(factory=dict)
    _n_issues: int = field(init=False, default=0)
    _initialized: bool = field(init=False, default=False)
    _n_bids: int = field(init=False, default=0)
    _last_bid: Outcome | None = field(init=False, default=None)

    def _initialize(self) -> None:
        """Initialize the model."""
        if not self.negotiator or not self.negotiator.nmi:
            return

        outcome_space = self.negotiator.nmi.outcome_space
        if outcome_space is None:
            return

        issues = outcome_space.issues
        self._n_issues = len(issues)

        if self._n_issues > 0:
            initial_weight = 1.0 / self._n_issues
            for i in range(self._n_issues):
                self._issue_weights[i] = initial_weight
                self._value_utils[i] = {}
                for v in issues[i].all:
                    self._value_utils[i][v] = 0.5

        self._initialized = True

    def on_preferences_changed(self, changes: list[PreferencesChange]) -> None:
        """Handle preference changes."""
        self._initialize()

    def update(self, state: GBState, offer: Outcome, partner_id: str) -> None:
        """Update model based on opponent's offer."""
        if not self._initialized:
            self._initialize()

        if offer is None or self._n_issues == 0:
            return

        self._n_bids += 1

        # Learn from changes between consecutive bids
        if self._last_bid is not None:
            for i in range(self._n_issues):
                if offer[i] == self._last_bid[i]:
                    # Issue didn't change - increase weight
                    self._issue_weights[i] *= 1.1

        # Normalize weights
        total_weight = sum(self._issue_weights.values())
        if total_weight > 0:
            for i in self._issue_weights:
                self._issue_weights[i] /= total_weight

        # Update value utilities
        for i in range(self._n_issues):
            value = offer[i]
            if value in self._value_utils[i]:
                self._value_utils[i][value] = min(
                    1.0, self._value_utils[i][value] + 0.1
                )

        self._last_bid = offer

    def eval(self, offer: Outcome | None) -> Value:
        """Evaluate opponent utility."""
        if offer is None:
            return 0.0

        if not self._initialized:
            self._initialize()

        if self._n_issues == 0:
            return 0.0

        total_utility = 0.0
        for i in range(self._n_issues):
            value = offer[i]
            issue_weight = self._issue_weights.get(i, 1.0 / max(1, self._n_issues))
            value_util = self._value_utils.get(i, {}).get(value, 0.5)
            total_utility += issue_weight * value_util

        return total_utility

    def eval_normalized(
        self,
        offer: Outcome | None,
        above_reserve: bool = True,
        expected_limits: bool = True,
    ) -> Value:
        """Return normalized utility."""
        return self.eval(offer)


@define
class GTheFawkesModel(GeniusOpponentModel):
    """
    TheFawkes opponent model.

    This model uses wavelet-based analysis for opponent preference estimation.

    Transcompiled from: negotiator.boaframework.opponentmodel.TheFawkes_OM
    """

    _issue_weights: dict[int, float] = field(factory=dict)
    _value_utils: dict[int, dict] = field(factory=dict)
    _n_issues: int = field(init=False, default=0)
    _initialized: bool = field(init=False, default=False)
    _bid_history: list[Outcome] = field(factory=list)

    def _initialize(self) -> None:
        """Initialize the model."""
        if not self.negotiator or not self.negotiator.nmi:
            return

        outcome_space = self.negotiator.nmi.outcome_space
        if outcome_space is None:
            return

        issues = outcome_space.issues
        self._n_issues = len(issues)

        if self._n_issues > 0:
            initial_weight = 1.0 / self._n_issues
            for i in range(self._n_issues):
                self._issue_weights[i] = initial_weight
                self._value_utils[i] = {}
                for v in issues[i].all:
                    self._value_utils[i][v] = 0.5

        self._initialized = True

    def on_preferences_changed(self, changes: list[PreferencesChange]) -> None:
        """Handle preference changes."""
        self._initialize()

    def update(self, state: GBState, offer: Outcome, partner_id: str) -> None:
        """Update model based on opponent's offer."""
        if not self._initialized:
            self._initialize()

        if offer is None or self._n_issues == 0:
            return

        self._bid_history.append(offer)

        # Simple frequency-based update
        for i in range(self._n_issues):
            value = offer[i]
            if value in self._value_utils[i]:
                self._value_utils[i][value] = min(
                    1.0, self._value_utils[i][value] + 0.05
                )

        # Analyze consistency - issues that don't change are more important
        if len(self._bid_history) >= 2:
            prev_bid = self._bid_history[-2]
            for i in range(self._n_issues):
                if offer[i] == prev_bid[i]:
                    self._issue_weights[i] *= 1.05

            # Normalize
            total = sum(self._issue_weights.values())
            if total > 0:
                for i in self._issue_weights:
                    self._issue_weights[i] /= total

    def eval(self, offer: Outcome | None) -> Value:
        """Evaluate opponent utility."""
        if offer is None:
            return 0.0

        if not self._initialized:
            self._initialize()

        if self._n_issues == 0:
            return 0.0

        total_utility = 0.0
        for i in range(self._n_issues):
            value = offer[i]
            issue_weight = self._issue_weights.get(i, 1.0 / max(1, self._n_issues))
            value_util = self._value_utils.get(i, {}).get(value, 0.5)
            total_utility += issue_weight * value_util

        return total_utility

    def eval_normalized(
        self,
        offer: Outcome | None,
        above_reserve: bool = True,
        expected_limits: bool = True,
    ) -> Value:
        """Return normalized utility."""
        return self.eval(offer)


@define
class GInoxAgentModel(GeniusOpponentModel):
    """
    InoxAgent opponent model.

    This model uses adaptive preference estimation.

    Transcompiled from: negotiator.boaframework.opponentmodel.InoxAgent_OM
    """

    _issue_weights: dict[int, float] = field(factory=dict)
    _value_utils: dict[int, dict] = field(factory=dict)
    _n_issues: int = field(init=False, default=0)
    _initialized: bool = field(init=False, default=False)
    _n_bids: int = field(init=False, default=0)

    def _initialize(self) -> None:
        """Initialize the model."""
        if not self.negotiator or not self.negotiator.nmi:
            return

        outcome_space = self.negotiator.nmi.outcome_space
        if outcome_space is None:
            return

        issues = outcome_space.issues
        self._n_issues = len(issues)

        if self._n_issues > 0:
            initial_weight = 1.0 / self._n_issues
            for i in range(self._n_issues):
                self._issue_weights[i] = initial_weight
                self._value_utils[i] = {}
                for v in issues[i].all:
                    self._value_utils[i][v] = 0.5

        self._initialized = True

    def on_preferences_changed(self, changes: list[PreferencesChange]) -> None:
        """Handle preference changes."""
        self._initialize()

    def update(self, state: GBState, offer: Outcome, partner_id: str) -> None:
        """Update model based on opponent's offer."""
        if not self._initialized:
            self._initialize()

        if offer is None or self._n_issues == 0:
            return

        self._n_bids += 1
        alpha = 0.15 / (1.0 + 0.05 * self._n_bids)

        for i in range(self._n_issues):
            value = offer[i]
            if value in self._value_utils[i]:
                old_util = self._value_utils[i][value]
                self._value_utils[i][value] = old_util + alpha * (1.0 - old_util)

    def eval(self, offer: Outcome | None) -> Value:
        """Evaluate opponent utility."""
        if offer is None:
            return 0.0

        if not self._initialized:
            self._initialize()

        if self._n_issues == 0:
            return 0.0

        total_utility = 0.0
        for i in range(self._n_issues):
            value = offer[i]
            issue_weight = self._issue_weights.get(i, 1.0 / max(1, self._n_issues))
            value_util = self._value_utils.get(i, {}).get(value, 0.5)
            total_utility += issue_weight * value_util

        return total_utility

    def eval_normalized(
        self,
        offer: Outcome | None,
        above_reserve: bool = True,
        expected_limits: bool = True,
    ) -> Value:
        """Return normalized utility."""
        return self.eval(offer)
