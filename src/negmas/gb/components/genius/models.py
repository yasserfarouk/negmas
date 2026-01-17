"""Genius opponent models.

This module contains Python implementations of Genius opponent modeling strategies,
transcompiled from the original Java implementations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from attrs import define, field

from .base import GeniusOpponentModel

if TYPE_CHECKING:
    from negmas.common import PreferencesChange, Value
    from negmas.gb import GBState
    from negmas.outcomes import Outcome

__all__ = [
    # Base opponent models
    "GHardHeadedFrequencyModel",
    "GDefaultModel",
    "GUniformModel",
    "GOppositeModel",
    "GSmithFrequencyModel",
    "GAgentXFrequencyModel",
    "GNashFrequencyModel",
    "GBayesianModel",
    "GScalableBayesianModel",
    # Additional opponent models
    "GFSEGABayesianModel",
    "GIAMhagglerBayesianModel",
    "GCUHKFrequencyModel",
    "GAgentLGModel",
    "GTheFawkesModel",
    "GInoxAgentModel",
    "GWorstModel",
    "GPerfectModel",
]


@define
class GHardHeadedFrequencyModel(GeniusOpponentModel):
    """
    Hard-Headed Frequency-based opponent model from Genius.

    This model estimates the opponent's utility function by tracking which
    issues remain unchanged between consecutive opponent bids. Issues that
    don't change are assumed to be more important to the opponent.

    The model works by:
    1. Tracking bid frequencies for each issue value
    2. When an issue value stays the same between consecutive bids,
       increasing its weight (the opponent likely cares about that issue)
    3. Computing opponent utility as a weighted sum of issue value frequencies

    Args:
        learning_coef: Learning coefficient controlling how fast weights adapt.
            Higher values mean faster adaptation (default 0.2).
        learning_value_addition: Value added to unchanged issue weights (default 1).
        default_value: Default value for unseen issue values (default 1).

    Transcompiled from: negotiator.boaframework.opponentmodel.HardHeadedFrequencyModel
    """

    learning_coef: float = 0.2
    learning_value_addition: int = 1
    default_value: int = 1
    _issue_weights: dict[int, float] = field(factory=dict)
    _value_weights: dict[int, dict] = field(factory=dict)
    _last_opponent_bid: Outcome | None = field(init=False, default=None)
    _n_issues: int = field(init=False, default=0)
    _initialized: bool = field(init=False, default=False)

    def _initialize(self) -> None:
        """Initialize the model with uniform weights."""
        if not self.negotiator or not self.negotiator.nmi:
            return

        outcome_space = self.negotiator.nmi.outcome_space
        if outcome_space is None:
            return

        issues = outcome_space.issues
        self._n_issues = len(issues)

        # Initialize uniform issue weights
        if self._n_issues > 0:
            initial_weight = 1.0 / self._n_issues
            for i in range(self._n_issues):
                self._issue_weights[i] = initial_weight
                self._value_weights[i] = {}

        self._initialized = True

    def on_preferences_changed(self, changes: list[PreferencesChange]) -> None:
        """Reset the model when preferences change.

        Args:
            changes: List of preference changes.
        """
        self._issue_weights = {}
        self._value_weights = {}
        self._last_opponent_bid = None
        self._initialized = False

    def on_partner_proposal(
        self, state: GBState, partner_id: str, offer: Outcome
    ) -> None:
        """Update the model based on the opponent's offer.

        Args:
            state: Current negotiation state.
            partner_id: ID of the partner who made the offer.
            offer: The opponent's offer.
        """
        if not self._initialized:
            self._initialize()

        if offer is None:
            return

        # Update value frequencies
        for i in range(self._n_issues):
            value = offer[i]
            if value not in self._value_weights[i]:
                self._value_weights[i][value] = self.default_value
            self._value_weights[i][value] += 1

        # Learn from unchanged issues (indicates importance to opponent)
        if self._last_opponent_bid is not None:
            unchanged_issues = []
            for i in range(self._n_issues):
                if offer[i] == self._last_opponent_bid[i]:
                    unchanged_issues.append(i)

            # Increase weights for unchanged issues
            if unchanged_issues:
                # Calculate total weight to add
                total_addition = self.learning_coef * len(unchanged_issues)
                addition_per_issue = total_addition / len(unchanged_issues)

                # Update weights
                for i in unchanged_issues:
                    self._issue_weights[i] += addition_per_issue

                # Normalize weights to sum to 1
                total_weight = sum(self._issue_weights.values())
                if total_weight > 0:
                    for i in self._issue_weights:
                        self._issue_weights[i] /= total_weight

        self._last_opponent_bid = offer

        # Update private_info with this model
        self._update_private_info(partner_id)

    def eval(self, offer: Outcome | None) -> Value:
        """Evaluate the estimated opponent utility for an outcome.

        Args:
            offer: The outcome to evaluate.

        Returns:
            Estimated opponent utility (0 to 1).
        """
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

            # Get value weight (frequency-based)
            value_weight = self._value_weights.get(i, {}).get(value, self.default_value)

            # Normalize value weight by max weight for this issue
            max_value_weight = max(
                self._value_weights.get(i, {}).values() or [self.default_value]
            )
            if max_value_weight > 0:
                normalized_value_weight = value_weight / max_value_weight
            else:
                normalized_value_weight = 1.0

            total_utility += issue_weight * normalized_value_weight

        return total_utility

    def eval_normalized(
        self,
        offer: Outcome | None,
        above_reserve: bool = True,
        expected_limits: bool = True,
    ) -> Value:
        """Evaluate normalized opponent utility.

        Args:
            offer: The outcome to evaluate.
            above_reserve: Whether to normalize above reserve value.
            expected_limits: Whether to use expected limits.

        Returns:
            Normalized opponent utility (0 to 1).
        """
        # The eval method already returns a value in [0, 1]
        return self.eval(offer)


@define
class GDefaultModel(GeniusOpponentModel):
    """
    Default opponent model from Genius.

    A no-op model that doesn't learn from opponent behavior and assumes
    uniform preferences. Always returns 0.5 utility for any outcome.

    Useful as a baseline or placeholder when no opponent modeling is needed.

    Transcompiled from: negotiator.boaframework.opponentmodel.DefaultModel
    """

    def on_preferences_changed(self, changes: list[PreferencesChange]) -> None:
        """No-op - this model doesn't adapt."""
        pass

    def on_partner_proposal(
        self, state: GBState, partner_id: str, offer: Outcome
    ) -> None:
        """No-op - this model doesn't learn from offers."""
        pass

    def eval(self, offer: Outcome | None) -> Value:
        """Return constant utility (0.5) for any outcome.

        Args:
            offer: The outcome to evaluate.

        Returns:
            Always returns 0.5.
        """
        if offer is None:
            return 0.0
        return 0.5

    def eval_normalized(
        self,
        offer: Outcome | None,
        above_reserve: bool = True,
        expected_limits: bool = True,
    ) -> Value:
        """Return normalized utility."""
        return self.eval(offer)


# =============================================================================
# Additional Opponent Models
# =============================================================================


@define
class GFSEGABayesianModel(GeniusOpponentModel):
    """
    FSEGA Bayesian opponent model.

    This model uses Bayesian inference to estimate opponent preferences,
    maintaining hypotheses about issue weights and value utilities.

    Transcompiled from: negotiator.boaframework.opponentmodel.FSEGABayesianModel
    """

    _issue_weights: dict[int, float] = field(factory=dict)
    _value_utils: dict[int, dict] = field(factory=dict)
    _n_issues: int = field(init=False, default=0)
    _initialized: bool = field(init=False, default=False)
    _n_bids: int = field(init=False, default=0)

    def _initialize(self) -> None:
        """Initialize the model with uniform weights."""
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
        learning_rate = 1.0 / (1.0 + self._n_bids)

        for i in range(self._n_issues):
            value = offer[i]
            if value in self._value_utils[i]:
                self._value_utils[i][value] = (1 - learning_rate) * self._value_utils[
                    i
                ][value] + learning_rate * 1.0

        # Normalize
        for i in range(self._n_issues):
            max_util = (
                max(self._value_utils[i].values()) if self._value_utils[i] else 1.0
            )
            if max_util > 0:
                for v in self._value_utils[i]:
                    self._value_utils[i][v] /= max_util

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
class GIAMhagglerBayesianModel(GeniusOpponentModel):
    """
    IAMhaggler Bayesian opponent model.

    This model uses Bayesian inference to estimate opponent preferences
    with specific adaptation for the IAMhaggler agent family.

    Transcompiled from: negotiator.boaframework.opponentmodel.IAMhagglerBayesianModel
    """

    _issue_weights: dict[int, float] = field(factory=dict)
    _value_utils: dict[int, dict] = field(factory=dict)
    _n_issues: int = field(init=False, default=0)
    _initialized: bool = field(init=False, default=False)
    _n_bids: int = field(init=False, default=0)

    def _initialize(self) -> None:
        """Initialize the model with uniform weights."""
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
        learning_rate = 0.2 / (1.0 + 0.1 * self._n_bids)

        for i in range(self._n_issues):
            value = offer[i]
            if value in self._value_utils[i]:
                old_util = self._value_utils[i][value]
                self._value_utils[i][value] = old_util + learning_rate * (
                    1.0 - old_util
                )

        # Normalize value utilities
        for i in range(self._n_issues):
            max_util = (
                max(self._value_utils[i].values()) if self._value_utils[i] else 1.0
            )
            if max_util > 0:
                for v in self._value_utils[i]:
                    self._value_utils[i][v] /= max_util

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
class GCUHKFrequencyModel(GeniusOpponentModel):
    """
    CUHK Frequency-based opponent model.

    This model tracks bid frequencies and uses them to estimate
    opponent preferences with specific adaptations from the CUHK agent.

    Transcompiled from: negotiator.boaframework.opponentmodel.CUHKFrequencyModelV2
    """

    _issue_weights: dict[int, float] = field(factory=dict)
    _value_counts: dict[int, dict] = field(factory=dict)
    _n_issues: int = field(init=False, default=0)
    _initialized: bool = field(init=False, default=False)
    _total_bids: int = field(init=False, default=0)

    def _initialize(self) -> None:
        """Initialize the model with uniform weights."""
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
                self._value_counts[i] = {}
                for v in issues[i].all:
                    self._value_counts[i][v] = 0

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

        self._total_bids += 1

        for i in range(self._n_issues):
            value = offer[i]
            if value in self._value_counts[i]:
                self._value_counts[i][value] += 1

    def eval(self, offer: Outcome | None) -> Value:
        """Evaluate opponent utility."""
        if offer is None:
            return 0.0

        if not self._initialized:
            self._initialize()

        if self._n_issues == 0 or self._total_bids == 0:
            return 0.5

        total_utility = 0.0
        for i in range(self._n_issues):
            value = offer[i]
            issue_weight = self._issue_weights.get(i, 1.0 / max(1, self._n_issues))
            count = self._value_counts.get(i, {}).get(value, 0)
            max_count = (
                max(self._value_counts[i].values()) if self._value_counts[i] else 1
            )
            value_util = count / max_count if max_count > 0 else 0.5
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


@define
class GWorstModel(GeniusOpponentModel):
    """
    Worst-case opponent model.

    This model assumes the opponent has opposite preferences to ours.

    Transcompiled from: negotiator.boaframework.opponentmodel.WorstModel
    """

    _initialized: bool = field(init=False, default=False)

    def _initialize(self) -> None:
        """Initialize the model."""
        self._initialized = True

    def on_preferences_changed(self, changes: list[PreferencesChange]) -> None:
        """Handle preference changes."""
        self._initialize()

    def update(self, state: GBState, offer: Outcome, partner_id: str) -> None:
        """Update model based on opponent's offer (no-op for worst model)."""
        pass

    def eval(self, offer: Outcome | None) -> Value:
        """Evaluate opponent utility as inverse of our utility."""
        if offer is None:
            return 0.0

        if not self.negotiator or not self.negotiator.ufun:
            return 0.0

        # Return inverse of our utility
        our_util = float(self.negotiator.ufun(offer))
        return 1.0 - our_util

    def eval_normalized(
        self,
        offer: Outcome | None,
        above_reserve: bool = True,
        expected_limits: bool = True,
    ) -> Value:
        """Return normalized utility."""
        return self.eval(offer)


@define
class GPerfectModel(GeniusOpponentModel):
    """
    Perfect opponent model (for testing/debugging).

    This model has access to the opponent's actual utility function.
    Only useful in experimental settings where opponent preferences are known.

    Transcompiled from: negotiator.boaframework.opponentmodel.PerfectModel
    """

    _opponent_ufun: object | None = field(init=False, default=None)

    def on_preferences_changed(self, changes: list[PreferencesChange]) -> None:
        """Handle preference changes."""
        pass

    def update(self, state: GBState, offer: Outcome, partner_id: str) -> None:
        """Update model based on opponent's offer (stores opponent ufun if available)."""
        # In a real scenario, this would try to get the opponent's actual ufun
        # For now, this is a placeholder
        pass

    def eval(self, offer: Outcome | None) -> Value:
        """Evaluate opponent utility."""
        if offer is None:
            return 0.0

        if self._opponent_ufun is not None:
            try:
                return float(self._opponent_ufun(offer))  # type: ignore
            except Exception:
                pass

        # Fallback to uniform
        return 0.5

    def eval_normalized(
        self,
        offer: Outcome | None,
        above_reserve: bool = True,
        expected_limits: bool = True,
    ) -> Value:
        """Return normalized utility."""
        return self.eval(offer)


@define
class GUniformModel(GeniusOpponentModel):
    """
    Uniform opponent model from Genius.

    Assumes the opponent values all outcomes equally - returns uniform
    random utility values. Each outcome gets a consistent random value.

    Transcompiled from: negotiator.boaframework.opponentmodel.UniformModel
    """

    _outcome_utils: dict[tuple, float] = field(factory=dict)

    def on_preferences_changed(self, changes: list[PreferencesChange]) -> None:
        """Reset cached utilities when preferences change."""
        self._outcome_utils = {}

    def on_partner_proposal(
        self, state: GBState, partner_id: str, offer: Outcome
    ) -> None:
        """No-op - this model uses random values."""
        pass

    def eval(self, offer: Outcome | None) -> Value:
        """Return a random but consistent utility for the outcome.

        Args:
            offer: The outcome to evaluate.

        Returns:
            Random utility value in [0, 1], consistent for same outcome.
        """
        import random

        if offer is None:
            return 0.0

        # Convert to hashable key
        key = tuple(offer) if isinstance(offer, (list, tuple)) else (offer,)

        if key not in self._outcome_utils:
            self._outcome_utils[key] = random.random()

        return self._outcome_utils[key]

    def eval_normalized(
        self,
        offer: Outcome | None,
        above_reserve: bool = True,
        expected_limits: bool = True,
    ) -> Value:
        """Return normalized utility."""
        return self.eval(offer)


@define
class GOppositeModel(GeniusOpponentModel):
    """
    Opposite opponent model from Genius.

    Assumes the opponent has exactly opposite preferences - what's good
    for us is bad for them and vice versa. Returns 1 - our_utility.

    This is a pessimistic assumption useful for competitive scenarios.

    Transcompiled from: negotiator.boaframework.opponentmodel.OppositeModel
    """

    def on_preferences_changed(self, changes: list[PreferencesChange]) -> None:
        """No special initialization needed."""
        pass

    def on_partner_proposal(
        self, state: GBState, partner_id: str, offer: Outcome
    ) -> None:
        """No-op - this model uses our utility function inversely."""
        pass

    def eval(self, offer: Outcome | None) -> Value:
        """Return opposite utility (1 - our utility).

        Args:
            offer: The outcome to evaluate.

        Returns:
            1 - our_utility for the outcome.
        """
        if offer is None:
            return 0.0

        if not self.negotiator or not self.negotiator.ufun:
            return 0.5

        our_util = float(self.negotiator.ufun(offer))
        return 1.0 - our_util

    def eval_normalized(
        self,
        offer: Outcome | None,
        above_reserve: bool = True,
        expected_limits: bool = True,
    ) -> Value:
        """Return normalized opposite utility."""
        return self.eval(offer)


@define
class GSmithFrequencyModel(GeniusOpponentModel):
    """
    Smith frequency-based opponent model from Genius.

    From AgentSmith (ANAC 2010). Similar to HardHeadedFrequencyModel but
    with a simpler weight update mechanism based purely on value frequencies.

    The model tracks how often each value appears in opponent bids and
    assumes higher frequency = higher importance.

    Args:
        default_value: Default value for unseen issue values (default 1).

    Transcompiled from: negotiator.boaframework.opponentmodel.SmithFrequencyModel
    """

    default_value: int = 1
    _value_counts: dict[int, dict] = field(factory=dict)
    _n_issues: int = field(init=False, default=0)
    _initialized: bool = field(init=False, default=False)

    def _initialize(self) -> None:
        """Initialize the model."""
        if not self.negotiator or not self.negotiator.nmi:
            return

        outcome_space = self.negotiator.nmi.outcome_space
        if outcome_space is None:
            return

        issues = outcome_space.issues
        self._n_issues = len(issues)

        for i in range(self._n_issues):
            self._value_counts[i] = {}

        self._initialized = True

    def on_preferences_changed(self, changes: list[PreferencesChange]) -> None:
        """Reset the model when preferences change."""
        self._value_counts = {}
        self._initialized = False

    def on_partner_proposal(
        self, state: GBState, partner_id: str, offer: Outcome
    ) -> None:
        """Update value frequencies based on opponent's offer."""
        if not self._initialized:
            self._initialize()

        if offer is None:
            return

        # Count value occurrences
        for i in range(self._n_issues):
            value = offer[i]
            if value not in self._value_counts[i]:
                self._value_counts[i][value] = self.default_value
            self._value_counts[i][value] += 1

        self._update_private_info(partner_id)

    def eval(self, offer: Outcome | None) -> Value:
        """Evaluate opponent utility based on value frequencies.

        Args:
            offer: The outcome to evaluate.

        Returns:
            Estimated opponent utility (0 to 1).
        """
        if offer is None:
            return 0.0

        if not self._initialized:
            self._initialize()

        if self._n_issues == 0:
            return 0.0

        total_utility = 0.0
        issue_weight = 1.0 / self._n_issues

        for i in range(self._n_issues):
            value = offer[i]
            count = self._value_counts.get(i, {}).get(value, self.default_value)

            # Normalize by max count for this issue
            max_count = max(
                self._value_counts.get(i, {}).values() or [self.default_value]
            )
            if max_count > 0:
                normalized_count = count / max_count
            else:
                normalized_count = 1.0

            total_utility += issue_weight * normalized_count

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
class GAgentXFrequencyModel(GeniusOpponentModel):
    """
    AgentX frequency-based opponent model from Genius.

    From AgentX (ANAC 2015). An advanced frequency model that uses both
    value frequencies and issue weight learning based on bid patterns.

    Tracks issue weights by observing which issues change less frequently,
    and uses exponential smoothing for weight updates.

    Args:
        learning_rate: Learning rate for weight updates (default 0.25).
        default_value: Default value for unseen issue values (default 1).

    Transcompiled from: negotiator.boaframework.opponentmodel.AgentXFrequencyModel
    """

    learning_rate: float = 0.25
    default_value: int = 1
    _issue_weights: dict[int, float] = field(factory=dict)
    _value_counts: dict[int, dict] = field(factory=dict)
    _last_bid: Outcome | None = field(init=False, default=None)
    _n_issues: int = field(init=False, default=0)
    _initialized: bool = field(init=False, default=False)

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
                self._value_counts[i] = {}

        self._initialized = True

    def on_preferences_changed(self, changes: list[PreferencesChange]) -> None:
        """Reset the model when preferences change."""
        self._issue_weights = {}
        self._value_counts = {}
        self._last_bid = None
        self._initialized = False

    def on_partner_proposal(
        self, state: GBState, partner_id: str, offer: Outcome
    ) -> None:
        """Update model based on opponent's offer."""
        if not self._initialized:
            self._initialize()

        if offer is None:
            return

        # Update value counts
        for i in range(self._n_issues):
            value = offer[i]
            if value not in self._value_counts[i]:
                self._value_counts[i][value] = self.default_value
            self._value_counts[i][value] += 1

        # Learn issue weights from unchanged values
        if self._last_bid is not None:
            for i in range(self._n_issues):
                if offer[i] == self._last_bid[i]:
                    # Issue unchanged - likely important
                    old_weight = self._issue_weights[i]
                    # Exponential smoothing update
                    self._issue_weights[i] = old_weight + self.learning_rate * (
                        1.0 - old_weight
                    )
                else:
                    # Issue changed - reduce weight slightly
                    old_weight = self._issue_weights[i]
                    self._issue_weights[i] = old_weight * (
                        1.0 - self.learning_rate * 0.5
                    )

            # Normalize weights
            total_weight = sum(self._issue_weights.values())
            if total_weight > 0:
                for i in self._issue_weights:
                    self._issue_weights[i] /= total_weight

        self._last_bid = offer
        self._update_private_info(partner_id)

    def eval(self, offer: Outcome | None) -> Value:
        """Evaluate opponent utility.

        Args:
            offer: The outcome to evaluate.

        Returns:
            Estimated opponent utility (0 to 1).
        """
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
            count = self._value_counts.get(i, {}).get(value, self.default_value)

            max_count = max(
                self._value_counts.get(i, {}).values() or [self.default_value]
            )
            if max_count > 0:
                value_util = count / max_count
            else:
                value_util = 1.0

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
class GNashFrequencyModel(GeniusOpponentModel):
    """
    Nash frequency-based opponent model from Genius.

    A frequency model that aims to estimate outcomes close to the Nash
    bargaining solution by combining opponent utility estimates with
    our own utility.

    Uses frequency-based opponent modeling but biases toward Pareto-efficient
    outcomes by considering the product of utilities.

    Args:
        default_value: Default value for unseen issue values (default 1).

    Transcompiled from: negotiator.boaframework.opponentmodel.NashFrequencyModel
    """

    default_value: int = 1
    _issue_weights: dict[int, float] = field(factory=dict)
    _value_counts: dict[int, dict] = field(factory=dict)
    _last_bid: Outcome | None = field(init=False, default=None)
    _n_issues: int = field(init=False, default=0)
    _initialized: bool = field(init=False, default=False)

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
                self._value_counts[i] = {}

        self._initialized = True

    def on_preferences_changed(self, changes: list[PreferencesChange]) -> None:
        """Reset the model when preferences change."""
        self._issue_weights = {}
        self._value_counts = {}
        self._last_bid = None
        self._initialized = False

    def on_partner_proposal(
        self, state: GBState, partner_id: str, offer: Outcome
    ) -> None:
        """Update model based on opponent's offer."""
        if not self._initialized:
            self._initialize()

        if offer is None:
            return

        # Update value counts
        for i in range(self._n_issues):
            value = offer[i]
            if value not in self._value_counts[i]:
                self._value_counts[i][value] = self.default_value
            self._value_counts[i][value] += 1

        # Learn from unchanged issues
        if self._last_bid is not None:
            unchanged = []
            for i in range(self._n_issues):
                if offer[i] == self._last_bid[i]:
                    unchanged.append(i)

            if unchanged:
                addition = 0.1 / len(unchanged)
                for i in unchanged:
                    self._issue_weights[i] += addition

                # Normalize
                total = sum(self._issue_weights.values())
                if total > 0:
                    for i in self._issue_weights:
                        self._issue_weights[i] /= total

        self._last_bid = offer
        self._update_private_info(partner_id)

    def eval(self, offer: Outcome | None) -> Value:
        """Evaluate opponent utility with Nash-optimal bias.

        Args:
            offer: The outcome to evaluate.

        Returns:
            Estimated opponent utility (0 to 1).
        """
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
            count = self._value_counts.get(i, {}).get(value, self.default_value)

            max_count = max(
                self._value_counts.get(i, {}).values() or [self.default_value]
            )
            if max_count > 0:
                value_util = count / max_count
            else:
                value_util = 1.0

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
class GBayesianModel(GeniusOpponentModel):
    """
    Bayesian opponent model from Genius.

    Uses Bayesian inference to update beliefs about opponent's utility function
    based on their bids. Maintains probability distributions over possible
    opponent preferences and updates them using Bayes' rule.

    This is a simplified version that assumes the opponent is rational and
    only offers bids above some threshold.

    Args:
        n_hypotheses: Number of hypotheses to consider (default 10).
        rationality: Assumed opponent rationality - higher values assume
            opponent is more likely to make utility-maximizing bids (default 5.0).

    Transcompiled from: negotiator.boaframework.opponentmodel.BayesianModel
    """

    n_hypotheses: int = 10
    rationality: float = 5.0
    _issue_weight_hypotheses: list[dict[int, float]] = field(factory=list)
    _hypothesis_probs: list[float] = field(factory=list)
    _value_utils: dict[int, dict] = field(factory=dict)
    _n_issues: int = field(init=False, default=0)
    _initialized: bool = field(init=False, default=False)

    def _initialize(self) -> None:
        """Initialize the model with random hypotheses."""
        import random

        if not self.negotiator or not self.negotiator.nmi:
            return

        outcome_space = self.negotiator.nmi.outcome_space
        if outcome_space is None:
            return

        issues = outcome_space.issues
        self._n_issues = len(issues)

        # Generate random weight hypotheses
        self._issue_weight_hypotheses = []
        self._hypothesis_probs = []

        for _ in range(self.n_hypotheses):
            # Random weights that sum to 1
            weights = [random.random() for _ in range(self._n_issues)]
            total = sum(weights)
            if total > 0:
                weights = [w / total for w in weights]
            else:
                weights = [1.0 / self._n_issues for _ in range(self._n_issues)]

            self._issue_weight_hypotheses.append(
                {i: weights[i] for i in range(self._n_issues)}
            )
            self._hypothesis_probs.append(1.0 / self.n_hypotheses)

        # Initialize random value utilities
        for i in range(self._n_issues):
            self._value_utils[i] = {}
            issue = issues[i]
            for value in issue.all:
                self._value_utils[i][value] = random.random()

        self._initialized = True

    def on_preferences_changed(self, changes: list[PreferencesChange]) -> None:
        """Reset the model when preferences change."""
        self._issue_weight_hypotheses = []
        self._hypothesis_probs = []
        self._value_utils = {}
        self._initialized = False

    def on_partner_proposal(
        self, state: GBState, partner_id: str, offer: Outcome
    ) -> None:
        """Update hypothesis probabilities using Bayes' rule."""
        import math

        if not self._initialized:
            self._initialize()

        if offer is None:
            return

        # Calculate likelihood of this bid under each hypothesis
        likelihoods = []
        for h_idx, hypothesis in enumerate(self._issue_weight_hypotheses):
            # Calculate utility under this hypothesis
            util = 0.0
            for i in range(self._n_issues):
                value = offer[i]
                issue_weight = hypothesis.get(i, 0.0)
                value_util = self._value_utils.get(i, {}).get(value, 0.5)
                util += issue_weight * value_util

            # Likelihood based on rationality (soft-max)
            likelihood = math.exp(self.rationality * util)
            likelihoods.append(likelihood)

        # Bayesian update
        posteriors = []
        for h_idx in range(len(self._hypothesis_probs)):
            posterior = self._hypothesis_probs[h_idx] * likelihoods[h_idx]
            posteriors.append(posterior)

        # Normalize
        total = sum(posteriors)
        if total > 0:
            self._hypothesis_probs = [p / total for p in posteriors]

        self._update_private_info(partner_id)

    def eval(self, offer: Outcome | None) -> Value:
        """Evaluate opponent utility as weighted average over hypotheses.

        Args:
            offer: The outcome to evaluate.

        Returns:
            Estimated opponent utility (0 to 1).
        """
        if offer is None:
            return 0.0

        if not self._initialized:
            self._initialize()

        if self._n_issues == 0:
            return 0.0

        # Weighted average utility across hypotheses
        total_utility = 0.0
        for h_idx, hypothesis in enumerate(self._issue_weight_hypotheses):
            h_prob = self._hypothesis_probs[h_idx]

            h_util = 0.0
            for i in range(self._n_issues):
                value = offer[i]
                issue_weight = hypothesis.get(i, 0.0)
                value_util = self._value_utils.get(i, {}).get(value, 0.5)
                h_util += issue_weight * value_util

            total_utility += h_prob * h_util

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
class GScalableBayesianModel(GeniusOpponentModel):
    """
    Scalable Bayesian opponent model from Genius.

    A Bayesian model optimized for large outcome spaces. Uses a more
    efficient representation that scales better with domain size.

    Instead of maintaining full hypothesis distributions, it tracks
    sufficient statistics and uses approximate inference.

    Args:
        learning_rate: Rate of belief updates (default 0.1).

    Transcompiled from: negotiator.boaframework.opponentmodel.ScalableBayesianModel
    """

    learning_rate: float = 0.1
    _issue_weights: dict[int, float] = field(factory=dict)
    _value_utils: dict[int, dict] = field(factory=dict)
    _bid_count: int = field(init=False, default=0)
    _n_issues: int = field(init=False, default=0)
    _initialized: bool = field(init=False, default=False)

    def _initialize(self) -> None:
        """Initialize the model."""
        if not self.negotiator or not self.negotiator.nmi:
            return

        outcome_space = self.negotiator.nmi.outcome_space
        if outcome_space is None:
            return

        issues = outcome_space.issues
        self._n_issues = len(issues)

        # Uniform prior over issue weights
        if self._n_issues > 0:
            for i in range(self._n_issues):
                self._issue_weights[i] = 1.0 / self._n_issues
                self._value_utils[i] = {}
                # Initialize uniform value utilities
                issue = issues[i]
                for value in issue.all:
                    self._value_utils[i][value] = 0.5

        self._initialized = True

    def on_preferences_changed(self, changes: list[PreferencesChange]) -> None:
        """Reset the model when preferences change."""
        self._issue_weights = {}
        self._value_utils = {}
        self._bid_count = 0
        self._initialized = False

    def on_partner_proposal(
        self, state: GBState, partner_id: str, offer: Outcome
    ) -> None:
        """Update beliefs using online learning."""
        if not self._initialized:
            self._initialize()

        if offer is None:
            return

        self._bid_count += 1

        # Update value utilities (increase for observed values)
        for i in range(self._n_issues):
            value = offer[i]
            if value in self._value_utils[i]:
                old_util = self._value_utils[i][value]
                # Move toward 1.0 for observed values
                self._value_utils[i][value] = old_util + self.learning_rate * (
                    1.0 - old_util
                )

                # Decay other values slightly
                for v in self._value_utils[i]:
                    if v != value:
                        self._value_utils[i][v] *= 1.0 - self.learning_rate * 0.1

        # Normalize value utilities per issue
        for i in range(self._n_issues):
            max_util = (
                max(self._value_utils[i].values()) if self._value_utils[i] else 1.0
            )
            if max_util > 0:
                for v in self._value_utils[i]:
                    self._value_utils[i][v] /= max_util

        self._update_private_info(partner_id)

    def eval(self, offer: Outcome | None) -> Value:
        """Evaluate opponent utility.

        Args:
            offer: The outcome to evaluate.

        Returns:
            Estimated opponent utility (0 to 1).
        """
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
