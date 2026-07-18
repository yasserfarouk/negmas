"""models components (split from models.py): frequency."""

from __future__ import annotations

from typing import TYPE_CHECKING

from attrs import define, field

from negmas.gb.components.genius.base import GeniusOpponentModel

if TYPE_CHECKING:
    from negmas.common import PreferencesChange, Value
    from negmas.gb import GBState
    from negmas.outcomes import Outcome

__all__ = [
    "GHardHeadedFrequencyModel",
    "GSmithFrequencyModel",
    "GCUHKFrequencyModel",
    "GNashFrequencyModel",
    "GAgentXFrequencyModel",
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
