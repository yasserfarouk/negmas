"""models components (split from models.py): bayesian."""

from __future__ import annotations

from typing import TYPE_CHECKING

from attrs import define, field

from negmas.gb.components.genius.base import GeniusOpponentModel

if TYPE_CHECKING:
    from negmas.common import PreferencesChange, Value
    from negmas.gb import GBState
    from negmas.outcomes import Outcome

__all__ = [
    "GBayesianModel",
    "GScalableBayesianModel",
    "GFSEGABayesianModel",
    "GIAMhagglerBayesianModel",
]


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
