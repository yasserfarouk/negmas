"""models components (split from models.py): baselines."""

from __future__ import annotations

from typing import TYPE_CHECKING

from attrs import define, field

from negmas.gb.components.genius.base import GeniusOpponentModel
from negmas.gb.components.models.ufun import PeekingOpponentModel

if TYPE_CHECKING:
    from negmas.common import PreferencesChange, Value
    from negmas.gb import GBState
    from negmas.outcomes import Outcome

__all__ = [
    "GDefaultModel",
    "GUniformModel",
    "GOppositeModel",
    "GWorstModel",
    "GPerfectModel",
]


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
        # Update private_info so negotiators can access this model
        self._update_private_info(partner_id)

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
        # Update private_info so negotiators can access this model
        self._update_private_info(partner_id)

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
        # Update private_info so negotiators can access this model
        self._update_private_info(partner_id)

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

    def on_partner_proposal(
        self, state: GBState, partner_id: str, offer: Outcome
    ) -> None:
        """Handle partner proposal by updating private_info."""
        self._update_private_info(partner_id)

    def update(self, state: GBState, offer: Outcome, partner_id: str) -> None:
        """Update model based on opponent's offer (no-op for worst model)."""
        # Update private_info so negotiators can access this model
        self._update_private_info(partner_id)

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


class GPerfectModel(PeekingOpponentModel):
    """
    Perfect (oracle) opponent model — has access to the opponent's true ufun.

    This is the Genius ``PerfectModel``: an oracle for testing/analysis where
    the opponent's preferences are known. Pass the opponent's true
    `BaseUtilityFunction` as ``ufun`` (at construction or later,
    ``model.ufun = ...``); ``eval`` / ``eval_normalized`` then delegate to it.
    When no ``ufun`` is set it falls back to ``0.5`` (uniform).

    Implemented as a thin alias of `PeekingOpponentModel` (the working oracle in
    ``gb.components.models.ufun``) so the two share one code path; the only
    addition is the multilateral per-partner ``private_info`` update inherited
    from the Genius model family.

    Transcompiled from: negotiator.boaframework.opponentmodel.PerfectModel
    """

    def on_partner_proposal(
        self, state: GBState, partner_id: str, offer: Outcome
    ) -> None:
        """Register the oracle in ``private_info`` under ``partner_id`` (multilateral)."""
        self._update_private_info(partner_id)
