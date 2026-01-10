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
    "GTimeDependentOffering",
    "GRandomOffering",
    "GBoulwareOffering",
    "GConcederOffering",
    "GLinearOffering",
    "GHardlinerOffering",
    "GChoosingAllBids",
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

    Transcompiled from: bilateralexamples.boacomponents.TimeDependent_Offering
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
