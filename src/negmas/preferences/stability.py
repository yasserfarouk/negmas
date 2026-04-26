"""Stability criteria for utility functions.

This module defines stability flags that describe which aspects of a utility function
remain stable during a negotiation. These flags can be combined using bitwise OR (|)
to describe complex stability guarantees.

Key Concepts:
    - VOLATILE: No stability guarantees at all (value = 0). The ufun is session-dependent,
      state-dependent, and no aspect is guaranteed stable.
    - STATIONARY: All flags set - fully stable, session-independent, state-independent.
      Calling it twice with the same outcome always returns the same value.

Independence Flags (Orthogonal Concepts):
    - SESSION_INDEPENDENT: The ufun does not depend on negotiation session details
      accessible via the NMI (NegotiatorMechanismInterface), such as:
      - Number of negotiators (n_negotiators)
      - Mechanism parameters (time_limit, n_steps, etc.)
      - Other negotiator information

    - STATE_INDEPENDENT: The ufun does not depend on negotiation state variables
      accessible via MechanismState, such as:
      - Current time/step (state.time, state.step)
      - Offer history (state.offers)
      - Current offer (state.current_offer)

    These two concepts are ORTHOGONAL - a ufun can be:
    - Session-independent AND state-independent (stationary ufuns)
    - Session-independent BUT state-dependent (e.g., discounted ufuns - discount
      depends on state.step but not on NMI parameters)
    - Session-dependent BUT state-independent (e.g., ufun that scales based on
      number of negotiators but doesn't change over time)
    - Session-dependent AND state-dependent (fully volatile)
"""

from __future__ import annotations

from enum import IntFlag
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

__all__ = [
    "Stability",
    # Independence flags
    "SESSION_INDEPENDENT",
    "STATE_INDEPENDENT",
    # Stability flags for specific aspects
    "STABLE_MIN",
    "STABLE_MAX",
    "STABLE_EXTREMES",
    "STABLE_RESERVED_VALUE",
    "FIXED_RESERVED_VALUE",
    "STABLE_RATIONAL_OUTCOMES",
    "STABLE_IRRATIONAL_OUTCOMES",
    "STABLE_ORDERING",
    "STABLE_DIFF_RATIOS",
    # Predefined combinations
    "STATIONARY",
    "VOLATILE",
    "STABLE_SCALE",
    "SCALE_INVARIANT",
]


class Stability(IntFlag):
    """
    Flags describing stability and independence properties of utility functions.

    These flags can be combined using bitwise OR (|) to describe which aspects
    of a utility function remain stable during a negotiation.

    Independence Flags:
        SESSION_INDEPENDENT: The ufun does not depend on negotiation session details
            (NMI values, number of negotiators, mechanism parameters, etc.)
        STATE_INDEPENDENT: The ufun does not depend on negotiation state variables
            (time, step, current offers, history, etc.). Note: anything that is
            NOT state-independent is considered VOLATILE.

    Stability Flags (for specific aspects):
        STABLE_MIN: The minimum utility value does not change
        STABLE_MAX: The maximum utility value does not change
        STABLE_EXTREMES: The extreme (best/worst) outcomes do not change. This is
            weaker than STABLE_ORDERING - ordering may change but extremes stay fixed.
        STABLE_RESERVED_VALUE: The reserved value relative to min/max does not change
        FIXED_RESERVED_VALUE: The absolute numeric reserved value does not change
        STABLE_RATIONAL_OUTCOMES: Rational outcomes (utility > reserved) stay rational
        STABLE_IRRATIONAL_OUTCOMES: Irrational outcomes (utility <= reserved) stay irrational
        STABLE_ORDERING: The ordering of outcomes by utility does not change.
            Implies STABLE_EXTREMES (if ordering is stable, extremes are stable).
        STABLE_DIFF_RATIOS: Relative differences between utilities stay the same.
            Implies STABLE_ORDERING (if diff ratios are stable, ordering is stable).

    Predefined Combinations:
        VOLATILE: No stability guarantees (value = 0, all aspects may change)
        STATIONARY: All flags set - fully stable, session/state independent
        STABLE_SCALE: STABLE_MIN | STABLE_MAX | STABLE_RESERVED_VALUE
        SCALE_INVARIANT: STABLE_DIFF_RATIOS | STABLE_RESERVED_VALUE

    Examples:
        >>> from negmas.preferences import (
        ...     Stability,
        ...     STABLE_MIN,
        ...     STABLE_MAX,
        ...     STATIONARY,
        ...     VOLATILE,
        ... )
        >>> # Create a ufun with stable min and max
        >>> stability = STABLE_MIN | STABLE_MAX
        >>> bool(stability & STABLE_MIN)
        True
        >>> stability.is_volatile  # Not volatile - has some stability flags
        False
        >>> stability.is_stationary  # Not fully stationary either
        False
        >>> # Check volatile (no flags set)
        >>> VOLATILE.is_volatile
        True
        >>> VOLATILE == 0
        True
        >>> # Check if fully stationary
        >>> STATIONARY.is_stationary
        True
        >>> STATIONARY.is_volatile
        False
    """

    # Independence flags (higher bits)
    SESSION_INDEPENDENT = 1 << 8  # Does not depend on NMI/session details
    STATE_INDEPENDENT = 1 << 9  # Does not depend on state (time, step, etc.)

    # Individual stability flags (lower bits, powers of 2 for bitwise operations)
    STABLE_MIN = 1 << 0  # Minimum utility value is stable
    STABLE_MAX = 1 << 1  # Maximum utility value is stable
    STABLE_EXTREMES = 1 << 10  # Extreme (best/worst) outcomes are stable
    STABLE_RESERVED_VALUE = 1 << 2  # Reserved value (relative to min/max) is stable
    FIXED_RESERVED_VALUE = 1 << 3  # Absolute numeric reserved value is stable
    STABLE_RATIONAL_OUTCOMES = 1 << 4  # Rational outcomes remain rational
    STABLE_IRRATIONAL_OUTCOMES = 1 << 5  # Irrational outcomes remain irrational
    STABLE_ORDERING = 1 << 6  # Outcome ordering is stable (implies STABLE_EXTREMES)
    STABLE_DIFF_RATIOS = (
        1 << 7
    )  # Relative utility differences are stable (implies STABLE_ORDERING)

    # Special value for no stability (volatile - depends on state)
    VOLATILE = 0

    @property
    def is_session_independent(self) -> bool:
        """Check if the ufun does not depend on session details (NMI)."""
        return bool(self & Stability.SESSION_INDEPENDENT)

    @property
    def is_state_independent(self) -> bool:
        """Check if the ufun does not depend on state variables."""
        return bool(self & Stability.STATE_INDEPENDENT)

    @property
    def is_stationary(self) -> bool:
        """Check if fully stationary (all flags set, session and state independent)."""
        return self == STATIONARY

    @property
    def is_volatile(self) -> bool:
        """Check if volatile (no stability guarantees, value == 0)."""
        return self == Stability.VOLATILE

    @property
    def is_session_dependent(self) -> bool:
        """Check if the ufun depends on session details (NMI)."""
        return not self.is_session_independent

    @property
    def is_state_dependent(self) -> bool:
        """Check if the ufun depends on state variables."""
        return not self.is_state_independent

    @property
    def has_stable_min(self) -> bool:
        """Check if minimum utility is stable."""
        return bool(self & Stability.STABLE_MIN)

    @property
    def has_stable_max(self) -> bool:
        """Check if maximum utility is stable."""
        return bool(self & Stability.STABLE_MAX)

    @property
    def has_stable_extremes(self) -> bool:
        """Check if extreme (best/worst) outcomes are stable.

        Note: STABLE_ORDERING implies STABLE_EXTREMES, and STABLE_DIFF_RATIOS
        implies STABLE_ORDERING, so this returns True if any of these are set.
        """
        return bool(
            self
            & (
                Stability.STABLE_EXTREMES
                | Stability.STABLE_ORDERING
                | Stability.STABLE_DIFF_RATIOS
            )
        )

    @property
    def has_stable_reserved_value(self) -> bool:
        """Check if reserved value (relative) is stable."""
        return bool(self & Stability.STABLE_RESERVED_VALUE)

    @property
    def has_fixed_reserved_value(self) -> bool:
        """Check if reserved value (absolute) is fixed."""
        return bool(self & Stability.FIXED_RESERVED_VALUE)

    @property
    def has_stable_rational_outcomes(self) -> bool:
        """Check if rational outcomes remain rational."""
        return bool(self & Stability.STABLE_RATIONAL_OUTCOMES)

    @property
    def has_stable_irrational_outcomes(self) -> bool:
        """Check if irrational outcomes remain irrational."""
        return bool(self & Stability.STABLE_IRRATIONAL_OUTCOMES)

    @property
    def has_stable_ordering(self) -> bool:
        """Check if outcome ordering is stable."""
        return bool(self & Stability.STABLE_ORDERING)

    @property
    def has_stable_diff_ratios(self) -> bool:
        """Check if relative utility differences are stable."""
        return bool(self & Stability.STABLE_DIFF_RATIOS)

    @property
    def has_stable_scale(self) -> bool:
        """Check if scale is stable (min, max, and relative reserved value)."""
        return bool(self & STABLE_SCALE == STABLE_SCALE)

    @property
    def is_scale_invariant(self) -> bool:
        """Check if scale-invariant (stable diff ratios and relative reserved value)."""
        return bool(self & SCALE_INVARIANT == SCALE_INVARIANT)

    def __str__(self) -> str:
        """Return a human-readable string representation."""
        if self == STATIONARY:
            return "STATIONARY"
        if self == Stability.VOLATILE:
            return "VOLATILE"

        parts = []
        if self.is_session_independent:
            parts.append("SESSION_INDEPENDENT")
        if self.is_state_independent:
            parts.append("STATE_INDEPENDENT")
        if self.has_stable_min:
            parts.append("STABLE_MIN")
        if self.has_stable_max:
            parts.append("STABLE_MAX")
        # Only show STABLE_EXTREMES if set directly (not implied by STABLE_ORDERING)
        if bool(self & Stability.STABLE_EXTREMES) and not bool(
            self & Stability.STABLE_ORDERING
        ):
            parts.append("STABLE_EXTREMES")
        if self.has_stable_reserved_value:
            parts.append("STABLE_RESERVED_VALUE")
        if self.has_fixed_reserved_value:
            parts.append("FIXED_RESERVED_VALUE")
        if self.has_stable_rational_outcomes:
            parts.append("STABLE_RATIONAL_OUTCOMES")
        if self.has_stable_irrational_outcomes:
            parts.append("STABLE_IRRATIONAL_OUTCOMES")
        if self.has_stable_ordering:
            parts.append("STABLE_ORDERING")
        if self.has_stable_diff_ratios:
            parts.append("STABLE_DIFF_RATIOS")

        return " | ".join(parts) if parts else "VOLATILE"


# Module-level constants for convenient access
# Independence flags
SESSION_INDEPENDENT = Stability.SESSION_INDEPENDENT
STATE_INDEPENDENT = Stability.STATE_INDEPENDENT

# Stability flags
STABLE_MIN = Stability.STABLE_MIN
STABLE_MAX = Stability.STABLE_MAX
STABLE_EXTREMES = Stability.STABLE_EXTREMES
STABLE_RESERVED_VALUE = Stability.STABLE_RESERVED_VALUE
FIXED_RESERVED_VALUE = Stability.FIXED_RESERVED_VALUE
STABLE_RATIONAL_OUTCOMES = Stability.STABLE_RATIONAL_OUTCOMES
STABLE_IRRATIONAL_OUTCOMES = Stability.STABLE_IRRATIONAL_OUTCOMES
STABLE_ORDERING = Stability.STABLE_ORDERING
STABLE_DIFF_RATIOS = Stability.STABLE_DIFF_RATIOS

# Special value for volatile (no stability, depends on state)
VOLATILE = Stability.VOLATILE

# Predefined combinations
# STATIONARY = session-independent + state-independent + all stability flags
STATIONARY = (
    Stability.SESSION_INDEPENDENT
    | Stability.STATE_INDEPENDENT
    | Stability.STABLE_MIN
    | Stability.STABLE_MAX
    | Stability.STABLE_EXTREMES
    | Stability.STABLE_RESERVED_VALUE
    | Stability.FIXED_RESERVED_VALUE
    | Stability.STABLE_RATIONAL_OUTCOMES
    | Stability.STABLE_IRRATIONAL_OUTCOMES
    | Stability.STABLE_ORDERING
    | Stability.STABLE_DIFF_RATIOS
)

STABLE_SCALE = (
    Stability.STABLE_MIN | Stability.STABLE_MAX | Stability.STABLE_RESERVED_VALUE
)

SCALE_INVARIANT = Stability.STABLE_DIFF_RATIOS | Stability.STABLE_RESERVED_VALUE
