"""Mixins for utility function behavior classification (volatility, stationarity, state-dependence).

.. deprecated::
    These mixins are deprecated and will be removed in a future version.
    Use the ``stability`` parameter directly when constructing utility functions instead:

    - Instead of ``VolatileUFunMixin``, pass ``stability=VOLATILE``
    - Instead of ``StationaryMixin``, use the default (``stability=STATIONARY``)
    - Instead of ``SessionDependentUFunMixin`` or ``StateDependentUFunMixin``,
      pass an appropriate ``stability`` value with the relevant independence flags cleared

These mixins provide backward compatibility with the inheritance-based approach.
The preferred approach is to use the ``stability`` parameter when constructing utility functions.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from negmas.common import Distribution, Value

from .stability import (
    SESSION_INDEPENDENT,
    STATE_INDEPENDENT,
    STATIONARY,
    VOLATILE,
    Stability,
)

if TYPE_CHECKING:
    from negmas.common import MechanismState, NegotiatorMechanismInterface
    from negmas.negotiators import Negotiator
    from negmas.outcomes.common import Outcome
    from negmas.preferences import UtilityFunction

__all__ = [
    "VolatileUFunMixin",
    "SessionDependentUFunMixin",
    "StateDependentUFunMixin",
    "StationaryMixin",
]


class VolatileUFunMixin:
    """
    Indicates that the ufun is volatile.

    .. deprecated::
        Use ``stability=VOLATILE`` parameter instead::

            # Old way (deprecated):
            class MyUfun(VolatileUFunMixin, BaseUtilityFunction):
                pass


            # New way:
            ufun = MyUtilityFunction(..., stability=VOLATILE)

    This mixin sets stability to VOLATILE (no stability guarantees).
    The preferred approach is to pass ``stability=VOLATILE`` to the constructor.
    """

    _stability: Stability

    def __init__(self, *args, **kwargs):
        """Initialize with VOLATILE stability."""
        # Force stability to VOLATILE
        kwargs["stability"] = VOLATILE
        super().__init__(*args, **kwargs)

    def is_volatile(self):
        """Check if volatile."""
        return True


class SessionDependentUFunMixin:
    """
    Indicates that the ufun is session-dependent (i.e. utility value of outcomes depend on the NMI).

    .. deprecated::
        Use appropriate ``stability`` parameter instead::

            # Old way (deprecated):
            class MyUfun(SessionDependentUFunMixin, BaseUtilityFunction):
                def eval_on_session(self, offer, nmi): ...


            # New way: Clear SESSION_INDEPENDENT flag in constructor
            ufun = MyUtilityFunction(..., stability=STATIONARY & ~SESSION_INDEPENDENT)

    This mixin clears the SESSION_INDEPENDENT flag from stability while preserving
    other stability guarantees. Being session-dependent does not imply volatility -
    the ufun may still have stable ordering, diff ratios, etc.

    The preferred approach is to pass an appropriate ``stability`` value to the constructor.
    """

    _stability: Stability

    def __init__(self, *args, **kwargs):
        """Initialize and clear SESSION_INDEPENDENT flag from stability."""
        super().__init__(*args, **kwargs)
        # Clear SESSION_INDEPENDENT flag while preserving other stability flags
        self._stability = Stability(self._stability & ~SESSION_INDEPENDENT)

    @abstractmethod
    def eval_on_session(
        self, offer: Outcome, nmi: NegotiatorMechanismInterface | None = None
    ) -> Distribution:
        """Evaluates the offer given a session"""

    def eval(self, offer: Outcome) -> Distribution:
        """Eval.

        Args:
            offer: Offer being considered.

        Returns:
            Distribution: The result.
        """
        if not self.owner or not self.owner.nmi:
            return self.eval_on_session(offer, None)
        self.owner: Negotiator
        return self.eval_on_session(offer, self.owner.nmi)

    def is_session_dependent(self):
        """
        Does the utiltiy of an outcome depend on the `NegotiatorMechanismInterface`?
        """
        return True

    def is_state_dependent(self):
        """
        Does the utiltiy of an outcome depend on the negotiation state?
        """
        return False


class StateDependentUFunMixin:
    """
    Indicates that the ufun is state-dependent (i.e. utility value of outcomes depend on the mechanism state).

    .. deprecated::
        Use appropriate ``stability`` parameter instead::

            # Old way (deprecated):
            class MyUfun(StateDependentUFunMixin, BaseUtilityFunction):
                def eval_on_state(self, offer, nmi, state): ...


            # New way: Clear STATE_INDEPENDENT and SESSION_INDEPENDENT flags
            ufun = MyUtilityFunction(
                ..., stability=STATIONARY & ~STATE_INDEPENDENT & ~SESSION_INDEPENDENT
            )

    This mixin clears the STATE_INDEPENDENT flag from stability while preserving
    other stability guarantees. Being state-dependent does not imply volatility -
    the ufun may still have stable ordering, diff ratios, etc.

    Note: State-dependent ufuns are also implicitly session-dependent since state
    is accessed through the session, so SESSION_INDEPENDENT is also cleared.

    The preferred approach is to pass an appropriate ``stability`` value to the constructor.
    """

    _stability: Stability

    def __init__(self, *args, **kwargs):
        """Initialize and clear STATE_INDEPENDENT flag from stability."""
        super().__init__(*args, **kwargs)
        # Clear STATE_INDEPENDENT flag while preserving other stability flags
        # Also clear SESSION_INDEPENDENT since state-dependent implies session-dependent
        self._stability = Stability(
            self._stability & ~STATE_INDEPENDENT & ~SESSION_INDEPENDENT
        )

    @abstractmethod
    def eval_on_state(
        self,
        offer: Outcome,
        nmi: NegotiatorMechanismInterface | None = None,
        state: MechanismState | None = None,
    ) -> Value:
        """Evaluates the offer given a session and state"""

    def eval(self, offer: Outcome) -> Value:
        """Eval.

        Args:
            offer: Offer being considered.

        Returns:
            Distribution: The result.
        """
        if not self.owner or not self.owner.nmi:
            return self.eval_on_state(offer, None, None)
        self.owner: Negotiator
        return self.eval_on_state(offer, self.owner.nmi, self.owner.nmi.state)

    def is_session_dependent(self):
        """
        Does the utiltiy of an outcome depend on the `NegotiatorMechanismInterface`?
        """
        return True

    def is_state_dependent(self):
        """
        Does the utiltiy of an outcome depend on the negotiation state?
        """
        return True


class StationaryMixin:
    """
    Indicates that the ufun is stationary which means that it is not session or state dependent and not volatile.

    Negotiators using this type of ufuns can assume that if they call it twice with the same outcome, it will always return the same value.

    .. deprecated::
        This mixin is no longer needed since ``STATIONARY`` is the default stability.
        Simply don't pass any ``stability`` parameter, or use the default::

            # Old way (deprecated):
            class MyUfun(StationaryMixin, UtilityFunction):
                pass


            # New way (stability=STATIONARY is the default):
            ufun = MyUtilityFunction(...)  # Already stationary by default

    This mixin sets stability to STATIONARY (all stability flags set) by default.
    The preferred approach is to pass ``stability=STATIONARY`` to the constructor (which is the default).

    Note: When using this mixin, if you pass a custom ``stability`` parameter to the constructor,
    the stability-related methods (is_stationary, is_volatile, etc.) will reflect that custom stability.
    """

    _stability: Stability

    def __init__(self, *args, **kwargs):
        """Initialize with STATIONARY stability if not explicitly provided."""
        # Set stability to STATIONARY if not explicitly provided
        if "stability" not in kwargs:
            kwargs["stability"] = STATIONARY
        super().__init__(*args, **kwargs)

    def to_stationary(self) -> UtilityFunction:
        """To stationary.

        Returns:
            UtilityFunction: The result.
        """
        return self  # type: ignore

    # Note: is_session_dependent, is_volatile, is_state_dependent, is_stationary
    # are inherited from the Preferences base class which uses the _stability attribute.
    # This allows custom stability to be passed via constructor while still
    # defaulting to STATIONARY when using this mixin.

    # @lru_cache(maxsize=100)
    # def eval_normalized(
    #     self: BaseUtilityFunction, # type: ignore
    #     offer: Outcome | None,
    #     above_reserve: bool = True,
    #     expected_limits: bool = True,
    # ) -> Value:
    #     """
    #     Caches the top 100 results from the ufun because we know they are never going to change.
    #     """
    #     return super().eval_normalized(offer, above_reserve, expected_limits)
