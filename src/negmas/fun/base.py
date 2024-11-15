from typing import Any, Protocol, Sequence

from negmas.common import (
    MechanismState,
    NegotiatorMechanismInterface,
    PreferencesChange,
)
from negmas.preferences.preferences import Preferences
from negmas.sao.common import SAOResponse, SAOState


class ReactiveStrategy(Protocol):
    """A python protocol defining the signature of a negotiation strategy.

    Args:
        nmi: The `NegotiatorMechanismInterface` storing static information about the negotiation (e.g. outcome-space) and provides services to the strategy (e.g. sampling random outcomes).
        state: The current `MechanismState` defining dynamic aspects of the negotiation (e.g. number of negotiators, relative time, current step, ...)
        preferences: The `Preferences` of the strategy (it can be ordinal or cardinal).
        changes: An optional list of changes that happened in the `preferences` since the last call. This is only used when the preferences change **during** the negotiation.
        kwargs: Any protocol-specific arguments

    Returns:
        - Protocol-defined return value
    """

    def __call__(
        self,
        nmi: NegotiatorMechanismInterface,
        state: MechanismState,
        preferences: Preferences,
        changes: Sequence[PreferencesChange] = tuple(),
        **kwargs,
    ) -> Any:
        ...


class AOStrategy(Protocol):
    """A python protocol defining an alternating-offers policy (returns a counter offer)

    Args:
        nmi: The `NegotiatorMechanismInterface` storing static information about the negotiation (e.g. outcome-space) and provides services to the strategy (e.g. sampling random outcomes).
        preferences: The `Preferences` of the strategy (it can be ordinal or cardinal).
        state: The current `MechanismState` defining dynamic aspects of the negotiation (e.g. number of negotiators, relative time, current step, ...)
        changes: An optional list of changes that happened in the `preferences` since the last call. This is only used when the preferences change **during** the negotiation.
        offer: The offer to counter

    Returns:
        An `SAOResponse`
    """

    def __call__(
        self,
        nmi: NegotiatorMechanismInterface,
        state: SAOState,
        preferences: Preferences,
        changes: Sequence[PreferencesChange] = tuple(),
    ) -> SAOResponse:
        ...
