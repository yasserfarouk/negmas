"""Meta-negotiator that combines multiple full negotiators."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Any, Iterable

from negmas.common import (
    MechanismState,
    NegotiatorMechanismInterface,
    PreferencesChange,
)

from .negotiator import Negotiator

if TYPE_CHECKING:
    from negmas.preferences import BaseUtilityFunction, Preferences

__all__ = ["MetaNegotiator"]


class MetaNegotiator(Negotiator):
    """
    A meta-negotiator that contains and delegates to multiple full `Negotiator` objects.

    Unlike `ModularNegotiator` which uses `Component` objects (behavior pieces),
    `MetaNegotiator` works with complete `Negotiator` instances, allowing for
    ensemble or delegation-based negotiation strategies.

    The meta-negotiator delegates all lifecycle callbacks to its sub-negotiators.
    Subclasses must implement aggregation strategies for actions like `propose`
    and `respond` (in protocol-specific subclasses like `GBMetaNegotiator`).

    Args:
        negotiators: An iterable of `Negotiator` instances to manage.
        negotiator_names: Optional names for the negotiators.
        share_ufun: If True (default), sub-negotiators will share the parent's ufun.
        share_nmi: If True (default), sub-negotiators will receive the parent's NMI on join.
        *args: Additional positional arguments passed to the base class.
        **kwargs: Additional keyword arguments passed to the base class.

    Remarks:
        - Sub-negotiators share the parent's NMI and ufun when `share_nmi` and
          `share_ufun` are True (the defaults).
        - All lifecycle callbacks are delegated to all sub-negotiators.
        - Protocol-specific subclasses (e.g., `GBMetaNegotiator`) must implement
          aggregation strategies for `propose` and `respond`.
    """

    def __init__(
        self,
        *args,
        negotiators: Iterable[Negotiator],
        negotiator_names: Iterable[str] | None = None,
        share_ufun: bool = True,
        share_nmi: bool = True,
        **kwargs,
    ):
        """Initialize the MetaNegotiator with sub-negotiators.

        Args:
            *args: Positional arguments for the base Negotiator.
            negotiators: The sub-negotiators to manage.
            negotiator_names: Optional names for the sub-negotiators.
            share_ufun: Whether sub-negotiators should share the parent's ufun.
            share_nmi: Whether sub-negotiators should receive the parent's NMI.
            **kwargs: Keyword arguments for the base Negotiator.
        """
        super().__init__(*args, **kwargs)
        self._negotiators: list[Negotiator] = []
        self.__negotiator_map: dict[str, int] = {}
        self._share_ufun = share_ufun
        self._share_nmi = share_nmi

        for neg, name in zip(
            negotiators,
            negotiator_names if negotiator_names else itertools.repeat(None),
        ):
            self.add_negotiator(neg, name=name)

    def add_negotiator(
        self, negotiator: Negotiator, name: str | None = None, index: int = -1
    ) -> None:
        """Add a sub-negotiator at the given index.

        Args:
            negotiator: The negotiator to add.
            name: Optional name for the negotiator.
            index: Position to insert at. If negative, appends at the end.
        """
        if index < 0:
            index = len(self._negotiators)
        self._negotiators.insert(index, negotiator)
        if not name:
            name = negotiator.name or str(index)
        self.__negotiator_map[name] = len(self._negotiators) - 1

    def remove_negotiator_at(self, index: int) -> Negotiator | None:
        """Remove and return the negotiator at the given index.

        Args:
            index: The index of the negotiator to remove.

        Returns:
            The removed negotiator, or None if index is out of bounds.
        """
        if 0 <= index < len(self._negotiators):
            neg = self._negotiators.pop(index)
            # Rebuild the map
            self.__negotiator_map = {
                name: i
                for i, (name, _) in enumerate(
                    (name, idx)
                    for name, idx in self.__negotiator_map.items()
                    if idx != index
                )
            }
            return neg
        return None

    def remove_negotiator(self, name: str) -> Negotiator | None:
        """Remove and return the negotiator with the given name.

        Args:
            name: The name of the negotiator to remove.

        Returns:
            The removed negotiator, or None if not found.
        """
        if name in self.__negotiator_map:
            return self.remove_negotiator_at(self.__negotiator_map[name])
        return None

    def get_negotiator(self, name: str) -> Negotiator | None:
        """Get a sub-negotiator by name.

        Args:
            name: The name of the negotiator.

        Returns:
            The negotiator, or None if not found.
        """
        if name in self.__negotiator_map:
            return self._negotiators[self.__negotiator_map[name]]
        return None

    @property
    def negotiators(self) -> tuple[Negotiator, ...]:
        """Return the tuple of sub-negotiators.

        Returns:
            A tuple of all sub-negotiators.
        """
        return tuple(self._negotiators)

    @property
    def negotiator_names(self) -> tuple[str, ...]:
        """Return the names of all sub-negotiators.

        Returns:
            A tuple of negotiator names.
        """
        return tuple(self.__negotiator_map.keys())

    # Lifecycle callbacks - delegate to all sub-negotiators

    def on_preferences_changed(self, changes: list[PreferencesChange]) -> None:
        """Notify all sub-negotiators of preference changes.

        Args:
            changes: The list of preference changes.
        """
        for neg in self._negotiators:
            neg.on_preferences_changed(changes)

    def join(
        self,
        nmi: NegotiatorMechanismInterface,
        state: MechanismState,
        *,
        preferences: Preferences | None = None,
        ufun: BaseUtilityFunction | None = None,
        role: str = "negotiator",
    ) -> bool:
        """Join a negotiation and optionally have sub-negotiators join too.

        When `share_nmi` is True, sub-negotiators will also join the negotiation
        with the same NMI. When `share_ufun` is True, they will use the parent's
        ufun.

        Args:
            nmi: The negotiator-mechanism interface.
            state: The current mechanism state.
            preferences: Optional preferences for this negotiator.
            ufun: Optional utility function (overrides preferences).
            role: The role in the negotiation.

        Returns:
            True if successfully joined, False otherwise.
        """
        joined = super().join(nmi, state, preferences=preferences, ufun=ufun, role=role)
        if not joined:
            return False

        if self._share_nmi:
            # Have sub-negotiators join with shared NMI and optionally shared ufun
            sub_ufun = self.ufun if self._share_ufun else None
            sub_prefs = self.preferences if self._share_ufun and not sub_ufun else None
            for neg in self._negotiators:
                # Sub-negotiators join but we don't fail the parent if they can't
                neg.join(nmi, state, preferences=sub_prefs, ufun=sub_ufun, role=role)

        return True

    def on_negotiation_start(self, state: MechanismState) -> None:
        """Notify all sub-negotiators that negotiation has started.

        Args:
            state: The current mechanism state.
        """
        for neg in self._negotiators:
            neg.on_negotiation_start(state)

    def on_round_start(self, state: MechanismState) -> None:
        """Notify all sub-negotiators that a round has started.

        Args:
            state: The current mechanism state.
        """
        for neg in self._negotiators:
            neg.on_round_start(state)

    def on_round_end(self, state: MechanismState) -> None:
        """Notify all sub-negotiators that a round has ended.

        Args:
            state: The current mechanism state.
        """
        for neg in self._negotiators:
            neg.on_round_end(state)

    def on_leave(self, state: MechanismState) -> None:
        """Notify all sub-negotiators that we're leaving the negotiation.

        Args:
            state: The current mechanism state.
        """
        for neg in self._negotiators:
            neg.on_leave(state)
        super().on_leave(state)

    def on_negotiation_end(self, state: MechanismState) -> None:
        """Notify all sub-negotiators that negotiation has ended.

        Args:
            state: The current mechanism state.
        """
        for neg in self._negotiators:
            neg.on_negotiation_end(state)

    def on_mechanism_error(self, state: MechanismState) -> None:
        """Notify all sub-negotiators of a mechanism error.

        Args:
            state: The current mechanism state.
        """
        for neg in self._negotiators:
            neg.on_mechanism_error(state)

    def before_death(self, cntxt: dict[str, Any]) -> bool:
        """Ask all sub-negotiators if they accept death.

        Returns False if any sub-negotiator returns False, but the controller
        can still force-kill the meta-negotiator.

        Args:
            cntxt: Context information about the death.

        Returns:
            True if all sub-negotiators agree to death, False otherwise.
        """
        result = super().before_death(cntxt)
        for neg in self._negotiators:
            if not neg.before_death(cntxt):
                result = False
        return result

    def cancel(self, reason=None) -> None:
        """Cancel processing in all sub-negotiators.

        Args:
            reason: Optional reason for the cancellation.
        """
        super().cancel(reason)
        for neg in self._negotiators:
            neg.cancel(reason)
