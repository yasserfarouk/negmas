"""Base class and core functionality for preference representations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from pathlib import Path
from negmas.common import PreferencesChange, PreferencesChangeType
from negmas.helpers import snake_case
from negmas.helpers.types import get_full_type_name
from negmas.outcomes import Outcome
from negmas.outcomes.common import check_one_at_most, os_or_none
from negmas.serialization import PYTHON_CLASS_IDENTIFIER, deserialize, serialize
from negmas.types import NamedObject

from .stability import Stability, STATIONARY

__all__ = ["Preferences"]

if TYPE_CHECKING:
    from negmas import Rational
    from negmas.outcomes.base_issue import Issue
    from negmas.outcomes.common import Outcome
    from negmas.outcomes.protocols import OutcomeSpace


class Preferences(NamedObject, ABC):
    """
    Base class for all preferences.

    Args:
        outcome_space: The outcome-space over which the preferences are defined
        stability: Stability flags describing which aspects of the preferences remain stable.
                   Default is STATIONARY (all stability flags set).
    """

    def __init__(
        self,
        *args,
        outcome_space: OutcomeSpace | None = None,
        issues: tuple[Issue] | None = None,
        outcomes: tuple[Outcome] | int | None = None,
        reserved_outcome: Outcome | None = None,
        owner: Rational | None = None,
        path: Path | None = None,
        stability: Stability | int = STATIONARY,
        **kwargs,
    ) -> None:
        """Initialize the instance.

        Args:
            *args: Additional positional arguments.
            outcome_space: The outcome space for these preferences.
            issues: Alternative to outcome_space - tuple of issues.
            outcomes: Alternative to outcome_space - tuple of outcomes or count.
            reserved_outcome: The outcome when no agreement is reached.
            owner: The rational agent that owns these preferences.
            path: Path to the file from which preferences were loaded.
            stability: Stability flags describing which aspects remain stable.
                       Default is STATIONARY. Can be an int (will be converted to Stability).
            **kwargs: Additional keyword arguments.
        """
        self._owner = owner
        check_one_at_most(outcome_space, issues, outcomes)
        super().__init__(*args, **kwargs)
        self.outcome_space = os_or_none(outcome_space, issues, outcomes)
        self.reserved_outcome = reserved_outcome  # type: ignore
        self._changes: list[PreferencesChange] = []
        self.path = path
        # Convert int to Stability if needed (for deserialization)
        self._stability = (
            Stability(stability) if isinstance(stability, int) else stability
        )

    @property
    def owner(self) -> Rational | None:
        """Returns the owner of these preferences.

        The owner is the negotiator currently using these preferences in a negotiation.

        Note:
            The owner is only valid during an active negotiation. It is set when
            the negotiator enters a negotiation (during ``_on_negotiation_start``)
            and cleared when the negotiation ends (during ``_on_negotiation_end``).
            Outside of an active negotiation, owner will be ``None``.
        """
        return self._owner

    @owner.setter
    def owner(self, value: Rational | None) -> None:
        """Sets the owner of these preferences.

        When owner is set to None (dissociation), notifies the previous owner
        via on_preferences_changed with Dissociated change type.

        Note:
            This is typically managed automatically by the negotiation framework.
            The owner is set when entering a negotiation and cleared when exiting.
        """
        old_owner = self._owner
        self._owner = value

        # Notify old owner when being dissociated (owner set to None)
        if old_owner is not None and value is None:
            old_owner.on_preferences_changed(
                [PreferencesChange(PreferencesChangeType.Dissociated)]
            )

    @property
    def stability(self) -> Stability:
        """Returns the stability flags for these preferences."""
        return self._stability

    @stability.setter
    def stability(self, value: Stability | int) -> None:
        """Sets the stability flags for these preferences.

        This setter:
        1. Skips if the new value equals the old value (no change)
        2. Saves the new stability value
        3. Determines the correct change type:
           - StabilityIncreased: all old bits are in new (and more in new)
           - StabilityReduced: all new bits are in old (and more in old)
           - StabilityChanged: bits differ in both directions
        4. Records the change in _changes list
        5. Notifies the owner (if set) via on_preferences_changed callback

        Args:
            value: The new stability flags (Stability enum or int)
        """
        _old = self._stability
        _new = Stability(value) if isinstance(value, int) else value

        # Skip if no change
        if _old == _new:
            return

        self._stability = _new

        # Determine the correct change type based on bit comparison
        old_has_extra = bool(_old & ~_new)  # old has bits not in new
        new_has_extra = bool(_new & ~_old)  # new has bits not in old

        if new_has_extra and not old_has_extra:
            change_type = PreferencesChangeType.StabilityIncreased
        elif old_has_extra and not new_has_extra:
            change_type = PreferencesChangeType.StabilityReduced
        else:
            change_type = PreferencesChangeType.StabilityChanged

        self._changes.append(
            PreferencesChange(change_type, data=dict(old=_old, new=_new))
        )

        # Only notify if owner is set (i.e., negotiator is in a negotiation)
        if self.owner is not None:
            self.owner.on_preferences_changed(
                [PreferencesChange(change_type, data=dict(old=_old, new=_new))]
            )

    def is_volatile(self) -> bool:
        """
        Does the utility of an outcome depend on factors outside the negotiation?

        Remarks:
            - A volatile preferences is one that can change even for the same mechanism state due to outside influence
            - Returns True if stability is VOLATILE (value = 0, no stability guarantees)
        """
        return self._stability.is_volatile

    @abstractmethod
    def is_not_worse(self, first: Outcome | None, second: Outcome | None) -> bool:
        """
        Is `first` at least as good as `second`
        """
        ...

    def is_session_dependent(self) -> bool:
        """
        Does the utility of an outcome depend on the `NegotiatorMechanismInterface`?

        Returns True if not all stability flags are set (i.e., not fully stationary).
        """
        return not self._stability.is_stationary

    def is_state_dependent(self) -> bool:
        """
        Does the utility of an outcome depend on the negotiation state?

        Returns True if not all stability flags are set (i.e., not fully stationary).
        """
        return not self._stability.is_stationary

    def is_stationary(self) -> bool:
        """Are the preferences stationary (i.e. repeated calls return the same value for any preferences comparison or evaluation method)?"""
        return self._stability.is_stationary

    # Stability-related properties for convenient access
    @property
    def has_stable_min(self) -> bool:
        """Check if minimum utility is stable."""
        return self._stability.has_stable_min

    @property
    def has_stable_max(self) -> bool:
        """Check if maximum utility is stable."""
        return self._stability.has_stable_max

    @property
    def has_stable_extremes(self) -> bool:
        """Check if extreme (best/worst) outcomes are stable."""
        return self._stability.has_stable_extremes

    @property
    def has_stable_reserved_value(self) -> bool:
        """Check if reserved value (relative) is stable."""
        return self._stability.has_stable_reserved_value

    @property
    def has_fixed_reserved_value(self) -> bool:
        """Check if reserved value (absolute) is fixed."""
        return self._stability.has_fixed_reserved_value

    @property
    def has_stable_rational_outcomes(self) -> bool:
        """Check if rational outcomes remain rational."""
        return self._stability.has_stable_rational_outcomes

    @property
    def has_stable_irrational_outcomes(self) -> bool:
        """Check if irrational outcomes remain irrational."""
        return self._stability.has_stable_irrational_outcomes

    @property
    def has_stable_ordering(self) -> bool:
        """Check if outcome ordering is stable."""
        return self._stability.has_stable_ordering

    @property
    def has_stable_diff_ratios(self) -> bool:
        """Check if relative utility differences are stable."""
        return self._stability.has_stable_diff_ratios

    @property
    def has_stable_scale(self) -> bool:
        """Check if scale is stable (min, max, and relative reserved value)."""
        return self._stability.has_stable_scale

    @property
    def is_scale_invariant(self) -> bool:
        """Check if scale-invariant (stable diff ratios and relative reserved value)."""
        return self._stability.is_scale_invariant

    def to_dict(
        self, python_class_identifier=PYTHON_CLASS_IDENTIFIER
    ) -> dict[str, Any]:
        """To dict.

        Args:
            python_class_identifier: Python class identifier.

        Returns:
            dict[str, Any]: The result.
        """
        d = {python_class_identifier: get_full_type_name(type(self))}
        return dict(
            **d,
            outcome_space=serialize(
                self.outcome_space, python_class_identifier=python_class_identifier
            ),
            reserved_outcome=self.reserved_outcome,
            stability=int(self._stability),
            name=self.name,
            id=self.id,
        )

    @classmethod
    def from_dict(cls, d, python_class_identifier=PYTHON_CLASS_IDENTIFIER):
        """From dict.

        Args:
            d: D.
            python_class_identifier: Python class identifier.
        """
        d.pop(python_class_identifier, None)
        d["outcome_space"] = deserialize(
            d.get("outcome_space", None),
            python_class_identifier=python_class_identifier,
        )
        if "stability" in d:
            d["stability"] = Stability(d["stability"])
        return cls(**d)

    def changes(self) -> list[PreferencesChange]:
        """
        Returns a list of changes to the preferences (if any) since last call.

        Remarks:
            - If the ufun is stationary, the return list will always be empty.
            - If the ufun is not stationary, the ufun itself is responsible for saving the
              changes in _changes whenever they happen.

        """
        if self.is_stationary():
            return []
        r = [_ for _ in self._changes]
        self.reset_changes()
        return r

    def reset_changes(self) -> None:
        """
        Will be called whenever we need to reset changes.
        """
        self._changes = []

    @property
    def type(self) -> str:
        """Returns the utility_function type.

        Each class inheriting from this ``UtilityFunction`` class will have its own type. The default type is the empty
        string.

        Examples:
            >>> from negmas.preferences import *
            >>> from negmas.outcomes import make_issue
            >>> print(
            ...     LinearAdditiveUtilityFunction(
            ...         (lambda x: x, lambda x: x), issues=[make_issue((0, 1), (0, 1))]
            ...     ).type
            ... )
            linear_additive
            >>> print(
            ...     MappingUtilityFunction(
            ...         [lambda x: x], issues=[make_issue((0.0, 1.0))]
            ...     ).type
            ... )
            mapping

        Returns:
            str: utility_function type
        """
        return snake_case(
            self.__class__.__name__.replace("Function", "").replace("Utility", "")
        )

    @property
    def base_type(self) -> str:
        """Returns the utility_function base type ignoring discounting and similar wrappings."""
        from .discounted import DiscountedUtilityFunction

        u = self
        while isinstance(u, DiscountedUtilityFunction):
            u = u.ufun
        return self.type

    def is_better(self, first: Outcome | None, second: Outcome | None) -> bool:
        """
        Compares two offers using the `ufun` returning whether the first is strictly better than the second

        Args:
            first: First outcome to be compared
            second: Second outcome to be compared

        Remarks:

            - Should raise `ValueError` if the comparison cannot be done

        """
        return self.is_not_worse(first, second) and not self.is_not_worse(second, first)

    def is_equivalent(self, first: Outcome | None, second: Outcome | None) -> bool:
        """
        Compares two offers using the `ufun` returning whether the first is strictly equivelent than the second

        Args:
            first: First outcome to be compared
            second: Second outcome to be compared

        Remarks:

            - Should raise `ValueError` if the comparison cannot be done

        """
        return self.is_not_worse(first, second) and self.is_not_worse(second, first)

    def is_not_better(self, first: Outcome | None, second: Outcome | None) -> bool:
        """
        Compares two offers using the `ufun` returning whether the first is worse or equivalent than the second

        Args:
            first: First outcome to be compared
            second: Second outcome to be compared

        Remarks:

            - Should raise `ValueError` if the comparison cannot be done

        """
        return self.is_not_worse(second, first)

    def is_worse(self, first: Outcome | None, second: Outcome | None) -> bool:
        """
        Compares two offers using the `ufun` returning whether the first is strictly worse than the second

        Args:
            first: First outcome to be compared
            second: Second outcome to be compared

        Remarks:

            - Should raise `ValueError` if the comparison cannot be done

        """
        return not self.is_not_worse(first, second)
