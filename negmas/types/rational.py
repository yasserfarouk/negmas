"""
Base class for all rational agents in NegMAS.

A rational agent is one that has some preferences.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from ..common import PreferencesChange
from .named import NamedObject

if TYPE_CHECKING:
    from ..outcomes import Outcome
    from ..preferences import (
        BaseUtilityFunction,
        Preferences,
        ProbUtilityFunction,
        UtilityFunction,
    )

__all__ = ["Rational"]


class Rational(NamedObject):
    """
    A rational object is an object that can have preferences.


    Args:
        name: Object name. Used for printing and logging but not internally by the system
        preferences: An optional preferences to attach to the object
        ufun: An optinoal utility function (overrides preferences if given)
        id: Object ID (must be unique in the whole system). If None, the system will generate it


    Remarks:

        - Rational objects can be created with a predefined preferences or without them
          but the preferences sould be set for any kind of rational object involved in
          negotiations **before** the negotiation starts and should not be changed after
          that.
        - `ufun` is aliased to `preferences`

    """

    def _set_pref_owner(self):
        if not self._preferences:
            return
        # todo: re-enable this after making sure scml does not raise it
        # we here assume that two entities that share an ID can share preferences without warning
        # if (
        #     self._preferences.owner is not None
        #     and self._preferences.owner.id != self.id
        # ):
        #     warnings.warn(
        #         f"Entity {self.name} ({self.__class__.__name__}) is "
        #         f"assigned preferences belonging to another entity "
        #         f"({self._preferences.owner.name} of type {self.__class__.__name__})!!",
        #         warnings.NegmasDoubleAssignmentWarning,
        #     )

        self._preferences.owner = self

    def on_preferences_changed(self, changes: list[PreferencesChange]):
        """
        Called to inform the entity that its ufun has changed.

        Args:
            changes: An ordered list of changes that happened.

        Remarks:

            - You MUST call the super() version of this function either before or after your code when you are overriding
              it.
            - The most general form of change is `PreferencesChange.General` which indicates that you cannot trust anything you knew about the ufun anymore
        """
        _ = changes

    def set_preferences(
        self, value: Preferences | None, force=False
    ) -> Preferences | None:
        """
        Sets tha utility function/Preferences.

        Args:

            value: The value to set to
            force: If true, `on_preferecnes_changed()` will always be called even if `value` == `self.preferences`

        """
        if value == self._preferences:
            if force:
                self.on_preferences_changed([PreferencesChange()])
            return self._preferences
        old = self._preferences
        self._preferences = value
        if value and value.owner != self:
            self._set_pref_owner()
        if id(value) != id(old):
            self.on_preferences_changed([PreferencesChange()])
        return self._preferences

    def __init__(
        self,
        name: str | None = None,
        preferences: Preferences | None = None,
        ufun: BaseUtilityFunction | None = None,
        id: str | None = None,
        type_name: str | None = None,
    ):
        super().__init__(name, type_name=type_name, id=id)
        if ufun:
            preferences = ufun
        self._init_preferences = preferences
        self._preferences = None
        if preferences is not None:
            self.set_preferences(preferences)

    @property
    def preferences(self) -> Preferences | None:
        """The utility function attached to that object"""
        return self._preferences

    @property
    def crisp_ufun(self) -> UtilityFunction | None:
        """Returns the preferences if it is a CrispUtilityFunction else None"""
        from negmas.preferences import UtilityFunction

        return (
            self._preferences
            if isinstance(self._preferences, UtilityFunction)
            else None
        )

    @crisp_ufun.setter
    def crisp_ufun(self, v: UtilityFunction):
        from negmas.preferences import UtilityFunction

        if not isinstance(v, UtilityFunction):
            raise ValueError(f"Cannot assign a {type(v)} to crisp_ufun")
        self.set_preferences(v)

    @property
    def prob_ufun(self) -> ProbUtilityFunction | None:
        """Returns the preferences if it is a ProbUtilityFunction else None"""
        from negmas.preferences import ProbUtilityFunction

        return (
            self._preferences
            if isinstance(self._preferences, ProbUtilityFunction)
            else None
        )

    @prob_ufun.setter
    def prob_ufun(self, v: ProbUtilityFunction):
        from negmas.preferences import ProbUtilityFunction

        if not isinstance(v, ProbUtilityFunction):
            raise ValueError(f"Cannot assign a {type(v)} to prob_ufun")
        self.set_preferences(v)

    @property
    def ufun(self) -> BaseUtilityFunction | None:
        """Returns the preferences if it is a `BaseUtilityFunction` else None"""
        from ..preferences import BaseUtilityFunction

        if self._preferences is None:
            return None
        if isinstance(self._preferences, BaseUtilityFunction):
            return self._preferences
        raise ValueError(
            f"Preferences are not for type `BaseUtilityFunction` ({self._preferences.__class__.__name__})"
        )

    @ufun.setter
    def ufun(self, v: BaseUtilityFunction):
        self.set_preferences(v)

    @property
    def has_preferences(self) -> bool:
        """Does the entity has an associated ufun?"""
        return self._preferences is not None

    @property
    def has_ufun(self) -> bool:
        """Does the entity has an associated ufun?"""
        return self.ufun is not None

    @property
    def has_cardinal_preferences(self) -> bool:
        """Does the entity has an associated ufun?"""
        from negmas.preferences.protocols import CardinalCrisp

        return self._preferences is not None and isinstance(
            self._preferences, CardinalCrisp
        )

    @property
    def reserved_outcome(self) -> Outcome | None:
        """
        Reserved outcome is the outcome that will be realized by default for this agent.

        Remarks:

            - Reserved outcomes are defined for `OrdinalPreferences`.

        See Also:
            `reserved_value`

        """
        from negmas.preferences import HasReservedOutcome

        if self._preferences is None or not isinstance(
            self._preferences, HasReservedOutcome
        ):
            return None
        return self._preferences.reserved_outcome

    @property
    def reserved_value(self) -> float:
        """
        Reserved value is what the entity gets if no agreement is reached in the negotiation.

        The reserved value can either be explicity defined for the ufun or it can be the output of the ufun
        for `None` outcome.

        """
        from negmas.preferences import HasReservedValue

        if self._preferences is None or not isinstance(
            self._preferences, HasReservedValue
        ):
            return float("nan")
        return self._preferences.reserved_value
