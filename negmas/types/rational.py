"""
Base class for all rational agents in NegMAS.

A rational agent is one that has some preferences.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from negmas.preferences.protocols import UFunCrisp, UFunProb

from ..preferences import BaseUtilityFunction, Preferences
from .named import NamedObject

if TYPE_CHECKING:
    from ..outcomes import Outcome

__all__ = ["Rational"]


class Rational(NamedObject):
    """
    A rational object is an object that can have preferences.


    Args:
        name: Object name. Used for printing and logging but not internally by the system
        preferences: An optional preferences to attach to the object
        ufun: An optinoal utility function (overrides preferences if given)


    Remarks:

        - Rational objects can be created with a predefined preferences or without them
          but the preferences sould be set for any kind of rational object involved in
          negotiations **before** the negotiation starts and should not be changed after
          that.
        - `ufun` is aliased to `preferences`

    """

    def __init__(
        self,
        name: str = None,
        preferences: Preferences | None = None,
        ufun: BaseUtilityFunction | None = None,
        id: str = None,
    ):
        super().__init__(name, id=id)
        if ufun:
            preferences = ufun
        self._preferences = preferences
        self._init_preferences = preferences
        self._preferences_modified = preferences is not None

    @property
    def preferences(self):
        """The utility function attached to that object"""
        return self._preferences

    @preferences.setter
    def preferences(self, value: Preferences):
        """Sets tha utility function."""
        self._preferences = value
        self._preferences_modified = True
        self.on_preferences_changed()

    @property
    def crisp_ufun(self) -> UFunCrisp | None:
        """Returns the preferences if it is a CrispUtilityFunction else None"""
        return self._preferences if isinstance(self._preferences, UFunCrisp) else None

    @crisp_ufun.setter
    def crisp_ufun(self, v: UFunCrisp):
        if not isinstance(v, UFunCrisp):
            raise ValueError(f"Cannot assign a {type(v)} to crisp_ufun")
        self._preferences = v

    @property
    def prob_ufun(self) -> UFunProb | None:
        """Returns the preferences if it is a ProbUtilityFunction else None"""
        return self._preferences if isinstance(self._preferences, UFunProb) else None

    @prob_ufun.setter
    def prob_ufun(self, v: UFunProb):
        if not isinstance(v, UFunProb):
            raise ValueError(f"Cannot assign a {type(v)} to prob_ufun")
        self._preferences = v

    @property
    def ufun(self) -> BaseUtilityFunction | None:
        """Returns the preferences if it is a UtilityFunction else None"""
        return (
            self._preferences
            if isinstance(self._preferences, BaseUtilityFunction)
            else None
        )

    @ufun.setter
    def ufun(self, v: BaseUtilityFunction):
        self._preferences = v

    @property
    def has_preferences(self) -> bool:
        """Does the entity has an associated ufun?"""
        return self._preferences is not None

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

    def reset_preferences_changed_flag(self):
        """Called to reset the internal preferences changed flag after processing a change"""
        self._preferences_modified = False

    def on_preferences_changed(self):
        """
        Called to inform the entity that its ufun has changed.

        Remarks:

            - You MUST call the super() version of this function either before or after your code when you are overriding
              it.
        """
        self.reset_preferences_changed_flag()
