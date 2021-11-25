"""
Base class for all rational agents in NegMAS.

A rational agent is one that has some preferences.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .named import NamedObject

if TYPE_CHECKING:
    from ..outcomes import Outcome
    from ..preferences import Preferences

__all__ = ["Rational"]


class Rational(NamedObject):
    """
    A rational object is an object that can have preferences.


    Args:
        name: Object name. Used for printing and logging but not internally by the system
        preferences: An optional preferences to attach to the object


    Remarks:

        - Rational objects can be created with a predefined preferences or without them
          but the preferences sould be set for any kind of rational object involved in
          negotiations **before** the negotiation starts and should not be changed after
          that.
        - `ufun` is aliased to `preferences`

    """

    def __init__(
        self, name: str = None, preferences: Preferences | None = None, id: str = None
    ):
        super().__init__(name, id=id)
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

    ufun = preferences
    """An alias to preferences"""

    @property
    def has_preferences(self) -> bool:
        """Does the entity has an associated ufun?"""
        return self._preferences is not None

    @property
    def has_cardinal_preferences(self) -> bool:
        """Does the entity has an associated ufun?"""
        from negmas.preferences import CardinalPreferences

        return self._preferences is not None and isinstance(
            self._preferences, CardinalPreferences
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
        from negmas.preferences import CardinalPreferences

        if self._preferences is None or isinstance(
            self._preferences, CardinalPreferences
        ):
            return None
        return self._preferences.reserved_outocme

    @property
    def reserved_value(self) -> float:
        """
        Reserved value is what the entity gets if no agreement is reached in the negotiation.

        The reserved value can either be explicity defined for the ufun or it can be the output of the ufun
        for `None` outcome.

        """
        from negmas.preferences import CardinalPreferences

        if self._preferences is None or not isinstance(
            self._preferences, CardinalPreferences
        ):
            return float("nan")
        return self._preferences.reserved_value

    def on_preferences_changed(self):
        """
        Called to inform the entity that its ufun has changed.

        Remarks:

            - You MUST call the super() version of this function either before or after your code when you are overriding
              it.
        """
        self._preferences_modified = False
