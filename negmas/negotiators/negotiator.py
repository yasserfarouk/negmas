from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Any

import negmas.warnings as warnings
from negmas.common import (
    MechanismState,
    NegotiatorMechanismInterface,
    PreferencesChange,
)
from negmas.events import Notifiable, Notification
from negmas.types import Rational

if TYPE_CHECKING:
    from negmas.outcomes import Outcome
    from negmas.preferences import BaseUtilityFunction, Preferences
    from negmas.situated import Agent

    from .controller import Controller

__all__ = [
    "Negotiator",
]


class Negotiator(Rational, Notifiable, ABC):
    """
    Abstract negotiation agent. Base class for all negotiators

    Args:

           name: Negotiator name. If not given it is assigned by the system (unique 16 characters).
           preferences: The preferences of the agent (pass either this or ufun)
           ufun: The ufun of the agent (overrides preferences if given)
           parent: The `Controller` that controls this neogtiator (if any)
           owner: The `Agent` that own this negotiator (if any)
           id: The unique ID of the negotiator

       Returns:
           bool: True if participating in the given negotiation (or any negotiation if it was None)

       Remarks:
           - `ufun` overrides `preferences`. You need to pass only one of them

    """

    def __init__(
        self,
        name: str | None = None,
        preferences: Preferences | None = None,
        ufun: BaseUtilityFunction | None = None,
        parent: Controller | None = None,
        owner: Agent | None = None,
        id: str | None = None,
        type_name: str | None = None,
        private_info: dict[str, Any] | None = None,
    ) -> None:
        if ufun is not None:
            preferences = ufun
        self.__parent = parent
        self._capabilities = {"enter": True, "leave": True, "ultimatum": True}
        self._nmi: NegotiatorMechanismInterface | None = None
        self._initial_state = None
        self._role = None
        self.__owner = owner
        super().__init__(
            name=name, ufun=None, preferences=None, id=id, type_name=type_name
        )
        self._preferences = preferences
        self._private_info = private_info if private_info else dict()
        self.__saved_pref_os = None
        self.__saved_prefs = None

    @property
    def ami(self) -> NegotiatorMechanismInterface:
        warnings.deprecated(
            "`ami` is depricated and will not be a member of `Negotiator` in the future. Use `nmi` instead."
        )
        return self._nmi  # type: ignore

    @property
    def opponent_ufun(self) -> BaseUtilityFunction | None:
        return self.private_info.get("opponent_ufun", None)

    @property
    def nmi(self) -> NegotiatorMechanismInterface:
        return self._nmi  # type: ignore

    @property
    def owner(self) -> Agent | None:
        """Returns the owner agent of the negotiator"""
        return self.__owner

    @owner.setter
    def owner(self, owner):
        """Sets the owner"""
        self.__owner = owner

    def _set_pref_os(self):
        if self.nmi and self._preferences:
            self.__saved_pref_os = self._preferences.outcome_space
            self.__saved_prefs = self._preferences
            self._preferences.outcome_space = self.nmi.outcome_space

    def set_preferences(self, value: Preferences | None, force=False) -> None:
        if self._nmi is None:
            self._preferences = value
            return
        if self._nmi.state.started:
            warnings.deprecated(
                "Changing the utility function by direct assignment after the negotiation is "
                "started is deprecated."
            )
        self._set_pref_os()
        super().set_preferences(value, force=force)

    def _reset_pref_os(self):
        if self.__saved_prefs is not None:
            self.__saved_prefs.outcome_space = self.__saved_pref_os
            self.__saved_prefs = None

    @property
    def annotation(self) -> dict[str, Any]:
        """Returns the private information (annotation) not shared with other negotiators"""
        return self._private_info

    @property
    def private_info(self) -> dict[str, Any]:
        """Returns the private information (annotation) not shared with other negotiators"""
        return self._private_info

    @property
    def parent(self) -> Controller | None:
        """Returns the parent controller"""
        return self.__parent

    def before_death(self, cntxt: dict[str, Any]) -> bool:
        """
        Called whenever the parent is about to kill this negotiator.

        It should return False if the negotiator
        does not want to be killed but the controller can still force-kill it
        """
        return True

    def _dissociate(self):
        self._nmi = None
        self._reset_pref_os()
        self._preferences = self._init_preferences
        self._role = None

    def is_acceptable_as_agreement(self, outcome: Outcome) -> bool:
        """
        Whether the given outcome is acceptable as a final agreement of a negotiation.

        The default behavior is to reject only if a reserved value is defined for the agent and is known to be higher
        than the utility of the outcome.

        """
        if not self.preferences:
            return False
        if self.reserved_outcome is not None:
            if not hasattr(self.preferences, "is_not_worse"):
                return False
            return self.preferences.is_not_worse(outcome, self.reserved_outcome)  # type: ignore
        if not self.ufun:
            return False
        return self.ufun(outcome) >= self.reserved_value

    def isin(self, negotiation_id: str | None) -> bool:
        """
        Is that agent participating in the given negotiation?
        Tests if the agent is participating in the given negotiation.

        Args:

            negotiation_id (Optional[str]): The negotiation ID tested. If
             None, it means ANY negotiation

        Returns:
            bool: True if participating in the given negotiation (or any
                negotiation if it was None)

        """
        if not self.nmi:
            return False
        return self.nmi.id == negotiation_id

    @property
    def capabilities(self) -> dict[str, Any]:
        """Agent capabilities"""
        return self._capabilities

    def remove_capability(self, name: str) -> None:
        """Removes named capability from the negotiator

        Args:
            capabilities: The capabilities to be added as a dict

        Returns:
            None

        Remarks:
            It is the responsibility of the caller to be really capable of added capabilities.

        """
        if hasattr(self, "_capabilities"):
            self._capabilities.pop(name, None)

    def add_capabilities(self, capabilities: dict) -> None:
        """Adds named capabilities to the negotiator.

        Args:
            capabilities: The capabilities to be added as a dict

        Returns:
            None

        Remarks:
            It is the responsibility of the caller to be really capable of added capabilities.

        """
        if hasattr(self, "_capabilities"):
            self._capabilities.update(capabilities)
        else:
            self._capabilities = capabilities

            # CALL BACKS

    def join(
        self,
        nmi: NegotiatorMechanismInterface,
        state: MechanismState,
        *,
        preferences: Preferences | None = None,
        ufun: BaseUtilityFunction | None = None,
        role: str = "negotiator",
    ) -> bool:
        """
        Called by the mechanism when the agent is about to enter a negotiation. It can prevent the agent from entering

        Args:
            nmi  (AgentMechanismInterface): The negotiation.
            state (MechanismState): The current state of the negotiation
            preferences (Preferences): The preferences used by the negotiator (see `ufun` )
            ufun (UtilityFunction): The ufun function to use (overrides `preferences` )
            role (str): role of the negotiator.

        Returns:
            bool indicating whether or not the agent accepts to enter.
            If False is returned it will not enter the negotiation

        Remarks:

            - Joining a neogiation will fail in the following conditions:

              1. The negotiator already has preferences and is asked to join with new ones
              2. The negotiator is already in a negotiation

        """
        if self.nmi is not None:
            return False
        if ufun:
            preferences = ufun
        if preferences is None:
            preferences = self._preferences
        # elif self.preferences and preferences != self.preferences:
        #     warnings.warn(
        #         f"Setting preferenes to {preferences} but the agent already has preferences {self.preferences}",
        #         warnings.NegmasDoubleAssignmentWarning,
        #     )
        self._role = role
        self._nmi = nmi
        self._initial_state = state
        if preferences is not None:
            self._preferences = preferences
        return True

    def on_negotiation_start(self, state: MechanismState) -> None:
        """
        A call back called at each negotiation start

        Args:

            state: `MechanismState` giving current state of the negotiation.

        Remarks:

            - You MUST call the super() version of this function either before or after your code when you are
              overriding it.
            - `on_negotiation_start` and `on_negotiation_end` will always be called once for every agent.

        """

    def _on_negotiation_start(self, state: MechanismState) -> None:
        """
        Internally called by the mechanism when the negotiation is about to start
        """
        if self._preferences:
            self._set_pref_os()
            super().set_preferences(self._preferences, force=True)
        self.on_negotiation_start(state)

    def on_round_start(self, state: MechanismState) -> None:
        """
        A call back called at each negotiation round start

        Args:
            state: `MechanismState` giving current state of the negotiation.

        Remarks:
            - The default behavior is to do nothing.
            - Override this to hook some action.

        """

    def on_mechanism_error(self, state: MechanismState) -> None:
        """
        A call back called whenever an error happens in the mechanism. The error and its explanation are accessible in
        `state`

        Args:
            state: `MechanismState` giving current state of the negotiation.

        Remarks:
            - The default behavior is to do nothing.
            - Override this to hook some action

        """

    def on_round_end(self, state: MechanismState) -> None:
        """
        A call back called at each negotiation round end

        Args:
            state: `MechanismState` giving current state of the negotiation.

        Remarks:
            - The default behavior is to do nothing.
            - Override this to hook some action

        """

    def on_leave(self, state: MechanismState) -> None:
        """
        A call back called after leaving a negotiation.

        Args:
            state: `MechanismState` giving current state of the negotiation.

        Remarks:
            - **MUST** call the baseclass `on_leave` using `super` () if you are going to override this.
            - The default behavior is to do nothing.
            - Override this to hook some action

        """
        self._dissociate()

    def on_negotiation_end(self, state: MechanismState) -> None:
        """
        A call back called at each negotiation end

        Args:
            state: `MechanismState` or one of its descendants giving the state at which the negotiation ended.

        Remarks:
            - The default behavior is to do nothing.
            - Override this to hook some action
            - `on_negotiation_start` and `on_negotiation_end` will always be called once for every agent.

        """

    def _on_negotiation_end(self, state: MechanismState) -> None:
        """
        A call back called at each negotiation end

        Args:
            state: `MechanismState` or one of its descendants giving the state at which the negotiation ended.

        Remarks:
            - The default behavior is to do nothing.
            - Override this to hook some action
            - `on_negotiation_start` and `on_negotiation_end` will always be called once for every agent.

        """
        self.on_negotiation_end(state)
        self._reset_pref_os()

    def on_notification(self, notification: Notification, notifier: str):
        """
        Called whenever the agent receives a notification

        Args:
            notification: The notification!!
            notifier: The notifier!!

        Returns:
            None

        Remarks:

            - You MUST call the super() version of this function either before or after your code when you are
              overriding it.

        """
        # if not self.nmi or notifier != self.nmi.id:
        #     raise ValueError(f"Notification is coming from unknown {notifier}")
        if notification.type == "negotiation_start":
            self.on_negotiation_start(state=notification.data)
        elif notification.type == "round_start":
            self.on_round_start(state=notification.data)
        elif notification.type == "round_end":
            self.on_round_end(state=notification.data)
        elif notification.type == "negotiation_end":
            self.on_negotiation_end(state=notification.data)
        elif notification.type == "ufun_modified":
            self.on_preferences_changed(
                changes=notification.data
                if notification.data
                else [PreferencesChange()]
            )

    def cancel(self, reason=None) -> None:
        """
        A method that may be called by a mechanism to make the negotiator cancel whatever it is currently
        processing.

        Negotiators can just ignore this message (default behavior) but if there is a way to actually cancel
        work, it should be implemented here to improve the responsiveness of the negotiator.
        """

    def __str__(self):
        return f"{self.name}"

    # def __call__(self, state: MechanismState) -> Action:
    #     """
    #     Implements the negotiation strategy
    #     """
    #     raise NotImplementedError(
    #         f"__call__ is not implemented. The MRO is\n {type(self).__mro__}"
    #     )
