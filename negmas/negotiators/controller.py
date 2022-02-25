from __future__ import annotations

from collections import namedtuple
from typing import TYPE_CHECKING, Any, TypeVar

from negmas.common import MechanismState, NegotiatorMechanismInterface
from negmas.events import Notification
from negmas.helpers import get_class
from negmas.preferences import Preferences
from negmas.types import Rational

from .negotiator import Negotiator

if TYPE_CHECKING:
    from negmas.preferences import BaseUtilityFunction
    from negmas.situated import Agent

    from .controlled import ControlledNegotiator


__all__ = [
    "Controller",
]

NegotiatorInfo = namedtuple("NegotiatorInfo", ["negotiator", "context"])
"""
The return type of `negotiators` member of `Controller`.
"""

ControlledNegotiatorType = TypeVar("ControlledNegotiatorType", bound=Negotiator)


class Controller(Rational):
    """
    Controls the behavior of multiple negotiators in multiple negotiations.

    The controller class MUST implement any methods of the negotiator class it
    is controlling with one added argument negotiator_id (str) which represents
    ID of the negotiator on which the method is being invoked (passed first).

    Controllers for specific classes should inherit from this class and
    implement whatever methods they want to override on their
    `ControlledNegotiator` objects. For example, the SAO module defines
    `SAOController` that needs only to implement `propose` and `respond` .

    Args:
        default_negotiator_type: The negotiator type to use for adding negotiator
                                 if no type is explicitly given.
        default_negotiator_params: The parameters to use to construct the
                                   default negotiator type.
        parent: The parent which can be an `Agent` or another `Controller`
        auto_kill: If True, negotiators will be killed once their negotiation
                   finishes.
        name: The controller name

    Remarks:

     - Controllers should always call negotiator methods using the `call`
       method defined in this class. Direct calls may lead to infinite loops


    """

    def __init__(
        self,
        default_negotiator_type: str | type[ControlledNegotiator] = None,
        default_negotiator_params: dict[str, Any] = None,
        parent: Controller | Agent = None,
        auto_kill: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._negotiators: dict[str, NegotiatorInfo] = {}
        if default_negotiator_params is None:
            default_negotiator_params = {}
        if isinstance(default_negotiator_type, str):
            default_negotiator_type = get_class(default_negotiator_type)
        self.__default_negotiator_type = default_negotiator_type
        self.__default_negotiator_params = default_negotiator_params
        self.__parent = parent
        self._auto_kill = auto_kill

    @property
    def negotiators(self) -> dict[str, NegotiatorInfo]:
        """
        Returns a dictionary mapping negotiator ID to the a tuple containing
        the negotiator and its context.
        """
        return self._negotiators

    @property
    def active_negotiators(self) -> dict[str, NegotiatorInfo]:
        """
        Returns the negotiators whose negotiations are running.

        Returns a dictionary mapping negotiator ID to the a tuple containing the negotiator
        and its context
        """
        return {
            k: v
            for k, v in self._negotiators.items()
            if v[0].nmi is not None
            and (v[0].nmi.state.running or not v[0].nmi.state.started)
        }

    @property
    def states(self) -> dict[str, MechanismState]:
        """
        Gets the current states of all negotiations as a mapping from negotiator ID to mechanism.
        """
        return dict(
            zip(
                self._negotiators.keys(),
                (self._negotiators[k][0]._nmi.state for k in self._negotiators.keys()),
            )
        )

    def make_negotiator(
        self,
        negotiator_type: str | ControlledNegotiatorType | None = None,
        name: str = None,
        **kwargs,
    ) -> ControlledNegotiatorType:
        """
        Creates a negotiator but does not add it to the controller. Call
        `add_negotiator` to add it.

        Args:
            negotiator_type: Type of the negotiator to be created. If None, A `ControlledNegotiator` negotiator will be controlled (which is **fully** controlled by the controller).
            name: negotiator name
            **kwargs: any key-value pairs to be passed to the negotiator constructor

        Returns:

            The negotiator to be controlled. None for failure

        """
        if negotiator_type is None:
            negotiator_type = self.__default_negotiator_type  # type: ignore
        elif isinstance(negotiator_type, str):
            negotiator_type = get_class(negotiator_type)
        if negotiator_type is None:
            raise ValueError(
                "No negotiator type is passed and no default negotiator type is defined for this "
                "controller"
            )
        args = self.__default_negotiator_params
        if kwargs:
            args.update(kwargs)
        return negotiator_type(name=name, parent=self, **args)  # type: ignore I already make sure it is a class in advance

    def add_negotiator(
        self,
        negotiator: Negotiator,
        cntxt: Any = None,
    ) -> None:
        """
        Adds a negotiator to the controller.

        Args:
            negotaitor: The negotaitor to add
            name: negotiator name
            cntxt: The context to be associated with this negotiator.
            **kwargs: any key-value pairs to be passed to the negotiator constructor

        """
        if negotiator is not None:
            self._negotiators[negotiator.id] = NegotiatorInfo(negotiator, cntxt)

    def create_negotiator(
        self,
        negotiator_type: str | ControlledNegotiatorType | None = None,
        name: str = None,
        cntxt: Any = None,
        **kwargs,
    ) -> ControlledNegotiatorType:
        """
        Creates a negotiator passing it the context

        Args:
            negotiator_type: Type of the negotiator to be created
            name: negotiator name
            cntxt: The context to be associated with this negotiator.
            **kwargs: any key-value pairs to be passed to the negotiator constructor

        Returns:

            The negotiator to be controlled. None for failure

        """
        new_negotiator = self.make_negotiator(negotiator_type, name, **kwargs)
        self.add_negotiator(new_negotiator)
        return new_negotiator

    def call(self, negotiator: ControlledNegotiator, method: str, *args, **kwargs):
        """
        Calls the given method on the given negotiator safely without causing
        recursion. The controller MUST use this function to access any callable
        on the negotiator.

        Args:
            negotiator:
            method:
            *args:
            **kwargs:

        Returns:

        """
        negotiator._Negotiator__parent = None  # type: ignore (little bad python magic)
        result = getattr(negotiator, method)(*args, **kwargs)
        negotiator._Negotiator__parent = self  # type: ignore (little bad python magic)
        return result

    def kill_negotiator(self, negotiator_id: str, force: bool = False) -> None:
        """
        Kills the negotiator sending it an `before_death` message.

        Args:
            negotiator_id: The ID of the negotiator to kill.
            force: Whether to kill the negotiator in case it refused to die.

        Remarks:

            - Killing a negotiator amounts to nothing more than removing it form the list of negotiators maintained by
              the controller.

        """
        negotiator, cntxt = self._negotiators.get(negotiator_id, (None, None))
        if negotiator is None:
            return
        response = negotiator.before_death(cntxt=cntxt)
        if response or force:
            negotiator._Negotiator__parent = None
            self._negotiators.pop(negotiator_id, None)

    def partner_negotiator_ids(self, negotiator_id: str) -> list[str] | None:
        """
        Finds the negotiator ID negotiating with one of our negotiators.

        Args:

            negotiator_id: Our negotiator ID
        """
        negotiator, cntxt = self._negotiators.get(negotiator_id, (None, None))
        if not negotiator or not negotiator.nmi:
            return None
        return [_ for _ in negotiator.nmi.negotiator_ids if _ != negotiator_id]

    def partner_negotiator_names(self, negotiator_id: str) -> list[str] | None:
        """
        Finds the negotiator names negotiating with one of our negotiators.

        Args:

            negotiator_id: Our negotiator ID
        """
        negotiator, cntxt = self._negotiators.get(negotiator_id, (None, None))
        if not negotiator or not negotiator.nmi:
            return None
        return [_ for _ in negotiator.nmi.negotiator_names if _ != negotiator.name]

    def partner_agent_ids(self, negotiator_id: str) -> list[str] | None:
        """
        Finds the agent ID negotiating with one of our negotiators.

        Args:

            negotiator_id: Our negotiator ID
        """
        negotiator, cntxt = self._negotiators.get(negotiator_id, (None, None))
        if not negotiator or not negotiator.nmi:
            return None
        me = negotiator.owner.id if negotiator.owner else ""
        return [_ for _ in negotiator.nmi.agent_ids if _ and _ != me]

    def partner_agent_names(self, negotiator_id: str) -> list[str] | None:
        """
        Finds the negotiator names negotiating with one of our negotiators.

        Args:

            negotiator_id: Our negotiator ID
        """
        negotiator, cntxt = self._negotiators.get(negotiator_id, (None, None))
        if not negotiator or not negotiator.nmi:
            return None
        me = negotiator.owner.name if negotiator.owner else ""
        return [_ for _ in negotiator.nmi.agent_names if _ and _ != me]

    def before_join(
        self,
        negotiator_id: str,
        nmi: NegotiatorMechanismInterface,
        state: MechanismState,
        *,
        preferences: Preferences | None = None,
        role: str = "negotiator",
    ) -> bool:
        """
        Called by children negotiators to get permission to join negotiations

        Args:
            negotiator_id: The negotiator ID
            nmi  (AgentMechanismInterface): The negotiation.
            state (MechanismState): The current state of the negotiation
            preferences (UtilityFunction): The prefrences to use before any discounting.
            role (str): role of the agent.

        Returns:
            True if the negotiator is allowed to join the negotiation otherwise
            False

        """
        return True

    def after_join(
        self,
        negotiator_id: str,
        nmi: NegotiatorMechanismInterface,
        state: MechanismState,
        *,
        preferences: Preferences | None = None,
        role: str = "negotiator",
    ) -> None:
        """
        Called by children negotiators after joining a negotiation to inform
        the controller

        Args:
            negotiator_id: The negotiator ID
            nmi  (AgentMechanismInterface): The negotiation.
            state (MechanismState): The current state of the negotiation
            preferences (UtilityFunction): The ufun function to use before any discounting.
            role (str): role of the agent.
        """

    def join(
        self,
        negotiator_id: str,
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
            negotiator_id: The negotiator ID
            nmi  (AgentMechanismInterface): The negotiation.
            state (MechanismState): The current state of the negotiation
            preferences (Preferences): The preferences.
            ufun (BaseUtilityFunction): The ufun function to use before any discounting (overrides preferences)
            role (str): role of the agent.

        Returns:
            bool indicating whether or not the agent accepts to enter.If False is returned it will not enter the
            negotiation.

        """
        if ufun is not None:
            preferences = ufun
        negotiator, _ = self._negotiators.get(negotiator_id, (None, None))
        if negotiator is None:
            raise ValueError(f"Unknown negotiator {negotiator_id}")
        permission = self.before_join(
            negotiator, nmi, state, preferences=preferences, role=role
        )
        if not permission:
            return False
        if hasattr(negotiator, "join") and self.call(
            negotiator, "join", nmi=nmi, state=state, preferences=preferences, role=role
        ):
            self.after_join(negotiator, nmi, state, preferences=preferences, role=role)
            return True
        return False

    #     def _on_negotiation_start(self, negotiator_id: str, state: MechanismState) -> None:
    #         """
    #         A call back called at each negotiation start dirctly by the mechanism
    #
    #         Args:
    #             negotiator_id: The negotiator ID
    #             state: `MechanismState` giving current state of the negotiation.
    #
    #         """
    #         negotiator, cntxt = self._negotiators.get(negotiator_id, (None, None))
    #         if negotiator is None:
    #             raise ValueError(f"Unknown negotiator {negotiator_id}")
    #         return self.call(negotiator, "_on_negotiation_start", state=state)

    def on_negotiation_start(self, negotiator_id: str, state: MechanismState) -> None:
        """
        A call back called at each negotiation start

        Args:
            negotiator_id: The negotiator ID
            state: `MechanismState` giving current state of the negotiation.

        """
        negotiator, cntxt = self._negotiators.get(negotiator_id, (None, None))
        if negotiator is None:
            raise ValueError(f"Unknown negotiator {negotiator_id}")
        return self.call(negotiator, "on_negotiation_start", state=state)

    def on_round_start(self, negotiator_id: str, state: MechanismState) -> None:
        """
        A call back called at each negotiation round start

        Args:
            negotiator_id: The negotiator ID
            state: `MechanismState` giving current state of the negotiation.

        """
        negotiator, cntxt = self._negotiators.get(negotiator_id, (None, None))
        if negotiator is None:
            raise ValueError(f"Unknown negotiator {negotiator_id}")
        return self.call(negotiator, "on_round_start", state=state)

    def on_mechanism_error(self, negotiator_id: str, state: MechanismState) -> None:
        """
        A call back called whenever an error happens in the mechanism. The error and its explanation are accessible in
        `state`

        Args:
            negotiator_id: The negotiator ID
            state: `MechanismState` giving current state of the negotiation.

        """
        negotiator, cntxt = self._negotiators.get(negotiator_id, (None, None))
        if negotiator is None:
            raise ValueError(f"Unknown negotiator {negotiator_id}")
        return self.call(negotiator, "on_mechanism_error", state=state)

    def on_round_end(self, negotiator_id: str, state: MechanismState) -> None:
        """
        A call back called at each negotiation round end

        Args:
            negotiator_id: The negotiator ID
            state: `MechanismState` giving current state of the negotiation.

        """
        negotiator, cntxt = self._negotiators.get(negotiator_id, (None, None))
        if negotiator is None:
            raise ValueError(f"Unknown negotiator {negotiator_id}")
        return self.call(negotiator, "on_round_end", state=state)

    def on_leave(self, negotiator_id: str, state: MechanismState) -> None:
        """
        A call back called after leaving a negotiation.

        Args:
            negotiator_id: The negotiator ID
            state: `MechanismState` giving current state of the negotiation.
        """
        negotiator, cntxt = self._negotiators.get(negotiator_id, (None, None))
        if negotiator is None:
            raise ValueError(f"Unknown negotiator {negotiator_id}")
        return self.call(negotiator, "on_leave", state=state)

    def on_negotiation_end(self, negotiator_id: str, state: MechanismState) -> None:
        """
        A call back called at each negotiation end

        Args:
            negotiator_id: The negotiator ID
            state: `MechanismState` or one of its descendants giving the state at which the negotiation ended.
        """
        negotiator, cntxt = self._negotiators.get(negotiator_id, (None, None))
        if negotiator is None:
            raise ValueError(f"Unknown negotiator {negotiator_id}")
        result = self.call(negotiator, "on_negotiation_end", state=state)
        if self._auto_kill:
            self.kill_negotiator(negotiator_id=negotiator_id, force=True)
        return result

    def on_notification(
        self, negotiator_id: str, notification: Notification, notifier: str
    ):
        negotiator, cntxt = self._negotiators.get(negotiator_id, (None, None))
        if negotiator is None:
            raise ValueError(f"Unknown negotiator {negotiator_id}")
        return self.call(
            negotiator, "on_notification", notification=notification, notifier=notifier
        )

    def __str__(self):
        return f"{self.name}"
