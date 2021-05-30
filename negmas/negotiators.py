"""
This module defines the interfaces to all negotiation agents (negotiators)
in negmas.

"""
from collections import namedtuple
import functools
import math
import warnings
from abc import ABC
from random import sample
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)

import numpy as np

from negmas.common import AgentMechanismInterface, MechanismState, Rational
from negmas.events import Notifiable, Notification
from negmas.helpers import get_class
from negmas.outcomes import Issue

if TYPE_CHECKING:
    from negmas.outcomes import Outcome
    from negmas.utilities import UtilityFunction
    from negmas.situated import Agent

__all__ = [
    "Negotiator",
    "AspirationMixin",
    "Controller",
    "NegotiatorInfo",
    "PassThroughNegotiator",
    "EvaluatorMixin",
    "RealComparatorMixin",
    "BinaryComparatorMixin",
    "NLevelsComparatorMixin",
    "RankerMixin",
    "RankerWithWeightsMixin",
    "SorterMixin",
    "EvaluatorNegotiator",
    "RealComparatorNegotiator",
    "BinaryComparatorNegotiator",
    "NLevelsComparatorNegotiator",
    "RankerNegotiator",
    "RankerWithWeightsNegotiator",
    "SorterNegotiator",
]

NegotiatorInfo = namedtuple("NegotiatorInfo", ["negotiator", "context"])
"""The return type of `negotiators` member of `Controller`."""

class Negotiator(Rational, Notifiable, ABC):
    r"""Abstract negotiation agent. Base class for all negotiators

    Args:

           name: Negotiator name. If not given it is assigned by the system (unique 16 characters).

       Returns:
           bool: True if participating in the given negotiation (or any negotiation if it was None)

       Remarks:

    """

    def __init__(
        self,
        name: str = None,
        ufun: Optional["UtilityFunction"] = None,
        parent: "Controller" = None,
        owner: "Agent" = None,
        id: str = None,
    ) -> None:
        super().__init__(name=name, ufun=ufun, id=id)
        self.__parent = parent
        self._capabilities = {"enter": True, "leave": True, "ultimatum": True}
        self._mechanism_id = None
        self._ami = None
        self._initial_state = None
        self._role = None
        self.__owner = owner

    @property
    def ami(self):
        return self._ami

    @property
    def owner(self):
        """Returns the owner agent of the negotiator"""
        return self.__owner

    @owner.setter
    def owner(self, owner):
        """Sets the owner"""
        self.__owner = owner

    @Rational.utility_function.setter
    def utility_function(self, value: "UtilityFunction"):
        """Sets tha utility function."""
        if self._ami is not None and self._ami.state.started:
            warnings.warn(
                "Changing the utility function by direct assignment after the negotiation is "
                "started is deprecated."
            )
        if self._ami is not None:
            Rational.utility_function.fset(self, value)
        else:
            self._utility_function = value
            self._ufun_modified = True

    @property
    def parent(self) -> "Controller":
        """Returns the parent controller"""
        return self.__parent

    def before_death(self, cntxt: Dict[str, Any]) -> bool:
        """Called whenever the parent is about to kill this negotiator. It should return False if the negotiator
        does not want to be killed but the controller can still force-kill it"""

    def _dissociate(self):
        self._mechanism_id = None
        self._ami = None
        self._utility_function = self._init_utility
        self._role = None

    def is_acceptable_as_agreement(self, outcome: "Outcome") -> bool:
        """Whether the given outcome is acceptable as a final agreement of a negotiation.

        The default behavior is to reject only if a reserved value is defined for the agent and is known to be higher
        than the utility of the outcome.

        """
        return self._utility_function(outcome) >= self.reserved_value

    def isin(self, negotiation_id: Optional[str]) -> bool:
        """Is that agent participating in the given negotiation?
        Tests if the agent is participating in the given negotiation.

        Args:

            negotiation_id (Optional[str]): The negotiation ID tested. If
             None, it means ANY negotiation

        Returns:
            bool: True if participating in the given negotiation (or any
                negotiation if it was None)

        """
        return self._mechanism_id == negotiation_id

    @property
    def capabilities(self) -> Dict[str, Any]:
        """Agent capabilities"""
        return self._capabilities

    def add_capabilities(self, capabilities: dict) -> None:
        """Adds named capabilities to the agent.

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
        ami: AgentMechanismInterface,
        state: MechanismState,
        *,
        ufun: Optional["UtilityFunction"] = None,
        role: str = "agent",
    ) -> bool:
        """
        Called by the mechanism when the agent is about to enter a negotiation. It can prevent the agent from entering

        Args:
            ami  (AgentMechanismInterface): The negotiation.
            state (MechanismState): The current state of the negotiation
            ufun (UtilityFunction): The ufun function to use before any discounting.
            role (str): role of the agent.

        Returns:
            bool indicating whether or not the agent accepts to enter.
            If False is returned it will not enter the negotiation

        """
        if self._mechanism_id is not None:
            return False
        self._role = role
        self._mechanism_id = ami.id
        self._ami = ami
        self._initial_state = state
        if ufun is not None:
            self.utility_function = ufun
        if self._utility_function:
            self._utility_function.ami = ami
            if self._ufun_modified:
                self.on_ufun_changed()
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
        if self._ufun_modified:
            self.on_ufun_changed()

    def on_round_start(self, state: MechanismState) -> None:
        """A call back called at each negotiation round start

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
        """A call back called after leaving a negotiation.

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
        if notifier != self._mechanism_id:
            raise ValueError(f"Notification is coming from unknown {notifier}")
        if notification.type == "negotiation_start":
            self.on_negotiation_start(state=notification.data)
        elif notification.type == "round_start":
            self.on_round_start(state=notification.data)
        elif notification.type == "round_end":
            self.on_round_end(state=notification.data)
        elif notification.type == "negotiation_end":
            self.on_negotiation_end(state=notification.data)
        elif notification.type == "ufun_modified":
            self.on_ufun_changed()

    def on_ufun_changed(self):
        """
        Called to inform the agent that its ufun has changed.

        Remarks:

            - You MUST call the super() version of this function either before or after your code when you are overriding
              it.
        """
        if hasattr(self._utility_function, "outcome_type"):
            if self._ami and self._utility_function.outcome_type is None:
                self._utility_function.outcome_type = self._ami.outcome_type
                self._utility_function.issue_names = [_.name for _ in self._ami.issues]
            elif (
                self._ami
                and self._utility_function.outcome_type != self._ami.outcome_type
            ):
                raise ValueError(
                    f"UFun uses outcome type {self._utility_function.outcome_type}, but the mechanism uses "
                    f"{self._ami.outcome_type}"
                )
        super().on_ufun_changed()

    def cancel(self, reason=None) -> None:
        """
        A method that may be called by a mechanism to make the negotiator cancel whatever it is currently
        processing.

        Negotiators can just ignore this message (default behavior) but if there is a way to actually cancel
        work, it should be implemented here to improve the responsiveness of the negotiator.
        """

    def __str__(self):
        return f"{self.name}"

    class Java:
        implements = ["jnegmas.negotiators.Negotiator"]


class PassThroughNegotiator(Negotiator):
    """
    A negotiator that can be used to pass all method calls to a parent (Controller).

    It uses magic dunder methods to implement a general way of passing calls to the parent. This method is slow.

    It is recommended to implement a PassThrough*Negotiator for each mechanism that does this passing explicitly which
    will be much faster.

    For an example, see the implementation of `PassThroughSAONegotiator` .

    """

    def __getattribute__(self, item):
        if (
            item
            in (
                "id",
                "name",
                "on_ufun_changed",
                "has_ufun",
                "utility_function",
                "reserved_value",
            )
            or item.startswith("_")
        ):
            return super().__getattribute__(item)
        parent = super().__getattribute__("__dict__").get("_Negotiator__parent", None)
        if parent is None:
            return super().__getattribute__(item)
        attr = getattr(parent, item, None)
        if attr is None:
            return super().__getattribute__(item)
        if isinstance(attr, Callable):
            return functools.partial(
                attr,
                negotiator_id=super().__getattribute__("__dict__")[
                    "_NamedObject__uuid"
                ],
            )
        return super().__getattribute__(item)


class Controller(Rational):
    """Controls the behavior of multiple negotiators in multiple negotiations

    The controller class MUST implement any methods of the negotiator class it
    is controlling with one added argument negotiator_id (str) which represents
    ID of the negotiator on which the method is being invoked (passed first).

    Controllers for specific classes should inherit from this class and
    implement whatever methods they want to override on their
    `PassThroughNegotiator` objects. For example, the SAO module defines
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
        default_negotiator_type: Union[str, Type[PassThroughNegotiator]] = None,
        default_negotiator_params: Dict[str, Any] = None,
        parent: Union["Controller", "Agent"] = None,
        auto_kill: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._negotiators: Dict[str, NegotiatorInfo] = {}
        if default_negotiator_params is None:
            default_negotiator_params = {}
        if isinstance(default_negotiator_type, str):
            default_negotiator_type = get_class(default_negotiator_type)
        self.__default_negotiator_type = default_negotiator_type
        self.__default_negotiator_params = default_negotiator_params
        self.__parent = parent
        self._auto_kill = auto_kill

    @property
    def negotiators(self) -> Dict[str, NegotiatorInfo]:
        """
        Returns a dictionary mapping negotiator ID to the a tuple containing
        the negotiator and its context
        """
        return self._negotiators

    @property
    def active_negotiators(self) -> Dict[str, NegotiatorInfo]:
        """
        Returns the negotiators whose negotiations are running.
        Returns a dictionary mapping negotiator ID to the a tuple containing the negotiator
        and its context
        """
        return {
            k: v
            for k, v in self._negotiators.items()
            if v[0].ami is not None and (v[0].ami.state.running or not v[0].ami.state.started)
        }

    @property
    def states(self) -> Dict[str, MechanismState]:
        """Gets the current states of all negotiations as a mapping from negotiator ID to mechanism"""
        return dict(
            zip(
                self._negotiators.keys(),
                (self._negotiators[k][0]._ami.state for k in self._negotiators.keys()),
            )
        )

    def create_negotiator(
        self,
        negotiator_type: Union[str, Type[PassThroughNegotiator]] = None,
        name: str = None,
        cntxt: Any = None,
        **kwargs,
    ) -> PassThroughNegotiator:
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

    def make_negotiator(
        self,
        negotiator_type: Union[str, Type[PassThroughNegotiator]] = None,
        name: str = None,
        **kwargs,
    ) -> PassThroughNegotiator:
        """
        Creates a negotiator but does not add it to the controller. Call 
        `add_negotiator` to add it.

        Args:
            negotiator_type: Type of the negotiator to be created
            name: negotiator name
            **kwargs: any key-value pairs to be passed to the negotiator constructor

        Returns:

            The negotiator to be controlled. None for failure

        """
        if negotiator_type is None:
            negotiator_type = self.__default_negotiator_type
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
        return negotiator_type(name=name, parent=self, **args)

    def add_negotiator(
        self,
        negotiator: Negotiator,
        cntxt: Any = None,
    ) -> None:
        """
        Adds a negotiator to the controller

        Args:
            negotaitor: The negotaitor to add
            name: negotiator name
            cntxt: The context to be associated with this negotiator.
            **kwargs: any key-value pairs to be passed to the negotiator constructor

        """
        if negotiator is not None:
            self._negotiators[negotiator.id] = NegotiatorInfo(negotiator, cntxt)

    def call(self, negotiator: PassThroughNegotiator, method: str, *args, **kwargs):
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
        negotiator._Negotiator__parent = None
        result = getattr(negotiator, method)(*args, **kwargs)
        negotiator._Negotiator__parent = self
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

    def partner_negotiator_ids(self, negotiator_id: str) -> Optional[List[str]]:
        """
        Finds the negotiator ID negotiating with one of our negotiators.

        Args:

            negotiator_id: Our negotiator ID
        """
        negotiator, cntxt = self._negotiators.get(negotiator_id, (None, None))
        if not negotiator or not negotiator.ami:
            return None
        return [_ for _ in negotiator.ami.negotiator_ids if _ != negotiator_id]

    def partner_negotiator_names(self, negotiator_id: str) -> Optional[List[str]]:
        """
        Finds the negotiator names negotiating with one of our negotiators.

        Args:

            negotiator_id: Our negotiator ID
        """
        negotiator, cntxt = self._negotiators.get(negotiator_id, (None, None))
        if not negotiator or not negotiator.ami:
            return None
        return [_ for _ in negotiator.ami.negotiator_names if _ != negotiator.name]

    def partner_agent_ids(self, negotiator_id: str) -> Optional[List[str]]:
        """
        Finds the agent ID negotiating with one of our negotiators.

        Args:

            negotiator_id: Our negotiator ID
        """
        negotiator, cntxt = self._negotiators.get(negotiator_id, (None, None))
        if not negotiator or not negotiator.ami:
            return None
        me = negotiator.owner.id if negotiator.owner else ""
        return [_ for _ in negotiator.ami.agent_ids if _ and _ != me]

    def partner_agent_names(self, negotiator_id: str) -> Optional[List[str]]:
        """
        Finds the negotiator names negotiating with one of our negotiators.

        Args:

            negotiator_id: Our negotiator ID
        """
        negotiator, cntxt = self._negotiators.get(negotiator_id, (None, None))
        if not negotiator or not negotiator.ami:
            return None
        me = negotiator.owner.name if negotiator.owner else ""
        return [_ for _ in negotiator.ami.agent_names if _ and _ != me]

    def before_join(
        self,
        negotiator_id: str,
        ami: AgentMechanismInterface,
        state: MechanismState,
        *,
        ufun: Optional["UtilityFunction"] = None,
        role: str = "agent",
    ) -> bool:
        """
        Called by children negotiators to get permission to join negotiations

        Args:
            negotiator_id: The negotiator ID
            ami  (AgentMechanismInterface): The negotiation.
            state (MechanismState): The current state of the negotiation
            ufun (UtilityFunction): The ufun function to use before any discounting.
            role (str): role of the agent.

        Returns:
            True if the negotiator is allowed to join the negotiation otherwise
            False

        """
        return True

    def after_join(
        self,
        negotiator_id: str,
        ami: AgentMechanismInterface,
        state: MechanismState,
        *,
        ufun: Optional["UtilityFunction"] = None,
        role: str = "agent",
    ) -> None:
        """
        Called by children negotiators after joining a negotiation to inform
        the controller

        Args:
            negotiator_id: The negotiator ID
            ami  (AgentMechanismInterface): The negotiation.
            state (MechanismState): The current state of the negotiation
            ufun (UtilityFunction): The ufun function to use before any discounting.
            role (str): role of the agent.
        """

    def join(
        self,
        negotiator_id: str,
        ami: AgentMechanismInterface,
        state: MechanismState,
        *,
        ufun: Optional["UtilityFunction"] = None,
        role: str = "agent",
    ) -> bool:
        """
        Called by the mechanism when the agent is about to enter a negotiation. It can prevent the agent from entering

        Args:
            negotiator_id: The negotiator ID
            ami  (AgentMechanismInterface): The negotiation.
            state (MechanismState): The current state of the negotiation
            ufun (UtilityFunction): The ufun function to use before any discounting.
            role (str): role of the agent.

        Returns:
            bool indicating whether or not the agent accepts to enter.If False is returned it will not enter the
            negotiation.

        """
        negotiator, _ = self._negotiators.get(negotiator_id, (None, None))
        if negotiator is None:
            raise ValueError(f"Unknown negotiator {negotiator_id}")
        permission = self.before_join(negotiator, ami, state, ufun=ufun, role=role)
        if not permission:
            return False
        if hasattr(negotiator, "join") and self.call(
            negotiator, "join", ami=ami, state=state, ufun=ufun, role=role
        ):
            self.after_join(negotiator, ami, state, ufun=ufun, role=role)
            return True
        return False

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
        """A call back called at each negotiation round start

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
        """A call back called after leaving a negotiation.

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


class AspirationMixin:
    """Adds aspiration level calculation. This Mixin MUST be used with a `Negotiator` class."""

    def aspiration_init(
        self,
        max_aspiration: float,
        aspiration_type: Union[str, int, float],
        above_reserved_value=True,
    ):
        """

        Args:
            max_aspiration:
            aspiration_type:
            above_reserved_value:
        """
        if hasattr(self, "add_capabilities"):
            self.add_capabilities({"aspiration": True})
        self.max_aspiration = max_aspiration
        self.aspiration_type = aspiration_type
        self.exponent = 1.0
        if isinstance(aspiration_type, int):
            self.exponent = float(aspiration_type)
        elif isinstance(aspiration_type, float):
            self.exponent = aspiration_type
        elif aspiration_type == "boulware":
            self.exponent = 4.0
        elif aspiration_type == "linear":
            self.exponent = 1.0
        elif aspiration_type == "conceder":
            self.exponent = 0.25
        else:
            raise ValueError(f"Unknown aspiration type {aspiration_type}")
        self.above_reserved = above_reserved_value

    def aspiration(self, t: float) -> float:
        """
        The aspiration level

        Args:
            t: relative time (a number between zero and one)

        Returns:
            aspiration level
        """
        if t is None:
            raise ValueError(
                f"Aspiration negotiators cannot be used in negotiations with no time or #steps limit!!"
            )
        return self.max_aspiration * (1.0 - math.pow(t, self.exponent))


class EvaluatorMixin:
    """A mixin that can be used to have the negotiator respond to evaluate messages from the server"""

    def init(self):
        self.capabilities["evaluate"] = True

    def evaluate(self, outcome: "Outcome") -> Optional["UtilityValue"]:
        if self._utility_function is None:
            return None
        return self._utility_function(outcome)


class RealComparatorMixin:
    def init(self):
        self.capabilities["compare-real"] = True
        self.capabilities["compare-binary"] = True

    def compare_real(self, first: "Outcome", second: "Outcome") -> Optional[float]:
        """
        Compares two offers using the `ufun` returning the difference in their utility

        Args:
            first: First outcome to be compared
            second: Second outcome to be compared

        Returns:
            "UtilityValue": An estimate of the differences between the two outcomes. It can be a real number between -1, 1
            or a probability distribution over the same range.
        """
        if not self.has_ufun:
            return None
        return self._utility_function.compare_real(first, second)

    def is_better(
        self, first: "Outcome", second: "Outcome", epsilon: float = 1e-10
    ) -> Optional[bool]:
        """
        Compares two offers using the `ufun` returning whether the first is better than the second

        Args:
            first: First outcome to be compared
            second: Second outcome to be compared
            epsilon: comparison threshold. If the utility difference within the range [-epsilon, epsilon] the two
                     outcomes are assumed to be compatible

        Returns:
            True if utility(first) > utility(second) + epsilon
            None if |utility(first) - utility(second)| <= epsilon or the utun is not defined
            False if utility(first) < utility(second) - epsilon
        """
        if not self.has_ufun:
            return None
        return self._utility_function.is_better(first, second, epsilon)


class BinaryComparatorMixin:
    def init(self):
        self.capabilities["compare-binary"] = True

    def is_better(
        self, first: "Outcome", second: "Outcome", epsilon: float = 1e-10
    ) -> Optional[bool]:
        """
        Compares two offers using the `ufun` returning whether the first is better than the second

        Args:
            first: First outcome to be compared
            second: Second outcome to be compared
            epsilon: comparison threshold. If the utility difference within the range [-epsilon, epsilon] the two
                     outcomes are assumed to be compatible

        Returns:
            True if utility(first) > utility(second) + epsilon
            None if |utility(first) - utility(second)| <= epsilon or the utun is not defined
            False if utility(first) < utility(second) - epsilon
        """
        if not self.has_ufun:
            return None
        return self._utility_function.is_better(first, second, epsilon)


class NLevelsComparatorMixin:
    def init(self):
        self.capabilities["compare-nlevels"] = True
        self.capabilities["compare-binary"] = True
        self.__ufun_thresholds = None

    @classmethod
    def generate_thresholds(
        cls,
        n: int,
        ufun_min: float = 0.0,
        ufun_max: float = 1.0,
        scale: Union[str, Callable[[float], float]] = None,
    ) -> List[float]:
        """
        Generates thresholds for the n given levels assuming the ufun ranges and scale function

        Args:
            n: Number of scale levels (one side)
            ufun_min: minimum value of all utilities
            ufun_max: maximum value of all utilities
            scale: Scales the ufun values. Can be a callable or 'log', 'exp', 'linear'. If None, it is 'linear'

        """
        if scale is not None:
            if isinstance(scale, str):
                scale = dict(
                    linear=lambda x: x,
                    log=math.log,
                    exp=math.exp,
                ).get(scale, None)
                if scale is None:
                    raise ValueError(f"Unknown scale function {scale}")
        thresholds = np.linspace(ufun_min, ufun_max, num=n + 2)[1:-1].tolist()
        if scale is not None:
            thresholds = [scale(_) for _ in thresholds]
        return thresholds

    @classmethod
    def equiprobable_thresholds(
        cls, n: int, ufun: "UtilityFunction", issues: List[Issue], n_samples: int = 1000
    ) -> List[float]:
        """
        Generates thresholds for the n given levels where levels are equally likely approximately

        Args:
            n: Number of scale levels (one side)
            ufun: The utility function to use
            issues: The issues to generate the thresholds for
            n_samples: The number of samples to use during the process

        """
        samples = list(
            Issue.sample(
                issues, n_samples, with_replacement=False, fail_if_not_enough=False
            )
        )
        n_samples = len(samples)
        diffs = []
        for i, first in enumerate(samples):
            n_diffs = min(10, n_samples - i - 1)
            for second in sample(samples[i + 1 :], k=n_diffs):
                diffs.append(abs(ufun.compare_real(first, second)))
        diffs = np.array(diffs)
        _, edges = np.histogram(diffs, bins=n + 1)
        return edges[1:-1].tolist()

    @property
    def thresholds(self) -> Optional[List[float]]:
        """Returns the internal thresholds and None if they do  not exist"""
        return self.__ufun_thresholds

    @thresholds.setter
    def thresholds(self, thresholds: List[float]) -> None:
        self.__ufun_thresholds = thresholds

    def compare_nlevels(
        self, first: "Outcome", second: "Outcome", n: int = 2
    ) -> Optional[int]:
        """
        Compares two offers using the `ufun` returning an integer in [-n, n] (i.e. 2n+1 possible values) which defines
        which outcome is better and the strength of the difference (discretized using internal thresholds)

        Args:
            first: First outcome to be compared
            second: Second outcome to be compared
            n: number of levels to use

        Returns:

            - None if either there is no ufun defined or the number of thresholds required cannot be satisfied
            - 0 iff |u(first) - u(second)| <= thresholds[0]
            - -i if  - thresholds[i-1] < u(first) - u(second) <= -thresholds[i]
            - +i if  thresholds[i-1] > u(first) - u(second) >= thresholds[i]

        Remarks:

            - thresholds is an internal array that can be set using `thresholds` property
            - thresholds[n] is assumed to equal infinity
            - n must be <= the length of the internal thresholds array. If n > that length, a ValueError will be raised.
              If n < the length of the internal thresholds array, the first n values of the array will be used
        """
        if not self.has_ufun:
            return None
        if self.thresholds is None:
            raise ValueError(
                f"Internal thresholds array is not set. Please set the threshold property with an array"
                f" of length >= {n}"
            )
        if len(self.thresholds) < n:
            raise ValueError(
                f"Internal thresholds array is only of length {len(self.thresholds)}. It cannot be used"
                f" to compare outcomes with {n} levels. len(self.thresholds) MUST be >= {n}"
            )
        diff = self._utility_function(first) - self._utility_function(second)
        sign = 1 if diff > 0.0 else -1
        for i, th in enumerate(self.thresholds):
            if diff < th:
                return sign * i
        return sign * n

    def is_better(
        self, first: "Outcome", second: "Outcome", epsilon: float = 1e-10
    ) -> Optional[bool]:
        """
        Compares two offers using the `ufun` returning whether the first is better than the second

        Args:
            first: First outcome to be compared
            second: Second outcome to be compared
            epsilon: comparison threshold. If the utility difference within the range [-epsilon, epsilon] the two
                     outcomes are assumed to be compatible

        Returns:
            True if utility(first) > utility(second) + epsilon
            None if |utility(first) - utility(second)| <= epsilon or the utun is not defined
            False if utility(first) < utility(second) - epsilon
        """
        if not self.has_ufun:
            return None
        return self._utility_function.is_better(first, second, epsilon)


class RankerWithWeightsMixin:
    """Adds the ability to rank outcomes returning the ranks and weights"""

    def init(self):
        self.capabilities["rank-weighted"] = True
        self.capabilities["compare-binary"] = True

    def rank_with_weights(
        self, outcomes: List[Optional["Outcome"]], descending=True
    ) -> List[Tuple[int, float]]:
        """Ranks the given list of outcomes with weights. None stands for the null outcome. Outcomes of equal utility
        are ordered arbitrarily.

        Returns:

            - A list of tuples each with two values:
                - an integer giving the index in the input array (outcomes) of an outcome
                - the weight of that outcome
            - The list is sorted by weights descendingly

        """
        if not self.has_ufun:
            return None
        return self._utility_function.rank_with_weights(outcomes, descending)

    def is_better(
        self, first: "Outcome", second: "Outcome", epsilon: float = 1e-10
    ) -> Optional[bool]:
        """
        Compares two offers using the `ufun` returning whether the first is better than the second

        Args:
            first: First outcome to be compared
            second: Second outcome to be compared
            epsilon: comparison threshold. If the utility difference within the range [-epsilon, epsilon] the two
                     outcomes are assumed to be compatible

        Returns:
            True if utility(first) > utility(second) + epsilon
            None if |utility(first) - utility(second)| <= epsilon or the utun is not defined
            False if utility(first) < utility(second) - epsilon
        """
        if not self.has_ufun:
            return None
        return self._utility_function.is_better(first, second, epsilon)


class RankerMixin:
    """Adds the ability to rank outcomes returning the ranks without weights. Outcomes of equal utility are ordered
    arbitrarily. None stands for the null outcome"""

    def init(self):
        self.capabilities["rank"] = True
        self.capabilities["compare-binary"] = True

    def rank(self, outcomes: List[Optional["Outcome"]], descending=True) -> List[int]:
        """Ranks the given list of outcomes. None stands for the null outcome.

        Returns:

            - A list of integers in the specified order of utility values of outcomes

        """
        if not self.has_ufun:
            return None
        return self._utility_function.rank(outcomes, descending)

    def is_better(
        self, first: "Outcome", second: "Outcome", epsilon: float = 1e-10
    ) -> Optional[bool]:
        """
        Compares two offers using the `ufun` returning whether the first is better than the second

        Args:
            first: First outcome to be compared
            second: Second outcome to be compared
            epsilon: comparison threshold. If the utility difference within the range [-epsilon, epsilon] the two
                     outcomes are assumed to be compatible

        Returns:
            True if utility(first) > utility(second) + epsilon
            None if |utility(first) - utility(second)| <= epsilon or the utun is not defined
            False if utility(first) < utility(second) - epsilon
        """
        if not self.has_ufun:
            return None
        return self._utility_function.is_better(first, second, epsilon)


class SorterMixin:
    """Adds the ability to sort outcomes according to utility. Outcomes of equal utility are ordered
    arbitrarily. None stands for the null outcome"""

    def init(self):
        self.capabilities["sort"] = True

    def sort(self, outcomes: List[Optional["Outcome"]], descending=True) -> None:
        """Ranks the given list of outcomes. None stands for the null outcome.

        Returns:

            - The outcomes are sorted IN PLACE.
            - There is no way to know if the ufun is not defined from the return value. Use `has_ufun` to check for
              the availability of the ufun

        """
        if not self.has_ufun:
            return None
        self._utility_function.sort(outcomes, descending)


class EvaluatorNegotiator(EvaluatorMixin, Negotiator):
    """A negotiator that can be asked to evaluate outcomes using its internal ufun.

    Th change the way it evaluates outcomes, override `evaluate`.

    It has the `evaluate` capability
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        EvaluatorMixin.init(self)


class RealComparatorNegotiator(RealComparatorMixin, Negotiator):
    """A negotiator that can be asked to evaluate outcomes using its internal ufun.

    Th change the way it evaluates outcomes, override `compare_real`

    It has the `compare-real` capability
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        RealComparatorMixin.init(self)


class BinaryComparatorNegotiator(BinaryComparatorMixin, Negotiator):
    """A negotiator that can be asked to compare two outcomes using is_better. By default is just consults the ufun.

    To change that behavior, override `is_better`.

    It has the `compare-binary` capability.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        BinaryComparatorMixin.init(self)


class NLevelsComparatorNegotiator(NLevelsComparatorMixin, Negotiator):
    """A negotiator that can be asked to compare two outcomes using compare_nlevels which returns the strength of
    the difference between two outcomes as an integer from [-n, n] in the C compare sense.
    By default is just consults the ufun.

    To change that behavior, override `compare_nlevels`.

    It has the `compare-nlevels` capability.

    """

    def __init__(self, *args, thresholds: List[float] = None, **kwargs):
        super().__init__(*args, **kwargs)
        NLevelsComparatorMixin.init(self)
        self.thresholds = thresholds


class RankerWithWeightsNegotiator(RankerWithWeightsMixin, Negotiator):
    """A negotiator that can be asked to rank outcomes returning rank and weight. By default is just consults the ufun.

    To change that behavior, override `rank_with_weights`.

    It has the `rank-weighted` capability.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        RankerWithWeightsMixin.init(self)


class RankerNegotiator(RankerMixin, Negotiator):
    """A negotiator that can be asked to rank outcomes. By default is just consults the ufun.

    To change that behavior, override `rank`.

    It has the `rank` capability.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        RankerMixin.init(self)


class SorterNegotiator(SorterMixin, Negotiator):
    """A negotiator that can be asked to rank outcomes returning rank without weight.
    By default is just consults the ufun.

    To change that behavior, override `sort`.

    It has the `sort` capability.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        SorterMixin.init(self)
