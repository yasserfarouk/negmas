"""This module defines the interfaces to all negotiation agents (negotiators) in the platform.

Few examples of negotiation agents are provided  in.sample.agent to make the
interface more concrete. In general a user of the platform can implement a new
agent by inheriting from one of the negotiators provided here and/or implement the
functions specified (in languages with Duck typing like python inheretance is
not needed).


"""
import functools
import math
from abc import ABC
from copy import copy
from typing import Optional, Tuple, Union, Type
from typing import TYPE_CHECKING, Dict, Any, Callable

from negmas.common import *
from negmas.events import Notifiable, Notification
from negmas.helpers import get_class
from negmas.utilities import make_discounted_ufun

if TYPE_CHECKING:
    from negmas.outcomes import Outcome
    from negmas.utilities import UtilityValue, UtilityFunction

__all__ = [
    'Negotiator',  # Most abstract kind of agent
    'AspirationMixin',
    'Controller',
]


class Negotiator(NamedObject, Notifiable, ABC):
    r"""Abstract negotiation agent. Base class for all negotiators

     Args:

            name: Negotiator name. If not given it is assigned by the system (unique 16 characters).

        Returns:
            bool: True if participating in the given negotiation (or any negotiation if it was None)

        Remarks:

    """

    def __init__(self, name: str = None, ufun: Optional['UtilityFunction'] = None
                 , parent: 'Controller' = None) -> None:
        super().__init__(name=name)
        self.__parent = parent
        self._capabilities = {'enter': True, 'leave': True}
        self.add_capabilities({'evaluate': True, 'compare': True})
        self._mechanism_id = None
        self._mechanism_info = None
        self._initial_info = None
        self._initial_state = None
        self.utility_function = ufun
        self._init_utility = ufun
        self._role = None

    def __getattribute__(self, item):
        if item in ('id', 'name') or item.startswith('_'):
            return super().__getattribute__(item)
        parent = super().__getattribute__('__dict__')['_Negotiator__parent']
        if parent is None:
            return super().__getattribute__(item)
        attr = getattr(parent, item, None)
        if attr is None:
            return super().__getattribute__(item)
        if isinstance(attr, Callable):
            return functools.partial(attr, negotiator_id=super().__getattribute__('__dict__')['_NamedObject__uuid'])
        return super().__getattribute__(item)

    def before_death(self, cntxt: Dict[str, Any]) -> bool:
        """Called whenever the parent is about to kill this negotiator. It should return False if the negotiator
        does not with to be killed but the controller can still force-kill it"""

    def _dissociate(self):
        self._mechanism_id = None
        self._mechanism_info = None
        self._initial_info = None
        self.utility_function = self._init_utility
        self._role = None

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
    def ufun(self) -> Callable[['Outcome'], Optional['UtilityValue']]:
        """
        The utility function in the given negotiation. 
        
        Remarks:
            - If no utility_function is internally stored, `ufun` still returns a valid callable that returns None 
              for everything.
            - This is what you should always call to get the utility of a given outcome.
            - ufun(None) gives the `reserved_value` of this agent.
        """
        if self.utility_function is not None:
            return self.utility_function
        else:
            return lambda x: None

    @property
    def reserved_value(self):
        """Reserved value is what the agent gets if no agreement is reached in the negotiation."""
        if self.utility_function is None:
            return None
        if self.utility_function.reserved_value is not None:
            return self.utility_function.reserved_value
        return self.utility_function(None)

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
        if hasattr(self, '_capabilities'):
            self._capabilities.update(capabilities)
        else:
            self._capabilities = capabilities

            # CALL BACKS

    def on_enter(self, info: MechanismInfo, state: MechanismState
                 , *, ufun: Optional['UtilityFunction'] = None, role: str = 'agent') -> bool:
        """
        Called by the mechanism when the agent is about to enter a negotiation. It can prevent the agent from entering

        Args:
            info  (MechanismInfo): The negotiation.
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
        self._mechanism_id = info.id
        self._mechanism_info = info
        self._initial_info = copy(info)
        self._initial_state = state
        if ufun is not None:
            self.utility_function = ufun
        if self.utility_function:
            self.utility_function.info = info
        return True

    def on_negotiation_start(self, state: MechanismState) -> None:
        """
        A call back called at each negotiation start

        Args:
            state: `MechanismState` giving current state of the negotiation.

        Remarks:
            - The default behavior is to do nothing.
            - Override this to hook some action.

        """

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
            - **MUST** call the baseclass `on_leave` using `super`() if you are going to override this.
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

        """

    def on_notification(self, notification: Notification, notifier: str):
        if notifier != self._mechanism_id:
            raise ValueError(f'Notification is coming from unknown {notifier}')
        if notification.type == 'negotiation_start':
            self.on_negotiation_start(state=notification.data)
        elif notification.type == 'round_start':
            self.on_round_start(state=notification.data)
        elif notification.type == 'round_end':
            self.on_round_end(state=notification.data)
        elif notification.type == 'negotiation_end':
            self.on_negotiation_end(state=notification.data)

    def __str__(self):
        return f'{self.name}'

    def compare(
        self,
        first: 'Outcome',
        second: 'Outcome',
    ) -> Optional['UtilityValue']:
        """
        Compares two offers using the `ufun`

        Args:
            first: First outcome to be compared
            second: Second outcome to be compared

        Returns:
            UtilityValue: An estimate of the differences between the two outcomes. It can be a real number between -1, 1
            or a probability distribution over the same range.
        """
        if self.utility_function is None:
            return None
        return self.utility_function.compare(first, second)

    class Java:
        implements = ['jnegmas.negotiators.Negotiator']


class AspirationMixin:
    """Adds aspiration level calculation. This Mixin MUST be used with a `Negotiator` class."""

    def aspiration_init(self
                        , max_aspiration: float, aspiration_type: Union[str, int, float], above_reserved_value=True):
        """

        Args:
            max_aspiration:
            aspiration_type:
            above_reserved_value:
        """
        self.add_capabilities({'aspiration': True})
        self.max_aspiration = max_aspiration
        self.aspiration_type = aspiration_type
        self.e = 1.0
        if isinstance(aspiration_type, int):
            self.e = float(aspiration_type)
        elif isinstance(aspiration_type, float):
            self.e = aspiration_type
        elif aspiration_type == 'boulware':
            self.e = 4.0
        elif aspiration_type == 'linear':
            self.e = 1.0
        elif aspiration_type == 'conceder':
            self.e = 0.25
        else:
            raise ValueError(f'Unknown aspiration type {aspiration_type}')
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
            raise ValueError(f'Aspiration negotiators cannot be used in negotiations with no time or #steps limit!!')
        if self.e < 1e-7:
            return 0.0
        pmin = self.reserved_value if self.above_reserved and self.reserved_value is not None else 0.0
        return pmin + (self.max_aspiration - pmin) * (1.0 - math.pow(t, self.e))


class Controller(NamedObject):
    """Controls the behavior of multiple negotiators in multiple negotiations

    The controller class MUST implement all methods of the negotiator class it is controlling with one added
    argument negotiator_id (str) which represents ID of the negotiator on which the method is being invoked.


    """

    def __init__(self, default_negotiator_type: Union[str, Type[Negotiator]] = None
                 , default_negotiator_params: Dict[str, Any] = None
                 , name: str = None):
        super().__init__(name=name)
        self._negotiators: Dict[str, Tuple['Negotiator', Dict[str, Any]]] = {}
        if default_negotiator_params is None:
            default_negotiator_params = {}
        if isinstance(default_negotiator_type, str):
            default_negotiator_type = get_class(default_negotiator_type)
        self.__default_negotiator_type = default_negotiator_type
        self.__default_negotiator_params = default_negotiator_params

    def create_negotiator(self, negotiator_type: Union[str, Type[Negotiator]] = None
                          , name: str = None
                          , cntxt: Dict[str, None] = None, **kwargs) -> Negotiator:
        """
        Creates a negotiator passing it the context

        Args:
            negotiator_type: Type of the negotiator to be created
            name: negotiator name
            cntxt: The context to be associated with this negotiator. It will not be passed to the negotiator
            constructor.
            **kwargs: any key-value pairs to be passed to the negotiator constructor

        Returns:

            Negotiator: The negotiator to be controlled

        """
        if negotiator_type is None:
            negotiator_type = self.__default_negotiator_type
        elif isinstance(negotiator_type, str):
            negotiator_type = get_class(negotiator_type)
        if negotiator_type is None:
            raise ValueError('No negotiator type is passed and no default negotiator type is defined for this '
                             'controller')
        new_negotiator = negotiator_type(name=name, parent=self, **self.__default_negotiator_params, **kwargs)
        if new_negotiator is not None:
            self._negotiators[new_negotiator.id] = (new_negotiator, cntxt)
        return new_negotiator

    def call(self, negotiator: Negotiator, method: str, *args, **kwargs):
        """
        Calls the given method on the given negotiator safely without causing recursion. The controller MUST use this
        function to access any callable on the negotiator

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
        response = negotiator.on_kill(cntxt=cntxt)
        if response or force:
            del self._negotiators[negotiator_id]

    def on_enter(self, negotiator_id: str, info: MechanismInfo, state: MechanismState
                 , *, ufun: Optional['UtilityFunction'] = None, role: str = 'agent') -> bool:
        """
        Called by the mechanism when the agent is about to enter a negotiation. It can prevent the agent from entering

        Args:
            negotiator_id: The negotiator ID
            info  (MechanismInfo): The negotiation.
            state (MechanismState): The current state of the negotiation
            ufun (UtilityFunction): The ufun function to use before any discounting.
            role (str): role of the agent.

        Returns:
            bool indicating whether or not the agent accepts to enter.If False is returned it will not enter the
            negotiation.

        """
        negotiator, cntxt = self._negotiators.get(negotiator_id, (None, None))
        if negotiator is None:
            raise ValueError(f'Unknown negotiator {negotiator_id}')
        return self.call(negotiator, 'on_enter', info=info, state=state, ufun=ufun, role=role)

    def on_negotiation_start(self, negotiator_id: str, state: MechanismState) -> None:
        """
        A call back called at each negotiation start

        Args:
            negotiator_id: The negotiator ID
            state: `MechanismState` giving current state of the negotiation.

        Remarks:
            - The default behavior is to do nothing.
            - Override this to hook some action.

        """
        negotiator, cntxt = self._negotiators.get(negotiator_id, (None, None))
        if negotiator is None:
            raise ValueError(f'Unknown negotiator {negotiator_id}')
        return self.call(negotiator, 'on_negotiation_start', state=state)

    def on_round_start(self, negotiator_id: str, state: MechanismState) -> None:
        """A call back called at each negotiation round start

        Args:
            negotiator_id: The negotiator ID
            state: `MechanismState` giving current state of the negotiation.

        Remarks:
            - The default behavior is to do nothing.
            - Override this to hook some action.

        """
        negotiator, cntxt = self._negotiators.get(negotiator_id, (None, None))
        if negotiator is None:
            raise ValueError(f'Unknown negotiator {negotiator_id}')
        return self.call(negotiator, 'on_round_start', state=state)

    def on_mechanism_error(self, negotiator_id: str, state: MechanismState) -> None:
        """
        A call back called whenever an error happens in the mechanism. The error and its explanation are accessible in
        `state`

        Args:
            negotiator_id: The negotiator ID
            state: `MechanismState` giving current state of the negotiation.

        Remarks:
            - The default behavior is to do nothing.
            - Override this to hook some action

        """
        negotiator, cntxt = self._negotiators.get(negotiator_id, (None, None))
        if negotiator is None:
            raise ValueError(f'Unknown negotiator {negotiator_id}')
        return self.call(negotiator, 'on_mechanism_error', state=state)

    def on_round_end(self, negotiator_id: str, state: MechanismState) -> None:
        """
        A call back called at each negotiation round end

        Args:
            negotiator_id: The negotiator ID
            state: `MechanismState` giving current state of the negotiation.

        Remarks:
            - The default behavior is to do nothing.
            - Override this to hook some action

        """
        negotiator, cntxt = self._negotiators.get(negotiator_id, (None, None))
        if negotiator is None:
            raise ValueError(f'Unknown negotiator {negotiator_id}')
        return self.call(negotiator, 'on_round_end', state=state)

    def on_leave(self, negotiator_id: str, state: MechanismState) -> None:
        """A call back called after leaving a negotiation.

        Args:
            negotiator_id: The negotiator ID
            state: `MechanismState` giving current state of the negotiation.

        Remarks:
            - **MUST** call the baseclass `on_leave` using `super`() if you are going to override this.
            - The default behavior is to do nothing.
            - Override this to hook some action

        """
        negotiator, cntxt = self._negotiators.get(negotiator_id, (None, None))
        if negotiator is None:
            raise ValueError(f'Unknown negotiator {negotiator_id}')
        return self.call(negotiator, 'on_leave', state=state)

    def on_negotiation_end(self, negotiator_id: str, state: MechanismState) -> None:
        """
        A call back called at each negotiation end

        Args:
            negotiator_id: The negotiator ID
            state: `MechanismState` or one of its descendants giving the state at which the negotiation ended.

        Remarks:
            - The default behavior is to do nothing.
            - Override this to hook some action

        """
        negotiator, cntxt = self._negotiators.get(negotiator_id, (None, None))
        if negotiator is None:
            raise ValueError(f'Unknown negotiator {negotiator_id}')
        return self.call(negotiator, 'on_negotiation_end', state=state)

    def on_notification(self, negotiator_id: str, notification: Notification, notifier: str):
        negotiator, cntxt = self._negotiators.get(negotiator_id, (None, None))
        if negotiator is None:
            raise ValueError(f'Unknown negotiator {negotiator_id}')
        return self.call(negotiator, 'on_notification', notification=notification, notifier=notifier)

    def __str__(self):
        return f'{self.name}'


Controller = Controller
"""Proxy for a `Controller`"""
