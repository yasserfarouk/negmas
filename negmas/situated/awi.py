from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Collection

from negmas.common import NegotiatorMechanismInterface
from negmas.negotiators import Negotiator
from negmas.outcomes import Issue
from negmas.preferences import Preferences

from .action import Action
from .common import NegotiationRequestInfo, RunningNegotiationInfo
from .contract import Contract

if TYPE_CHECKING:
    from .agent import Agent
    from .world import World

__all__ = ["AgentWorldInterface"]


class AgentWorldInterface:
    """Agent World Interface class"""

    # def __getstate__(self):
    #     return self._world, self.agent.id
    #
    # def __setstate__(self, state):
    #     self._world, agent_id = state
    #     self.agent = self._world.agents[agent_id]

    def __init__(self, world: World, agent: Agent):
        self._world, self.agent = world, agent

    def execute(
        self, action: Action, callback: Callable[[Action, bool], Any] = None
    ) -> bool:
        """Executes an action in the world simulation"""
        return self._world.execute_action(
            action=action, agent=self.agent, callback=callback
        )

    @property
    def state(self) -> Any:
        """Returns the private state of the agent in that world"""
        return self._world.get_private_state(self.agent)

    @property
    def relative_time(self) -> float:
        """Relative time of the simulation going from 0 to 1"""
        return self._world.relative_time

    @property
    def current_step(self) -> int:
        """Current simulation step"""
        return self._world.current_step

    @property
    def n_steps(self) -> int:
        """Number of steps in a simulation"""
        return self._world.n_steps

    @property
    def default_signing_delay(self) -> int:
        return self._world.default_signing_delay if not self._world.force_signing else 0

    def run_negotiation(
        self,
        issues: Collection[Issue],
        partners: Collection[str | Agent],
        negotiator: Negotiator,
        preferences: Preferences = None,
        caller_role: str = None,
        roles: Collection[str] = None,
        annotation: dict[str, Any] | None = None,
        mechanism_name: str = None,
        mechanism_params: dict[str, Any] = None,
    ) -> tuple[Contract, NegotiatorMechanismInterface] | None:
        """
        Runs a negotiation until completion

        Args:
            partners: A list of partners to participate in the negotiation.
                      Note that the caller itself may not be in this list which
                      makes it possible for an agent to request a negotaition
                      that it does not participate in. If that is not to be
                      allowed in some world, override this method and explicitly
                      check for these kinds of negotiations and return False.
                      If partners is passed as a single string/`Agent` or as a list
                      containing a single string/`Agent`, then he caller will be added
                      at the beginning of the list. This will only be done if
                      `roles` was passed as None.
            negotiator: The negotiator to be used in the negotiation
            preferences: The preferences. Only needed if the negotiator does not already know it
            caller_role: The role of the caller in the negotiation
            issues: Negotiation issues
            annotation: Extra information to be passed to the `partners` when asking them to join the negotiation
            partners: A list of partners to participate in the negotiation
            roles: The roles of different partners. If None then each role for each partner will be None
            mechanism_name: Name of the mechanism to use. It must be one of the mechanism_names that are supported by the
            `World` or None which means that the `World` should select the mechanism. If None, then `roles` and `my_role`
            must also be None
            mechanism_params: A dict of parameters used to initialize the mechanism object

        Returns:

            A Tuple of a contract and the nmi of the mechanism used to get it in case of success. None otherwise

        """
        a, b = self._world.run_negotiation(
            caller=self.agent,
            issues=issues,
            partners=partners,
            annotation=annotation,
            roles=roles,
            mechanism_name=mechanism_name,
            mechanism_params=mechanism_params,
            negotiator=negotiator,
            preferences=preferences,
            caller_role=caller_role,
        )
        if a is None or b is None:
            return None
        return a, b

    def run_negotiations(
        self,
        issues: list[Issue] | list[list[Issue]],
        partners: list[list[str | Agent]],
        negotiators: list[Negotiator],
        preferences: list[Preferences] = None,
        caller_roles: list[str] = None,
        roles: list[list[str] | None] | None = None,
        annotations: list[dict[str, Any] | None] | None = None,
        mechanism_names: str | list[str] | None = None,
        mechanism_params: dict[str, Any] | list[dict[str, Any]] | None = None,
        all_or_none: bool = False,
    ) -> list[tuple[Contract, NegotiatorMechanismInterface]]:
        """
        Requests to run a set of negotiations simultaneously. Returns after all negotiations are run to completion

        Args:
            partners: A list of partners to participate in the negotiation.
                      Note that the caller itself may not be in this list which
                      makes it possible for an agent to request a negotaition
                      that it does not participate in. If that is not to be
                      allowed in some world, override this method and explicitly
                      check for these kinds of negotiations and return False.
                      If partners is passed as a single string/`Agent` or as a list
                      containing a single string/`Agent`, then he caller will be added
                      at the beginning of the list. This will only be done if
                      `roles` was passed as None.
            issues: Negotiation issues
            negotiators: The negotiator to be used in the negotiation
            ufuns: The utility function. Only needed if the negotiator does not already know it
            caller_roles: The role of the caller in the negotiation
            annotations: Extra information to be passed to the `partners` when asking them to join the negotiation
            partners: A list of partners to participate in the negotiation
            roles: The roles of different partners. If None then each role for each partner will be None
            mechanism_names: Name of the mechanism to use. It must be one of the mechanism_names that are supported by the
            `World` or None which means that the `World` should select the mechanism. If None, then `roles` and `my_role`
            must also be None
            mechanism_params: A dict of parameters used to initialize the mechanism object
            all_or_none: If true, either no negotiations will be started execpt if all partners accepted

        Returns:

             A list of tuples each with two values: contract (None for failure) and nmi (The mechanism info [None if the
             corresponding partner refused to negotiation])

        """
        return self._world.run_negotiations(
            caller=self.agent,
            issues=issues,
            roles=roles,
            annotations=annotations,
            mechanism_names=mechanism_names,
            mechanism_params=mechanism_params,
            partners=partners,
            negotiators=negotiators,
            caller_roles=caller_roles,
            preferences=preferences,
            all_or_none=all_or_none,
        )

    def request_negotiation_about(
        self,
        issues: list[Issue],
        partners: list[str],
        req_id: str,
        roles: list[str] = None,
        annotation: dict[str, Any] | None = None,
        mechanism_name: str = None,
        mechanism_params: dict[str, Any] = None,
        group: str | None = None,
    ) -> bool:
        """
        Requests to start a negotiation with some other agents

        Args:
            req_id:
            issues: Negotiation issues
            annotation: Extra information to be passed to the `partners` when asking them to join the negotiation
            partners: A list of partners to participate in the negotiation.
                      Note that the caller itself may not be in this list which
                      makes it possible for an agent to request a negotaition
                      that it does not participate in. If that is not to be
                      allowed in some world, override this method and explicitly
                      check for these kinds of negotiations and return False.
                      If partners is passed as a single string/`Agent` or as a list
                      containing a single string/`Agent`, then he caller will be added
                      at the beginning of the list. This will only be done if
                      `roles` was passed as None.
            roles: The roles of different partners. If None then each role for each partner will be None
            mechanism_name: Name of the mechanism to use. It must be one of the mechanism_names that are supported by the
            `World` or None which means that the `World` should select the mechanism. If None, then `roles` and `my_role`
            must also be None
            mechanism_params: A dict of parameters used to initialize the mechanism object
            group: An opational identifier for the group to which this negotiation belongs. It is not used by the system
                   but is logged for debugging purposes. Moreover, the agent have access to it through its `negotiations`
                   property.

        Returns:

            List["Agent"] the list of partners who rejected the negotiation if any. If None then the negotiation was
            accepted. If empty then the negotiation was not started from the world manager


        Remarks:

            - The function will create a request ID that will be used in callbacks `on_neg_request_accepted` and
            `on_neg_request_rejected`


        """
        partner_agents = [
            self._world.agents[_] if isinstance(_, str) else _ for _ in partners
        ]
        return self._world.request_negotiation_about(
            req_id=req_id,
            caller=self.agent,
            partners=partner_agents,
            roles=roles,
            issues=issues,
            group=group,
            annotation=annotation,
            mechanism_name=mechanism_name,
            mechanism_params=mechanism_params,
        )

    @property
    def params(self) -> dict[str, Any]:
        """Returns the basic parameters of the world"""
        return self._world.params

    def loginfo(self, msg: str) -> None:
        """
        Logs an INFO message

        Args:
            msg: The message to log

        Returns:

        """
        self._world.loginfo(msg)

    def logwarning(self, msg: str) -> None:
        """
        Logs a WARNING message

        Args:
            msg: The message to log

        Returns:

        """
        self._world.logwarning(msg)

    def logdebug(self, msg: str) -> None:
        """
        Logs a WARNING message

        Args:
            msg: The message to log

        Returns:

        """
        self._world.logdebug(msg)

    def logerror(self, msg: str) -> None:
        """
        Logs a WARNING message

        Args:
            msg: The message to log

        Returns:

        """
        self._world.logerror(msg)

    def loginfo_agent(self, msg: str) -> None:
        """
        Logs an INFO message to the agent's log

        Args:
            msg: The message to log

        Returns:

        """
        self._world.loginfo_agent(self.agent.id, msg)

    def logwarning_agent(self, msg: str) -> None:
        """
        Logs a WARNING message to the agent's log

        Args:
            msg: The message to log

        Returns:

        """
        self._world.logwarning_agent(self.agent.id, msg)

    def logdebug_agent(self, msg: str) -> None:
        """
        Logs a WARNING message to the agent's log

        Args:
            msg: The message to log

        Returns:

        """
        self._world.logdebug_agent(self.agent.id, msg)

    def logerror_agent(self, msg: str) -> None:
        """
        Logs a WARNING message to the agent's log

        Args:
            msg: The message to log

        Returns:

        """
        self._world.logerror_agent(self.agent.id, msg)

    def bb_query(
        self, section: str | list[str] | None, query: Any, query_keys=False
    ) -> dict[str, Any] | None:
        """
        Returns all records in the given section/sections of the bulletin-board that satisfy the query

        Args:
            section: Either a section name, a list of sections or None specifying ALL public sections (see remarks)
            query: The query which is USUALLY a dict with conditions on it when querying values and a RegExp when
            querying keys
            query_keys: Whether the query is to be applied to the keys or values.

        Returns:

            - A dictionary with key:value pairs giving all records that satisfied the given requirements.

        Remarks:

            - A public section is a section with a name that does not start with an underscore
            - If a set of sections is given, and two records in different sections had the same key, only one of them
              will be returned
            - Key queries use regular expressions and match from the beginning using the standard re.match function

        """
        if not self._world.bulletin_board:
            return None
        return self._world.bulletin_board.query(
            section=section, query=query, query_keys=query_keys
        )

    def bb_read(self, section: str, key: str) -> Any | None:
        """
        Reads the value associated with given key from the bulletin board

        Args:
            section: section name
            key: key

        Returns:

            Content of that key in the bulletin-board

        """
        if not self._world.bulletin_board:
            return None
        return self._world.bulletin_board.read(section=section, key=key)

    def bb_record(self, section: str, value: Any, key: str | None = None) -> None:
        """
        Records data in the given section of the bulletin board

        Args:
            section: section name (can contain subsections separated by '/')
            key: The key
            value: The value

        """
        if not self._world.bulletin_board:
            return None
        return self._world.bulletin_board.record(section=section, value=value, key=key)

    def bb_remove(
        self,
        section: list[str] | str | None,
        *,
        query: Any | None = None,
        key: str = None,
        query_keys: bool = False,
        value: Any = None,
    ) -> bool:
        """
        Removes a value or a set of values from the bulletin Board

        Args:
            section: The section
            query: the query to use to select what to remove
            key: the key to remove (no need to give a full query)
            query_keys: Whether to apply the query (if given) to keys or values
            value: Value to be removed

        Returns:
            bool: Success of failure
        """
        if not self._world.bulletin_board:
            return False
        return self._world.bulletin_board.remove(
            section=section, query=query, key=key, query_keys=query_keys, value=value
        )

    @property
    def settings(self):
        if not self._world.bulletin_board:
            return None
        return self._world.bulletin_board.data.get("settings", dict())

    @property
    def initialized(self) -> bool:
        """Was the agent initialized (i.e. was init_() called)"""
        return self.agent._initialized

    @property
    def unsigned_contracts(self) -> list[Contract]:
        """
        All contracts that are not yet signed.
        """
        return list(self.agent._unsigned_contracts)

    @property
    def requested_negotiations(self) -> list[NegotiationRequestInfo]:
        """The negotiations currently requested by the agent.

        Returns:

            A list of negotiation request information objects (`NegotiationRequestInfo`)
        """
        return list(self.agent._requested_negotiations.values())

    @property
    def accepted_negotiation_requests(self) -> list[NegotiationRequestInfo]:
        """A list of negotiation requests sent to this agent that are already accepted by it.

        Remarks:
            - These negotiations did not start yet as they are still not accepted  by all partners.
              Once that happens, they will be moved to `running_negotiations`
        """
        return list(self.agent._accepted_requests.values())

    @property
    def negotiation_requests(self) -> list[NegotiationRequestInfo]:
        """A list of the negotiation requests sent by this agent that are not yet accepted or rejected.

        Remarks:
            - These negotiations did not start yet as they are still not accepted  by all partners.
              Once that happens, they will be moved to `running_negotiations`
        """
        return list(self.agent._requested_negotiations.values())

    @property
    def running_negotiations(self) -> list[RunningNegotiationInfo]:
        """The negotiations currently requested by the agent.

        Returns:

            A list of negotiation information objects (`RunningNegotiationInfo`)
        """
        return list(self.agent._running_negotiations.values())
