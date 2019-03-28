"""
Implements an agent-world-interface (see `AgentWorldInterface`) for the SCM world.
"""
from negmas import Issue, Agent
from negmas.apps.scml.common import FinancialReport
from negmas.situated import AgentWorldInterface, Contract
from negmas.apps.scml.common import *
from typing import Optional, List, Dict, Any

__all__ = [
    'SCMLAWI',
]


class SCMLAWI(AgentWorldInterface):
    """A single contact point between SCML agents and the world simulation."""

    def register_cfp(self, cfp: CFP) -> None:
        """Registers a CFP"""
        self._world.n_new_cfps += 1
        cfp.money_resolution = self._world.money_resolution
        self.bb_record(section='cfps', key=cfp.id, value=cfp)

    def register_interest(self, products: List[int]) -> None:
        """registers interest in receiving callbacks about CFPs related to these products"""
        self._world.register_interest(agent=self.agent, products=products)

    def unregister_interest(self, products: List[int]) -> None:
        """registers interest in receiving callbacks about CFPs related to these products"""
        self._world.unregister_interest(agent=self.agent, products=products)

    def remove_cfp(self, cfp: CFP) -> bool:
        """Removes a CFP"""
        if self.agent.id != cfp.publisher:
            return False
        return self.bb_remove(section='cfps', key=str(hash(cfp)))

    def evaluate_insurance(self, contract: Contract, t: int = None) -> Optional[float]:
        """Can be called to evaluate the premium for insuring the given contract against breachs committed by others

        Args:

            contract: hypothetical contract
            t: time at which the policy is to be bought. If None, it means current step
        """
        return self._world.evaluate_insurance(contract=contract, agent=self.agent, t=t)

    def buy_insurance(self, contract: Contract) -> bool:
        """Buys insurance for the contract by the premium calculated by the insurance company.

        Remarks:
            The agent can call `evaluate_insurance` to find the premium that will be used.
        """
        return self._world.buy_insurance(contract=contract, agent=self.agent)

    def _create_annotation(self, cfp: 'CFP'):
        """Creates full annotation based on a cfp that the agent is receiving"""
        partners = [self.agent.id, cfp.publisher]
        annotation = {'cfp': cfp, 'partners': partners}
        if cfp.is_buy:
            annotation['seller'] = self.agent.id
            annotation['buyer'] = cfp.publisher
        else:
            annotation['buyer'] = self.agent.id
            annotation['seller'] = cfp.publisher
        return annotation

    def request_negotiation(self, cfp: CFP, req_id: str, roles: List[str] = None, mechanism_name: str = None
                            , mechanism_params: Dict[str, Any] = None)  -> bool:
        """
        Requests a negotiation with the publisher of a given CFP

        Args:

            cfp: The CFP to negotiate about
            req_id: A string that is passed back to the caller in all callbacks related to this negotiation
            roles: The roles of the CFP publisher and the agent (in that order). By default no roles are passed (None)
            mechanism_name: The mechanism type to use. If not given the default mechanism from the world will be used
            mechanism_params: Parameters of the mechanism

        Returns:
            Success of failure of the negotiation

        Remarks:

            - The `SCMLAgent` class implements another request_negotiation method that does not receive a `req_id`. This
              helper method is recommended as it generates the required req_id and passes it keeping track of requested
              negotiations (and later of running negotiations). Call this method direclty *only* if you do not
              intend to use the `requested_negotiations` and `running_negotiations` properties of the `SCMLAgent` class

        """
        default_annotation = self._create_annotation(cfp)
        return super().request_negotiation_about(issues=cfp.issues, req_id=req_id, partners=default_annotation['partners']
                                          , roles=roles, annotation=default_annotation, mechanism_name=mechanism_name
                                          , mechanism_params=mechanism_params)

    def request_negotiation_about(self
                                  , issues: List[Issue]
                                  , partners: List[str]
                                  , req_id: str
                                  , roles: List[str] = None
                                  , annotation: Optional[Dict[str, Any]] = None
                                  , mechanism_name: str = None
                                  , mechanism_params: Dict[str, Any] = None
                                  ):
        """
        Overrides the method of the same name in the base class to disable it in SCM Worlds.

        **Do not call this method**

        """
        raise RuntimeError('request_negotiation_about should never be called directly in the SCM world'
                           ', call request_negotiation instead.')

    def is_bankrupt(self, agent_id: str) -> bool:
        """
        Checks whether the given agent is bankrupt

        Args:
            agent_id: Agent ID

        Returns:
            The bankruptcy state of the agent

        """
        return bool(self.bb_read('bankruptcy', key=agent_id))

    def reports_for(self, agent_id: str) -> List[FinancialReport]:
        """
        Gets all financial reports of an agent (in the order of their publication)

        Args:
            agent_id: Agent ID

        Returns:

        """
        reports = self.bb_read('reports_agent', key=agent_id)
        if reports is None:
            return []
        return reports

    def reports_at(self, step: int = None) -> Dict[str, FinancialReport]:
        """
        Gets all financial reports of all agents at a given step

        Args:

            step: Step at which the reports are required. If None, the last set of reports is returned

        Returns:

            A dictionary with agent IDs in keys and their financial reports at the given time as values
        """
        if step is None:
            reports = self.bb_query(section='reports_time', query=None)
            reports = self.bb_read('reports_time', key=str(max([int(_) for _ in reports.keys()])))
        else:
            reports = self.bb_read('reports_time', key=str(step))
        if reports is None:
            return {}
        return reports

    def receive_financial_reports(self, receive: bool = True, agents: Optional[List[str]] = None) -> None:
        """
        Registers/unregisters interest in receiving financial reports

        Args:
            receive: True to receive and False to stop receiving
            agents: If given reception is enabled/disabled only for the given set of agents.

        Remarks:

            - by default financial reports are not sent to any agents. To opt-in to receive financial reports, call this
              method.

        """
        self._world.receive_financial_reports(self.agent, receive, agents)

    @property
    def state(self) -> Factory:
        """Returns the private state of the agent in that world.

        In the SCML world, that is a reference to its factory"""
        return self._world.get_private_state(self.agent)

    @property
    def products(self) -> List[Product]:
        """Products in the world"""
        return self._world.products

    @property
    def processes(self) -> List[Process]:
        """Processes in the world"""
        return self._world.processes


