"""
Implements an agent-world-interface (see `AgentWorldInterface`) for the SCM world.
"""
from typing import Optional, List, Dict, Any

from negmas import Issue
from negmas.apps.scml.common import *
from negmas.apps.scml.common import FactoryState
from negmas.java import to_java, from_java, to_dict
from negmas.situated import AgentWorldInterface, Contract, Action

__all__ = [
    'SCMLAWI',
]


class SCMLAWI(AgentWorldInterface):
    """A single contact point between SCML agents and the world simulation."""

    def register_cfp(self, cfp: CFP) -> None:
        """Registers a CFP"""
        self._world.n_new_cfps += 1
        cfp.money_resolution = self._world.money_resolution
        cfp.publisher = self.agent.id # force the publisher to be the agent using this AWI.
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
                            , mechanism_params: Dict[str, Any] = None) -> bool:
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
        return super().request_negotiation_about(issues=cfp.issues, req_id=req_id,
                                                 partners=default_annotation['partners']
                                                 , roles=roles, annotation=default_annotation,
                                                 mechanism_name=mechanism_name
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
    def state(self) -> FactoryState:
        """Returns the private state of the agent in that world.

        In the SCML world, that is a reference to its factory. You are allowed to read information from the returned
        `Factory` but **not to modify it or call ANY methods on it that modify the state**.


        """
        return self._world.get_private_state(self.agent)

    @property
    def products(self) -> List[Product]:
        """Products in the world"""
        return self._world.products

    @property
    def processes(self) -> List[Process]:
        """Processes in the world"""
        return self._world.processes

    # sugar functions (implementing actions that can all be done through execute

    def schedule_production(self, profile: int, step: int, contract: Optional[Contract] = None,
                            override: bool = True) -> None:
        """
        Schedules production on the agent's factory

        Args:
            profile: Index of the profile in the agent's `compiled_profiles` list
            step: The step to start production according to the given profile
            contract: The contract for which the production is scheduled (optional)
            override: Whether to override existing production jobs schedules at the same time.

        """
        self.execute(action=Action(type='run', params={'profile': profile, 'time': step
                                                       , 'contract': contract, 'override': override}))

    def stop_production(self, line: int, step: int, contract: Optional[Contract], override: bool = True):
        """
        Stops/cancels production scheduled at the given line at the given time.

        Args:
            line: One of the factory lines (index)
            step: Step to stop/cancel production at
            contract: The contract for which the job is scheduled (optional)
            override: Whether to override existing production jobs schedules at the same time.
        """
        self.execute(action=Action(type='stop', params={'line': line, 'time': step}))

    cancel_production = stop_production
    """
    Stops/cancels production scheduled at the given line at the given time.

    Args:
        line: One of the factory lines (index) 
        step: Step to stop/cancel production at
    """

    def schedule_job(self, job: Job, contract: Optional[Contract]):
        """
        Schedules production using a `Job` object. This can be used to schedule any kind of job

        Args:
            job: The job description
            contract: The contract for which the job is scheduled (optional)

        Remarks:

            - Notice that actions that require the profile member of Job (run) never use the line member and vice versa.
        """
        self.execute(action=Action(type=job.action, params={'profile': job.profile, 'time': job.time
                                                            , 'line': job.line
                                                            , 'contract': contract, 'override': job.override}))

    def hide_inventory(self, product: int, quantity: int) -> None:
        """
        Hides the given quantity of the given product so that it is not accessible by the simulator and does not appear
        in reports etc.

        Args:
            product: product index
            quantity: the amount of the product to hide

        Remarks:

            - if the current quantity in storage of the product is less than the amount to be hidden, whatever quantity
              exists is hidden
            - hiding is always immediate
        """
        self.execute(action=Action(type='hide_product', params={'product': product, 'quantity': quantity}))

    def hide_funds(self, amount: float) -> None:
        """
        Hides the given amount of money so that it is not accessible by the simulator and does not appear
        in reports etc.

        Args:
            amount: The amount of money to hide

        Remarks:

            - if the current cash in the agent's wallet is less than the amount to be hidden, all the cash is hidden.
            - hiding is always immediate
        """
        self.execute(action=Action(type='hide_funds', params={'amount': amount}))

    def unhide_inventory(self, product: int, quantity: int) -> None:
        """
        Un-hides the given quantity of the given product so that it is not accessible by the simulator and does not appear
        in reports etc.

        Args:
            product: product index
            quantity: the amount of the product to hide

        Remarks:

            - if the current quantity in storage of the product is less than the amount to be hidden, whatever quantity
              exists is hidden
            - hiding is always immediate
        """
        self.execute(action=Action(type='unhide_product', params={'product': product, 'quantity': quantity}))

    def unhide_funds(self, amount: float) -> None:
        """
        Un-hides the given amount of money so that it is not accessible by the simulator and does not appear
        in reports etc.

        Args:
            amount: The amount of money to unhide

        Remarks:

            - if the current cash in the agent's wallet is less than the amount to be hidden, all the cash is hidden.
            - hiding is always immediate
        """
        self.execute(action=Action(type='unhide_funds', params={'amount': amount}))


class _ShadowSCMLAWI:
    """An SCMLAWI As seen by JNegMAS.

    This is an object that is not visible to python code. It is not directly called from python ever. It is only called
    from a corresponding Java object to represent an internal python object. Because of he way py4j works, we cannot
    just use dunders to implement this kind of object in general. We will have to implement each such class
    independently.

    This kind of classes will always have an internal Java class implementing a Java interface in Jnegmas that starts
    with Py.

    """

    def to_java(self):
        return to_dict(self.shadow)

    def __init__(self, awi: SCMLAWI):
        self.shadow = awi

    def getProducts(self):
        return to_java(self.shadow.products)

    def getProcesses(self):
        return to_java(self.shadow.processes)

    def getState(self):
        return to_java(self.shadow.state)

    def relativeTime(self):
        return self.shadow.relative_time

    def getCurrentStep(self):
        return self.shadow.current_step

    def getNSteps(self):
        return self.shadow.n_steps

    def getDefaultSigningDelay(self):
        return self.shadow.default_signing_delay

    def requestNegotiation(self, cfp, req_id: str, roles=None, mechanism_name=None, mechanism_params=None):
        return self.shadow.request_negotiation(from_java(cfp), req_id, roles, mechanism_name, mechanism_params)

    def registerCFP(self, cfp: Dict[str, Any]) -> None:
        """Registers a CFP"""
        self.shadow.register_cfp(from_java(cfp))

    def removeCFP(self, cfp: Dict[str, Any]) -> bool:
        """Removes a CFP"""
        return self.shadow.remove_cfp(from_java(cfp))

    def registerInterest(self, products: List[int]) -> None:
        """registers interest in receiving callbacks about CFPs related to these products"""
        self.shadow.register_interest(from_java(products))

    def unregisterInterest(self, products: List[int]) -> None:
        """registers interest in receiving callbacks about CFPs related to these products"""
        self.shadow.unregister_interest(from_java(products))

    def evaluateInsurance(self, contract: Dict[str, Any], t: int = None) -> Optional[float]:
        """Can be called to evaluate the premium for insuring the given contract against breaches committed by others

        Args:

            contract: hypothetical contract
            t: time at which the policy is to be bought. If None, it means current step
        """
        result = self.shadow.evaluate_insurance(from_java(contract), t)
        if result < 0:
            return None
        return result

    def buyInsurance(self, contract: Dict[str, Any]) -> bool:
        """Buys insurance for the contract by the premium calculated by the insurance company.

        Remarks:
            The agent can call `evaluate_insurance` to find the premium that will be used.
        """
        return self.shadow.buy_insurance(from_java(contract))

    def loginfo(self, msg: str):
        return self.shadow.loginfo(msg)

    def logwarning(self, msg: str):
        return self.shadow.logwarning(msg)

    def logdebug(self, msg: str):
        return self.shadow.logdebug(msg)

    def logerror(self, msg: str):
        return self.shadow.logerror(msg)

    class Java:
        implements = ['jnegmas.apps.scml.awi.SCMLAWI']

