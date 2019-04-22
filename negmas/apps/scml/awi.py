"""
Implements an agent-world-interface (see `AgentWorldInterface`) for the SCM world.
"""
from typing import Optional, List, Dict, Any

from negmas import Issue
from negmas.apps.scml.common import *
from negmas.apps.scml.common import FactoryState
from negmas.java import to_java, from_java, to_dict
from negmas.situated import AgentWorldInterface, Contract, Action

__all__ = ["SCMLAWI"]


class SCMLAWI(AgentWorldInterface):
    """A single contact point between SCML agents and the world simulation.

    The agent can access the world simulation in one of two ways:

    1. Attributes and methods available in this Agent-World-Interface
    2. Attributes and methods in the `FactoryManager` object itself which provide handy shortcuts
       to the agent-world interface

    **Attributes**

    *Simulation settings*

    - `current_step` : Current simulation step
    - `default_signing_delay` : The grace period allowed between contract conclusion and signature
      by default (i.e. if not agreed upon during the negotiation)
    - `n_steps` : Total number of simulation steps.
    - `relative_time` : The fraction of total simulation time elapsed (it will be a number between 0 and 1)

    *Production Graph*

    - `products` : A list of `Product` objects giving all products defined in the world simulation
    - `processes` : A list of `Process` objects giving all products defined in the world simulation

    *Agent Related*

    - `state` : The current private state available to the agent. In SCML it is a `FactoryState` object.

    **Methods**

    *Production Control*

    - `schedule_job` : Schedules a `Job` for production sometime in the future
    - `schedule_production` : Schedules production using profile number instead of a `Job` object
    - `cancel_production` : Cancels already scheduled production (if it did not start yet) or stop a running
      production.
    - `execute` : A general function to execute any command on the factory. There is no need to directly call
      this function as the SCMLAWI provides convenient functions (e.g. `schedule_job` , `hide_funds` , etc)
      to achieve the same goal without having to worry about creating `Action` objects

    *Storage and Wallet Control*

    - `hide_funds` : Hides funds from the view of the simulator. Note that when bankruptcy is considered, hidden
      funds are visible to the simulator.
    - `hide_inventory` : Hides inventory from the view of the simulator. Note that when bankruptcy is considered, hidden
      funds are visible to the simulator.
    - `unhide_funds` : Un-hides funds hidden earlier with a call to `hide_funds`
    - `unhide_inventory` : Un-hides inventory hidden earlier with a call to `hide_inventory`

    *Negotiation and CFP Control*

    - `register_cfp` : Registers a Call-for-Proposals on the bulletin board.
    - `remove_cfp` : Removes a Call-for-Proposals from the  bulletin board.
    - `request_negotiation` : Requests a negotiation based on the content of a CFP published on the bulletin-board.
      *It is recommended not to use this method directly and to request negotiations using the
      request_negotiation method of `FactoryManager` (i.e. use self.request_negotiation instead
      of self.awi.request_negotiation). This makes it possible for NegMAS to keep track of
      existing `requested_negotiations` and `running_negotiations` for you.

    *Notification Control*

    - `receive_financial_reports` : Register/unregisters interest in receiving financial reports for an agent, a set of
      agents or all agents.
    - `register_interest` : registers interest in receiving CFPs about a set of products. By default all
      `FactoryManager` objects are registered to receive all CFPs for any product they
      can produce or need to consumer according to their line-profiles.
    - `unregister_interest` : unregisters interest in receiving CFPs about a set of products.

    *Information about Other Agents*

    - `is_bankrupt` : Asks about the bankruptcy status of an agent
    - `receive_financial_reports` : Register/unregisters interest in receiving financial reports for an agent, a set of
      agents or all agents.
    - `reports_at` : reads *all* financial reports produced at a given time-step
    - `reports_for` : reads *all* financial reports of a given agent

    *Financial Control*

    - `evaluate_insurance` : Asks for the premium to be paid for insuring against partner breaches
      for a given contract
    - `buy_insurance` : Buys an insurance against partner breaches for a given contract

    *Bulletin-Board*

    The bulletin-board is a key-value store. These methods allows the agent to interact with it. *The `SCMLAWI` provides
    convenient functions for recording to the bulletin-board so you mostly need to use read/query functions*.

    - `bb_read` : Reads a complete section or a single value from the bulletin-board
    - `bb_query` : Returns all records in the given section/sections of the bulletin-board that satisfy a query
    - `bb_record` : Registers a record in the bulletin-board.
    - `bb_remove` : Removes a record from the bulletin-board.

    The following list of sections are available in the SCML Bulletin-Board (Use the exact string for the ``section``
    parameter of any method starting with ``bb_``):

    - **cfps**: All CFPs currently on the board. The key is the CFP ID
    - **products**: A list of all products. The key is the product index/ID
    - **processes**: A list of all processes. The key is the product index/ID
    - **bankruptcy**: The bankruptcy list giving names of all bankrupt agents.
    - **reports_time**: Financial reports indexed by time.
    - **reports_agent**: Financial reports indexed by agent
    - **breaches**: Breach-list indexed by breach ID giving all breaches committed in the system
    - **settings**: Static settings of the simulation.

      The following settings are currently available:

      - *breach_penalty_society*: Penalty of breaches paid to society (as a fraction of contract value). This is always
        paid for every breach whether or not there is a negotiated breach.
      - *breach_penalty_victim*: Penalty of breaches paid to victim (as a fraction of contract value). This is always
        paid for every breach whether or not there is a negotiated breach.
      - *immediate_negotiations*: Whether negotiations start immediately when registered (the other possibility -- which
        is the default -- is for them to start at the next production step).
      - *negotiation_speed_multiple*: Number of negotiation steps that finish in a single production step.
      - *negotiation_n_steps*: Maximum allowed number of steps (rounds) in any negotiation
      - *negotiation_step_time_limit*: The maximum real-time allowed for each negotiation step (round)
      - *negotiation_time_limit*: The time limit for a complete negotiation.
      - *transportation_delay*: Transportation delay when products are moved between factories. Default is zero.
      - *transfer_delay*: The delay in transferring funds between factories when executing a contract. Default is zero.
      - *n_steps*: Number of simulation steps
      - *time_limit*: Time limit for the complete simulation

    - stats: Global statistics about the simulation. **Not available for SCML 2019 league**.

    *Logging*

    - `logerror` : Logs an error in the world simulation log file
    - `logwarning` : Logs a warning in the world simulation log file
    - `loginfo` : Logs information in the world simulation log file
    - `logdebug` : Logs debug information in the world simulation log file

    """

    def register_cfp(self, cfp: CFP) -> None:
        """Registers a CFP"""
        self._world.n_new_cfps += 1
        cfp.money_resolution = self._world.money_resolution
        cfp.publisher = (
            self.agent.id
        )  # force the publisher to be the agent using this AWI.
        self.bb_record(section="cfps", key=cfp.id, value=cfp)

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
        return self.bb_remove(section="cfps", key=str(hash(cfp)))

    def evaluate_insurance(self, contract: Contract, t: int = None) -> Optional[float]:
        """Can be called to evaluate the premium for insuring the given contract against breaches committed by others

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

    def _create_annotation(self, cfp: "CFP"):
        """Creates full annotation based on a cfp that the agent is receiving"""
        partners = [self.agent.id, cfp.publisher]
        annotation = {"cfp": cfp, "partners": partners}
        if cfp.is_buy:
            annotation["seller"] = self.agent.id
            annotation["buyer"] = cfp.publisher
        else:
            annotation["buyer"] = self.agent.id
            annotation["seller"] = cfp.publisher
        return annotation

    def request_negotiation(
        self,
        cfp: CFP,
        req_id: str,
        roles: List[str] = None,
        mechanism_name: str = None,
        mechanism_params: Dict[str, Any] = None,
    ) -> bool:
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
        return super().request_negotiation_about(
            issues=cfp.issues,
            req_id=req_id,
            partners=default_annotation["partners"],
            roles=roles,
            annotation=default_annotation,
            mechanism_name=mechanism_name,
            mechanism_params=mechanism_params,
        )

    def request_negotiation_about(
        self,
        issues: List[Issue],
        partners: List[str],
        req_id: str,
        roles: List[str] = None,
        annotation: Optional[Dict[str, Any]] = None,
        mechanism_name: str = None,
        mechanism_params: Dict[str, Any] = None,
    ):
        """
        Overrides the method of the same name in the base class to disable it in SCM Worlds.

        **Do not call this method**

        """
        raise RuntimeError(
            "request_negotiation_about should never be called directly in the SCM world"
            ", call request_negotiation instead."
        )

    def is_bankrupt(self, agent_id: str) -> bool:
        """
        Checks whether the given agent is bankrupt

        Args:
            agent_id: Agent ID

        Returns:
            The bankruptcy state of the agent

        """
        return bool(self.bb_read("bankruptcy", key=agent_id))

    def reports_for(self, agent_id: str) -> List[FinancialReport]:
        """
        Gets all financial reports of an agent (in the order of their publication)

        Args:
            agent_id: Agent ID

        Returns:

        """
        reports = self.bb_read("reports_agent", key=agent_id)
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
            reports = self.bb_query(section="reports_time", query=None)
            reports = self.bb_read(
                "reports_time", key=str(max([int(_) for _ in reports.keys()]))
            )
        else:
            reports = self.bb_read("reports_time", key=str(step))
        if reports is None:
            return {}
        return reports

    def receive_financial_reports(
        self, receive: bool = True, agents: Optional[List[str]] = None
    ) -> None:
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

    def schedule_production(
        self,
        profile: int,
        step: int,
        contract: Optional[Contract] = None,
        override: bool = True,
    ) -> None:
        """
        Schedules production on the agent's factory

        Args:
            profile: Index of the profile in the agent's `compiled_profiles` list
            step: The step to start production according to the given profile
            contract: The contract for which the production is scheduled (optional)
            override: Whether to override existing production jobs schedules at the same time.

        """
        self.execute(
            action=Action(
                type="run",
                params={
                    "profile": profile,
                    "time": step,
                    "contract": contract,
                    "override": override,
                },
            )
        )

    def stop_production(
        self, line: int, step: int, contract: Optional[Contract], override: bool = True
    ):
        """
        Stops/cancels production scheduled at the given line at the given time.

        Args:
            line: One of the factory lines (index)
            step: Step to stop/cancel production at
            contract: The contract for which the job is scheduled (optional)
            override: Whether to override existing production jobs schedules at the same time.
        """
        self.execute(action=Action(type="stop", params={"line": line, "time": step}))

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
        self.execute(
            action=Action(
                type=job.action,
                params={
                    "profile": job.profile,
                    "time": job.time,
                    "line": job.line,
                    "contract": contract,
                    "override": job.override,
                },
            )
        )

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
        self.execute(
            action=Action(
                type="hide_product", params={"product": product, "quantity": quantity}
            )
        )

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
        self.execute(action=Action(type="hide_funds", params={"amount": amount}))

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
        self.execute(
            action=Action(
                type="unhide_product", params={"product": product, "quantity": quantity}
            )
        )

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
        self.execute(action=Action(type="unhide_funds", params={"amount": amount}))


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

    def requestNegotiation(
        self, cfp, req_id: str, roles=None, mechanism_name=None, mechanism_params=None
    ):
        return self.shadow.request_negotiation(
            from_java(cfp), req_id, roles, mechanism_name, mechanism_params
        )

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

    def evaluateInsurance(
        self, contract: Dict[str, Any], t: int = None
    ) -> Optional[float]:
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
        implements = ["jnegmas.apps.scml.awi.SCMLAWI"]
