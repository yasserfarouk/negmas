"""The base class agent needed for all SCML agents."""
import itertools
import math
from abc import abstractmethod
from collections import defaultdict
from typing import Dict, List, Optional, Any, TYPE_CHECKING

from negmas import UtilityFunction
from ...common import AgentMechanismInterface
from ...negotiators import Negotiator
from ...outcomes import Issue
from ...situated import Agent
from ...situated import Contract, Breach
from .common import (
    ManufacturingProfileCompiled,
    ProductManufacturingInfo,
    Process,
    Product,
    Loan,
    CFP,
    FinancialReport,
)

if TYPE_CHECKING:
    from .awi import SCMLAWI

__all__ = ["SCMLAgent"]


class SCMLAgent(Agent):
    """The base for all SCM Agents"""

    # @todo remove negotiator_type from here and add it independently to consumer, miner, and greedy_factory_manager
    def __init__(self, name: str = None):
        super().__init__(name=name)
        self.line_profiles: Dict[int, ManufacturingProfileCompiled] = {}
        """A mapping specifying for each `Line` index, all the profiles used to run it in the factory"""
        self.process_profiles: Dict[int, ManufacturingProfileCompiled] = {}
        """A mapping specifying for each `Process` index, all the profiles used to run it in the factory"""
        self.producing: Dict[int, List[ProductManufacturingInfo]] = defaultdict(list)
        """Mapping from a product to all manufacturing processes that can generate it"""
        self.consuming: Dict[int, List[ProductManufacturingInfo]] = defaultdict(list)
        """Mapping from a product to all manufacturing processes that can consume it"""
        self.compiled_profiles: List[ManufacturingProfileCompiled] = []
        """All the profiles to be used by the factory belonging to this agent compiled to use indices"""
        self.immediate_negotiations = False
        """Whether or not negotiations start immediately upon registration (default is to start on the next production 
        step)"""
        self.negotiation_speed_multiple: int = 1
        """The number of negotiation rounds (steps) conducted in a single production step"""
        self.transportation_delay: int = 0
        """Transportation delay in the system. Default is zero"""
        self.products: List[Product] = []
        """List of products in the system"""
        self.processes: List[Process] = []
        """List of processes in the system"""

    @property
    def awi(self) -> "SCMLAWI":
        """Returns the Agent-World-Interface through which the agent does all of its actions in the world.

        A single excption is request_negotiation for which it is recommended to actually call the helper method
        on the agent itself instead of directly calling the AWI version."""
        return self._awi

    @awi.setter
    def awi(self, awi: "SCMLAWI"):
        """Sets the AWI. Not to be used by agents. Only used by the world simulation itself."""
        self._awi = awi

    def init_(self):
        """The initialization function called by the world directly.

        It does the following actions by default:

            1. copies some of the static world settings to the agent to make them available without calling the AWI.
            2. prepares production related properties like producing, consuming, line_profiles, compiled_profiles, etc.
            3. registers interest in all products that the agent can produce or consume in its factory.
            4. finally it calls any custom initialization logic implemented in `init`()

        See Also:

            `init`, `step`

        """
        # noinspection PyUnresolvedReferences
        self.products = self.awi.products  # type: ignore
        # noinspection PyUnresolvedReferences
        self.processes = self.awi.processes  # type: ignore
        self.negotiation_speed_multiple = self.awi.bb_read(
            "settings", "negotiation_speed_multiple"
        )
        self.immediate_negotiations = self.awi.bb_read(
            "settings", "immediate_negotiations"
        )
        self.transportation_delay = self.awi.bb_read(
            section="settings", key="transportation_delay"
        )

        factory = self.awi.state
        if factory is None:
            raise ValueError("Cannot init any SCMLAgent without specifying a factory")
        profiles = factory.profiles
        self.line_profiles = defaultdict(list)
        self.process_profiles = defaultdict(list)
        self.compiled_profiles = []
        self.producing = defaultdict(list)
        self.consuming = defaultdict(list)
        p2i = dict(zip(self.processes, range(len(self.processes))))
        for index, profile in enumerate(profiles):
            compiled = ManufacturingProfileCompiled.from_manufacturing_profile(
                profile=profile, process2ind=p2i
            )
            self.compiled_profiles.append(compiled)
            self.line_profiles[profile.line].append(compiled)
            self.process_profiles[profile.process].append(compiled)
            process = profile.process
            for outpt in process.outputs:
                step = int(math.ceil(outpt.step * profile.n_steps))
                self.producing[outpt.product].append(
                    ProductManufacturingInfo(
                        profile=index, quantity=outpt.quantity, step=step
                    )
                )

            for inpt in process.inputs:
                step = int(math.floor(inpt.step * profile.n_steps))
                self.consuming[inpt.product].append(
                    ProductManufacturingInfo(
                        profile=index, quantity=inpt.quantity, step=step
                    )
                )
        self.awi.register_interest(
            list(set(itertools.chain(self.producing.keys(), self.consuming.keys())))
        )
        self.init()

    def can_expect_agreement(self, cfp: "CFP", margin: int):
        """
        Checks if it is possible in principle to get an agreement on this CFP by the time it becomes executable
        Args:
            margin:
            cfp:

        Returns:

        """
        return (
            cfp.max_time
            >= self.awi.current_step + 1 - int(self.immediate_negotiations) + margin
        )

    def _create_annotation(self, cfp: "CFP"):
        """Creates full annotation based on a cfp that the agent is receiving"""
        partners = [self.id, cfp.publisher]
        annotation = {"cfp": cfp, "partners": partners}
        if cfp.is_buy:
            annotation["seller"] = self.id
            annotation["buyer"] = cfp.publisher
        else:
            annotation["buyer"] = self.id
            annotation["seller"] = cfp.publisher
        return annotation

    def _respond_to_negotiation_request(
        self,
        initiator: str,
        partners: List[str],
        issues: List[Issue],
        annotation: Dict[str, Any],
        mechanism: AgentMechanismInterface,
        role: Optional[str],
        req_id: Optional[str],
    ) -> Optional[Negotiator]:
        """
        Called by the mechanism to ask for joining a negotiation. The agent can refuse by returning a None

        Args:
            initiator: The ID of the agent that initiated the negotiation request
            partners: The partner list (will include this agent)
            issues: The list of issues
            annotation: Any annotation specific to this negotiation.
            mechanism: The mechanism that started the negotiation
            role: The role of this agent in the negotiation
            req_id: The req_id passed to the AWI when starting the negotiation (only to the initiator).

        Returns:
            None to refuse the negotiation or a `Negotiator` object appropriate to the given mechanism to accept it.

        Remarks:

            - It is expected that world designers will introduce a better way to respond and override this function to
              call it

        """
        cfp = annotation["cfp"]
        return self.respond_to_negotiation_request(cfp=cfp, partner=cfp.publisher)

    def request_negotiation(
        self, cfp: CFP, negotiator: Negotiator = None, ufun: UtilityFunction = None
    ) -> bool:
        """
        Requests a negotiation from the AWI while keeping track of available negotiation requests

        Args:

            cfp:
            negotiator:
            ufun:

        Returns:

            Whether the negotiation request was successful indicating that the partner accepted the negotiation
        """
        if negotiator is not None and ufun is not None:
            negotiator.utility_function = ufun
        req_id = self._add_negotiation_request_info(
            issues=cfp.issues,
            partners=[self.id, cfp.publisher],
            annotation=None,
            negotiator=negotiator,
            extra=None,
        )
        return self.awi.request_negotiation(cfp=cfp, req_id=req_id)

    # ------------------------------------------------------------------
    # EVENT CALLBACKS (Called by the `World` when certain events happen)
    # ------------------------------------------------------------------

    @abstractmethod
    def on_contract_executed(self, contract: Contract) -> None:
        pass

    @abstractmethod
    def on_contract_breached(
        self, contract: Contract, breaches: List[Breach], resolution: Optional[Contract]
    ) -> None:
        pass

    @abstractmethod
    def confirm_loan(self, loan: Loan, bankrupt_if_rejected: bool) -> bool:
        """called by the world manager to confirm a loan if needed by the buyer of a contract that is about to be
        breached"""

    @abstractmethod
    def on_contract_nullified(
        self, contract: Contract, bankrupt_partner: str, compensation: float
    ) -> None:
        """Will be called whenever a contract the agent is involved in is nullified because another partner went
        bankrupt"""

    @abstractmethod
    def on_agent_bankrupt(self, agent_id: str) -> None:
        """
        Will be called whenever any agent goes bankrupt

        Args:

            agent_id: The ID of the agent that went bankrupt

        Remarks:

            - Agents can go bankrupt in two cases:

                1. Failing to pay one installments of a loan they bought and refusing (or being unable to) get another
                   loan to pay it.
                2. Failing to pay a penalty on a sell contract they failed to honor (and refusing or being unable to get
                   a loan to pay for it).

            - All built-in agents ignore this call and they use the bankruptcy list ONLY to decide whether or not to
              negotiate in their `on_new_cfp` and `respond_to_negotiation_request` callbacks by pulling the
              bulletin-board using the helper function `is_bankrupt` of their AWI.
        """

    @abstractmethod
    def confirm_partial_execution(
        self, contract: Contract, breaches: List[Breach]
    ) -> bool:
        """Will be called whenever a contract cannot be fully executed due to breaches by the other partner.

        Args:

            contract: The contract that was breached
            breaches: A list of all the breaches committed.

        Remarks:

            - Will not be called if both partners committed breaches.
        """

    @abstractmethod
    def confirm_contract_execution(self, contract: Contract) -> bool:
        """Called before executing any agreement"""
        return True

    @abstractmethod
    def respond_to_negotiation_request(
        self, cfp: "CFP", partner: str
    ) -> Optional[Negotiator]:
        """Called when a prospective partner requests a negotiation to start"""

    @abstractmethod
    def on_new_cfp(self, cfp: "CFP"):
        """Called when a new CFP for a product for which the agent registered interest is published"""

    @abstractmethod
    def on_remove_cfp(self, cfp: "CFP"):
        """Called when a new CFP for a product for which the agent registered interest is removed"""

    @abstractmethod
    def on_new_report(self, report: FinancialReport):
        """Called whenever a financial report is published.

        Args:

            report: The financial report giving details of the standing of an agent at some time (see `FinancialReport`)

        Remarks:

            - Agents must opt-in to receive these calls by calling `receive_financial_reports` on their AWI
        """

    @abstractmethod
    def on_inventory_change(self, product: int, quantity: int, cause: str) -> None:
        """
        Received whenever something moves in or out of the factory's storage

        Args:
            product: Product index.
            quantity: Negative value for products moving out and positive value for products moving in
            cause: The cause of the change. Possibilities include:

                   - contract: Contract execution
                   - insurance: Received from insurance company
                   - bankruptcy: Liquidated due to bankruptcy
                   - transport: Arrival of goods (when transportation delay in the system is > 0).
        """

    @abstractmethod
    def on_cash_transfer(self, amount: float, cause: str) -> None:
        """
        Received whenever money is transferred to the factory or from it.

        Args:
            amount: Amount of money (negative for transfers out of the factory, positive for transfers to it).
            cause: The cause of the change. Possibilities include:

                   - contract: Contract execution
                   - insurance: Received from insurance company
                   - bankruptcy: Liquidated due to bankruptcy
                   - transfer: Arrival of transferred money (when transfer delay in the system is > 0).
        """
