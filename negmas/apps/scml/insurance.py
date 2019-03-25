from abc import ABC
from collections import defaultdict
from typing import TYPE_CHECKING, Any

from negmas import Mechanism
from negmas.situated import Agent, RenegotiationRequest, Breach
from .common import InsurancePolicy, SCMLAgent, Factory

if True: # if TYPE_CHECKING:
    from typing import Dict, Tuple, List, Optional
    from negmas.situated import Contract
    from negmas.outcomes import Issue
    from negmas.negotiators import Negotiator

__all__ = [
    'DefaultInsuranceCompany',
    'InsuranceCompany'
]


class InsuranceCompany(Agent, ABC):
    """Base class for all insurance companies"""


class DefaultInsuranceCompany(InsuranceCompany):
    """Represents an insurance company in the world"""

    def respond_to_negotiation_request(self, initiator: str, partners: List[str], issues: List[Issue],
                                       annotation: Dict[str, Any], mechanism: Mechanism, role: Optional[str],
                                       req_id: str) -> Optional[Negotiator]:
        pass

    def __init__(self, premium: float, premium_breach_increment: float, premium_time_increment: float
                 , a2f: Dict[str, Factory], name: str = None):
        super().__init__(name=name)
        self.premium_breach_increment = premium_breach_increment
        self.premium = premium
        self.premium_time_increment = premium_time_increment
        self.insured_contracts: Dict[Tuple[Contract, str], InsurancePolicy] = dict()
        self.storage: Dict[int, int] = defaultdict(int)
        self.wallet: float = 0.0
        self.a2f = a2f

    def set_renegotiation_agenda(self, contract: Contract, breaches: List[Breach]) -> Optional[RenegotiationRequest]:
        return None

    def respond_to_renegotiation_request(self, contract: Contract, breaches: List[Breach]
                                         , agenda: RenegotiationRequest) -> Optional[Negotiator]:
        raise ValueError('The insurance company does not receive callbacks')

    def evaluate_insurance(self, contract: Contract, insured: SCMLAgent, against: SCMLAgent
                           , t: int = None) -> Optional[float]:
        """Can be called to evaluate the premium for insuring the given contract against breaches committed by others

        Args:

            against: The `SCMLAgent` to insure against
            contract: hypothetical contract
            insured: The `SCMLAgent` to buy the insurance
            t: time at which the policy is to be bought. If None, it means current step
        """

        # fail if no premium
        if self.premium is None:
            return None

        # assume the insurance is to be bought now if needed
        if t is None:
            t = self.awi.current_step

        # find the delay from contract signing. The more this is the more expensive the insurance will be
        if contract.signed_at is None:
            dt = 0
        else:
            dt = max(0, t - contract.signed_at)

        # fail if the insurance is to be bought at or after the agreed upon delivery time
        if t >= contract.agreement.get('time', -1):
            return None

        # find the total breach of the agent I am insuring against. The more this is, the more expensive the insurance
        breaches = self.awi.bb_query(section='breaches', query={'perpetrator': against})
        b = 0
        if breaches is not None:
            for _, breach in breaches.items():
                b += breach.level
        return (self.premium + b * self.premium_breach_increment) * (1 + self.premium_time_increment * dt)

    def buy_insurance(self, contract: Contract, insured: SCMLAgent, against: SCMLAgent) -> Optional[InsurancePolicy]:
        """Buys insurance for the contract by the premium calculated by the insurance company.

        Remarks:
            The agent can call `evaluate_insurance` to find the premium that will be used.
        """
        premium = self.evaluate_insurance(contract=contract, t=self.awi.current_step, insured=insured, against=against)
        factory = self.a2f[insured.id]
        if premium is None or factory.wallet < premium:
            return None
        factory.pay(premium)
        self.wallet += premium
        policy = InsurancePolicy(contract=contract, at_time=self.awi.current_step, against=against, premium=premium)
        self.insured_contracts[(contract, against.id)] = policy
        return policy

    def pay_insurance(self, contract: Contract, perpetrator: SCMLAgent) -> bool:
        """

        Args:
            contract:
            perpetrator:

        Returns:

        """
        if (contract, perpetrator.id) in self.insured_contracts.keys():
            del self.insured_contracts[(contract, perpetrator.id)]
            return True
        return False

    def step(self):
        """does nothing"""
