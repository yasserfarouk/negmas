"""Implements all builtin banks."""
from abc import ABC
from collections import defaultdict
from typing import Dict, List, Optional, TYPE_CHECKING, Any

from negmas import Issue, Negotiator, Mechanism, AgentMechanismInterface, MechanismState
from negmas.situated import Agent, RenegotiationRequest, Contract, Breach
from .common import *

if TYPE_CHECKING:
    from .agent import SCMLAgent

__all__ = ["DefaultBank", "Bank"]


class Bank(Agent, ABC):
    """Base class for all banks"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._world = None

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
        pass

    def on_neg_request_rejected(self, req_id: str, by: Optional[List[str]]):
        pass

    def on_neg_request_accepted(self, req_id: str, mechanism: AgentMechanismInterface):
        pass

    def on_negotiation_failure(
        self,
        partners: List[str],
        annotation: Dict[str, Any],
        mechanism: AgentMechanismInterface,
        state: MechanismState,
    ) -> None:
        pass

    def on_negotiation_success(
        self, contract: Contract, mechanism: AgentMechanismInterface
    ) -> None:
        pass

    def on_contract_signed(self, contract: Contract) -> None:
        pass

    def on_contract_cancelled(self, contract: Contract, rejectors: List[str]) -> None:
        pass

    def sign_contract(self, contract: Contract) -> Optional[str]:
        pass

    def respond_to_negotiation_request(
        self,
        initiator: str,
        partners: List[str],
        issues: List[Issue],
        annotation: Dict[str, Any],
        mechanism: Mechanism,
        role: Optional[str],
        req_id: str,
    ) -> Optional[Negotiator]:
        pass

    def on_contract_executed(self, contract: Contract) -> None:
        pass

    def on_contract_breached(
        self, contract: Contract, breaches: List[Breach], resolution: Optional[Contract]
    ) -> None:
        pass


class DefaultBank(Bank):
    """Represents a bank in the world"""

    def init(self):
        pass

    def respond_to_negotiation_request(
        self,
        initiator: str,
        partners: List[str],
        issues: List[Issue],
        annotation: Dict[str, Any],
        mechanism: Mechanism,
        role: Optional[str],
        req_id: str,
    ) -> Optional[Negotiator]:
        pass

    def __init__(
        self,
        minimum_balance: float,
        interest_rate: float,
        interest_max: float,
        balance_at_max_interest: float,
        installment_interest: float,
        time_increment: float,
        a2f: Dict[str, Factory],
        disabled: bool = False,
        name: str = None,
    ):
        super().__init__(name=name)
        self.storage: Dict[int, int] = defaultdict(int)
        self.wallet: float = 0.0
        self.disabled = disabled
        self.loans: Dict[SCMLAgent, List[Loan]] = defaultdict(list)
        self.minimum_balance = minimum_balance
        self.interest_rate = interest_rate
        self.interest_max = interest_max
        self.installment_interest = installment_interest
        self.time_increment = time_increment
        self.balance_at_max_interest = balance_at_max_interest
        self._credit_rating: Dict[str, float] = defaultdict(float)
        self.a2f = a2f

    def set_renegotiation_agenda(
        self, contract: Contract, breaches: List[Breach]
    ) -> Optional[RenegotiationRequest]:
        return None

    def respond_to_renegotiation_request(
        self, contract: Contract, breaches: List[Breach], agenda: RenegotiationRequest
    ) -> Optional[Negotiator]:
        raise ValueError("The bank does not receive callbacks")

    def _evaluate_loan(
        self,
        agent: "SCMLAgent",
        amount: float,
        n_installments: int,
        starts_at: int,
        installment_loan=False,
    ) -> Optional[Loan]:
        """Evaluates the interest that will be imposed on the agent to buy_loan that amount"""
        if self.disabled:
            return None
        factory = self.a2f[agent.id]
        balance = factory.balance

        if self.interest_rate is None:
            return None
        interest = self.installment_interest if installment_loan else self.interest_rate

        if balance < 0 and self.interest_max is not None:
            interest += (
                balance * (interest - self.interest_max) / self.balance_at_max_interest
            )
        interest += max(0, starts_at - self.awi.current_step) * self.time_increment
        interest += self._credit_rating[agent.id]
        total = amount * (1 + interest) ** n_installments
        installment = total / n_installments
        if self.minimum_balance is not None and balance - total < -self.minimum_balance:
            return None
        return Loan(
            amount=amount,
            total=total,
            interest=interest,
            n_installments=n_installments,
            installment=installment,
            starts_at=starts_at,
        )

    def evaluate_loan(
        self, agent: "SCMLAgent", amount: float, start_at: int, n_installments: int
    ) -> Optional[Loan]:
        """Evaluates the interest that will be imposed on the agent to buy_loan that amount"""
        if self.disabled:
            return None
        return self._evaluate_loan(
            agent=agent,
            amount=amount,
            n_installments=n_installments,
            installment_loan=False,
            starts_at=start_at,
        )

    def _buy_loan(
        self,
        agent: "SCMLAgent",
        loan: Loan,
        beneficiary: Agent,
        contract: Optional[Contract],
        bankrupt_if_rejected=False,
    ) -> Optional[Loan]:
        if self.disabled:
            return None
        if loan is None:
            return loan
        factory = self.a2f[agent.id]
        if agent.confirm_loan(loan=loan, bankrupt_if_rejected=bankrupt_if_rejected):
            self.loans[agent].append(loan)
            self.awi.logdebug(f"Bank: {agent.name} borrowed {str(loan)}")
            factory.receive(loan.amount)
            factory.add_loan(loan.total)
            self.wallet -= loan.amount
            return loan
        elif bankrupt_if_rejected:
            # the agent rejected a loan with bankrupt_if_rejected, bankrupt the agent
            self._world.make_bankrupt(
                agent, amount=loan.amount, beneficiary=beneficiary, contract=contract
            )
            return None
        return None

    def buy_loan(
        self,
        agent: "SCMLAgent",
        amount: float,
        n_installments: int,
        beneficiary: Agent,
        contract: Optional[Contract],
        force: bool = False,
    ) -> Optional[Loan]:
        """Gives a loan of amount to agent at the interest calculated using `evaluate_loan`"""
        if self.disabled:
            return None
        loan = self.evaluate_loan(
            amount=amount,
            agent=agent,
            n_installments=n_installments,
            start_at=self.awi.current_step,
        )
        return self._buy_loan(
            agent=agent,
            loan=loan,
            bankrupt_if_rejected=force,
            beneficiary=beneficiary,
            contract=contract,
        )

    def step(self):
        """Takes payments from agents"""
        # apply interests and pay loans
        # -----------------------------
        if self.disabled:
            return
        t = self.awi.current_step
        delayed_payments = 0.0
        # for every agent with loans
        for agent, loans in self.loans.items():
            factory = self.a2f[agent.id]
            keep = [True] * len(
                loans
            )  # a flag to tell whether a loan is to be kept for future processing
            unpaid = []  # any new loans that may arise from failure to pay installments
            unavailable = 0.0
            for i, loan in enumerate(loans):
                # if there are no remaining installments or I am in the grace period do not do anything
                if loan.n_installments <= 0 or loan.starts_at > t:
                    continue

                # pay as much as possible from the agent's wallet (which may be zero)
                wallet = factory.wallet
                payment = max(0.0, min(loan.installment, wallet))
                loan.amount -= payment
                factory.pay(payment)
                factory.add_loan(-payment)
                self.wallet += payment

                if payment >= loan.installment:
                    # reduce the number of remaining installments if needed
                    loan.n_installments -= 1
                else:
                    # if the payment is not enough for the installment, try to get a new loan
                    unavailable += loan.installment - payment
                    unpaid.append((i, loan))
                # if the loan is completely paid, mark it for removal
                if loan.n_installments <= 0:
                    keep[i] = False
                self.awi.logdebug(
                    f"Bank: {agent.name} payed {payment} (of {loan.installment}) "
                    f"[{loan.n_installments} remain]"
                )
            if unavailable > 0.0:
                new_loan = self._evaluate_loan(
                    agent=agent,
                    amount=unavailable,
                    n_installments=1,
                    installment_loan=True,
                    starts_at=t + 1,
                )
                if new_loan is None:
                    self._reduce_credit_rating(agent=agent, unavailable=unavailable)
                    self.awi.logdebug(
                        f"Bank: CR of {agent.name} was reduced for failure to pay {unavailable}"
                    )
                elif (
                    self._buy_loan(
                        agent=agent,
                        loan=new_loan,
                        bankrupt_if_rejected=True,
                        beneficiary=self,
                        contract=None,
                    )
                    is not None
                ):
                    self.awi.logdebug(
                        f"Bank: {agent.name} payed an installment by a new loan {str(new_loan)}"
                    )
                    factory.add_loan(-new_loan.amount)
                    factory.pay(new_loan.amount)
                    self.wallet += new_loan.amount
                    for indx, loan in unpaid:
                        loan.amount -= loan.installment
                        loan.n_installments -= 1
                        if loan.n_installments <= 0:
                            keep[indx] = False

            # remove marked loans (that were completely paid)
            self.loans[agent] = [l for i, l in enumerate(loans) if keep[i]]

        # remove records of agents that paid all their loans
        self.loans = {k: v for k, v in self.loans.items() if len(v) > 0}

    def _reduce_credit_rating(self, agent: Agent, unavailable: float):
        """Updates the credit rating when the agent fails to pay an installment"""
        self._credit_rating[agent.id] -= unavailable

    def credit_rating(self, agent_id: str) -> float:
        if self.disabled:
            return 1
        return self._credit_rating.get(agent_id, 1.0)
