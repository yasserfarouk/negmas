from abc import ABC
from collections import defaultdict
from typing import TYPE_CHECKING, Any

from negmas import Issue, Negotiator, Mechanism
from negmas.situated import Agent, RenegotiationRequest, Contract, Breach
from .common import *

if True: # if TYPE_CHECKING:
    from typing import Dict, List, Optional

__all__ = [
    'DefaultBank', 'Bank'
]


class Bank(Agent, ABC):
    """Base class for all banks"""


class DefaultBank(Bank):
    """Represents a bank in the world"""

    def respond_to_negotiation_request(self, initiator: str, partners: List[str], issues: List[Issue],
                                       annotation: Dict[str, Any], mechanism: Mechanism, role: Optional[str],
                                       req_id: str) -> Optional[Negotiator]:
        pass

    def __init__(self, minimum_balance: float, interest_rate: float
                 , interest_max: float, balance_at_max_interest: float, installment_interest: float
                 , time_increment: float, a2f: Dict[str, Factory], name: str = None):
        super().__init__(name=name)
        self.storage: Dict[int, int] = defaultdict(int)
        self.wallet: float = 0.0
        self.loans: Dict[SCMLAgent, List[Loan]] = defaultdict(list)
        self.minimum_balance = minimum_balance
        self.interest_rate = interest_rate
        self.interest_max = interest_max
        self.installment_interest = installment_interest
        self.time_increment = time_increment
        self.balance_at_max_interest = balance_at_max_interest
        self.credit_rating: Dict[str, float] = defaultdict(float)
        self.a2f = a2f

    def set_renegotiation_agenda(self, contract: Contract, breaches: List[Breach]) -> Optional[RenegotiationRequest]:
        return None

    def respond_to_renegotiation_request(self, contract: Contract, breaches: List[Breach]
                                         , agenda: RenegotiationRequest) -> Optional[Negotiator]:
        raise ValueError('The bank does not receive callbacks')


    def _evaluate_loan(self, agent: SCMLAgent, amount: float, n_installments: int, starts_at: int
                       , installment_loan=False) -> Optional[Loan]:
        """Evaluates the interest that will be imposed on the agent to buy_loan that amount"""
        factory = self.a2f[agent.id]
        balance = factory.balance

        if self.interest_rate is None:
            return None
        interest = self.installment_interest if installment_loan else self.interest_rate

        if balance < 0 and self.interest_max is not None:
            interest += balance * (interest - self.interest_max) / self.balance_at_max_interest
        interest += max(0, starts_at - self.awi.current_step) * self.time_increment
        interest += self.credit_rating[agent.id]
        total = amount * (1 + interest) ** n_installments
        installment = total / n_installments
        if self.minimum_balance is not None and balance - total < - self.minimum_balance:
            return None
        return Loan(amount=amount, total=total, interest=interest
                    , n_installments=n_installments, installment=installment, starts_at=starts_at)

    def evaluate_loan(self, agent: SCMLAgent, amount: float, start_at: int, n_installments: int) -> Optional[Loan]:
        """Evaluates the interest that will be imposed on the agent to buy_loan that amount"""
        return self._evaluate_loan(agent=agent, amount=amount, n_installments=n_installments, installment_loan=False
                                   , starts_at=start_at)

    def _buy_loan(self, agent: SCMLAgent, loan: Loan, force=False) -> Optional[Loan]:
        if loan is None:
            return loan
        factory = self.a2f[agent.id]
        if force or agent.confirm_loan(loan=loan):
            self.loans[agent].append(loan)
            self.awi.logdebug(f'Bank: {agent.name} borrowed {str(loan)}')
            factory.receive(loan.amount)
            factory.add_loan(loan.total)
            self.wallet -= loan.amount
        return loan

    def buy_loan(self, agent: SCMLAgent, amount: float, n_installments: int
                 , force: bool = False) -> Optional[Loan]:
        """Gives a loan of amount to agent at the interest calculated using `evaluate_loan`"""
        loan = self.evaluate_loan(amount=amount, agent=agent, n_installments=n_installments
                                  , start_at=self.awi.current_step)
        return self._buy_loan(agent=agent, loan=loan, force=force)

    def step(self):
        """Takes payments from agents"""
        # apply interests and pay loans
        # -----------------------------
        t = self.awi.current_step
        delayed_payments = 0.0
        # for every agent with loans
        for agent, loans in self.loans.items():
            factory = self.a2f[agent.id]
            keep = [True] * len(loans)  # a flag to tell whether a loan is to be kept for future processing
            new_loans = []  # any new loans that may arise from failure to pay installments
            for i, loan in enumerate(loans):
                # if there are no remaining installments or I am in the grace period do not do anything
                if loan.n_installments <= 0 or loan.starts_at > t:
                    continue

                # pay as much as possible from the agent's wallet (which may be zero)
                wallet = factory.wallet
                payment = max(0, min(loan.installment, wallet))
                loan.amount -= payment
                factory.pay(payment)
                factory.add_loan(-payment)
                self.wallet += payment

                if payment >= loan.installment:
                    # reduce the number of remaining installments if needed
                    loan.n_installments -= 1
                else:
                    # if the payment is not enough for the installment, try to get a new loan
                    unavailable = loan.installment - payment
                    new_loan = self._evaluate_loan(agent=agent, amount=unavailable, n_installments=1
                                                   , installment_loan=True, starts_at=t + 1)
                    if new_loan is None:
                        # @todo delete l/lmax from interest calculation and base it only on cr. cr goes exponentially with
                        # a discounted version of the total failures to pay
                        # The agent does not have enough money and cannot get a new loan, blacklist it
                        self.credit_rating[agent.id] += unavailable
                        self.awi.logdebug(f'Bank: {agent.name} blacklisted for {unavailable} (of {loan.installment})')
                    else:
                        # The agent does not have enough money but can pay the installment by getting a loan
                        # we will get this loan later (the new_loans loop) so pre-pay it from the wallet
                        new_loans.append(new_loan)
                        self.awi.logdebug(f'Bank: {agent.name} payed an installment of {str(loan)} '
                                          f'by a new loan {str(new_loan)}')
                        loan.amount -= new_loan.amount
                        delayed_payments += new_loan.amount
                        self.wallet += new_loan.amount
                        loan.n_installments -= 1
                        payment = loan.installment
                # if the loan is completely paid, mark it for removal
                if loan.n_installments <= 0:
                    keep[i] = False
                self.awi.logdebug(f'Bank: {agent.name} payed {payment} (of {loan.installment}) '
                                  f'[{loan.n_installments} remain]')
            # remove marked loans (that were completely paid)
            self.loans[agent] = [l for i, l in enumerate(loans) if keep[i]]

            # add new loans (that are needed to pay installments)
            for loan in new_loans:
                self._buy_loan(agent=agent, loan=loan, force=True)
            factory.pay(delayed_payments)
            factory.add_loan(-delayed_payments)
        # remove records of agents that paid all their loans
        self.loans = {k: v for k, v in self.loans.items() if len(v) > 0}
