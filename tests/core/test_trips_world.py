from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from random import randint, random, sample, shuffle
from typing import Any, Callable, Collection, Dict, List, Optional, Set

import numpy as np

from negmas import MechanismState, Negotiator, NegotiatorMechanismInterface
from negmas.gb.negotiators.base import GBNegotiator
from negmas.outcomes import Issue, dict2outcome, make_issue
from negmas.outcomes.common import Outcome
from negmas.preferences import LinearUtilityFunction, UtilityFunction
from negmas.sao.negotiators import RandomNegotiator, SAONegotiator
from negmas.serialization import to_flat_dict
from negmas.situated import (
    Action,
    Agent,
    AgentWorldInterface,
    Breach,
    Contract,
    RenegotiationRequest,
    World,
)


# just repeating the code from the previous tutorial
class AWI(AgentWorldInterface):
    @property
    def n_negs(self) -> int:
        """Number of negotiations an agent can start in a step (holiday season)"""
        return self._world.neg_quota_step

    @property
    def agents(self):
        """List of all other agent IDs"""
        return list(_ for _ in self._world.agents.keys() if _ != self.agent.id)

    def request_negotiation(
        self, partners: list[str], negotiator: SAONegotiator
    ) -> bool:
        """A convenient way to request negotiations"""
        _world: TripsWorld = self._world  # type: ignore
        if self.agent.id not in partners:
            partners.append(self.agent.id)
        req_id = self.agent.create_negotiation_request(
            issues=_world.ISSUES,
            partners=partners,
            negotiator=negotiator,
            annotation=dict(),
            extra=dict(negotiator_id=negotiator.id),
        )
        return self.request_negotiation_about(
            issues=_world.ISSUES, partners=partners, req_id=req_id
        )


class TripsWorld(World):
    ISSUES = [
        make_issue((0.0, 1.0), "cost"),
        make_issue(2, "active"),
        make_issue((1, 7), "duration"),
    ]

    def __init__(self, *args, **kwargs):
        """Initialize the world"""
        kwargs["awi_type"] = AWI
        kwargs["negotiation_quota_per_step"] = kwargs.get(
            "negotiation_quota_per_step", 8
        )
        kwargs["force_signing"] = True
        kwargs["default_signing_delay"] = 0
        super().__init__(*args, **kwargs)
        self._contracts: dict[int, list[Contract]] = defaultdict(list)
        self._total_utility: dict[str, float] = defaultdict(float)
        self._ufuns: dict[str, UtilityFunction] = dict()
        self._breach_prob: dict[str, float] = dict()

    def join(self, x, ufun=None, breach_prob=None, **kwargs):
        """Define the ufun and breach-probability for each agent"""
        super().join(x, **kwargs)
        weights = (np.random.rand(len(self.ISSUES)) - 0.5).tolist()
        x.ufun = (
            LinearUtilityFunction(
                weights=weights, reserved_value=0.0, issues=self.ISSUES
            )
            if ufun is None
            else ufun
        )
        self._ufuns[x.id] = x.ufun
        self._breach_prob[x.id] = random() * 0.1 if breach_prob is None else breach_prob

    def simulation_step(self, stage: int = 0):
        """What happens in this world? Nothing"""
        pass

    def get_private_state(self, agent: Agent) -> dict:
        """What is the information available to agents? total utility points"""
        return dict(total_utility=self._total_utility[agent.id])

    def execute_action(
        self, action: Action, agent: Agent, callback: Callable | None = None
    ) -> bool:
        """Executing actions by agents? No actions available"""
        ...

    def on_contract_signed(self, contract: Contract) -> None:
        """Save the contract to be executed in the following hoiday season (step)"""
        super().on_contract_signed(contract)
        self._contracts[self.current_step + 1].append(contract)

    def executable_contracts(self) -> Collection[Contract]:
        """What contracts are to be executed in the current step?
        Ones that were signed the previous step"""
        return self._contracts[self.current_step]

    def order_contracts_for_execution(
        self, contracts: Collection[Contract]
    ) -> Collection[Contract]:
        """What should be the order of contract execution? Random"""
        contracts = list(contracts)
        shuffle(contracts)
        return contracts

    def start_contract_execution(self, contract: Contract) -> set[Breach] | None:
        """What should happen when a contract comes due?
        1. Find out if it will be breached
        2. If not, add to each agent its utility from the trip
        """
        breaches = []
        for aid in contract.partners:
            if random() < self._breach_prob[aid]:
                breaches.append(
                    Breach(
                        contract,
                        aid,
                        "breach",
                        victims=[_ for _ in contract.partners if _ != aid],
                    )
                )
        if len(breaches) > 0:
            return set(breaches)
        for aid in contract.partners:
            if not isinstance(contract.agreement, dict):
                continue
            self._total_utility[aid] += self._ufuns[aid](
                dict2outcome(contract.agreement, issues=self.ISSUES)
            )
        return set()

    def complete_contract_execution(
        self, contract: Contract, breaches: list[Breach], resolution: Contract
    ) -> None:
        """What happens if a breach was resolved? Nothing. They cannot"""
        pass

    def delete_executed_contracts(self) -> None:
        """Removes all contracts for the current step"""
        if self._current_step in self._contracts.keys():
            del self._contracts[self.current_step]

    def contract_record(self, contract: Contract) -> dict[str, Any]:
        """Convert the contract into a dictionary for saving"""
        return to_flat_dict(contract)

    def breach_record(self, breach: Breach) -> dict[str, Any]:
        """Convert the breach into a dictionary for saving"""
        return to_flat_dict(breach)

    def contract_size(self, contract: Contract) -> float:
        """How good is a contract? Welfare"""
        if contract.agreement is None:
            return 0.0
        return sum(
            self._ufuns[aid](dict2outcome(contract.agreement, issues=self.ISSUES))
            for aid in contract.partners
            if isinstance(contract.agreement, dict)
        )

    def post_step_stats(self):
        for aid, agent in self.agents.items():
            self._stats[f"total_utility_{agent.name}"].append(self._total_utility[aid])


class Person(Agent, ABC):
    @abstractmethod
    def step(self):
        ...

    @abstractmethod
    def init(self):
        ...

    @abstractmethod
    def respond_to_negotiation_request(
        self,
        initiator: str,
        partners: list[str],
        mechanism: NegotiatorMechanismInterface,
    ) -> Negotiator | None:
        ...

    def _respond_to_negotiation_request(
        self,
        initiator: str,
        partners: list[str],
        issues: list[Issue],
        annotation: dict[str, Any],
        mechanism: NegotiatorMechanismInterface,
        role: str | None,
        req_id: str | None,
    ) -> Negotiator | None:
        return self.respond_to_negotiation_request(initiator, partners, mechanism)

    def on_neg_request_rejected(self, req_id: str, by: list[str] | None):
        pass

    def on_neg_request_accepted(
        self, req_id: str, mechanism: NegotiatorMechanismInterface
    ):
        pass

    def on_negotiation_failure(
        self,
        partners: list[str],
        annotation: dict[str, Any],
        mechanism: NegotiatorMechanismInterface,
        state: MechanismState,
    ) -> None:
        pass

    def on_negotiation_success(
        self, contract: Contract, mechanism: NegotiatorMechanismInterface
    ) -> None:
        pass

    def set_renegotiation_agenda(
        self, contract: Contract, breaches: list[Breach]
    ) -> RenegotiationRequest | None:
        pass

    def respond_to_renegotiation_request(
        self, contract: Contract, breaches: list[Breach], agenda: RenegotiationRequest
    ) -> Negotiator | None:
        pass

    def on_contract_executed(self, contract: Contract) -> None:
        pass

    def on_contract_breached(
        self, contract: Contract, breaches: list[Breach], resolution: Contract | None
    ) -> None:
        pass


class RandomPerson(Person):
    def step(self):
        awi: AWI = self.awi  # type: ignore
        # get IDs of all ogher agents from the AWI
        agents = awi.agents
        # request the maximum number of negotiations possible
        for _ in range(awi.n_negs):
            # for each negotiation, use a random subset of partners and a random negotiator
            awi.request_negotiation(
                partners=sample(agents, k=randint(1, len(agents) - 1)),
                negotiator=RandomNegotiator(),
            )

    def init(self):
        # we need no initialization
        pass

    def respond_to_negotiation_request(
        self,
        initiator: str,
        partners: list[str],
        mechanism: NegotiatorMechanismInterface,
    ) -> Negotiator | None:
        # just us a random negotiator for everything
        return RandomNegotiator()


def test_trips_run_random():
    world = TripsWorld(n_steps=10, construct_graphs=True)
    for i in range(5):
        world.join(RandomPerson(name=f"a{i}"))
    world.run_with_progress()
