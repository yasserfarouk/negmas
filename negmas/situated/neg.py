"""
Defines a world for running negotiations directly
"""
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Collection, Dict, List, Optional, Set, Type, Union, Iterable

from negmas import AgentMechanismInterface, MechanismState
from negmas.helpers import get_class, get_full_type_name, instantiate
from negmas.negotiators import Negotiator
from negmas.outcomes import Issue
from negmas.serialization import serialize, to_flat_dict, deserialize
from negmas.situated import (
    Action,
    Agent,
    AgentWorldInterface,
    Breach,
    Contract,
    NoContractExecutionMixin,
    RenegotiationRequest,
    World,
)
from negmas.utilities import UtilityFunction

__all__ = [
    "NegWorld",
    "NegAgent",
    "NegDomain",
]


def _unwrap_negotiators(types, params):
    types = list(types)
    old_types = [_ if issubclass(_, NegAgent) else NegAgent for _ in types]
    n = len(types)

    if params is None:
        params = [dict() for _ in range(n)]

    params = [dict() if _ is None else _ for _ in params]

    for i, (t, p) in enumerate(zip(types, params)):
        t = types[i] = get_class(t)
        if issubclass(t, Negotiator):
            continue
        params[i] = p.get("negotiator_params", dict())
        types[i] = p["negotiator_type"]

    return types, params, old_types


def _wrap_in_agents(types, params, agent_types):
    types = list(types)
    n = len(types)

    if params is None:
        params = [dict() for _ in range(n)]

    if isinstance(agent_types, str):
        agent_types = get_class(agent_types)

    if not isinstance(agent_types, Iterable):
        agent_types = [agent_types] * n

    params = [dict() if _ is None else _ for _ in params]

    for i, (t, p, w) in enumerate(zip(types, params, agent_types)):
        t = types[i] = get_class(t)
        if issubclass(t, NegAgent):
            continue
        params[i].update(
            dict(negotiator_type=t, negotiator_params={k: v for k, v in p.items()})
        )
        types[i] = w
    return types, params


@dataclass
class NegDomain:
    name: str
    issues: List[Issue]
    ufuns: List[UtilityFunction]
    partner_types: List[Union[str, Negotiator]]
    index: int = 0
    partner_params: Optional[List[Optional[Dict[str, Any]]]] = None
    roles: Optional[List[str]] = None
    annotation: Optional[Dict[str, Any]] = None

    def to_dict(self):
        return dict(
            name=self.name,
            issues=[i.to_dict() for i in self.issues],
            ufuns=[serialize(u) for u in self.ufuns],
            partner_types=[get_full_type_name(_) for _ in self.partner_types],
            index=self.index,
            partner_params=serialize(self.partner_params),
            roles=self.roles,
            annotation=serialize(self.annotation),
        )

    @classmethod
    def from_dict(cls, d):
        return cls(
            name=d["name"],
            issues=[Issue.from_dict(_) for _ in d["issues"]],
            ufuns=[deserialize(u) for u in d["ufuns"]],
            partner_types=[get_class(_) for _ in d["partner_types"]],
            index=d["index"],
            partner_params=deserialize(d["partner_params"]),
            roles=d["roles"],
            annotation=deserialize(d["annotation"]),
        )


class NegAgent(Agent):
    def __init__(
        self,
        *args,
        negotiator_type: Union[str, Type[Negotiator]],
        negotiator_params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._negotiator_params = negotiator_params if negotiator_params else dict()
        self._negotiator_type = get_class(negotiator_type)

    @property
    def short_type_name(self):
        """Returns a short name of the type of this entity"""
        return self._negotiator_type.__name__.replace("Negotiator", "").replace(
            "Agent", ""
        )

    @property
    def type_name(self):
        """Returns a short name of the type of this entity"""
        return get_full_type_name(self._negotiator_type)

    @classmethod
    def _type_name(cls):
        return cls.__module__ + "." + cls.__name__

    def make_negotiator(self, ufun: Optional[UtilityFunction] = None):
        return instantiate(self._negotiator_type, ufun=ufun, **self._negotiator_params)

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
        return self.make_negotiator(self.awi.get_ufun(partners.index(self.id)))

    def step(self):
        """Called by the simulator at every simulation step"""

    def init(self):
        """Called to initialize the agent **after** the world is initialized. the AWI is accessible at this point."""

    def on_neg_request_rejected(self, req_id: str, by: Optional[List[str]]):
        """Called when a requested negotiation is rejected

        Args:
            req_id: The request ID passed to _request_negotiation
            by: A list of agents that refused to participate or None if the failure was for another reason


        """

    def on_neg_request_accepted(self, req_id: str, mechanism: AgentMechanismInterface):
        """Called when a requested negotiation is accepted"""

    def on_negotiation_failure(
        self,
        partners: List[str],
        annotation: Dict[str, Any],
        mechanism: AgentMechanismInterface,
        state: MechanismState,
    ) -> None:
        """Called whenever a negotiation ends without agreement"""

    def on_negotiation_success(
        self, contract: Contract, mechanism: AgentMechanismInterface
    ) -> None:
        """Called whenever a negotiation ends with agreement"""

    def on_contract_signed(self, contract: Contract) -> None:
        """Called whenever a contract is signed by all partners"""

    def on_contract_cancelled(self, contract: Contract, rejectors: List[str]) -> None:
        """Called whenever at least a partner did not sign the contract"""

    def respond_to_renegotiation_request(
        self, contract: Contract, breaches: List[Breach], agenda: RenegotiationRequest
    ) -> Optional[Negotiator]:
        """
        Called to respond to a renegotiation request

        Args:

            agenda:
            contract:
            breaches:

        Returns:

        """

    def on_contract_executed(self, contract: Contract) -> None:
        """
        Called after successful contract execution for which the agent is one of the partners.
        """

    def on_contract_breached(
        self, contract: Contract, breaches: List[Breach], resolution: Optional[Contract]
    ) -> None:
        """
        Called after complete processing of a contract that involved a breach.
        """

    def set_renegotiation_agenda(
        self, contract: Contract, breaches: List[Breach]
    ) -> Optional[RenegotiationRequest]:
        """
        Received by partners in ascending order of their total breach levels in order to set the
        renegotiation agenda when contract execution fails
        """


class _NegPartner(NegAgent):
    pass


class _NegAWI(AgentWorldInterface):
    def get_ufun(self, uid: int):
        return self._world._domain.ufuns[uid]


class NegWorld(NoContractExecutionMixin, World):
    def __init__(
        self,
        *args,
        domain: NegDomain,
        types: List[Union[Negotiator, NegAgent]],
        params: Optional[List[Optional[Dict[str, Any]]]] = None,
        agent_names_reveal_type: bool = True,
        compact: bool = False,
        no_logs: bool = False,
        **kwargs,
    ):
        kwargs["log_to_file"] = not no_logs
        if compact:
            kwargs["event_file_name"] = None
            kwargs["event_types"] = []
            kwargs["log_screen_level"] = logging.CRITICAL
            kwargs["log_file_level"] = logging.ERROR
            kwargs["log_negotiations"] = False
            kwargs["log_ufuns"] = False
            # kwargs["save_mechanism_state_in_contract"] = False
            kwargs["save_cancelled_contracts"] = False
            kwargs["save_resolved_breaches"] = False
            kwargs["save_negotiations"] = True
        else:
            kwargs["save_negotiations"] = True

        self.compact = compact
        super().__init__(*args, awi_type="negmas.situated.neg._NegAWI", **kwargs)

        if self.info is None:
            self.info = {}
        self.info["domain"] = serialize(domain)
        if domain.annotation is None:
            domain.annotation = dict()
        self._domain = domain
        self._received_utility: Dict[str, List[float]] = defaultdict(list)
        self._partner_utility: Dict[str, List[float]] = defaultdict(list)
        self._success: Dict[str, bool] = defaultdict(bool)
        self._received_advantage: Dict[str, List[float]] = defaultdict(list)
        self._partner_advantage: Dict[str, List[float]] = defaultdict(list)
        self._competitors: Dict[str, NegAgent] = dict()
        self._partners: Dict[str, NegAgent] = dict()
        partner_types = domain.partner_types
        partner_params = domain.partner_params

        if partner_params is None:
            partner_params = [dict() for _ in partner_types]

        for i, p in enumerate(partner_params):
            if p is None:
                partner_params[i] = dict()

        types = list(types)
        len(types)

        types, params, wrappers = _unwrap_negotiators(types, params)

        for i, t in enumerate(types):
            types[i] = get_full_type_name(t)

        self.agent_unique_types = [
            f"{t}:{hash(str(p)) if p else ''}" if len(p) > 0 else t
            for t, p in zip(types, params)
        ]

        def add_agents(types, params, wrappers, target):
            types, params = _wrap_in_agents(types, params, wrappers)
            for t, p in zip(types, params):
                agent = instantiate(t, **p)
                if agent_names_reveal_type:
                    agent.name = agent.short_type_name
                    agent.id = agent.short_type_name
                target[agent.id] = agent
                self.join(agent)

        add_agents(types, params, wrappers, self._competitors)

        add_agents(partner_types, partner_params, _NegPartner, self._partners)


    def simulation_step(self, stage):
        for aid, agent in self._competitors.items():
            partners = list(self._partners.keys())
            partners.insert(self._domain.index, aid)
            _, mechanism = self.run_negotiation(
                caller=agent,
                issues=self._domain.issues,
                partners=partners,
                roles=self._domain.roles,
                annotation=self._domain.annotation,
                negotiator=None,
            )
            agreement = mechanism.state.agreement if mechanism else None
            self._success[aid] = mechanism is not None
            agent = self.agents[aid]
            ufun = self._domain.ufuns[self._domain.index]
            u = float(ufun(agreement) if agreement else ufun.reserved_value)
            r = float(ufun.reserved_value)
            self._received_utility[aid].append(u)
            self._received_advantage[aid].append(u - r)
            pufuns = [
                p.awi.get_ufun(partners.index(pid)) for pid, p in self._partners.items()
            ]
            pu = sum(float(_(agreement)) for _ in pufuns)
            pa = sum(float(_(agreement)) - float(_.reserved_value) for _ in pufuns)
            self._partner_utility[aid].append(pu)
            self._partner_advantage[aid].append(pa)
            partner_names = [self.agents[_].name for _ in partners]
            self.loginfo(f"{agent.name} : {partner_names} -> {agreement}")

    def received_utility(self, aid: str):
        return sum(self._received_utility.get(aid, []))

    def partner_utility(self, aid: str):
        return sum(self._partner_utility.get(aid, []))

    def received_advantage(self, aid: str):
        return sum(self._received_advantage.get(aid, []))

    def partner_advantage(self, aid: str):
        return sum(self._partner_advantage.get(aid, []))

    @property
    def competitors(self):
        return self._competitors

    @property
    def partners(self):
        return self._partners

    def post_step_stats(self):
        for aid in self._competitors.keys():
            self._stats[f"has_agreement_{aid}"].append(self._success[aid])
            self._stats[f"received_utility_{aid}"].append(
                self._received_utility[aid][-1]
            )
            self._stats[f"partner_utility_{aid}"].append(self._partner_utility[aid][-1])
            self._stats[f"received_advantage_{aid}"].append(
                self._received_advantage[aid][-1]
            )
            self._stats[f"partner_advantage_{aid}"].append(
                self._partner_advantage[aid][-1]
            )

    def pre_step_stats(self):
        pass

    def order_contracts_for_execution(
        self, contracts: Collection[Contract]
    ) -> Collection[Contract]:
        return contracts

    def contract_record(self, contract: Contract) -> Dict[str, Any]:
        return to_flat_dict(contract, deep=True)

    def breach_record(self, breach: Breach) -> Dict[str, Any]:
        return to_flat_dict(breach, deep=True)

    def contract_size(self, contract: Contract) -> float:
        n = len(self._competitors)
        return 1.0 / (n if n else 1)

    def delete_executed_contracts(self) -> None:
        pass

    def execute_action(
        self, action: Action, agent: "Agent", callback: Callable = None
    ) -> bool:
        """Executes the given action by the given agent"""

    def get_private_state(self, agent: "Agent") -> dict:
        """Reads the private state of the given agent"""

    def executable_contracts(self) -> Collection[Contract]:
        return []

    def start_contract_execution(self, contract: Contract) -> Set[Breach]:
        return set()

    def complete_contract_execution(
        self, contract: Contract, breaches: List[Breach], resolution: Contract
    ) -> None:
        pass


if __name__ == "__main__":
    from negmas.sao import AspirationNegotiator, NaiveTitForTatNegotiator
    from negmas.genius.gnegotiators import Atlas3, NiceTitForTat
    from negmas.genius import genius_bridge_is_running
    from negmas.situated import save_stats
    from negmas.utilities import LinearUtilityAggregationFunction as U

    issues = [Issue(10, "quantity"), Issue(5, "price")]
    competitors = [AspirationNegotiator, NaiveTitForTatNegotiator]
    if genius_bridge_is_running():
        competitors += [Atlas3, NiceTitForTat]

    domain = NegDomain(
        name="d0",
        issues=issues,
        ufuns=[
            U.random(issues, reserved_value=(0.0, 0.2), normalized=True),
            U.random(issues, reserved_value=(0.0, 0.2), normalized=True),
        ],
        partner_types=[AspirationNegotiator],
        index=0,
    )
    world = NegWorld(
        domain=domain,
        types=competitors,
        n_steps=2,
        neg_n_steps=10,
        neg_time_limit=None,
    )
    world.run()
    print("World ran", flush=True)
    for aid in world.agents.keys():
        print(world._received_utility[aid])
    save_stats(world, world.log_folder)
