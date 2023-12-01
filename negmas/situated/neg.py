"""
Defines a world for running negotiations directly
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Collection, Iterable

from negmas import MechanismState, NegotiatorMechanismInterface
from negmas.helpers import get_class, get_full_type_name, instantiate
from negmas.negotiators import Negotiator
from negmas.outcomes import Issue
from negmas.preferences import Preferences
from negmas.sao import SAONegotiator
from negmas.serialization import deserialize, serialize, to_flat_dict
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

__all__ = [
    "NegWorld",
    "NegAgent",
    "Condition",
]


class NegAgent(Agent):
    """Wraps a negotiator for evaluaton"""

    def __init__(
        self,
        *args,
        negotiator_type: str | type[Negotiator],
        negotiator_params: dict[str, Any] | None = None,
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

    def make_negotiator(self, preferences: Preferences | None = None):
        """Makes a negotiator of the appropriate type passing it an optional ufun"""
        return instantiate(
            self._negotiator_type, preferences=preferences, **self._negotiator_params
        )

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
        """Responds to any negotiation request by creating a negotiator"""
        return self.make_negotiator(self.awi.get_preferences(partners.index(self.id)))

    def step(self):
        """Called by the simulator at every simulation step"""

    def init(self):
        """Called to initialize the agent **after** the world is initialized. the AWI is accessible at this point."""

    def on_neg_request_rejected(self, req_id: str, by: list[str] | None):
        """Called when a requested negotiation is rejected"""

    def on_neg_request_accepted(
        self, req_id: str, mechanism: NegotiatorMechanismInterface
    ):
        """Called when a requested negotiation is accepted"""

    def on_negotiation_failure(
        self,
        partners: list[str],
        annotation: dict[str, Any],
        mechanism: NegotiatorMechanismInterface,
        state: MechanismState,
    ) -> None:
        """Called whenever a negotiation ends without agreement"""

    def on_negotiation_success(
        self, contract: Contract, mechanism: NegotiatorMechanismInterface
    ) -> None:
        """Called whenever a negotiation ends with agreement"""

    def on_contract_signed(self, contract: Contract) -> None:
        """Called whenever a contract is signed by all partners"""

    def on_contract_cancelled(self, contract: Contract, rejectors: list[str]) -> None:
        """Called whenever at least a partner did not sign the contract"""

    def respond_to_renegotiation_request(
        self, contract: Contract, breaches: list[Breach], agenda: RenegotiationRequest
    ) -> Negotiator | None:
        """Called to respond to a renegotiation request"""

    def on_contract_executed(self, contract: Contract) -> None:
        """
        Called after successful contract execution for which the agent is one of the partners.
        """

    def on_contract_breached(
        self, contract: Contract, breaches: list[Breach], resolution: Contract | None
    ) -> None:
        """
        Called after complete processing of a contract that involved a breach.
        """

    def set_renegotiation_agenda(
        self, contract: Contract, breaches: list[Breach]
    ) -> RenegotiationRequest | None:
        """
        Received by partners in ascending order of their total breach levels in order to set the
        renegotiation agenda when contract execution fails
        """


def _unwrap_negotiators(types, params):
    """
    Removes the agent wrapping the negotiator for each type
    """
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
    """
    wraps each negotiator in `types` with an agent from `agent_types`
    """
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
        name = p.get("name", None)
        params[i] = dict(
            negotiator_type=t, negotiator_params={k: v for k, v in p.items()}
        )
        params[i]["name"] = name
        types[i] = w
    return types, params


@dataclass
class Condition:
    """
    A representation of a negotiation scenario in which a negotiator can be evaluated
    """

    name: str
    """Scenario name"""
    issues: tuple[Issue, ...]
    """The issue space as a list of issues"""
    ufuns: tuple[Preferences, ...]
    """The utility functions used by all negotiators in the scenario"""
    partner_types: tuple[str | type[Negotiator] | type[SAONegotiator], ...]
    """The types of all partners (other than the agent being evaluated). Its length must be one less than `ufuns`"""
    index: int = 0
    """The index of the negotiator being evaluated in the list of negotiators passed to the mechanism"""
    partner_params: tuple[dict[str, Any] | None, ...] | None = None
    """Any parameters used to construct partners (must be the same length as `partner_types`)"""
    roles: tuple[str, ...] | None = None
    """Roles of all negotiators (including the negotiator being evaluated) in order"""
    annotation: dict[str, Any] | None = None
    """Any extra annotation to add to the mechanism."""
    scored_indices: tuple[int, ...] | None = None
    """Indices of negotiators to be scored in this negotiation. `None` is equivalent to `[self.index]`"""

    def to_dict(self):
        return dict(
            name=self.name,
            issues=[i.to_dict() for i in self.issues],
            ufuns=[serialize(u) for u in self.ufuns],
            partner_types=[get_full_type_name(_) for _ in self.partner_types],  # type: ignore
            index=self.index,
            partner_params=serialize(self.partner_params),
            roles=self.roles,
            annotation=serialize(self.annotation),
            scored_indices=self.scored_indices,
        )

    @classmethod
    def from_dict(cls, d):
        return cls(
            name=d["name"],
            issues=tuple(Issue.from_dict(_) for _ in d["issues"]),
            ufuns=tuple(deserialize(u) for u in d["ufuns"]),  # type: ignore The type should be correct in the dict
            partner_types=tuple(get_class(_) for _ in d["partner_types"]),
            index=d["index"],
            partner_params=deserialize(d["partner_params"]),  # type: ignore The type should be correct in the dict
            roles=d["roles"],
            annotation=deserialize(d["annotation"]),  # type: ignore The type should be correct in the dict
            scored_indices=d["scored_indices"],
        )


class _NegPartner(NegAgent):
    """A `NegAgent` representing a partner that is not being evaluated"""


class _NegAWI(AgentWorldInterface):
    """The AWI for the `NegWorld`"""

    def get_preferences(self, uid: int):
        """Get the agent's ufun"""
        return self._world._scenario.ufuns[uid]


class NegWorld(NoContractExecutionMixin, World):
    """
    A world that runs a list of negotiators in a given scenario to evaluate them

    Args:
        scenario: The `NegScenario` specifying all information about the situation
                in which negotiators are to be evaluated including the partners.
        types: The negotiator types to be evaluated
        params: Any parameters needed to create negotiators
        agent_names_reveal_type: if given the agent name for each negotiator will
                                 simply be the negotiator's type
        compact: If given the system will strive to save memory and minimize logging
        no_logs: disables all logging
        kwargs: Any extra arguments to be passed to the `World` constructor
    """

    def __init__(
        self,
        *args,
        scenario: Condition,
        types: list[Negotiator | NegAgent],
        params: list[dict[str, Any] | None] | None = None,
        agent_names_reveal_type: bool = True,
        compact: bool = False,
        no_logs: bool = False,
        normalize_scores: bool = False,
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
        self.info["scenario"] = serialize(scenario)
        self.info["n_steps"] = self.n_steps
        self.info["n_negotiators"] = len(scenario.ufuns)

        if scenario.annotation is None:
            scenario.annotation = dict()
        self._scenario = scenario
        self._received_utility: dict[str, list[float]] = defaultdict(list)
        self._partner_utility: dict[str, list[float]] = defaultdict(list)
        self._success: dict[str, bool] = defaultdict(bool)
        self._received_advantage: dict[str, list[float]] = defaultdict(list)
        self._n_agreements_per_cometitor: dict[str, list[int]] = defaultdict(list)
        self._partner_advantage: dict[str, list[float]] = defaultdict(list)
        self._competitors: dict[str, NegAgent] = dict()
        self._partners: dict[str, NegAgent] = dict()
        self._n_negs_per_copmetitor = defaultdict(int)
        self._normalize_scores = normalize_scores
        self._preferences_ranges = None
        # if not all(isinstance(_, HasRange) for _ in scenario.ufuns):
        #     raise ValueError(f"Not all ufuns has a range!!!")
        if self._normalize_scores:
            self._preferences_ranges = [
                u.minmax() for u in scenario.ufuns  # type: ignore We already check just above
            ]
        partner_types = scenario.partner_types
        partner_params = scenario.partner_params

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
        def unormalize(u, indx):
            if self._normalize_scores:
                mn, mx = self._preferences_ranges[indx]
                if mx > mn:
                    u = (u - mn) / (mx - mn)
                else:
                    u = u / mx
            return u

        def calcu(ufun, indx, agreement):
            u = ufun(agreement)
            return unormalize(u, indx)

        def getid(indx, aid, special_index=self._scenario.index):
            lst = list(self._partners.keys())
            lst.insert(special_index, aid)
            return lst[indx]

        for aid, agent in self._competitors.items():
            partner_ids = list(self._partners.keys())
            partners = list(self._partners.values())
            partner_ids.insert(self._scenario.index, aid)
            partners.insert(self._scenario.index, agent)
            _, mechanism = self.run_negotiation(
                caller=agent,
                issues=self._scenario.issues,
                partners=partner_ids,
                roles=self._scenario.roles,
                annotation=self._scenario.annotation,
                negotiator=None,
            )
            agreement = mechanism.state.agreement if mechanism else None
            # agent = self.agents[aid]
            # index = self._scenario.index
            # caid = aid
            scored_indices = self._scenario.scored_indices
            if scored_indices is None:
                scored_indices = [self._scenario.index]
            for index in scored_indices:
                current_aid = getid(index, aid)
                if index == self._scenario.index:
                    assert (
                        current_aid == aid
                    ), f"{index=}, {self._scenario.index=} but the IDS are not equal: {aid=}, {current_aid=}"
                ufun = self._scenario.ufuns[index]
                u = float(
                    calcu(ufun, index, agreement)
                    if agreement
                    else unormalize(ufun.reserved_value, index)
                )
                r = unormalize(float(ufun.reserved_value), index)
                self._received_utility[current_aid].append(u)
                self._n_agreements_per_cometitor[current_aid].append(
                    int(mechanism is not None)
                )
                self._n_negs_per_copmetitor[current_aid] += 1
                self._success[current_aid] = mechanism is not None
                self._received_advantage[current_aid].append(u - r)
                pufuns = [
                    (
                        partner_ids.index(pid),
                        p.awi.get_preferences(partner_ids.index(pid)),
                    )
                    for pid, p in zip(partner_ids, partners)
                    if pid != current_aid
                ]
                pu = sum(unormalize(float(_(agreement)), i) for i, _ in pufuns)
                pa = sum(
                    unormalize(float(_(agreement)), i)
                    - unormalize(float(_.reserved_value), i)
                    for i, _ in pufuns
                )
                self._partner_utility[current_aid].append(pu)
                self._partner_advantage[current_aid].append(pa)
            partner_names = [self.agents[_].name for _ in partner_ids]
            self.loginfo(f"{partner_names} -> {agreement}")

    def received_utility(self, aid: str):
        return sum(
            self._received_utility.get(aid, [0])
        ) / self._n_negs_per_copmetitor.get(aid, 0)

    def agreement_rate(self, aid: str):
        return sum(
            self._n_agreements_per_cometitor.get(aid, [0])
        ) / self._n_negs_per_agent.get(aid, 0)

    def partner_utility(self, aid: str):
        return sum(
            self._partner_utility.get(aid, [0])
        ) / self._n_negs_per_copmetitor.get(aid, 0)

    def received_advantage(self, aid: str):
        return sum(
            self._received_advantage.get(aid, [0])
        ) / self._n_negs_per_copmetitor.get(aid, 0)

    def partner_advantage(self, aid: str):
        return sum(
            self._partner_advantage.get(aid, [0])
        ) / self._n_negs_per_copmetitor.get(aid, 0)

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
            self._stats[f"agrement_rate_{aid}"].append(
                self._n_agreements_per_cometitor[aid][-1]
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

    def contract_record(self, contract: Contract) -> dict[str, Any]:
        return to_flat_dict(contract, deep=True)

    def breach_record(self, breach: Breach) -> dict[str, Any]:
        return to_flat_dict(breach, deep=True)

    def contract_size(self, contract: Contract) -> float:
        n = len(self._competitors)
        return 1.0 / (n if n else 1)

    def delete_executed_contracts(self) -> None:
        pass

    def execute_action(
        self, action: Action, agent: Agent, callback: Callable | None = None
    ) -> bool:
        """Executes the given action by the given agent"""
        ...

    def get_private_state(self, agent: Agent) -> dict:
        """Reads the private state of the given agent"""
        ...

    def executable_contracts(self) -> Collection[Contract]:
        return []

    def start_contract_execution(self, contract: Contract) -> set[Breach]:
        return set()

    def complete_contract_execution(
        self, contract: Contract, breaches: list[Breach], resolution: Contract
    ) -> None:
        pass


if __name__ == "__main__":
    from negmas.genius import genius_bridge_is_running
    from negmas.genius.gnegotiators import Atlas3, NiceTitForTat
    from negmas.outcomes import Issue
    from negmas.preferences import LinearAdditiveUtilityFunction as U
    from negmas.sao import AspirationNegotiator, NaiveTitForTatNegotiator
    from negmas.situated import save_stats

    issues = [Issue(10, "quantity"), Issue(5, "price")]
    competitors = [AspirationNegotiator, NaiveTitForTatNegotiator]
    if genius_bridge_is_running():
        competitors += [Atlas3, NiceTitForTat]

    scenario = Condition(
        name="d0",
        issues=issues,
        ufuns=[
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=True),
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=True),
        ],
        partner_types=[AspirationNegotiator],
        index=0,
    )
    world = NegWorld(
        scenario=scenario,
        types=competitors,
        n_steps=2,
        neg_n_steps=10,
        neg_time_limit=None,
    )
    world.run()
    print("World ran", flush=True)
    for aid in world.agents.keys():
        print(world.received_utility(aid))
    save_stats(world, world.log_folder)
