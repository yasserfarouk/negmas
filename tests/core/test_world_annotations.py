"""Tests for world-level and per-agent annotations on the World/Agent side.

These mirror the mechanism-side annotations exposed through the NMI
(``Mechanism(annotation=...)`` -> ``nmi.annotation`` and per-negotiator
``Mechanism.add(negotiator, annotation=...)``).

On the world side:
- ``World(annotation=...)`` -> ``awi.annotation`` (common, shared with all agents)
- ``World.join(agent, annotation=...)`` -> per-agent keys merged into ``awi.annotation``
- ``Agent(private_info=...)`` -> ``agent.annotation`` / ``agent.private_info`` (fully private)
"""

from __future__ import annotations

from collections.abc import Collection
from typing import Any, Callable

import pytest

from negmas.situated import Action, Agent, Breach, Contract, World


class MiniWorld(World):
    """A minimal concrete world that does nothing — used only to exercise annotations."""

    def breach_record(self, breach: Breach) -> dict[str, Any]:
        return dict()

    def contract_record(self, contract: Contract) -> dict[str, Any]:
        return dict()

    def delete_executed_contracts(self) -> None:
        pass

    def executable_contracts(self) -> Collection[Contract]:
        return []

    def order_contracts_for_execution(
        self, contracts: Collection[Contract]
    ) -> Collection[Contract]:
        return contracts

    def start_contract_execution(self, contract: Contract) -> set[Breach] | None:
        return None

    def complete_contract_execution(
        self, contract: Contract, breaches: list[Breach], resolution: Contract
    ) -> None:
        pass

    def execute_action(
        self, action: Action, agent, callback: Callable | None = None
    ) -> bool:
        return True

    def get_private_state(self, agent: Agent) -> dict:
        return dict()

    def simulation_step(self, stage: int = 0):
        pass

    def contract_size(self, contract: Contract) -> float:
        return 0.0


class MiniAgent(Agent):
    def step(self):
        pass

    def init(self):
        pass

    def on_neg_request_rejected(self, req_id: str, by: list[str] | None):
        pass

    def on_neg_request_accepted(self, req_id: str, mechanism):
        pass

    def on_negotiation_failure(
        self, partners: list[str], annotation, mechanism, state
    ) -> None:
        pass

    def on_negotiation_success(self, contract: Contract, mechanism) -> None:
        pass

    def _respond_to_negotiation_request(
        self, initiator, partners, issues, annotation, mechanism, role, req_id
    ):
        return None

    def on_contract_executed(self, contract: Contract) -> None:
        pass

    def on_contract_breached(
        self, contract: Contract, breaches: list[Breach], resolution
    ) -> None:
        pass

    def set_renegotiation_agenda(self, contract: Contract, breaches: list[Breach]):
        return None

    def respond_to_renegotiation_request(self, contract, breaches, agenda):
        return None


def make_world(**kwargs) -> MiniWorld:
    kwargs.setdefault("n_steps", 2)
    kwargs.setdefault("no_logs", True)
    return MiniWorld(**kwargs)


def test_world_annotation_defaults_to_empty_dict():
    w = make_world()
    assert w.annotation == {}
    assert w.params["annotation"] == {}


def test_world_annotation_stored_and_in_params():
    w = make_world(annotation=dict(scenario="s1", seed=42))
    assert w.annotation == {"scenario": "s1", "seed": 42}
    assert w.params["annotation"] == {"scenario": "s1", "seed": 42}


def test_awi_annotation_exposes_world_annotation():
    w = make_world(annotation=dict(common="c"))
    a = MiniAgent(name="a0")
    w.join(a)
    assert a.awi.annotation == {"common": "c"}


def test_per_agent_annotation_merged_into_awi_annotation():
    w = make_world(annotation=dict(common="c", shared="world"))
    a0 = MiniAgent(name="a0")
    a1 = MiniAgent(name="a1")
    w.join(a0, annotation=dict(private_to_a0="x", shared="agent"))
    w.join(a1)  # no per-agent annotation

    # a0 sees world annotation merged with its own (per-agent overrides world)
    assert a0.awi.annotation == {"common": "c", "shared": "agent", "private_to_a0": "x"}
    # a1 sees only the world annotation
    assert a1.awi.annotation == {"common": "c", "shared": "world"}
    # the world-level annotation is not mutated by per-agent annotations
    assert w.annotation == {"common": "c", "shared": "world"}


def test_per_agent_annotation_does_not_leak_to_other_agents():
    w = make_world(annotation=dict(common="c"))
    a0 = MiniAgent(name="a0")
    a1 = MiniAgent(name="a1")
    w.join(a0, annotation=dict(secret="a0-only"))
    w.join(a1)
    assert "secret" in a0.awi.annotation
    assert "secret" not in a1.awi.annotation


def test_agent_private_info_defaults_to_empty_dict():
    a = MiniAgent(name="a0")
    assert a.private_info == {}
    assert a.annotation == {}


def test_agent_private_info_set_at_construction():
    a = MiniAgent(name="a0", private_info=dict(role="buyer", token=7))
    assert a.private_info == {"role": "buyer", "token": 7}
    assert a.annotation == {"role": "buyer", "token": 7}


def test_agent_private_info_is_distinct_from_awi_annotation():
    w = make_world(annotation=dict(common="c"))
    a = MiniAgent(name="a0", private_info=dict(secret="s"))
    w.join(a, annotation=dict(per_agent="p"))
    # private info is on the agent, shared annotation is on the awi
    assert a.private_info == {"secret": "s"}
    assert a.annotation == {"secret": "s"}
    assert a.awi.annotation == {"common": "c", "per_agent": "p"}
    # they are unrelated dicts
    assert "secret" not in a.awi.annotation
    assert "common" not in a.private_info


def test_world_agent_annotation_accessor():
    w = make_world(annotation=dict(common="c"))
    a = MiniAgent(name="a0")
    w.join(a, annotation=dict(extra="e"))
    assert w.agent_annotation(a.id) == {"common": "c", "extra": "e"}
    # unknown agent id just returns the world annotation
    assert w.agent_annotation("does-not-exist") == {"common": "c"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
