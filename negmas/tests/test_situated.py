from __future__ import annotations

from copy import deepcopy
from random import choice, random
from typing import Any, Callable, Collection

import pytest
from attrs import asdict

from negmas import (
    AspirationNegotiator,
    Issue,
    MappingUtilityFunction,
    MechanismState,
    Negotiator,
    NegotiatorMechanismInterface,
    RenegotiationRequest,
    SAOMechanism,
)
from negmas.events import Event, EventSink, EventSource
from negmas.gb.common import ResponseType
from negmas.outcomes import Outcome, make_issue
from negmas.sao import SAOState
from negmas.sao.common import ResponseType, SAOResponse
from negmas.sao.negotiators.base import SAONegotiator
from negmas.serialization import to_flat_dict
from negmas.situated import Action, Agent, Breach, Contract, Operations, World

results = []  # will keep results not to use printing
N_NEG_STEPS = 20


class NegPerStepWorld(World):
    def __init__(self, n_steps=100, **kwargs):
        super().__init__(
            n_steps=n_steps,
            neg_n_steps=N_NEG_STEPS,
            neg_time_limit=float("inf"),
            mechanisms={
                "negmas.sao.SAOMechanism": dict(
                    one_offer_per_step=True, sync_calls=True
                )
            },
            operations=(
                Operations.StatsUpdate,
                Operations.AgentSteps,
                Operations.Negotiations,
                Operations.ContractSigning,
                Operations.ContractExecution,
                Operations.SimulationStep,
                Operations.ContractSigning,
                Operations.StatsUpdate,
            ),
            **kwargs,
        )
        self.the_agents = []

    def join(self, x: Agent, simulation_priority: int = 0):
        super().join(x=x, simulation_priority=simulation_priority)
        self.the_agents.append(x)

    def complete_contract_execution(
        self, contract: Contract, breaches: list[Breach], resolved: bool
    ) -> None:
        pass

    def get_contract_finalization_time(self, contract: Contract) -> int:
        return self.current_step + 1

    def get_contract_execution_time(self, contract: Contract) -> int:
        return self.current_step

    def contract_size(self, contract: Contract) -> float:
        return 0.0

    def delete_executed_contracts(self) -> None:
        pass

    def executable_contracts(self) -> Collection[Contract]:
        return []

    def post_step_stats(self):
        pass

    def pre_step_stats(self):
        pass

    def order_contracts_for_execution(
        self, contracts: Collection[Contract]
    ) -> Collection[Contract]:
        return contracts

    def contract_record(self, contract: Contract) -> dict[str, Any]:
        return asdict(contract)

    def start_contract_execution(self, contract: Contract) -> set[Breach]:
        return set()

    def _process_breach(
        self, contract: Contract, breaches: list[Breach], force_immediate_signing=True
    ) -> Contract | None:
        return None

    def breach_record(self, breach: Breach):
        return to_flat_dict(breach)

    def execute_action(
        self, action: Action, agent: Agent, callback: Callable = None
    ) -> bool:
        return True

    def get_private_state(self, agent: Agent) -> Any:
        s = {"partners": [_ for _ in self.the_agents if _ is not agent]}
        return s

    def simulation_step(self, stage: int):
        pass


class NeverCalled(SAONegotiator):
    def __call__(self, state: SAOState) -> SAOResponse:
        raise AssertionError(
            f"Negotiator {self.id} of agent {self.owner} should have never been called"
        )

    def propose(self, state: SAOState) -> Outcome | None:
        raise AssertionError(
            f"Negotiator {self.id} of agent {self.owner} should have never been called"
        )

    def respond(self, state: SAOState, source: str | None = None) -> ResponseType:
        raise AssertionError(
            f"Negotiator {self.id} of agent {self.owner} should have never been called"
        )


class AlwaysRejectingNegotiator(SAONegotiator):
    def __call__(self, state: SAOState) -> SAOResponse:
        return SAOResponse(ResponseType.REJECT_OFFER, self.nmi.random_outcome())

    def propose(self, state: SAOState) -> Outcome | None:
        return self.nmi.random_outcome()

    def respond(self, state: SAOState, source: str | None = None) -> ResponseType:
        return ResponseType.REJECT_OFFER


class BilateralNegAgent(Agent):
    def __init__(
        self,
        *args,
        p_request=1.1,
        should_never_be_called=False,
        never_agree=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.id = self.name
        self.__current_step = 0
        self.p_request = p_request
        self.never_agree = never_agree
        self.n_rejected = 0
        self.should_never_be_called = should_never_be_called

    def init(self):
        pass

    def step(self):
        if self.awi is None:
            return
        if random() > self.p_request:
            return
        self.__current_step = self.awi.current_step
        issues = [make_issue(10, name="i1")]
        partners = self.awi.state["partners"]
        for partner in partners:
            self._request_negotiation(partners=[partner.name, self.name], issues=issues)

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
        neg_type = (
            NeverCalled
            if self.should_never_be_called
            else AlwaysRejectingNegotiator
            if self.never_agree
            else AspirationNegotiator
        )
        a = -1 if random() < 0.5 else 1
        negotiator = neg_type(
            preferences=MappingUtilityFunction(
                mapping=lambda x: random() - a * x[0] / (random() + 1.0 * 9.0),
                issues=issues,
            )
        )
        return negotiator

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
        self.n_rejected += 1

    def on_negotiation_success(
        self, contract: Contract, mechanism: NegotiatorMechanismInterface
    ) -> None:
        pass

    def on_contract_signed(self, contract: Contract) -> None:
        pass

    def on_contract_cancelled(self, contract: Contract, rejectors: list[str]) -> None:
        pass

    def sign_contract(self, contract: Contract) -> str | None:
        return self.id

    def on_contract_executed(self, contract: Contract) -> None:
        pass

    def on_contract_breached(
        self, contract: Contract, breaches: list[Breach], resolution: Contract | None
    ) -> None:
        pass

    def set_renegotiation_agenda(
        self, contract: Contract, breaches: list[Breach]
    ) -> RenegotiationRequest | None:
        return None

    def respond_to_renegotiation_request(
        self, contract: Contract, breaches: list[Breach], agenda: RenegotiationRequest
    ) -> Negotiator | None:
        return None


class NegAgent(Agent):
    def __init__(
        self,
        *args,
        p_request=1.1,
        should_never_be_called=False,
        never_agree=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.id = self.name
        self.__current_step = 0
        self.p_request = p_request
        self.never_agree = never_agree
        self.n_rejected = 0
        self.should_never_be_called = should_never_be_called

    def init(self):
        pass

    def step(self):
        if self.awi is None:
            return
        if random() > self.p_request:
            return
        self.__current_step = self.awi.current_step
        issues = [make_issue(10, name="i1")]
        partners = self.awi.state["partners"]
        self._request_negotiation(
            partners=[_.name for _ in partners] + [self.name], issues=issues
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
        neg_type = (
            NeverCalled
            if self.should_never_be_called
            else AlwaysRejectingNegotiator
            if self.never_agree
            else AspirationNegotiator
        )
        a = -1 if random() < 0.5 else 1
        negotiator = neg_type(
            preferences=MappingUtilityFunction(
                mapping=lambda x: random() - a * x[0] / (random() + 1.0 * 9.0),
                issues=issues,
            )
        )
        return negotiator

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
        self.n_rejected += 1

    def on_negotiation_success(
        self, contract: Contract, mechanism: NegotiatorMechanismInterface
    ) -> None:
        pass

    def on_contract_signed(self, contract: Contract) -> None:
        pass

    def on_contract_cancelled(self, contract: Contract, rejectors: list[str]) -> None:
        pass

    def sign_contract(self, contract: Contract) -> str | None:
        return self.id

    def on_contract_executed(self, contract: Contract) -> None:
        pass

    def on_contract_breached(
        self, contract: Contract, breaches: list[Breach], resolution: Contract | None
    ) -> None:
        pass

    def set_renegotiation_agenda(
        self, contract: Contract, breaches: list[Breach]
    ) -> RenegotiationRequest | None:
        return None

    def respond_to_renegotiation_request(
        self, contract: Contract, breaches: list[Breach], agenda: RenegotiationRequest
    ) -> Negotiator | None:
        return None


class DummyWorld(World):
    def __init__(self, n_steps=10000, negotiation_speed=20, **kwargs):
        super().__init__(
            n_steps=n_steps,
            negotiation_speed=negotiation_speed,
            neg_n_steps=N_NEG_STEPS,
            neg_time_limit=10,
            **kwargs,
        )
        self.the_agents = []

    def complete_contract_execution(
        self, contract: Contract, breaches: list[Breach], resolved: bool
    ) -> None:
        pass

    def get_contract_finalization_time(self, contract: Contract) -> int:
        return self.current_step + 1

    def get_contract_execution_time(self, contract: Contract) -> int:
        return self.current_step

    def contract_size(self, contract: Contract) -> float:
        return 0.0

    def join(self, x: Agent, simulation_priority: int = 0):
        super().join(x=x, simulation_priority=simulation_priority)
        self.the_agents.append(x)

    def delete_executed_contracts(self) -> None:
        pass

    def executable_contracts(self) -> Collection[Contract]:
        return []

    def post_step_stats(self):
        pass

    def pre_step_stats(self):
        pass

    def order_contracts_for_execution(
        self, contracts: Collection[Contract]
    ) -> Collection[Contract]:
        return contracts

    def contract_record(self, contract: Contract) -> dict[str, Any]:
        return asdict(contract)

    def start_contract_execution(self, contract: Contract) -> set[Breach]:
        return set()

    def _process_breach(
        self, contract: Contract, breaches: list[Breach], force_immediate_signing=True
    ) -> Contract | None:
        return None

    def breach_record(self, breach: Breach):
        return to_flat_dict(breach)

    def execute_action(
        self, action: Action, agent: Agent, callback: Callable = None
    ) -> bool:
        return True

    def get_private_state(self, agent: Agent) -> Any:
        s = {"partners": [_ for _ in self.the_agents if _ is not agent]}
        return s

    def simulation_step(self, stage: int):
        pass


class DummyAgent(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.id = self.name
        self.__current_step = 0

    def init(self):
        pass

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
        negotiator = AspirationNegotiator(
            preferences=MappingUtilityFunction(
                mapping=lambda x: 1.0 - x[0] / 10.0, issues=issues
            )
        )
        return negotiator

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

    def on_contract_signed(self, contract: Contract) -> None:
        pass

    def on_contract_cancelled(self, contract: Contract, rejectors: list[str]) -> None:
        pass

    def sign_contract(self, contract: Contract) -> str | None:
        return self.id

    def on_contract_executed(self, contract: Contract) -> None:
        pass

    def on_contract_breached(
        self, contract: Contract, breaches: list[Breach], resolution: Contract | None
    ) -> None:
        pass

    def set_renegotiation_agenda(
        self, contract: Contract, breaches: list[Breach]
    ) -> RenegotiationRequest | None:
        return None

    def respond_to_renegotiation_request(
        self, contract: Contract, breaches: list[Breach], agenda: RenegotiationRequest
    ) -> Negotiator | None:
        return None

    def step(self):
        global results
        if self.awi is None:
            return
        self.__current_step = self.awi.current_step
        if (self.__current_step == 2 and self.name.endswith("1")) or (
            self.__current_step == 4 and self.name.endswith("2")
        ):
            issues = [make_issue(10, name="i1")]
            partners = self.awi.state["partners"]
            self._request_negotiation(
                partners=[_.name for _ in partners] + [self.name], issues=issues
            )
            results.append(f"{self.name} started negotiation with {partners[0].name}")
        results.append(f"{self.name}: step {self.__current_step}")


class ExceptionAgent(Agent):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.id = name
        self.__current_step = 0

    def init(self):
        pass

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
        negotiator = AspirationNegotiator(
            preferences=MappingUtilityFunction(
                mapping=lambda x: 1.0 - x[0] / 10.0,
                issues=issues,
            ),
        )
        return negotiator

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

    def on_contract_signed(self, contract: Contract) -> None:
        pass

    def on_contract_cancelled(self, contract: Contract, rejectors: list[str]) -> None:
        pass

    def sign_contract(self, contract: Contract) -> str | None:
        return self.id

    def on_contract_executed(self, contract: Contract) -> None:
        pass

    def on_contract_breached(
        self, contract: Contract, breaches: list[Breach], resolution: Contract | None
    ) -> None:
        pass

    def set_renegotiation_agenda(
        self, contract: Contract, breaches: list[Breach]
    ) -> RenegotiationRequest | None:
        return None

    def respond_to_renegotiation_request(
        self, contract: Contract, breaches: list[Breach], agenda: RenegotiationRequest
    ) -> Negotiator | None:
        return None

    def step(self):
        global results
        if self.awi is None:
            return
        self.__current_step = self.awi.current_step
        if (self.__current_step == 2 and self.name.endswith("1")) or (
            self.__current_step == 4 and self.name.endswith("2")
        ):
            issues = [make_issue(10, name="i1")]
            partners = self.awi.state["partners"]
            self._request_negotiation(
                partners=[_.name for _ in partners] + [self.name], issues=issues
            )
            results.append(f"{self.name} started negotiation with {partners[0].name}")
        raise ValueError("error")
        results.append(f"{self.name}: step {self.__current_step}")


def test_world_has_times():
    import time

    world = DummyWorld(n_steps=N_NEG_STEPS)
    world.join(DummyAgent("A1"))
    world.join(DummyAgent("A2"))
    _strt = time.perf_counter()
    world.run()
    t = time.perf_counter() - _strt
    for aid in world.agents.keys():
        assert 0.0 < world.times[aid] < t
    assert 0.0 < sum(world.times.values()) < t


def test_world_records_exceptions():
    pass

    world = DummyWorld(n_steps=N_NEG_STEPS, ignore_agent_exceptions=True)
    world.join(ExceptionAgent("A1"))
    world.join(ExceptionAgent("A2"))
    world.run()
    assert len(world.simulation_exceptions) == 0
    assert len(world.contract_exceptions) == 0
    assert len(world.mechanism_exceptions) == 0
    for aid in world.agents.keys():
        assert len(world.negotiator_exceptions[aid]) == 0.0
        assert len(world.agent_exceptions[aid]) == N_NEG_STEPS
        assert sum(world.n_negotiator_exceptions.values()) == 0
        assert sum(world.n_agent_exceptions.values()) == N_NEG_STEPS * 2
        a = list(world.n_agent_exceptions.values())
        assert a[0] == a[1] == N_NEG_STEPS
        assert sum(world.n_total_simulation_exceptions.values()) == 0
        assert sum(world.n_total_agent_exceptions.values()) == N_NEG_STEPS * 2


def test_world_runs_with_some_negs():
    global results
    results = []
    world = DummyWorld(n_steps=N_NEG_STEPS)
    world.join(DummyAgent("A1"))
    world.join(DummyAgent("A2"))
    world.run()
    assert "A1: step 1" in results, "first step logged"
    assert "A2: step 1" in results, "first step logged"
    assert "A1 started negotiation with A2" in results, "negotiation started"
    assert f"A1: step {world.n_steps-1}" in results, "last step logged"
    assert f"A2: step {world.n_steps-1}" in results, "last step logged"
    assert len(world.saved_contracts) == 2
    assert sum(world.stats["n_negotiations"]) == 2


def test_config_reader_with_a_world():
    world = DummyWorld()
    assert world.bulletin_board is not None
    assert world.n_steps == 10000

    world = DummyWorld.from_config(scope=globals(), config={"n_steps": N_NEG_STEPS})
    assert world.bulletin_board is not None
    assert world.n_steps == N_NEG_STEPS


def test_config_reader_with_a_world_with_enum():
    world = DummyWorld()
    assert world.bulletin_board is not None
    assert world.n_steps == 10000

    world = DummyWorld.from_config(
        scope=globals(), config={"n_steps": N_NEG_STEPS, "negotiation_speed": 2}
    )
    assert world.bulletin_board is not None
    assert world.n_steps == N_NEG_STEPS
    assert world.negotiation_speed == 2


def test_world_picklable(tmp_path):
    import dill as pickle

    world = DummyWorld()
    world.step()
    world.step()
    file = tmp_path / "world.pckl"

    with open(file, "wb") as f:
        pickle.dump(world, f)
    with open(file, "rb") as f:
        w = pickle.load(f)
    assert world.current_step == w.current_step
    assert world.agents == w.agents
    assert w.bulletin_board is not None
    assert w.n_steps == world.n_steps
    assert w.negotiation_speed == world.negotiation_speed

    w.run()


def test_world_checkpoint(tmp_path):
    world = DummyWorld()
    world.step()
    world.step()

    file_name = world.checkpoint(tmp_path)

    info = DummyWorld.checkpoint_info(file_name)
    assert isinstance(info["time"], str)
    assert info["step"] == 2
    assert info["type"].endswith("DummyWorld")
    assert info["id"] == world.id
    assert info["name"] == world.name

    w = DummyWorld.from_checkpoint(file_name)

    assert world.current_step == w.current_step
    assert world.agents == w.agents
    assert w.bulletin_board is not None
    assert w.n_steps == world.n_steps
    assert w.negotiation_speed == world.negotiation_speed

    world = w
    file_name = world.checkpoint(tmp_path)
    w = DummyWorld.from_checkpoint(file_name)

    assert world.current_step == w.current_step
    assert world.agents == w.agents
    assert w.bulletin_board is not None
    assert w.n_steps == world.n_steps
    assert w.negotiation_speed == world.negotiation_speed

    w.step()

    world = w
    file_name = world.checkpoint(tmp_path)
    w = DummyWorld.from_checkpoint(file_name)

    assert world.current_step == w.current_step
    assert world.agents == w.agents
    assert w.bulletin_board is not None
    assert w.n_steps == world.n_steps
    assert w.negotiation_speed == world.negotiation_speed

    w.run()


def test_agent_checkpoint(tmp_path):
    a = DummyAgent(name="abcd")

    a.step_()

    file_name = a.checkpoint(tmp_path, info={"a": 3})

    info = SAOMechanism.checkpoint_info(file_name)
    assert isinstance(info["time"], str)
    assert info["step"] == 1
    assert info["type"].endswith("DummyAgent")
    assert info["id"] == a.id
    assert info["name"] == a.name == "abcd"
    assert info["a"] == 3

    b = a.from_checkpoint(file_name)

    assert a.id == b.id

    b.step_()


def test_agent_checkpoint_in_world(tmp_path):
    world = DummyWorld(n_steps=N_NEG_STEPS)
    world.join(DummyAgent("A1"))
    world.join(DummyAgent("A2"))
    assert len(world.agents) == 2
    for a in world.agents.values():
        a.step_()
        file_name = a.checkpoint(tmp_path, info={"a": 3})

        info = SAOMechanism.checkpoint_info(file_name)
        assert isinstance(info["time"], str)
        assert info["step"] == 1
        assert info["type"].endswith("DummyAgent")
        assert info["id"] == a.id
        assert info["name"] == a.name
        assert info["a"] == 3

        b = a.from_checkpoint(file_name)

        assert a.id == b.id

        b.step_()


class MyMonitor(EventSink):
    def on_event(self, event: Event, sender: EventSource):
        assert event.type == "agent-joined"


def test_cannot_start_a_neg_with_no_outcomes():
    world = DummyWorld(n_steps=N_NEG_STEPS)
    a, b = DummyAgent(name="a"), DummyAgent(name="b")
    world.join(a)
    world.join(b)
    with pytest.raises(ValueError):
        a.awi.request_negotiation_about(
            issues=[make_issue((1, 0))], partners=[a.id, b.id], req_id="1234"
        )


def test_world_monitor():
    monitor = MyMonitor()
    world = DummyWorld(n_steps=N_NEG_STEPS)
    world.register_listener("agent-joined", listener=monitor)
    world.join(DummyAgent("A1"))
    world.join(DummyAgent("A2"))
    assert len(world.agents) == 2
    world.run()


def test_neg_world_steps_serial_n_neg_steps_mode_all_requested_and_timeout():
    n_steps, n_agents = N_NEG_STEPS, 4
    world = NegPerStepWorld(n_steps)
    for _ in range(n_agents):
        world.join(NegAgent(p_request=1.0, never_agree=True, name=f"a{_}"))
    assert world.current_step == 0
    i, s, n = 0, 0, 0
    while world.step(n_neg_steps=1, n_mechanisms=1):
        i += 1
    # n. negotiations == n. agents
    assert i == (N_NEG_STEPS + 1) * n_steps * n_agents - 1
    assert world.current_step == n_steps
    assert (
        sum(_.n_rejected for _ in world.agents.values())  # type: ignore
        == n_steps * n_agents * n_agents
    )


def test_neg_world_steps_n_neg_steps_mode_all_requested_and_timeout():
    n_steps, n_agents = N_NEG_STEPS, 4
    world = NegPerStepWorld(n_steps)
    for _ in range(n_agents):
        world.join(NegAgent(p_request=1.0, never_agree=True, name=f"a{_}"))
    assert world.current_step == 0
    i, s, n = 0, 0, 0
    while world.step(n_neg_steps=1):
        i += 1
    assert i == (N_NEG_STEPS + 1) * n_steps - 1
    assert world.current_step == n_steps
    assert (
        sum(_.n_rejected for _ in world.agents.values())  # type: ignore
        == n_steps * n_agents * n_agents
    )


def test_neg_world_steps_n_neg_steps_mode_all_requested_and_timeout_with_action():
    n_steps, n_agents = N_NEG_STEPS, 3
    world = NegPerStepWorld(n_steps)
    agents = [
        BilateralNegAgent(p_request=1.0, never_agree=True, name=f"a{_}")
        for _ in range(n_agents)
    ]
    agent = choice(agents)
    for a in agents:
        # choose an agent that will never really negotiate as we will provide its negotiation actions for it
        if a.id == agent.id:
            a.should_never_be_called = True
        world.join(a)
    assert world.current_step == 0
    i, s, n = 0, 0, 0
    world.step(0)
    while True:
        # get all running negotiations for this agent
        infos_from_world = dict()
        for mid, ninfo in world._negotiations.items():
            if agent.id in {_.id for _ in ninfo.partners}:
                infos_from_world[mid] = ninfo
        infos = agent.awi.running_mechanism_dicts
        action_copy, action = dict(), dict()
        assert len(infos) <= 2 * (n_agents - 1)
        assert len(infos) == len(infos_from_world)
        assert set(infos.keys()) == set(infos_from_world.keys())
        # for every running negotiation (mechanism)
        for mechanism_id, info in infos.items():
            # find a random outcome for the agent
            m = world._negotiations[mechanism_id].mechanism
            if m is None:
                continue
            assert info.negotiator.id in m.negotiator_ids
            outcome = m.random_outcome()
            action[mechanism_id] = {
                info.negotiator.id: SAOResponse(ResponseType.REJECT_OFFER, outcome)
            }
            action_copy = deepcopy(action)
        old_step = world.current_step
        if not world.step(n_neg_steps=1, neg_actions=action):
            break
        new_step = world.current_step
        # print(i, action_copy)
        # When a new step starts, new negotiations start so we cannot do this check
        if new_step == old_step:
            for mechanism_id, info in infos.items():
                m = world._negotiations[mechanism_id].mechanism
                if m is None or info.negotiator is None:
                    continue
                negotiator_id = info.negotiator.id
                state: SAOState = m.state  # type: ignore
                # print(state)
                for nid, offer in state.new_offers:
                    if nid == negotiator_id:
                        assert nid not in action[mechanism_id].keys()
                        assert (
                            offer == action_copy[mechanism_id][negotiator_id].outcome
                        ), f"{action=}"

        i += 1
    assert i == (N_NEG_STEPS + 1) * n_steps - 1
    assert world.current_step == n_steps


def test_neg_world_steps_n_neg_steps_mode_all_requested():
    n_steps, n_agents = N_NEG_STEPS, 4
    world = NegPerStepWorld(n_steps)
    for _ in range(n_agents):
        world.join(NegAgent(p_request=1.0, never_agree=False, name=f"a{_}"))
    assert world.current_step == 0
    i, s, n = 0, 0, 0
    while world.step(n_neg_steps=1):
        i += 1
    assert i <= (N_NEG_STEPS + 1) * n_steps - 1
    assert world.current_step == n_steps
    assert (
        sum(_.n_rejected for _ in world.agents.values())  # type: ignore
        == n_steps * n_agents * n_agents
    )


if __name__ == "__main__":
    pytest.main(args=[__file__])
