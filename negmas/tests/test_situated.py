from pathlib import Path
from typing import Any, Callable, Collection, Dict, List, Optional, Set

import hypothesis.strategies as st
import pytest
from hypothesis import given, settings, HealthCheck

from negmas import (
    AgentMechanismInterface,
    AspirationNegotiator,
    Issue,
    MappingUtilityFunction,
    MechanismState,
    Negotiator,
    RenegotiationRequest,
    SAOMechanism,
)
from negmas.events import Event, EventSink, EventSource
from negmas.helpers import unique_name
from negmas.situated import Action, Agent, Breach, Contract, World

results = []  # will keep results not to use printing


class DummyWorld(World):
    def complete_contract_execution(
        self, contract: Contract, breaches: List[Breach], resolved: bool
    ) -> None:
        pass

    def get_contract_finalization_time(self, contract: Contract) -> int:
        return self.current_step + 1

    def get_contract_execution_time(self, contract: Contract) -> int:
        return self.current_step

    def contract_size(self, contract: Contract) -> float:
        return 0.0

    def __init__(self, n_steps=10000, negotiation_speed=20, **kwargs):
        super().__init__(
            n_steps=n_steps,
            negotiation_speed=negotiation_speed,
            neg_n_steps=10,
            neg_time_limit=10,
            **kwargs,
        )
        self.the_agents = []

    def join(self, x: "Agent", simulation_priority: int = 0):
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

    def contract_record(self, contract: Contract) -> Dict[str, Any]:
        return contract.__dict__

    def start_contract_execution(self, contract: Contract) -> Set[Breach]:
        return set()

    def _process_breach(
        self, contract: Contract, breaches: List[Breach], force_immediate_signing=True
    ) -> Optional[Contract]:
        return None

    def breach_record(self, breach: Breach):
        return breach.__dict__

    def execute_action(
        self, action: Action, agent: "Agent", callback: Callable = None
    ) -> bool:
        return True

    def get_private_state(self, agent: "Agent") -> Any:
        s = {"partners": [_ for _ in self.the_agents if _ is not agent]}
        return s

    def simulation_step(self, stage: int):
        pass


class DummyAgent(Agent):
    def init(self):
        pass

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
        negotiator = AspirationNegotiator(
            ufun=MappingUtilityFunction(mapping=lambda x: 1.0 - x[0] / 10.0)
        )
        return negotiator

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
        return self.id

    def on_contract_executed(self, contract: Contract) -> None:
        pass

    def on_contract_breached(
        self, contract: Contract, breaches: List[Breach], resolution: Optional[Contract]
    ) -> None:
        pass

    def set_renegotiation_agenda(
        self, contract: Contract, breaches: List[Breach]
    ) -> Optional[RenegotiationRequest]:
        return None

    def respond_to_renegotiation_request(
        self, contract: Contract, breaches: List[Breach], agenda: RenegotiationRequest
    ) -> Optional[Negotiator]:
        return None

    def __init__(self, name=None):
        super().__init__(name=name)
        self.id = name
        self.__current_step = 0

    def step(self):
        global results
        if self.awi is None:
            return
        self.__current_step = self.awi.current_step
        if (self.__current_step == 2 and self.name.endswith("1")) or (
            self.__current_step == 4 and self.name.endswith("2")
        ):
            issues = [Issue(10, name="i1")]
            partners = self.awi.state["partners"]
            self._request_negotiation(
                partners=[_.name for _ in partners] + [self.name], issues=issues
            )
            results.append(f"{self.name} started negotiation with {partners[0].name}")
        results.append(f"{self.name}: step {self.__current_step}")


class ExceptionAgent(Agent):
    def init(self):
        pass

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
        negotiator = AspirationNegotiator(
            ufun=MappingUtilityFunction(mapping=lambda x: 1.0 - x[0] / 10.0)
        )
        return negotiator

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
        return self.id

    def on_contract_executed(self, contract: Contract) -> None:
        pass

    def on_contract_breached(
        self, contract: Contract, breaches: List[Breach], resolution: Optional[Contract]
    ) -> None:
        pass

    def set_renegotiation_agenda(
        self, contract: Contract, breaches: List[Breach]
    ) -> Optional[RenegotiationRequest]:
        return None

    def respond_to_renegotiation_request(
        self, contract: Contract, breaches: List[Breach], agenda: RenegotiationRequest
    ) -> Optional[Negotiator]:
        return None

    def __init__(self, name=None):
        super().__init__(name=name)
        self.id = name
        self.__current_step = 0

    def step(self):
        global results
        if self.awi is None:
            return
        self.__current_step = self.awi.current_step
        if (self.__current_step == 2 and self.name.endswith("1")) or (
            self.__current_step == 4 and self.name.endswith("2")
        ):
            issues = [Issue(10, name="i1")]
            partners = self.awi.state["partners"]
            self._request_negotiation(
                partners=[_.name for _ in partners] + [self.name], issues=issues
            )
            results.append(f"{self.name} started negotiation with {partners[0].name}")
        raise ValueError("error")
        results.append(f"{self.name}: step {self.__current_step}")


def test_world_has_times(capsys):
    import time

    world = DummyWorld(n_steps=10)
    world.join(DummyAgent("A1"))
    world.join(DummyAgent("A2"))
    _strt = time.perf_counter()
    world.run()
    t = time.perf_counter() - _strt
    for aid in world.agents.keys():
        assert 0.0 < world.times[aid] < t
    assert 0.0 < sum(world.times.values()) < t


def test_world_records_exceptions(capsys):
    pass

    world = DummyWorld(n_steps=10, ignore_agent_exceptions=True)
    world.join(ExceptionAgent("A1"))
    world.join(ExceptionAgent("A2"))
    world.run()
    assert len(world.simulation_exceptions) == 0
    assert len(world.contract_exceptions) == 0
    assert len(world.mechanism_exceptions) == 0
    for aid in world.agents.keys():
        assert len(world.negotiator_exceptions[aid]) == 0.0
        assert len(world.agent_exceptions[aid]) == 10
        assert sum(world.n_negotiator_exceptions.values()) == 0
        assert sum(world.n_agent_exceptions.values()) == 20
        a = list(world.n_agent_exceptions.values())
        assert a[0] == a[1] == 10
        assert sum(world.n_total_simulation_exceptions.values()) == 0
        assert sum(world.n_total_agent_exceptions.values()) == 20


def test_world_runs_with_some_negs(capsys):
    global results
    results = []
    world = DummyWorld(n_steps=10)
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

    world = DummyWorld.from_config(scope=globals(), config={"n_steps": 10})
    assert world.bulletin_board is not None
    assert world.n_steps == 10


def test_config_reader_with_a_world_with_enum():
    world = DummyWorld()
    assert world.bulletin_board is not None
    assert world.n_steps == 10000

    world = DummyWorld.from_config(
        scope=globals(), config={"n_steps": 10, "negotiation_speed": 2}
    )
    assert world.bulletin_board is not None
    assert world.n_steps == 10
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


@settings(deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    single_checkpoint=st.booleans(),
    checkpoint_every=st.integers(0, 6),
    exist_ok=st.booleans(),
)
def test_world_auto_checkpoint(tmp_path, single_checkpoint, checkpoint_every, exist_ok):
    import shutil

    new_folder: Path = tmp_path / unique_name("empty", sep="")
    new_folder.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(new_folder)
    new_folder.mkdir(parents=True, exist_ok=True)
    filename = "mechanism"
    n_steps = 20

    world = DummyWorld(
        n_steps=n_steps,
        checkpoint_every=checkpoint_every,
        checkpoint_folder=new_folder,
        checkpoint_filename=filename,
        extra_checkpoint_info=None,
        exist_ok=exist_ok,
        single_checkpoint=single_checkpoint,
    )

    world.run()

    if 0 < checkpoint_every <= n_steps:
        if single_checkpoint:
            assert len(list(new_folder.glob("*"))) == 2, print(
                f"World ran for: {world.current_step}"
            )
        else:
            assert len(list(new_folder.glob("*"))) >= 2 * (
                max(1, world.current_step // checkpoint_every)
            )
    elif checkpoint_every > n_steps:
        assert len(list(new_folder.glob("*"))) == 2
    else:
        assert len(list(new_folder.glob("*"))) == 0


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
    world = DummyWorld(n_steps=10)
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
    world = DummyWorld(n_steps=10)
    a, b = DummyAgent(name="a"), DummyAgent(name="b")
    world.join(a)
    world.join(b)
    assert not a.awi.request_negotiation_about(
        issues=[Issue((1, 0))], partners=[a.id, b.id], req_id="1234"
    )


def test_world_monitor():
    monitor = MyMonitor()
    world = DummyWorld(n_steps=10)
    world.register_listener("agent-joined", listener=monitor)
    world.join(DummyAgent("A1"))
    world.join(DummyAgent("A2"))
    assert len(world.agents) == 2
    world.run()


if __name__ == "__main__":
    pytest.main(args=[__file__])
