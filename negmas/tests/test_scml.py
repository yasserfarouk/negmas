import copy
from pathlib import Path
from pprint import pprint
from typing import List, Dict

import numpy as np
import pkg_resources
import pytest
from hypothesis import given, settings

from negmas.apps.scml import *
from negmas.apps.scml import (
    InputOutput,
    Job,
    FactoryStatusUpdate,
    GreedyScheduler,
    ProductionFailure,
)
from negmas.helpers import unique_name
from negmas.situated import Contract
import hypothesis.strategies as st

# def test_can_create_a_random_scml_world():
#     world = SCMLWorld.random()
#     assert len(world.products) > 0
#     assert len(world.processes) > 0
#     assert len(world.stats['market_size']) == 0
#     winners = [_.name for _ in world.winners]
#     assert len(winners) == len(world.factory_managers)
#     for x in ('n_contract_executions', 'n_breaches', 'n_cfps_on_board', 'n_cfps'):
#         assert len(world.stats[x]) == 0


# def test_can_run_a_random_small_scml_world():
#     world = SCMLWorld.random_small(log_file_name='', n_steps=200)
#     world.run()
#     print('')
#     for key in sorted(world.stats.keys()):
#         print(f'{key}:{world.stats[key]}')
#     winners = [_.name for _ in world.winners]
#     print('Winners: ', winners)
#     print('Losers: ', [_.name for _ in world.factory_managers if _.name not in winners])
#     data = pd.DataFrame(data=world.saved_contracts).loc[:, ["buyer", "seller", "product", "quantity", "delivery_time"
#     , "unit_price", "penalty", "signing_delay", "concluded_at", "signed_at", "issues", "cfp"]]
#     data.to_csv(f'{logdir()}/contracts.csv')
#
#
# def test_can_run_a_random_small_scml_world_with_delayed_signing():
#     world = SCMLWorld.random_small(log_file_name='', n_steps=20, default_signing_delay=1)
#     world.run()
#     print('')
#     for key in sorted(world.stats.keys()):
#         print(f'{key}:{world.stats[key]}')
#     winners = [_.name for _ in world.winners]
#     print('Winners: ', winners)
#     print('Losers: ', [_.name for _ in world.factory_managers if _.name not in winners])
#     data = pd.DataFrame(data=world.saved_contracts).loc[:, ["buyer", "seller", "product", "quantity", "delivery_time"
#     , "unit_price", "penalty", "signing_delay", "concluded_at", "signed_at", "issues", "cfp"]]
#     data.to_csv(f'{logdir()}/contracts.csv')
#
#
# def test_can_run_a_random_small_scml_world_with_money_resolution():
#     world = SCMLWorld.random_small(log_file_name='', n_steps=20, money_resolution=1)
#     world.run()
#     print('')
#     for key in sorted(world.stats.keys()):
#         print(f'{key}:{world.stats[key]}')
#     winners = [_.name for _ in world.winners]
#     print('Winners: ', winners)
#     print('Losers: ', [_.name for _ in world.factory_managers if _.name not in winners])
#     data = pd.DataFrame(data=world.saved_contracts).loc[:, ["buyer", "seller", "product", "quantity", "delivery_time"
#     , "unit_price", "penalty", "signing_delay", "concluded_at", "signed_at", "issues", "cfp"]]
#     data.to_csv(f'{logdir()}/contracts.csv')


def logdir():
    return pkg_resources.resource_filename("negmas", resource_name="tests")


@settings(deadline=None)
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
    filename = "scml"
    n_steps = 5

    world = SCMLWorld.chain_world(
        log_file_name="",
        n_steps=n_steps,
        n_factories_per_level=1,
        consumer_kwargs={
            "negotiator_type": "negmas.sao.NiceNegotiator",
            "consumption_horizon": 2,
        },
        miner_kwargs={"negotiator_type": "negmas.sao.NiceNegotiator"},
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
    world = SCMLWorld.chain_world(
        log_file_name="",
        n_steps=5,
        n_factories_per_level=1,
        consumer_kwargs={
            "negotiator_type": "negmas.sao.NiceNegotiator",
            "consumption_horizon": 2,
        }
        # , factory_kwargs={'max_insurance_premium': 100}
        ,
        miner_kwargs={"negotiator_type": "negmas.sao.NiceNegotiator"},
    )
    world.step()
    world.step()

    file_name = world.checkpoint(tmp_path)

    info = SCMLWorld.checkpoint_info(file_name)
    assert isinstance(info["time"], str)
    assert info["step"] == 2
    assert info["type"].endswith("SCMLWorld")
    assert info["id"] == world.id
    assert info["name"] == world.name

    w = SCMLWorld.from_checkpoint(file_name)

    assert world.current_step == w.current_step
    assert len(world.agents) == len(w.agents)
    assert world.agents.keys() == w.agents.keys()
    assert len(world.factories) == len(w.factories)
    assert w.bulletin_board is not None
    assert w.n_steps == world.n_steps
    assert w.negotiation_speed == world.negotiation_speed

    world = w
    file_name = world.checkpoint(tmp_path)
    w = SCMLWorld.from_checkpoint(file_name)

    assert world.current_step == w.current_step
    assert len(world.agents) == len(w.agents)
    assert world.agents.keys() == w.agents.keys()
    assert len(world.factories) == len(w.factories)
    assert w.bulletin_board is not None
    assert w.n_steps == world.n_steps
    assert w.negotiation_speed == world.negotiation_speed

    w.step()

    world = w
    file_name = world.checkpoint(tmp_path)
    w = SCMLWorld.from_checkpoint(file_name)

    assert world.current_step == w.current_step
    assert len(world.agents) == len(w.agents)
    assert world.agents.keys() == w.agents.keys()
    assert len(world.factories) == len(w.factories)
    assert w.bulletin_board is not None
    assert w.n_steps == world.n_steps
    assert w.negotiation_speed == world.negotiation_speed

    w.run()


def test_can_run_a_random_tiny_scml_world():
    world = SCMLWorld.chain_world(
        log_file_name="",
        n_steps=5,
        n_factories_per_level=1,
        consumer_kwargs={
            "negotiator_type": "negmas.sao.NiceNegotiator",
            "consumption_horizon": 2,
        }
        # , factory_kwargs={'max_insurance_premium': 100}
        ,
        miner_kwargs={"negotiator_type": "negmas.sao.NiceNegotiator"},
    )
    world.run()
    # print('')
    # for key in sorted(world.stats.keys()):
    #     print(f'{key}:{world.stats[key]}')
    winners = [_.name for _ in world.winners]
    # print('Winners: ', winners)
    # print('Losers: ', [_.name for _ in world.factory_managers if _.name not in winners])
    # if len(world.saved_contracts) > 0:
    #     data = pd.DataFrame(data=world.saved_contracts).loc[:, ["buyer", "seller", "product", "quantity", "delivery_time"
    #     , "unit_price", "penalty", "signing_delay", "concluded_at", "signed_at", "issues", "cfp"]]
    #     data.to_csv(f'{logdir()}/contracts.csv')


def test_can_run_a_random_tiny_scml_world_no_immediate():
    world = SCMLWorld.chain_world(log_file_name="", n_steps=5)
    world.run()


def test_can_run_a_random_tiny_scml_world_with_insurance():
    world = SCMLWorld.chain_world(
        log_file_name="",
        n_steps=5
        # , factory_kwargs={'max_insurance_premium': 1e6}
        ,
        consumer_kwargs={"negotiator_type": "negmas.sao.NiceNegotiator"},
        miner_kwargs={"negotiator_type": "negmas.sao.NiceNegotiator"},
    )
    world.run()


# @settings(max_examples=50)
# @given(horizon=st.sampled_from([None, 10, 50, 100, 500]),
#        immediate=st.booleans(),
#        signing_delay=st.integers(0, 3),
#        n_factory_levels=st.integers(1, 5),
#        n_factories_per_level=st.integers(1, 5)
#        )
# def test_can_run_a_random_tiny_scml_world_with_n_factories(horizon, immediate, signing_delay
#                                                            , n_factory_levels, n_factories_per_level):
#     n_steps = 500
#     world = SCMLWorld.tiny(n_intermediate_levels=n_factory_levels - 1, log_file_name='', n_steps=n_steps
#                            , n_factories_per_level=n_factories_per_level
#                            , default_signing_delay=signing_delay
#                            , consumer_kwargs={'immediate_cfp_update': immediate, 'consumption_horizon': horizon
#             , 'negotiator_type': 'NiceNegotiator'}
#                            , miner_kwargs={'negotiator_type': 'NiceNegotiator'
#             , 'use_immediate_negotiations': immediate}
#                            , factory_kwargs={'max_insurance_premium': -1
#             , 'use_immediate_negotiations': immediate}
#                            )
#     world.run()
#     print('')
#     for key in sorted(world.stats.keys()):
#         print(f'{key}:{world.stats[key]}')
#     winners = [_.name for _ in world.winners]
#     print('Winners: ', winners)
#     print('Losers: ', [_.name for _ in world.factory_managers if _.name not in winners])
#     # assert sum(world.stats['n_contracts_concluded']) > 0
#     # assert sum(world.stats['n_contracts_signed']) > 0
#     # assert len(winners) < len(world.factory_managers)


def test_can_run_a_random_tiny_scml_world_with_linear_production():
    horizon = None
    signing_delay = 0
    n_factory_levels = 0
    n_factories_per_level = 2
    n_steps = 10
    world = SCMLWorld.chain_world(
        n_intermediate_levels=n_factory_levels - 1,
        log_file_name="",
        n_steps=n_steps,
        n_factories_per_level=n_factories_per_level,
        default_signing_delay=signing_delay,
        consumer_kwargs={
            "consumption_horizon": horizon,
            "negotiator_type": "negmas.sao.NiceNegotiator",
        },
        miner_kwargs={"negotiator_type": "negmas.sao.NiceNegotiator"},
    )
    world.run()
    assert sum(world.stats["n_contracts_concluded"]) > 0


def test_can_run_a_random_tiny_scml_world_with_no_factory():
    n_steps = 10
    world = SCMLWorld.chain_world(
        n_intermediate_levels=-1,
        log_file_name="",
        n_steps=n_steps,
        negotiation_speed=None,
        n_miners=2,
        n_consumers=2,
    )
    world.run()
    # print('')
    # for key in sorted(world.stats.keys()):
    #     print(f'{key}:{world.stats[key]}')
    assert world.stats["n_negotiations"][0] >= n_steps - 1, (
        "at least n negotiations occur where n is the number of " "steps - 1"
    )
    assert world.stats["n_negotiations"][2:] == [0] * (
        len(world.stats["n_negotiations"]) - 2
    ), ("All negotiations " "happen in steps 0, 1")
    assert sum(world.stats["n_contracts_concluded"]) >= sum(
        world.stats["n_contracts_signed"]
    ), "some contracts signed"
    assert sum(world.stats["n_breaches"]) == 0, "No breaches"
    assert sum(world.stats["market_size"]) == 0, "No change in the market size"

    # data = pd.DataFrame(data=world.saved_contracts).loc[:, ["buyer", "seller", "product", "quantity", "delivery_time"
    # , "unit_price", "penalty", "signing_delay", "concluded_at", "signed_at", "issues", "cfp"]]
    # data.to_csv(f'{logdir()}/contracts.csv')


def test_can_run_a_random_tiny_scml_world_with_no_factory_finite_horizon():
    n_steps = 5
    horizon = n_steps // 2
    world = SCMLWorld.chain_world(
        n_intermediate_levels=-1,
        log_file_name="",
        n_steps=n_steps,
        consumer_kwargs={"consumption_horizon": horizon},
    )
    world.run()
    # print('')
    # for key in sorted(world.stats.keys()):
    #     print(f'{key}:{world.stats[key]}')
    # assert world.stats['n_negotiations'][0] >= horizon - 1, "at least n negotiations occur where n is the number of " \
    #                                                        "horizon - 1"
    assert sum(world.stats["n_contracts_concluded"]) >= sum(
        world.stats["n_contracts_signed"]
    ), "all contracts signed"
    assert sum(world.stats["n_breaches"]) == 0, "No breaches"
    assert sum(world.stats["market_size"]) == 0, "No change in the market size"

    # data = pd.DataFrame(data=world.saved_contracts).loc[:, ["buyer", "seller", "product", "quantity", "delivery_time"
    # , "unit_price", "penalty", "signing_delay", "concluded_at", "signed_at", "issues", "cfp"]]
    # data.to_csv(f'{logdir()}/contracts.csv')


def test_can_run_a_random_tiny_scml_world_with_no_factory_with_delay():
    n_steps = 10
    world = SCMLWorld.chain_world(
        n_intermediate_levels=-1,
        log_file_name="",
        n_steps=n_steps,
        default_signing_delay=1,
    )
    world.run()
    # print('')
    # for key in sorted(world.stats.keys()):
    #     print(f'{key}:{world.stats[key]}')
    assert world.stats["n_negotiations"][0] >= n_steps - 1, (
        "at least n negotiations occur where n is the number of " "steps - 1"
    )
    # assert world.stats['n_negotiations'][1:] == [0] * (len(world.stats['n_negotiations']) - 1), "All negotiations " \
    #                                                                                           "happen in step 0"
    assert sum(world.stats["n_contracts_concluded"]) >= sum(
        world.stats["n_contracts_signed"]
    ), "all contracts signed"
    assert sum(world.stats["n_breaches"]) == 0, "No breaches"
    assert sum(world.stats["market_size"]) == 0, "No change in the market size"

    # data = pd.DataFrame(data=world.saved_contracts).loc[:, ["buyer", "seller", "product", "quantity", "delivery_time"
    # , "unit_price", "penalty", "signing_delay", "concluded_at", "signed_at", "issues", "cfp"]]
    # data.to_csv(f'{logdir()}/contracts.csv')


def test_can_run_a_random_tiny_scml_world_with_no_factory_with_delay_no_immediate_neg():
    n_steps = 10
    horizon = 4
    world = SCMLWorld.chain_world(
        n_intermediate_levels=-1,
        log_file_name="",
        n_steps=n_steps,
        default_signing_delay=1,
        consumer_kwargs={"immediate_cfp_update": False, "consumption_horizon": horizon},
    )
    world.run()
    # print('')
    # for key in sorted(world.stats.keys()):
    #     print(f'{key}:{world.stats[key]}')
    # assert world.stats['n_negotiations'][0] >= horizon - 1, "at least n negotiations occur where n is the number of " \
    #                                                        "horizon - 1"
    assert sum(world.stats["n_contracts_concluded"]) >= world.stats[
        "n_contracts_concluded"
    ][-1] + sum(world.stats["n_contracts_signed"]), "some contracts signed"
    assert sum(world.stats["n_breaches"]) == 0, "No breaches"
    assert sum(world.stats["market_size"]) == 0, "No change in the market size"


def test_scml_picklable(tmp_path):
    import dill
    import pickle

    file = tmp_path / "world.pckl"

    n_steps = 10
    horizon = 4
    world = SCMLWorld.chain_world(
        n_intermediate_levels=-1,
        log_file_name="",
        n_steps=n_steps,
        default_signing_delay=1,
        consumer_kwargs={"immediate_cfp_update": False, "consumption_horizon": horizon},
    )
    world.step()
    world.step()
    with open(file, "wb") as f:
        dill.dump(world, f)
    with open(file, "rb") as f:
        w = dill.load(f)
    assert world.current_step == w.current_step
    assert sorted(world.agents.keys()) == sorted(w.agents.keys())
    assert w.bulletin_board is not None
    assert w.n_steps == world.n_steps
    assert w.negotiation_speed == world.negotiation_speed
    world.step()
    with open(file, "wb") as f:
        dill.dump(world, f)
    with open(file, "rb") as f:
        w = dill.load(f)
    assert world.current_step == w.current_step
    assert sorted(world.agents.keys()) == sorted(w.agents.keys())
    assert w.bulletin_board is not None
    assert w.n_steps == world.n_steps
    assert w.negotiation_speed == world.negotiation_speed
    w.run()
    assert sum(w.stats["n_contracts_concluded"]) >= w.stats["n_contracts_concluded"][
        -1
    ] + sum(w.stats["n_contracts_signed"]), "some contracts signed"
    assert sum(w.stats["n_breaches"]) == 0, "No breaches"
    assert sum(w.stats["market_size"]) == 0, "No change in the market size"


if __name__ == "__main__":
    pytest.main(args=[__file__])
