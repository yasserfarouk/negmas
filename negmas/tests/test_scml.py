import copy
from pprint import pprint
from typing import List, Dict

import numpy as np
import pkg_resources
import pytest

from negmas.apps.scml import *
from negmas.apps.scml import InputOutput, Job, FactoryStatusUpdate, GreedyScheduler, ProductionFailure
from negmas.situated import Contract


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
    return pkg_resources.resource_filename('negmas', resource_name='tests')


def test_can_run_a_random_tiny_scml_world():
    world = SCMLWorld.single_path_world(log_file_name='', n_steps=5, n_factories_per_level=1
                                        , consumer_kwargs={'negotiator_type': 'negmas.sao.NiceNegotiator', 'consumption_horizon': 2}
                                        # , factory_kwargs={'max_insurance_premium': 100}
                                        , miner_kwargs={'negotiator_type': 'negmas.sao.NiceNegotiator'}
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
    world = SCMLWorld.single_path_world(log_file_name='', n_steps=5)
    world.run()


def test_can_run_a_random_tiny_scml_world_with_insurance():
    world = SCMLWorld.single_path_world(log_file_name='', n_steps=5
                                        # , factory_kwargs={'max_insurance_premium': 1e6}
                                        , consumer_kwargs={'negotiator_type': 'negmas.sao.NiceNegotiator'}
                                        , miner_kwargs={'negotiator_type': 'negmas.sao.NiceNegotiator'})
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
    world = SCMLWorld.single_path_world(n_intermediate_levels=n_factory_levels - 1, log_file_name='', n_steps=n_steps
                                        , n_factories_per_level=n_factories_per_level
                                        , default_signing_delay=signing_delay
                                        , consumer_kwargs={'consumption_horizon': horizon
                                              , 'negotiator_type': 'negmas.sao.NiceNegotiator'}
                                        , miner_kwargs={'negotiator_type': 'negmas.sao.NiceNegotiator'}
                                        )
    world.run()
    assert sum(world.stats['n_contracts_concluded']) > 0


def test_can_run_a_random_tiny_scml_world_with_no_factory():
    n_steps = 10
    world = SCMLWorld.single_path_world(n_intermediate_levels=-1, log_file_name='', n_steps=n_steps
                                        , negotiation_speed=None)
    world.run()
    # print('')
    # for key in sorted(world.stats.keys()):
    #     print(f'{key}:{world.stats[key]}')
    assert world.stats['n_negotiations'][0] >= n_steps - 1, "at least n negotiations occur where n is the number of " \
                                                            "steps - 1"
    assert world.stats['n_negotiations'][1:] == [0] * (len(world.stats['n_negotiations']) - 1), "All negotiations " \
                                                                                                "happen in step 0"
    assert sum(world.stats['n_contracts_concluded']) >= sum(world.stats['n_contracts_signed']), "some contracts signed"
    assert sum(world.stats['n_breaches']) == 0, "No breaches"
    assert sum(world.stats['market_size']) == 0, "No change in the market size"

    # data = pd.DataFrame(data=world.saved_contracts).loc[:, ["buyer", "seller", "product", "quantity", "delivery_time"
    # , "unit_price", "penalty", "signing_delay", "concluded_at", "signed_at", "issues", "cfp"]]
    # data.to_csv(f'{logdir()}/contracts.csv')


def test_can_run_a_random_tiny_scml_world_with_no_factory_finite_horizon():
    n_steps = 5
    horizon = n_steps // 2
    world = SCMLWorld.single_path_world(n_intermediate_levels=-1, log_file_name='', n_steps=n_steps
                                        , consumer_kwargs={'consumption_horizon': horizon})
    world.run()
    # print('')
    # for key in sorted(world.stats.keys()):
    #     print(f'{key}:{world.stats[key]}')
    #assert world.stats['n_negotiations'][0] >= horizon - 1, "at least n negotiations occur where n is the number of " \
    #                                                        "horizon - 1"
    assert sum(world.stats['n_contracts_concluded']) >= sum(world.stats['n_contracts_signed']), "all contracts signed"
    assert sum(world.stats['n_breaches']) == 0, "No breaches"
    assert sum(world.stats['market_size']) == 0, "No change in the market size"

    # data = pd.DataFrame(data=world.saved_contracts).loc[:, ["buyer", "seller", "product", "quantity", "delivery_time"
    # , "unit_price", "penalty", "signing_delay", "concluded_at", "signed_at", "issues", "cfp"]]
    # data.to_csv(f'{logdir()}/contracts.csv')


def test_can_run_a_random_tiny_scml_world_with_no_factory_with_delay():
    n_steps = 10
    world = SCMLWorld.single_path_world(n_intermediate_levels=-1, log_file_name='', n_steps=n_steps, default_signing_delay=1)
    world.run()
    # print('')
    # for key in sorted(world.stats.keys()):
    #     print(f'{key}:{world.stats[key]}')
    assert world.stats['n_negotiations'][0] >= n_steps - 1, "at least n negotiations occur where n is the number of " \
                                                            "steps - 1"
    #assert world.stats['n_negotiations'][1:] == [0] * (len(world.stats['n_negotiations']) - 1), "All negotiations " \
    #                                                                                           "happen in step 0"
    assert sum(world.stats['n_contracts_concluded']) >= sum(world.stats['n_contracts_signed']), "all contracts signed"
    assert sum(world.stats['n_breaches']) == 0, "No breaches"
    assert sum(world.stats['market_size']) == 0, "No change in the market size"

    # data = pd.DataFrame(data=world.saved_contracts).loc[:, ["buyer", "seller", "product", "quantity", "delivery_time"
    # , "unit_price", "penalty", "signing_delay", "concluded_at", "signed_at", "issues", "cfp"]]
    # data.to_csv(f'{logdir()}/contracts.csv')


def test_can_run_a_random_tiny_scml_world_with_no_factory_with_delay_no_immediate_neg():
    n_steps = 10
    horizon = 4
    world = SCMLWorld.single_path_world(n_intermediate_levels=-1, log_file_name='', n_steps=n_steps
                                        , default_signing_delay=1
                                        , consumer_kwargs={'immediate_cfp_update': False
                                                           , 'consumption_horizon': horizon})
    world.run()
    # print('')
    # for key in sorted(world.stats.keys()):
    #     print(f'{key}:{world.stats[key]}')
    #assert world.stats['n_negotiations'][0] >= horizon - 1, "at least n negotiations occur where n is the number of " \
    #                                                        "horizon - 1"
    assert sum(world.stats['n_contracts_concluded']) >= world.stats['n_contracts_concluded'][-1] + \
           sum(world.stats['n_contracts_signed']), "some contracts signed"
    assert sum(world.stats['n_breaches']) == 0, "No breaches"
    assert sum(world.stats['market_size']) == 0, "No change in the market size"


if __name__ == '__main__':
    pytest.main(args=[__file__])
