import os
from pprint import pprint

import numpy as np
import pkg_resources
import pytest
import pathlib
from hypothesis import given, settings
import hypothesis.strategies as st
from negmas.apps.scml.factory_managers import *
from negmas.helpers import get_class
from negmas.apps.scml import SCMLWorld
from negmas.java import JNegmasGateway, jnegmas_connection, jnegmas_bridge_is_running

SHUTDOWN_AFTER_EVERY_TEST = False


# @pytest.fixture(scope='session', autouse=True)
# def init_jnegmas():
#     from negmas.java import init_jnegmas_bridge
#     init_jnegmas_bridge(None, 0)


# @pytest.mark.skipif(jnegmas_bridge_is_running(), reason='JNegMAS is already running')
# def donottest_can_init_jnegmas_on_default_port():
#     from negmas.java import init_jnegmas_bridge
#     init_jnegmas_bridge(None, 0)


@pytest.mark.skipif(not jnegmas_bridge_is_running(), reason='JNegMAS is not running')
@pytest.mark.timeout(timeout=10, method='thread')
def test_can_connect_and_disconnect():
    JNegmasGateway.connect()
    if SHUTDOWN_AFTER_EVERY_TEST:
        JNegmasGateway.shutdown()


@pytest.mark.skipif(not jnegmas_bridge_is_running(), reason='JNegMAS is not running')
@pytest.mark.timeout(timeout=10, method='thread')
def test_java_connection_context():
    with jnegmas_connection(shutdown=SHUTDOWN_AFTER_EVERY_TEST):
        pass


def python_name(java_class: str) -> str:
    parts = java_class.split('.')
    parts[0] = parts[0][1:]
    parts[-1] = 'Java' + parts[-1]
    return '.'.join(parts)


@pytest.mark.skipif(not jnegmas_bridge_is_running(), reason='JNegMAS is not running')
@pytest.mark.parametrize(argnames='java_class'
                         , argvalues=['jnegmas.apps.scml.factory_managers.MiddleMan',
                                      'jnegmas.apps.scml.factory_managers.GreedyFactoryManager',
                                      'jnegmas.apps.scml.factory_managers.DoNothingFactoryManager', ]
                         , ids=['java middleman (no shadow)', 'java greedy (with shadow)', 'java do-nothing'])
def test_java_factory_manager(java_class):
    with jnegmas_connection(shutdown=SHUTDOWN_AFTER_EVERY_TEST):
        python_class = get_class(python_name(java_class))
        world = SCMLWorld.single_path_world(manager_types=(python_class
                                                           , GreedyFactoryManager)
                                            , n_steps=15, agent_names_reveal_type=True)
        print('World created')
        world.run()
        pprint('World completed')
        pprint(world.stats)
        print(world.winners)

