import os
from pprint import pprint

import numpy as np
import pkg_resources
import pytest
import pathlib
from hypothesis import given, settings
import hypothesis.strategies as st


def donot_test_can_init_jnegmas_on_default_port():
    from negmas.java import init_jnegmas_bridge
    init_jnegmas_bridge(None, 0)


def test_do_nothing_java_facotry_manager():
    from negmas.apps.scml import SCMLWorld
    from negmas.apps.scml import JavaFactoryManager
    from negmas.apps.scml import GreedyFactoryManager
    world = SCMLWorld.single_path_world(manager_types=(JavaFactoryManager.do_nothing_manager().__class__
                                                       , GreedyFactoryManager)
                                        , n_steps=15, agent_names_reveal_type=True)
    print('World created')
    world.run()
    pprint('World completed')
    pprint(world.stats)
    print(world.winners)


