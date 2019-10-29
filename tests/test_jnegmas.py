import os
from pathlib import Path
from pprint import pprint

import pytest

from negmas.apps.scml import SCMLWorld
from negmas.apps.scml.factory_managers import *
from negmas.helpers import get_class
from negmas.java import jnegmas_connection, jnegmas_bridge_is_running

SHUTDOWN_AFTER_EVERY_TEST = False


# @pytest.fixture(scope='session', autouse=True)
# def init_jnegmas():
#     from negmas.java import init_jnegmas_bridge
#     init_jnegmas_bridge()
#     JNegmasGateway.connect()
#     yield
#     JNegmasGateway.shutdown()


def python_name(java_class: str) -> str:
    parts = java_class.split(".")
    parts[0] = parts[0][1:]
    parts[-1] = "Java" + parts[-1]
    return ".".join(parts)


@pytest.mark.skipif(not jnegmas_bridge_is_running(), reason="JNegMAS is not running")
@pytest.mark.parametrize(
    argnames="java_class",
    argvalues=[
        "jnegmas.apps.scml.factory_managers.GreedyFactoryManager",
        "jnegmas.apps.scml.factory_managers.DoNothingFactoryManager",
        "jnegmas.apps.scml.factory_managers.DummyMiddleMan",
        "jnegmas.apps.scml.factory_managers.RandomizingGFM",
    ],
    ids=[
        "java greedy (with shadow)",
        "java do-nothing",
        "java middleman (no shadow)",
        "based on java greedy",
    ],
)
def test_java_factory_manager(java_class):
    class JFM(JavaFactoryManager):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, java_class_name=java_class, **kwargs)

    with jnegmas_connection(shutdown=SHUTDOWN_AFTER_EVERY_TEST):
        log_file = os.path.expanduser(f'test{java_class.split(".")[-1]}.txt')
        print(log_file)
        world = SCMLWorld.chain_world(
            manager_types=(JFM, GreedyFactoryManager),
            n_steps=5,
            n_factories_per_level=3,
            n_default_per_level=1,
            n_intermediate_levels=1,
            agent_names_reveal_type=True,
            # log_folder=str(Path.home() / "negmas" / "logs" / "debug"),
            # log_file_name=log_file,
        )
        world.run()
        # pprint('World completed')
        # pprint(world.stats)
        # print(world.winners)


# def test_java_factory_manager_special():
#     java_class = "jnegmas.apps.scml.factory_managers.DummyMiddleMan"
#
#     class JFM(JavaFactoryManager):
#         def __init__(self, *args, **kwargs):
#             super().__init__(*args, java_class_name=java_class, **kwargs)
#
#     with jnegmas_connection(shutdown=SHUTDOWN_AFTER_EVERY_TEST):
#         log_file = os.path.expanduser(
#             f'~/negmas/logs/debug/test{java_class.split(".")[-1]}.txt'
#         )
#         print(log_file)
#         world = SCMLWorld.chain_world(
#             manager_types=(JFM, GreedyFactoryManager),
#             n_steps=5,
#             n_factories_per_level=3,
#             n_default_per_level=1,
#             n_intermediate_levels=1,
#             agent_names_reveal_type=True,
#             log_file_name=log_file,
#         )
#         world.run()
