from pprint import pprint

import pytest

from negmas.apps.scml import SCMLWorld
from negmas.apps.scml.factory_managers import *
from negmas.apps.scml.java import JavaFactoryManager
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
    java_class = java_class.replace('factory_managers', 'java')
    parts = java_class.split('.')
    parts[0] = parts[0][1:]
    parts[-1] = 'Java' + parts[-1]
    return '.'.join(parts)


@pytest.mark.skipif(not jnegmas_bridge_is_running(), reason='JNegMAS is not running')
@pytest.mark.parametrize(argnames='java_class'
    , argvalues=['jnegmas.apps.scml.factory_managers.DoNothingFactoryManager',
                 'jnegmas.apps.scml.factory_managers.MiddleMan',
                 # 'jnegmas.apps.scml.factory_managers.GreedyFactoryManager',
                 ]
    , ids=['java do-nothing',
           'java middleman (no shadow)',
           # 'java greedy (with shadow)'
    ])
def test_java_factory_manager(java_class):
    class JFM(JavaFactoryManager):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, java_class_name=java_class, **kwargs)
    with jnegmas_connection(shutdown=SHUTDOWN_AFTER_EVERY_TEST):
        python_class = get_class(python_name(java_class))
        world = SCMLWorld.single_path_world(manager_types=(JFM
                                                           , GreedyFactoryManager)
                                            , n_steps=10, agent_names_reveal_type=True
                                            , log_file_name=f'/Users/yasser/negmas/logs/test{java_class.split(".")[-1]}.txt')
        print('World created')
        world.run()
        pprint('World completed')
        pprint(world.stats)
        print(world.winners)
