import math
import sys
from os import PathLike
from random import randint

from negmas.helpers import get_class
from negmas.tournaments import WorldRunResults, TournamentResults, tournament
from .factory_managers import GreedyFactoryManager
from .world import SCMLWorld

if True:
    from typing import Tuple, Union, Type, Iterable, Sequence, Optional, Callable
    from .factory_managers import FactoryManager

__all__ = [
    'anac2019_world',
    'anac2019_tournament',
    'balance_calculator',
]


def anac2019_world(
    competitors: Sequence[Union[str, Type[FactoryManager]]] = ()
    , randomize: bool = True
    , log_file_name: str = None
    , name: str = None
    , agent_names_reveal_type: bool = False
    , n_intermediate: Tuple[int, int] = (1, 4)
    , n_miners=5
    , n_factories_per_level=11
    , n_consumers=5
    , n_lines_per_factory=10
    , guaranteed_contracts=False
    , use_consumer=True
    , max_insurance_premium=100, n_retrials=5
    , negotiator_type: str = 'negmas.sao.AspirationNegotiator'
    , transportation_delay=0
    , default_signing_delay=0
    , max_storage=sys.maxsize
    , consumption_horizon=15
    , consumption=(3, 5)
    , negotiation_speed=21
    , neg_time_limit=60 * 4
    , neg_n_steps=20
    , n_steps=100
    , time_limit=60 * 90
    , n_default_per_level: int = 5

) -> SCMLWorld:
    """
    Creates a world compatible with the ANAC 2019 competition. Note that

    Args:
        name: World name to use
        agent_names_reveal_type: If true, a snake_case version of the agent_type will prefix agent names
        randomize: If true, managers are assigned to factories randomly otherwise in the order
        they are giving (cycling).
        n_intermediate:
        n_default_per_level:
        competitors: A list of class names for the competitors
        n_miners: number of miners of the single raw material
        n_factories_per_level: number of factories at every production level
        n_consumers: number of consumers of the final product
        n_steps: number of simulation steps
        n_lines_per_factory: number of lines in each factory
        negotiation_speed: The number of negotiation steps per simulation step. None means infinite
        default_signing_delay: The number of simulation between contract conclusion and signature
        neg_n_steps: The maximum number of steps of a single negotiation (that is double the number of rounds)
        neg_time_limit: The total time-limit of a single negotiation
        time_limit: The total time-limit of the simulation
        transportation_delay: The transportation delay
        n_retrials: The number of retrials the `Miner` and `GreedyFactoryManager` will try if negotiations fail
        max_insurance_premium: The maximum insurance premium accepted by `GreedyFactoryManager` (-1 to disable)
        use_consumer: If true, the `GreedyFactoryManager` will use an internal consumer for buying its needs
        guaranteed_contracts: If true, the `GreedyFactoryManager` will only sign contracts that it can guaratnee not to
        break.
        consumption_horizon: The number of steps for which `Consumer` publishes `CFP` s
        consumption: The consumption schedule will be sampled from a uniform distribution with these limits inclusive
        log_file_name: File name to store the logs
        negotiator_type: The negotiation factory used to create all negotiators
        max_storage: maximum storage capacity for all factory negmas If None then it is unlimited


    Returns:
        SCMLWorld ready to run

    Remarks:

        - Every production level n has one process only that takes n steps to complete


    """
    competitors = list(competitors)
    if n_factories_per_level == n_default_per_level and len(competitors) > 0:
        raise ValueError(f'All factories in all levels are occupied by the default factory manager. Either decrease'
                         f' n_default_per_level ({n_default_per_level}) or increase n_factories_per_level '
                         f' ({n_factories_per_level})')
    if isinstance(n_intermediate, Iterable):
        n_intermediate = list(n_intermediate)
    else:
        n_intermediate = [n_intermediate, n_intermediate]
    max_insurance_premium = None if max_insurance_premium < 0 else max_insurance_premium
    n_competitors = len(competitors)
    n_intermediate_levels_min = int(math.ceil(n_competitors / (n_factories_per_level - n_default_per_level))) - 1
    if n_intermediate_levels_min > n_intermediate[1]:
        raise ValueError(f'Need {n_intermediate_levels_min} intermediate levels to run {n_competitors} competitors')
    n_intermediate[0] = max(n_intermediate_levels_min, n_intermediate[0])
    competitors = [get_class(c) if isinstance(c, str) else c for c in competitors]
    if len(competitors) < 1:
        competitors.append(GreedyFactoryManager)
    world = SCMLWorld.single_path_world(log_file_name=log_file_name, n_steps=n_steps
                                        , agent_names_reveal_type=agent_names_reveal_type
                                        , negotiation_speed=negotiation_speed
                                        , n_intermediate_levels=randint(*n_intermediate)
                                        , n_miners=n_miners
                                        , n_consumers=n_consumers
                                        , n_factories_per_level=n_factories_per_level
                                        , consumption=consumption
                                        , consumer_kwargs={'negotiator_type': negotiator_type
            , 'consumption_horizon': consumption_horizon}
                                        , miner_kwargs={'negotiator_type': negotiator_type, 'n_retrials': n_retrials}
                                        , manager_kwargs={'negotiator_type': negotiator_type, 'n_retrials': n_retrials
            , 'sign_only_guaranteed_contracts': guaranteed_contracts
            , 'use_consumer': use_consumer
            , 'max_insurance_premium': max_insurance_premium}
                                        , transportation_delay=transportation_delay
                                        , time_limit=time_limit
                                        , neg_time_limit=neg_time_limit
                                        , neg_n_steps=neg_n_steps
                                        , default_signing_delay=default_signing_delay
                                        , n_lines_per_factory=n_lines_per_factory
                                        , max_storage=max_storage
                                        , manager_types=competitors
                                        , n_default_per_level=n_default_per_level
                                        , randomize=randomize
                                        , name=name)

    return world


def balance_calculator(world: SCMLWorld) -> WorldRunResults:
    """A scoring function that scores factory managers' performance by the final balance only ignoring whatever still
    in their inventory.

    Args:
        world: The world which is assumed to be run up to the point at which the scores are to be calculated.

    Returns:
        WorldRunResults giving the names, scores, and types of factory managers.

    """
    result = WorldRunResults(world_name=world.name, log_file_name=world.log_file_name)
    initial_balances = []
    for manager in world.factory_managers:
        if '_default__preassigned__' in manager.id:
            continue
        initial_balances.append(world.a2f[manager.id].initial_balance)
    normalize = all(_ != 0 for _ in initial_balances)
    for manager in world.factory_managers:
        if '_default__preassigned__' in manager.id:
            continue
        factory = world.a2f[manager.id]
        result.names.append(manager.name)
        result.types.append(manager.__class__.__name__)
        if normalize:
            result.scores.append((factory.balance - factory.initial_balance) / factory.initial_balance)
        else:
            result.scores.append(factory.balance - factory.initial_balance)
    return result


def anac2019_tournament(competitors: Sequence[Union[str, Type[FactoryManager]]]
                        , randomize=True
                        , agent_names_reveal_type=False
                        , n_runs_per_config: int = 5, tournament_path: str = './logs/tournaments'
                        , max_n_runs: int = 100
                        , total_timeout: Optional[int] = None
                        , parallelism='parallel'
                        , scheduler_ip: Optional[str] = None
                        , scheduler_port: Optional[str] = None
                        , tournament_progress_callback: Callable[[Optional[WorldRunResults]], None] = None
                        , world_progress_callback: Callable[[Optional[SCMLWorld]], None] = None
                        , name: str = None
                        , verbose: bool = False
                        , configs_only=False
                        , **kwargs
                        ) -> Union[TournamentResults, PathLike]:
    """
    The function used to run ANAC 2019 SCML tournaments.

    Args:

        name: Tournament name
        competitors: A list of class names for the competitors
        randomize: If true, then instead of trying all possible permutations of assignment random shuffles will be used.
        agent_names_reveal_type: If true then the type of an agent should be readable in its name (most likely at its
        beginning).
        max_n_runs: No more than n_runs_max worlds will be run. If `randomize` then it cannot be None and that is exactly
        the number of worlds to run. If not `randomize` then at most this number of worlds will be run if it is not None
        n_runs_per_config: Number of runs per configuration.
        total_timeout: Total timeout for the complete process
        tournament_path: Path at which to store all results. A scores.csv file will keep the scores and logs folder will
        keep detailed logs
        parallelism: Type of parallelism. Can be 'none' for serial, 'local' for parallel and 'dist' for distributed
        scheduler_port: Port of the dask scheduler if parallelism is dask, dist, or distributed
        scheduler_ip:   IP Address of the dask scheduler if parallelism is dask, dist, or distributed
        world_progress_callback: A function to be called after everystep of every world run (only allowed for serial
        evaluation and should be used with cautious).
        tournament_progress_callback: A function to be called with `WorldRunResults` after each world finished
        processing
        verbose: Verbosity
        configs_only: If true, a config file for each
        kwargs: Arguments to pass to the `world_generator` function

    Returns:
        `TournamentResults` The results of the tournament or a `PathLike` giving the location where configs were saved

    Remarks:
        Default parameters will be used in the league with the exception of `parallelism` which may use distributed
        processing

    """
    return tournament(competitors=competitors, randomize=randomize, agent_names_reveal_type=agent_names_reveal_type
                      , max_n_runs=max_n_runs, n_runs_per_config=n_runs_per_config
                      , tournament_path=tournament_path, total_timeout=total_timeout
                      , parallelism=parallelism, scheduler_ip=scheduler_ip, scheduler_port=scheduler_port
                      , tournament_progress_callback=tournament_progress_callback
                      , world_progress_callback=world_progress_callback, name=name, verbose=verbose
                      , configs_only=configs_only
                      , world_generator=anac2019_world, score_calculator=balance_calculator, **kwargs)
