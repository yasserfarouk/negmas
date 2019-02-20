import itertools
import math
import pathlib
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from random import shuffle
from time import perf_counter
from typing import TYPE_CHECKING

import pandas as pd
from dataclasses import dataclass
from distributed import Client

from negmas.helpers import get_class, unique_name
from .factory_managers import GreedyFactoryManager
from .world import SCMLWorld

if True:
    from typing import Tuple, Union, Type, Iterable, Sequence, Optional, Dict, Any
    from .factory_managers import FactoryManager

__all__ = [
    'anac2019_world',
    'anac2019_tournament'
]


def anac2019_world(n_intermediate: Tuple[int, int] = (1, 4)
                   , n_miners=5, n_factories_per_level=5, n_consumers=5, n_lines_per_factory=10
                   , guaranteed_contracts=False, use_consumer=True, max_insurance_premium=-1, n_retrials=4
                   , competitors: Tuple[Union[str, Type[FactoryManager]]] = ()
                   , negotiator_type: str = 'negmas.sao.AspirationNegotiator'
                   , transportation_delay=0, default_signing_delay=1
                   , max_storage=None
                   , consumption_horizon=15
                   , consumption=(3, 5)
                   , negotiation_speed=21, neg_time_limit=60 * 4, neg_n_steps=20
                   , n_steps=60, time_limit=60 * 90
                   , n_greedy_per_level: int = 2
                   , random_factory_manager_assignment: bool = True
                   , log_file_name: str = None
                   ):
    """
    Creates a world compatible with the ANAC 2019 competition. Note that

    Args:
        random_factory_manager_assignment: If true, managers are assigned to factories randomly otherwise in the order
        they are giving (cycling).
        n_intermediate:
        n_greedy_per_level:
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
    if n_factories_per_level == n_greedy_per_level and len(competitors) > 0:
        raise ValueError(f'All factories in all levels are occupied by greedy_factory_managers')
    if isinstance(n_intermediate, Iterable):
        n_intermediate = list(n_intermediate)
    else:
        n_intermediate = [n_intermediate, n_intermediate]
    max_insurance_premium = None if max_insurance_premium < 0 else max_insurance_premium
    n_competitors = len(competitors)
    n_intermediate_levels_min = int(math.ceil(n_competitors / (n_factories_per_level - n_greedy_per_level))) - 1
    if n_intermediate_levels_min > n_intermediate[1]:
        raise ValueError(f'Need {n_intermediate_levels_min} intermediate levels to run {n_competitors} competitors')
    n_intermediate[0] = max(n_intermediate_levels_min, n_intermediate[0])
    competitors = [get_class(c) if isinstance(c, str) else c for c in competitors]
    if len(competitors) < 1:
        competitors.extend(GreedyFactoryManager)
    world = SCMLWorld.single_path_world(log_file_name=log_file_name, n_steps=n_steps
                                        , negotiation_speed=negotiation_speed
                                        , n_intermediate_levels=randint(*n_intermediate)
                                        , n_miners=n_miners
                                        , n_consumers=n_consumers
                                        , n_factories_per_level=n_factories_per_level
                                        , consumption=consumption
                                        , consumer_kwargs={'negotiator_type': negotiator_type
            , 'consumption_horizon': consumption_horizon}
                                        , miner_kwargs={'negotiator_type': negotiator_type, 'n_retrials': n_retrials}
                                        , factory_kwargs={'negotiator_type': negotiator_type, 'n_retrials': n_retrials
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
                                        , n_greedy_per_level=n_greedy_per_level
                                        , random_factory_manager_assignment=random_factory_manager_assignment)

    return world


def _run_world(world: SCMLWorld):
    """Runs a world and returns stats"""
    world.run()
    return world.stats, world.name, world.log_file_name


@dataclass
class TournamentResults:
    scores: pd.DataFrame
    total_scores: pd.DataFrame
    winner: str
    winner_score: float
    ttest: pd.DataFrame


def anac2019_tournament(competitors: Sequence[Union[str, Type[FactoryManager]]]
                        , randomize=True
                        , n_runs: int = 10, tournament_path: str = './logs/tournaments'
                        , total_timeout: Optional[int] = None
                        , parallelism='local'
                        , scheduler_ip: Optional[str] = None
                        , scheduler_port: Optional[str] = None
                        , n_intermediate: Tuple[int, int] = (1, 4)
                        , n_miners=5, n_factories_per_level=5, n_consumers=5, n_lines_per_factory=10
                        , guaranteed_contracts=False, use_consumer=True, max_insurance_premium=-1, n_retrials=4
                        , negotiator_type: str = 'negmas.sao.AspirationNegotiator'
                        , transportation_delay=0, default_signing_delay=1
                        , max_storage=None
                        , consumption_horizon=15
                        , consumption=(3, 5)
                        , negotiation_speed=21, neg_time_limit=60 * 4, neg_n_steps=20
                        , n_steps=60, time_limit=60 * 90
                        , n_greedy_per_level: int = 2
                        ) -> TournamentResults:
    """
    Runs a tournament

    Args:

        competitors: A list of class names for the competitors
        randomize: If true, then instead of trying all possible permutations of assignment random shuffles will be used.
        n_runs: No more than n_runs_max worlds will be run. If `randomize` then it cannot be None and that is exactly
        the number of worlds to run. If not `randomize` then at most this number of worlds will be run if it is not None
        total_timeout: Total timeout for the complete process
        tournament_path: Path at which to store all results. A scores.csv file will keep the scores and logs folder will
        keep detailed logs
        parallelism: Type of parallelism. Can be 'none' for serial, 'local' for parallel and 'dist' for distributed
        scheduler_port: Port of the dask scheduler if parallelism is dask, dist, or distributed
        scheduler_ip:   IP Address of the dask scheduler if parallelism is dask, dist, or distributed
        n_intermediate:
        n_greedy_per_level:
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
        negotiator_type: The negotiation factory used to create all negotiators
        max_storage: maximum storage capacity for all factory negmas If None then it is unlimited

    Returns:
        scores as a dataframe

    Remarks:

        - Every production level n has one process only that takes n steps to complete


    """
    tournament_path = pathlib.Path(tournament_path) / unique_name('', add_time=True, rand_digits=0)
    os.makedirs(str(tournament_path), exist_ok=True)
    worlds = []
    if randomize:
        for i in range(n_runs):
            shuffle(competitors)
            log_file_name = str(
                tournament_path / 'logs' / (unique_name(f'{i:05}', add_time=True, rand_digits=4) + '.log').replace('/',
                                                                                                                   ''))
            worlds.append(anac2019_world(competitors=competitors
                                         , log_file_name=log_file_name
                                         , random_factory_manager_assignment=True
                                         , n_intermediate=n_intermediate
                                         , n_miners=n_miners
                                         , n_factories_per_level=n_factories_per_level
                                         , n_consumers=n_consumers
                                         , n_lines_per_factory=n_lines_per_factory
                                         , guaranteed_contracts=guaranteed_contracts
                                         , use_consumer=use_consumer
                                         , max_insurance_premium=max_insurance_premium
                                         , n_retrials=n_retrials
                                         , negotiator_type=negotiator_type
                                         , transportation_delay=transportation_delay
                                         , default_signing_delay=default_signing_delay
                                         , max_storage=max_storage
                                         , consumption_horizon=consumption_horizon
                                         , consumption=consumption
                                         , negotiation_speed=negotiation_speed
                                         , neg_time_limit=neg_time_limit
                                         , neg_n_steps=neg_n_steps
                                         , n_steps=n_steps
                                         , time_limit=time_limit
                                         , n_greedy_per_level=n_greedy_per_level
                                         ))

    else:
        c_list = list(itertools.permutations(competitors))
        extra_runs = 0
        if n_runs is not None:
            if len(c_list) > n_runs:
                print(f'Need {len(c_list)} permutations but allowed to only use {n_runs} of them')
                c_list = shuffle(c_list)[:n_runs]
            elif len(c_list) < n_runs:
                extra_runs = n_runs - len(c_list)

        for i, c in enumerate(c_list):
            log_file_name = str(
                tournament_path / 'logs' / (unique_name(f'{i:05}', add_time=True, rand_digits=4) + '.log').replace('/',
                                                                                                                   ''))
            worlds.append(anac2019_world(competitors=c, log_file_name=log_file_name
                                         , random_factory_manager_assignment=False
                                         , n_intermediate=n_intermediate
                                         , n_miners=n_miners
                                         , n_factories_per_level=n_factories_per_level
                                         , n_consumers=n_consumers
                                         , n_lines_per_factory=n_lines_per_factory
                                         , guaranteed_contracts=guaranteed_contracts
                                         , use_consumer=use_consumer
                                         , max_insurance_premium=max_insurance_premium
                                         , n_retrials=n_retrials
                                         , negotiator_type=negotiator_type
                                         , transportation_delay=transportation_delay
                                         , default_signing_delay=default_signing_delay
                                         , max_storage=max_storage
                                         , consumption_horizon=consumption_horizon
                                         , consumption=consumption
                                         , negotiation_speed=negotiation_speed
                                         , neg_time_limit=neg_time_limit
                                         , neg_n_steps=neg_n_steps
                                         , n_steps=n_steps
                                         , time_limit=time_limit
                                         , n_greedy_per_level=n_greedy_per_level
                                         ))
            if extra_runs > 0:
                for j in range(extra_runs):
                    shuffle(competitors)
                    log_file_name = str(
                        tournament_path / 'logs' / (
                                unique_name(f'{j + len(c_list):05}', add_time=True, rand_digits=4) + '.log').replace(
                            '/', ''))
                    worlds.append(anac2019_world(competitors=competitors
                                                 , log_file_name=log_file_name
                                                 , random_factory_manager_assignment=True
                                                 , n_intermediate=n_intermediate
                                                 , n_miners=n_miners
                                                 , n_factories_per_level=n_factories_per_level
                                                 , n_consumers=n_consumers
                                                 , n_lines_per_factory=n_lines_per_factory
                                                 , guaranteed_contracts=guaranteed_contracts
                                                 , use_consumer=use_consumer
                                                 , max_insurance_premium=max_insurance_premium
                                                 , n_retrials=n_retrials
                                                 , negotiator_type=negotiator_type
                                                 , transportation_delay=transportation_delay
                                                 , default_signing_delay=default_signing_delay
                                                 , max_storage=max_storage
                                                 , consumption_horizon=consumption_horizon
                                                 , consumption=consumption
                                                 , negotiation_speed=negotiation_speed
                                                 , neg_time_limit=neg_time_limit
                                                 , neg_n_steps=neg_n_steps
                                                 , n_steps=n_steps
                                                 , time_limit=time_limit
                                                 , n_greedy_per_level=n_greedy_per_level
                                                 ))

    scores = []
    scores_file = str(tournament_path / 'scores.csv')

    def _process_stats(stats: Dict[str, Any], world_name: str, file_name: str):
        if file_name is not None:
            with open(file_name, 'a') as f:
                f.write('\nDONE SUCCESSFULLY\n')
        for k, v in stats.items():
            if not k.startswith('balance'):
                continue
            if k.endswith('insurance') or k.endswith('bank') or k[len('balance_'):].startswith('c_') \
                or k[len('balance_'):].startswith('m_'):
                continue
            name_ = k[len('balance_'):]
            type_ = '_'.join(name_.split('_')[:-2])
            score = v[-1] - v[0]
            scores.append({'name': name_, 'type': type_, 'score': score, 'log_file': file_name
                              , 'world': world_name})
        pd.DataFrame(data=scores).to_csv(scores_file, index_label='index')

    if parallelism in ('serial', 'none'):
        strt = perf_counter()
        for world in worlds:
            if total_timeout is not None and perf_counter() - strt > total_timeout:
                break
            try:
                _process_stats(*_run_world(world))
            except Exception as e:
                print(traceback.format_exc())
                print(e)
    elif parallelism in ('local', 'parallel', 'processes'):
        executor = ProcessPoolExecutor(max_workers=None)
        future_results = []
        for world in worlds:
            future_results.append(executor.submit(_run_world, world))
        for future in as_completed(future_results, timeout=total_timeout):
            try:
                _process_stats(*future.results())
            except Exception as e:
                print(traceback.format_exc())
                print(e)
    elif parallelism in ('dist', 'distributed', 'dask'):
        client = Client(scheduler=f'{scheduler_ip}:{scheduler_port}' if scheduler_ip is not None else None)
        future_results = []
        for world in worlds:
            future_results.append(client.submit(_run_world, world))
        for future in as_completed(future_results, timeout=total_timeout):
            try:
                _process_stats(*future.results())
            except Exception as e:
                print(traceback.format_exc())
                print(e)

    scores = pd.DataFrame(data=scores)
    scores = scores.loc[~scores['type'].isnull(), :]
    scores = scores.loc[scores.type.str.len() > 0, :]
    total_scores = scores.groupby(['type'])['score'].sum().sort_values(ascending=False)
    winner = total_scores.index[0]
    winner_score = total_scores.loc[winner]
    types = list(scores['type'].unique())

    ttest_results = []
    for i, t1 in enumerate(types):
        for j, t2 in enumerate(types[i + 1:]):
            from scipy.stats import ttest_ind
            t, p = ttest_ind(scores[scores['type'] == t1].score, scores[scores['type'] == t2].score)
            ttest_results.append({'a': t1, 'b': t2, 't': t, 'p': p})

    return TournamentResults(scores=scores, total_scores=total_scores, winner=winner, winner_score=winner_score
                             , ttest=pd.DataFrame(data=ttest_results))
