"""
Tournament generation and management.

"""
import concurrent.futures as futures
import copy
import itertools
import json
import pathlib
import random
import time
import traceback
from os import PathLike
from pathlib import Path
from typing import Optional, List, Callable, Union, Type, Sequence

import numpy as np
import pandas as pd
import yaml
from dataclasses import dataclass, field
from typing_extensions import Protocol

from .situated import Agent, World, save_stats

from negmas.helpers import get_class, unique_name, import_by_name, get_full_type_name, humanize_time

__all__ = [
    'tournament',
    'WorldGenerator',
    'WorldRunResults',
    'TournamentResults',
    'run_world',
    'process_world_run',
]

PROTOCOL_CLASS_NAME_FIELD = '__mechanism_class_name'

try:
    # disable a warning in yaml 1b1 version
    yaml.warnings({'YAMLLoadWarning': False})
except:
    pass


class WorldGenerator(Protocol):
    """A callback-protocol specifying the signature of a world generator function that can be passed to `tournament`

    Args:
            name: world name. If None, a random name should be generated
            competitors: A list of `Agent` types that can be used to create the agents of the competitor types
            log_File_name: A log file name to keep logs
            randomize: If true, competitors should be assigned randomly within the world. The meaning of "random
            assignment" can vary from a world to another. In general it should be the case that if randomize is False,
            all worlds generated given kwargs, and a selection competitors will be the same.
            agent_names_reveal_type: Whether the type of an agent should be apparent in its name
            kwargs: key-value pairs of arguments.

    See Also:
        `tournament`

    """

    def __call__(self, name: Optional[str] = None, competitors: Sequence[Type[Agent]] = ()
                 , log_file_name: Optional[str] = None, randomize: bool = True, agent_names_reveal_type: bool = False
                 , **kwargs) -> World: ...


@dataclass
class WorldRunResults:
    """Results of a world run"""
    world_name: str
    """World name"""
    log_file_name: str
    """Log file name"""
    names: List[str] = field(default_factory=list, init=False)
    """Agent names"""
    scores: List[float] = field(default_factory=list, init=False)
    """Agent scores"""
    types: List[str] = field(default_factory=list, init=False)
    """Agent type names"""


@dataclass
class TournamentResults:
    scores: pd.DataFrame
    total_scores: pd.DataFrame
    winners: List[str]
    """Winner type name(s) which may be a list"""
    winners_scores: np.array
    """Winner score (accumulated)"""
    ttest: pd.DataFrame


def run_world(world_info: dict):
    """Runs a world and returns stats. This function is designed to be used with distributed systems like dask.

    Args:
        world_info: World info dict. See remarks for its parameters

    Remarks:

        The `world_info` dict should have the following members:

            - name: world name [Defaults to random]
            - competitors: list of strings giving competitor types [Defaults to an empty list]
            - log_file_name: file name to store the world log [Defaults to random]
            - randomize: whether to randomize assignment [Defaults to True]
            - agent_names_reveal_type: whether agent names reveal type [Defaults to False]
            - __dir_name: directory to store the world stats [Defaults to random]
            - __world_generator: full name of the world generator function (including its module) [Required]
            - __score_calculator: full name of the score calculator function [Required]
            - __tournament_name: name of the tournament [Defaults to random]
            - others: values of all other keys are passed to the world generator as kwargs
    """
    world_generator = world_info.get('__world_generator', None)
    score_calculator = world_info.get('__score_calculator', None)
    tournament_name = world_info.get('__tournament_name', unique_name(base=""))
    assert world_generator and score_calculator, f'Cannot run without specifying both a world generator and a score ' \
        f'calculator'

    world_generator = import_by_name(world_generator)
    score_calculator = import_by_name(score_calculator)
    world_info['competitors'] = [get_class(_) for _ in world_info.get('competitors', [])]
    default_name = unique_name(base="")
    world_info['name'] = world_info.get('name', default_name)
    world_name = world_info['name']
    default_dir = (Path(f'~') / 'negmas' / 'tournaments' / tournament_name / world_name).absolute()
    world_info['log_file_name'] = world_info.get('log_file_name', str(default_dir / 'log.txt'))
    world_info['agent_names_reveal_type'] = world_info.get('agent_names_reveal_type', False)
    world_info['randomize'] = world_info.get('randomzie', True)
    world_info['__dir_name'] = world_info.get('__dir_name', str(default_dir))

    # delete the parameters not used by _run_world
    for k in ('__world_generator', '__tournament_name', '__score_calculator'):
        if k in world_info.keys():
            del world_info[k]
    return _run_world(world_info=world_info, world_generator=world_generator, score_calculator=score_calculator)


def _run_world(world_info: dict, world_generator: WorldGenerator,
               score_calculator: Callable[[World], WorldRunResults]
               , world_progress_callback: Callable[[Optional[World]], None] = None
               ):
    """Runs a world and returns stats

    Args:
        world_info: World info dict. See remarks for its parameters
        world_generator: World generator function.
        score_calculator: Score calculator function
        world_progress_callback: world progress callback

    Remarks:

        The `world_info` dict should have the following members:

            - name: world name
            - competitors: list of types giving competitor types
            - log_file_name: file name to store the world log
            - randomize: whether to randomize assignment
            - agent_names_reveal_type: whether agent names reveal type
            - __dir_name: directory to store the world stats
            - others: values of all other keys are passed to the world generator as kwargs
    """
    world_info = world_info.copy()
    dir_name = world_info['__dir_name']
    del world_info['__dir_name']
    world = world_generator(**world_info)
    if world_progress_callback is None:
        world.run()
    else:
        _start_time = time.monotonic()
        for _ in range(world.n_steps):
            if world.time_limit is not None and (time.monotonic() - _start_time) >= world.time_limit:
                break
            if not world.step():
                break
            world_progress_callback(world)
    save_stats(world=world, log_dir=dir_name)
    scores = score_calculator(world)
    return scores, dir_name


def process_world_run(results: WorldRunResults, tournament_name: str, dir_name: str) -> pd.DataFrame:
    """
    Generates a dataframe with the results of this world run

    Args:
        results: Results of the world run
        tournament_name: tournament name
        dir_name: directory name to store the stats.

    Returns:

        A pandas DataFrame with agent_name, agent_type, score, log_file, world, and stats_folder columns

    """
    log_file, world_name_ = results.log_file_name, results.world_name
    if log_file is not None:
        with open(log_file, 'a') as f:
            f.write(f'\nPART of TOURNAMENT {tournament_name}. This world run completed successfully\n')
    scores = []
    for name_, type_, score in zip(results.names, results.types, results.scores):
        scores.append({'agent_name': name_, 'agent_type': type_, 'score': score, 'log_file': log_file
                          , 'world': world_name_, 'stats_folder': dir_name})
    return pd.DataFrame(data=scores)


def _run_dask(scheduler_ip, scheduler_port, verbose, world_infos, world_generator, tournament_progress_callback
              , n_worlds, name, score_calculator) -> List[pd.DataFrame]:
    """Runs the tournament on dask"""

    import distributed
    scores = []
    if scheduler_ip is None and scheduler_port is None:
        address = None
    else:
        if scheduler_ip is None:
            scheduler_ip = '127.0.0.1'
        if scheduler_port is None:
            scheduler_port = '8786'
        address = f'{scheduler_ip}:{scheduler_port}'
    if verbose:
        print(f'Will use DASK on {address}')
    client = distributed.Client(address=address, set_as_default=True)
    future_results = []
    for world_info in world_infos:
        future_results.append(client.submit(_run_world, world_info, world_generator, score_calculator))
    print(f'Submitted all processes to DASK ({len(world_infos)})')
    _strt = time.perf_counter()
    for i, (future, result) in enumerate(
        distributed.as_completed(future_results, with_results=True, raise_errors=False)):
        try:
            score_, dir_name = result
            if tournament_progress_callback is not None:
                tournament_progress_callback(score_, i, n_worlds)
            scores.append(process_world_run(score_, tournament_name=name, dir_name=str(dir_name)))
            if verbose:
                _duration = time.perf_counter() - _strt
                print(f'{i + 1:003} of {n_worlds:003} [{100 * (i + 1) / n_worlds:0.3}%] completed in '
                      f'{humanize_time(_duration)} [ETA {humanize_time(_duration * n_worlds  / (i + 1))}]')
        except Exception as e:
            if tournament_progress_callback is not None:
                tournament_progress_callback(None, i, n_worlds)
            print(traceback.format_exc())
            print(e)
    client.shutdown()
    return scores


def tournament(competitors: Sequence[Union[str, Type[Agent]]]
               , world_generator: WorldGenerator
               , score_calculator: Callable[[World], WorldRunResults]
               , randomize=False
               , agent_names_reveal_type=False
               , max_n_runs: int = 1000
               , n_runs_per_config: int = 5
               , tournament_path: str = './logs/tournaments'
               , total_timeout: Optional[int] = None
               , parallelism='local'
               , scheduler_ip: Optional[str] = None
               , scheduler_port: Optional[str] = None
               , tournament_progress_callback: Callable[[Optional[WorldRunResults], int, int], None] = None
               , world_progress_callback: Callable[[Optional[World]], None] = None
               , name: str = None
               , verbose: bool = False
               , configs_only: bool = False
               , **kwargs
               ) -> Union[TournamentResults, PathLike]:
    """
    Runs a tournament

    Args:

        name: Tournament name
        world_generator: A functions to generate worlds for the tournament
        score_calculator: A function for calculating the score of a world *After it finishes running*
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

    """
    dask_options = ('dist', 'distributed', 'dask', 'd')
    multiprocessing_options = ('local', 'parallel', 'par', 'p')
    serial_options = ('none', 'serial', 's')
    assert total_timeout is None or parallelism not in dask_options, f'Cannot use {parallelism} with a total-timeout'
    assert world_progress_callback is None or parallelism not in dask_options, f'Cannot use {parallelism} with a world callback'
    if name is None:
        name = unique_name('', add_time=True, rand_digits=0)
    competitors = list(set(competitors))
    if tournament_path.startswith('~'):
        tournament_path = Path.home() / ('/'.join(tournament_path.split('/')[1:]))
    tournament_path = (pathlib.Path(tournament_path) / name).absolute()
    tournament_path.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f'Results of Tournament {name} will be saved to {str(tournament_path)}')
    params = {
        'competitors': [get_class(_).__name__ if not isinstance(_, str) else _ for _ in competitors],
        'randomize': randomize,
        'n_runs': max_n_runs,
        'tournament_path': str(tournament_path),
        'total_timeout': total_timeout,
        'parallelism': parallelism,
        'scheduler_ip': scheduler_ip,
        'scheduler_port': scheduler_port,
        'name': name,
        'n_worlds_to_run': None
    }
    params.update(kwargs)
    with (tournament_path / 'params.json').open('w') as f:
        json.dump(params, f, sort_keys=True, indent=4)
    world_infos = []
    if randomize:
        for i in range(max_n_runs):
            random.shuffle(competitors)
            world_name = unique_name(f'{i:05}', add_time=True, rand_digits=4)
            dir_name = tournament_path / world_name
            world_info = {'name': world_name, 'competitors': competitors, 'log_file_name': str(dir_name / 'log.txt')
                , 'randomize': True, 'agent_names_reveal_type': agent_names_reveal_type
                , '__dir_name': str(dir_name)}
            world_info.update(kwargs)
            world_infos += [world_info.copy() for _ in range(n_runs_per_config)]
    else:
        c_list = list(itertools.permutations(competitors))
        if max_n_runs is not None:
            if len(c_list) > max_n_runs:
                print(f'Need {len(c_list)} permutations but allowed to only use {max_n_runs} of them'
                      f' ({max_n_runs / len(c_list):0.2%})')
                c_list = random.shuffle(c_list)[:max_n_runs]

        for i, c in enumerate(c_list):
            world_name = unique_name(f'{i:05}', add_time=True, rand_digits=4)
            dir_name = tournament_path / world_name
            world_info = {'name': world_name, 'competitors': list(c), 'log_file_name': str(dir_name / 'log.txt')
                , 'randomize': False, 'agent_names_reveal_type': agent_names_reveal_type
                , '__dir_name': str(dir_name)}
            world_info.update(kwargs)
            world_infos += [world_info.copy() for _ in range(n_runs_per_config)]

    saved_configs = [{k: copy.copy(v) if k != 'competitors' else
    [get_full_type_name(c) if not isinstance(c, str) else c for c in v]
                      for k, v in _.items()} for _ in world_infos]
    score_calculator_name = get_full_type_name(score_calculator) if not isinstance(score_calculator,
                                                                                   str) else score_calculator
    world_generator_name = get_full_type_name(world_generator) if not isinstance(world_generator,
                                                                                 str) else world_generator
    for d in saved_configs:
        d['__score_calculator'] = score_calculator_name
        d['__world_generator'] = world_generator_name
        d['__tournament_name'] = name
    config_path = tournament_path / 'configs'
    config_path.mkdir(exist_ok=True, parents=True)
    for i, conf in enumerate(saved_configs):
        f_name = config_path / f'{i:06}.json'
        with open(f_name, 'w') as f:
            json.dump(conf, f, sort_keys=True, indent=4)

    if configs_only:
        return config_path

    scores = []
    scores_file = str(tournament_path / 'scores.csv')
    n_worlds = len(world_infos)
    params['n_worlds_to_run'] = n_worlds
    with (tournament_path / 'params.json').open('w') as f:
        json.dump(params, f, sort_keys=True, indent=4)
    if verbose:
        print(f'Will run {n_worlds} worlds')
    if parallelism in serial_options:
        strt = time.perf_counter()
        for i, world_info in enumerate(world_infos):
            if total_timeout is not None and time.perf_counter() - strt > total_timeout:
                break
            try:
                score_, _ = _run_world(world_info=world_info, world_generator=world_generator
                                       , world_progress_callback=world_progress_callback
                                       , score_calculator=score_calculator)
                if tournament_progress_callback is not None:
                    tournament_progress_callback(score_, i, n_worlds)
                scores.append(process_world_run(score_, tournament_name=name, dir_name=str(world_info['__dir_name'])))
                if verbose:
                    _duration = time.perf_counter() - strt
                    print(f'{i + 1:003} of {n_worlds:003} [{100 * (i + 1) / n_worlds:0.3}%] completed '
                          f'in {humanize_time(_duration)}'
                          f' [ETA {humanize_time(_duration * n_worlds  / (i + 1))}]')
            except Exception as e:
                if tournament_progress_callback is not None:
                    tournament_progress_callback(None, i, n_worlds)
                print(traceback.format_exc())
                print(e)
    elif parallelism in multiprocessing_options:
        executor = futures.ProcessPoolExecutor(max_workers=None)
        future_results = []
        for world_info in world_infos:
            future_results.append(executor.submit(_run_world, world_info, world_generator, score_calculator
                                                  , world_progress_callback))
        if verbose:
            print(f'Submitted all processes ({len(world_infos)})')
        _strt = time.perf_counter()
        for i, future in enumerate(futures.as_completed(future_results, timeout=total_timeout)):
            try:
                score_, dir_name = future.result()
                if tournament_progress_callback is not None:
                    tournament_progress_callback(score_, i, n_worlds)
                scores.append(process_world_run(score_, tournament_name=name, dir_name=str(dir_name)))
                if verbose:
                    _duration = time.perf_counter() - _strt
                    print(f'{i+1:003} of {n_worlds:003} [{100*(i+1)/n_worlds:0.3}%] completed in '
                          f'{humanize_time(_duration)}'
                          f' [ETA {humanize_time(_duration * n_worlds  / (i + 1))}]')
            except futures.TimeoutError:
                if tournament_progress_callback is not None:
                    tournament_progress_callback(None, i, n_worlds)
                print('Tournament timed-out')
                break
            except Exception as e:
                if tournament_progress_callback is not None:
                    tournament_progress_callback(None, i, n_worlds)
                print(traceback.format_exc())
                print(e)
    elif parallelism in dask_options:
        scores = _run_dask(scheduler_ip, scheduler_port, verbose, world_infos, world_generator
                           , tournament_progress_callback, n_worlds, name, score_calculator)
    if verbose:
        print(f'Finding winners')
    if len(scores) < 1:
        return TournamentResults(scores=pd.DataFrame(), total_scores=pd.DataFrame()
                          , winners=[], winners_scores=np.array([])
                          , ttest=pd.DataFrame())
    scores: pd.DataFrame = pd.concat(scores, ignore_index=True)
    scores = pd.DataFrame(data=scores)
    scores.to_csv(scores_file, index_label='index')
    scores = scores.loc[~scores['agent_type'].isnull(), :]
    scores = scores.loc[scores['agent_type'].str.len() > 0, :]
    total_scores = scores.groupby(['agent_type'])['score'].mean().sort_values(ascending=False).reset_index()
    winner_table = total_scores.loc[total_scores['score'] == total_scores['score'].max(), :]
    winners = winner_table['agent_type'].values.tolist()
    winner_scores = winner_table['score'].values
    types = list(scores['agent_type'].unique())

    ttest_results = []
    for i, t1 in enumerate(types):
        for j, t2 in enumerate(types[i + 1:]):
            from scipy.stats import ttest_ind
            t, p = ttest_ind(scores[scores['agent_type'] == t1].score, scores[scores['agent_type'] == t2].score)
            ttest_results.append({'a': t1, 'b': t2, 't': t, 'p': p})

    if verbose:
        print(f'Tournament completed successfully\nWinners: {list(zip(winners, winner_scores))}')

    scores.to_csv(str(tournament_path / 'scores.csv'), index_label='index')
    total_scores.to_csv(str(tournament_path / 'total_scores.csv'), index_label='index')
    winner_table.to_csv(str(tournament_path / 'winners.csv'), index_label='index')
    ttest_results = pd.DataFrame(data=ttest_results)
    ttest_results.to_csv(str(tournament_path / 'ttest.csv'), index_label='index')

    return TournamentResults(scores=scores, total_scores=total_scores, winners=winners, winners_scores=winner_scores
                             , ttest=ttest_results)
