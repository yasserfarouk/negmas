"""
Tournament generation and management.

"""
import concurrent.futures as futures
import copy
import hashlib
import itertools
import math
import pathlib
import random
import time
import traceback
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from multiprocessing import cpu_count
from os import PathLike
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

import numpy as np
import pandas as pd
import yaml
from scipy.stats import ks_2samp, ttest_ind
from typing_extensions import Protocol

from negmas.helpers import (
    exception2str,
    add_records,
    dump,
    get_class,
    get_full_type_name,
    humanize_time,
    import_by_name,
    load,
    unique_name,
)
from negmas.java import to_flat_dict

from .situated import Agent, World, save_stats

__all__ = [
    "tournament",
    "WorldGenerator",
    "WorldRunResults",
    "TournamentResults",
    "run_world",
    "process_world_run",
    "evaluate_tournament",
    "combine_tournaments",
    "combine_tournament_stats",
    "create_tournament",
    "run_tournament",
]

PROTOCOL_CLASS_NAME_FIELD = "__mechanism_class_name"

try:
    # disable a warning in yaml 1b1 version
    yaml.warnings({"YAMLLoadWarning": False})
except:
    pass


def _hash(*args) -> str:
    """Generates a unique ID given any inputs"""
    return hashlib.sha1(
        ("h" + "".join([str(_) for _ in args])).encode("utf-8")
    ).hexdigest()


class WorldGenerator(Protocol):
    """A callback-protocol specifying the signature of a world generator function that can be passed to `tournament`

    Args:
            kwargs: key-value pairs of arguments.

    See Also:
        `tournament`

    """

    def __call__(self, **kwargs) -> World:
        ...


class ConfigGenerator(Protocol):
    """A callback-protocol specifying the signature of a config generator function that can be passed to `tournament`

    Args:

            n_competitors: Number of competitor types
            n_agents_per_competitor: Number of agents to instantiate for each competitor
            agent_names_reveal_type: whether agent names contain their types (used for debugging purposes).
            non_competitors: A list of agent types that will not be competing in the sabotage competition but will exist
                             in the world
            non_competitor_params: paramters of non competitor agents
            compact: Whether to try to reduce memory footprint (and avoid logging)
            kwargs: key-value pairs of arguments.

    See Also:

        `tournament` `ConfigAssigner`

    """

    def __call__(
        self,
        n_competitors: int,
        n_agents_per_competitor: int,
        agent_names_reveal_type: bool = False,
        non_competitors: Optional[Tuple[Union[str, Any]]] = None,
        non_competitor_params: Optional[Tuple[Dict[str, Any]]] = None,
        compact: bool = False,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        ...


class ConfigAssigner(Protocol):
    """A callback-protocol specifying the signature of a function that can be used to assign competitors to a config
     generated using a `ConfigGenerator`

     Args:

         config: The dict returned from the `ConfigGenerator`
         max_n_worlds: Maximum allowed number of worlds to generate from this config
         n_agents_per_competitor: Number of agents to instantiate for each competitor
         competitors: A list of `Agent` types that can be used to create the competitors
         fair: If true, each competitor must be assigned to each unique config the same number of times. If max_n_worlds
               is None, this parameter has no effect, otherwise the nearest number of worlds to max_n_worlds that
               guarantee fairness will be used which may be > max_n_worlds
         params: A list of parameters to pass to the agent types

     See Also:

         `ConfigGenerator` `tournament`

    """

    def __call__(
        self,
        config: List[Dict[str, Any]],
        max_n_worlds: int,
        n_agents_per_competitor: int = 1,
        fair: bool = True,
        competitors: Sequence[Union[str, Type[Agent]]] = (),
        params: Sequence[Dict[str, Any]] = (),
        dynamic_non_competitors: Sequence[Union[str, Type[Agent]]] = None,
        dynamic_non_competitor_params: Sequence[Dict[str, Any]] = None,
        exclude_competitors_from_reassignment: bool = True,
    ) -> List[List[Dict[str, Any]]]:
        ...


@dataclass
class WorldRunResults:
    """Results of a world run"""

    world_names: List[str]
    """World names (there can be multiple worlds for each scoring call)"""
    log_file_names: List[str]
    """Log file names"""
    names: List[str] = field(default_factory=list, init=False)
    """Agent names"""
    ids: List[str] = field(default_factory=list, init=False)
    """Agent IDs"""
    scores: List[float] = field(default_factory=list, init=False)
    """Agent scores"""
    types: List[str] = field(default_factory=list, init=False)
    """Agent type names"""


@dataclass
class WorldSetRunStats:
    """Statistics kept in the tournament about the set of worlds"""

    name: str
    """Names of the world set separated by ;"""
    planned_n_steps: int
    """Planned number of steps for each world"""
    executed_n_steps: int
    """Actually executed number of steps for each world"""
    execution_time: int
    """Total execution time of each world"""
    simulation_exceptions: Dict[int, List[str]] = field(default_factory=dict)
    """Exceptions thrown by the simulator (not including mechanism creation and contract exceptions)"""
    contract_exceptions: Dict[int, List[str]] = field(default_factory=dict)
    """Exceptions thrown by the simulator during contract execution"""
    mechanism_exceptions: Dict[int, List[str]] = field(default_factory=dict)
    """Exceptions thrown by the simulator during mechanism creation or execution"""
    other_exceptions: List[str] = field(default_factory=list)
    """Exceptions raised by tournament running code itself not any world"""
    agent_exceptions: Dict[str, List[Tuple[int, str]]] = field(
        default_factory=lambda: defaultdict(list)
    )
    """All exceptions thrown per agent (not including negotiator exceptions)"""
    negotiator_exceptions: Dict[str, List[Tuple[int, str]]] = field(
        default_factory=lambda: defaultdict(list)
    )
    """All exceptions thown by negotiators of an agent"""
    agent_times: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    """Total execution time per agent"""


@dataclass
class TournamentResults:
    scores: pd.DataFrame
    """Scores of individual agent instantiations"""
    total_scores: pd.DataFrame
    """Total scores collected by competitor types"""
    winners: List[str]
    """Winner type name(s) which may be a list"""
    winners_scores: np.array
    """Winner score (accumulated)"""
    ttest: pd.DataFrame = None
    """Results of ttest analysis of the scores"""
    kstest: pd.DataFrame = None
    """Results of the nonparametric kstest"""
    stats: pd.DataFrame = None
    """Stats of all worlds"""
    agg_stats: pd.DataFrame = None
    """Aggregated stats per world"""
    score_stats: pd.DataFrame = None
    """Score statistics for different competitor types"""
    path: str = None
    """Path at which tournament results are stored"""
    world_stats: pd.DataFrame = None
    """Some statistics about each world run"""
    params: Dict[str, Any] = None
    """Parameters of the tournament"""

    def __str__(self):
        import tabulate

        results = ""
        results += tabulate.tabulate(self.total_scores)
        results += f"The winner(s) is: {self.winners}"
        if self.kstest is not None:
            results += tabulate.tabulate(self.kstest)
        elif self.ttest is not None:
            results += tabulate.tabulate(self.ttest)
        results += f"\n See stats at {self.path}"
        return results


def run_world(
    world_params: dict, dry_run: bool = False, save_world_stats: bool = True
) -> Tuple[str, Optional[WorldRunResults], WorldSetRunStats]:
    """Runs a world and returns stats. This function is designed to be used with distributed systems like dask.

    Args:
        world_params: World info dict. See remarks for its parameters
        dry_run: If true, the world will not be run. Only configs will be saved
        save_world_stats: If true, saves individual world stats

    Remarks:

        The `world_params` dict should have the following members:

            - name: world name [Defaults to random]
            - log_file_name: file name to store the world log [Defaults to random]
            - __dir_name: directory to store the world stats [Defaults to random]
            - __world_generator: full name of the world generator function (including its module) [Required]
            - __score_calculator: full name of the score calculator function [Required]
            - __tournament_name: name of the tournament [Defaults to random]
            - others: values of all other keys are passed to the world generator as kwargs
    """
    world_generator = world_params.get("__world_generator", None)
    score_calculator = world_params.get("__score_calculator", None)
    tournament_name = world_params.get("__tournament_name", unique_name(base=""))
    assert world_generator and score_calculator, (
        f"Cannot run without specifying both a world generator and a score "
        f"calculator"
    )

    world_generator = import_by_name(world_generator)
    score_calculator = import_by_name(score_calculator)
    default_name = unique_name(base="")
    world_params["name"] = world_params.get("name", default_name)
    world_name = world_params["name"]
    default_dir = (
        Path(f"~") / "negmas" / "tournaments" / tournament_name / world_name
    ).absolute()
    world_params["log_file_name"] = world_params.get("log_file_name", str("log.txt"))
    world_params["log_folder"] = str(default_dir)
    world_params["__dir_name"] = world_params.get("__dir_name", str(default_dir))
    # delete the parameters not used by _run_worlds
    for k in ("__world_generator", "__tournament_name", "__score_calculator"):
        if k in world_params.keys():
            world_params.pop(k, None)
    return _run_worlds(
        worlds_params=[world_params],
        world_generator=world_generator,
        score_calculator=score_calculator,
        dry_run=dry_run,
        save_world_stats=save_world_stats,
    )


def run_worlds(
    worlds_params: List[dict], dry_run: bool = False, save_world_stats: bool = True
) -> Tuple[str, Optional[WorldRunResults], WorldSetRunStats]:
    """Runs a set of worlds and returns stats. This function is designed to be used with distributed systems like dask.

    Args:
        worlds_params: list of World info dicts. See remarks for its parameters
        dry_run: If true, the world will not be run. Only configs will be saved
        save_world_stats: If true, saves individual world stats

    Remarks:

        Each dict in `worlds_params` dict should have the following members:

            - name: world name [Defaults to random]
            - log_file_name: file name to store the world log [Defaults to random]
            - __dir_name: directory to store the world stats [Defaults to random]
            - __world_generator: full name of the world generator function (including its module) [Required]
            - __score_calculator: full name of the score calculator function [Required]
            - __tournament_name: name of the tournament [Defaults to random]
            - others: values of all other keys are passed to the world generator as kwargs
    """
    params = []
    if len(worlds_params) < 1:
        return (
            _hash(worlds_params),
            # WorldRunResults(world_names=[""], log_file_names=[""]),
            None,
            None,
        )
    world_generator, score_calculator = None, None
    for world_params in worlds_params:
        world_generator = world_params.get("__world_generator", None)
        score_calculator = world_params.get("__score_calculator", None)
        tournament_name = world_params.get("__tournament_name", unique_name(base=""))
        assert world_generator and score_calculator, (
            f"Cannot run without specifying both a world generator and a score "
            f"calculator"
        )
        world_generator = import_by_name(world_generator)
        score_calculator = import_by_name(score_calculator)
        default_name = unique_name(base="")
        world_params["name"] = world_params.get("name", default_name)
        world_name = world_params["name"]
        default_dir = (
            Path(f"~") / "negmas" / "tournaments" / tournament_name / world_name
        ).absolute()
        world_params["log_file_name"] = world_params.get(
            "log_file_name", str("log.txt")
        )
        world_params["__dir_name"] = world_params.get("__dir_name", str(default_dir))
        # delete the parameters not used by _run_worlds
        for k in ("__world_generator", "__tournament_name", "__score_calculator"):
            if k in world_params.keys():
                world_params.pop(k, None)
        params.append(world_params)
    return _run_worlds(
        worlds_params=params,
        world_generator=world_generator,
        score_calculator=score_calculator,
        dry_run=dry_run,
        save_world_stats=save_world_stats,
    )


def _run_worlds(
    worlds_params: List[Dict[str, Any]],
    world_generator: WorldGenerator,
    score_calculator: Callable[[List[World], Dict[str, Any], bool], WorldRunResults],
    world_progress_callback: Callable[[Optional[World]], None] = None,
    dry_run: bool = False,
    save_world_stats: bool = True,
    override_ran_worlds: bool = False,
    save_progress_every: int = 1,
) -> Tuple[str, Optional[WorldRunResults], WorldSetRunStats]:
    """Runs a set of worlds (generated from a world generator) and returns stats

    Args:
        worlds_params: A list of World info dicts. See remarks for its parameters
        world_generator: World generator function.
        score_calculator: Score calculator function.
        world_progress_callback: world progress callback
        dry_run: If true, the world is not run. Its config is saved instead.
        save_world_stats: If true, saves individual world stats
        override_ran_worlds: If true, run the worlds even if they are already ran before.
        save_progress_every: If true, progress will be saved every this number of steps.

    Returns:
        A tuple with the following components in order:

            - The run ID for this world-set
            - The results (scores) for this world set (will be None in case of exception)
            - The stats for this world set (will not be None even in case of exception)


    Remarks:

        - Each `worlds_params` dict should have the following members:

            - name: world name
            - log_file_name: file name to store the world log
            - __dir_name: directory to store the world stats
            - others: values of all other keys are passed to the world generator as kwargs

        - The system knows that a world is already ran using `is_already_run` which checks that
          the folder to store the results of the world exists and contains a file called stats.json

    """
    worlds, dir_names = [], []
    scoring_context = {}
    run_id = _hash(worlds_params)
    video_savers, video_saver_params_list, save_videos = [], [], []
    scores: Optional[WorldRunResults] = None

    for world_params in worlds_params:
        world_params = world_params.copy()
        dir_name = world_params["__dir_name"]
        dir_names.append(dir_name)
        world_params.pop("__dir_name", None)
        video_savers.append(world_params.get("__video_saver", None))
        video_saver_params_list.append(world_params.get("__video_saver_params", dict()))
        save_videos.append(world_params.get("__save_video", False))
        scoring_context.update(world_params.get("scoring_context", {}))
        world_params.pop("__video_saver", None)
        world_params.pop("__video_saver_params", None)
        world_params.pop("__save_video", None)
        world = world_generator(**world_params)
        worlds.append(world)
        if dry_run:
            world.save_config(dir_name)
            continue
    try:
        for (
            world,
            world_params_,
            dir_name,
            save_video,
            video_saver,
            video_saver_params,
        ) in zip(
            worlds,
            world_params,
            dir_names,
            save_videos,
            video_savers,
            video_saver_params_list,
        ):
            _start_time = time.perf_counter()
            for _ in range(world.n_steps):
                if not world.step():
                    break
                if _ % save_progress_every == 0:
                    save_stats(world, world.log_folder, params=world_params_)
                    # TODO reorganize the code so that the worlds are run in parallel when there are multiple of them
                    if not dry_run:
                        scores_ = to_flat_dict(
                            score_calculator(worlds, scoring_context, False)
                        )
                        scores_["n_steps"] = world.n_steps
                        scores_["step"] = world.current_step
                        scores_["relative_time"] = world.relative_time
                        scores_["time_limit"] = world.time_limit
                        scores_["time"] = world.time
                        dump(
                            to_flat_dict(scores_),
                            Path(world.log_folder) / "_current_scores.json",
                            sort_keys=True,
                        )
                    if world_progress_callback:
                        world_progress_callback(world)
                if world.time >= world.time_limit:
                    break
            if save_world_stats:
                save_stats(world=world, log_dir=dir_name)
            if save_video:
                if video_saver is None:
                    video_saver = World.save_gif
                if video_saver_params is None:
                    video_saver_params = {}
                video_saver(world, **video_saver_params)

        scores = score_calculator(worlds, scoring_context, dry_run)
        agent_exceptions: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
        for w in worlds:
            for aid, v in w.agent_exceptions.items():
                if v:
                    agent_exceptions[w.agents[aid].type_name] += v
        negotiator_exceptions: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
        for w in worlds:
            for aid, v in w.negotiator_exceptions.items():
                if v:
                    negotiator_exceptions[w.agents[aid].type_name] += v

        simulation_exceptions: Dict[int, List[str]] = defaultdict(list)
        for w in worlds:
            for k, l in w.simulation_exceptions.items():
                if l:
                    simulation_exceptions[k] += l
        mechanism_exceptions: Dict[int, List[str]] = defaultdict(list)
        for w in worlds:
            for k, l in w.mechanism_exceptions.items():
                if l:
                    mechanism_exceptions[k] += l
        contract_exceptions: Dict[int, List[str]] = defaultdict(list)
        for w in worlds:
            for k, l in w.contract_exceptions.items():
                if l:
                    contract_exceptions[k] += l

        agent_times: Dict[str, float] = defaultdict(float)
        for w in worlds:
            for aid, _ in w.times.items():
                if _:
                    agent_times[w.agents[aid].type_name] += _

        world_stats = WorldSetRunStats(
            name=";".join(_.name for _ in worlds),
            planned_n_steps=sum(_.n_steps for _ in worlds),
            executed_n_steps=sum(_.current_step for _ in worlds),
            execution_time=sum(_.frozen_time for _ in worlds),
            agent_exceptions=agent_exceptions,
            negotiator_exceptions=negotiator_exceptions,
            simulation_exceptions=simulation_exceptions,
            contract_exceptions=contract_exceptions,
            mechanism_exceptions=mechanism_exceptions,
            agent_times=agent_times,
            other_exceptions=[],
        )
    except Exception as e:
        scores = None
        print(traceback.format_exc())
        print(e)
        world_stats = WorldSetRunStats(
            name=";".join(_.name for _ in worlds),
            planned_n_steps=sum(_.n_steps for _ in worlds),
            executed_n_steps=sum(_.current_step for _ in worlds),
            execution_time=sum(_.frozen_time for _ in worlds),
            other_exceptions=[exception2str()],
        )
    return run_id, scores, world_stats


def process_world_run(
    run_id: str,
    results: Optional[WorldRunResults],
    tournament_name: str,
    world_stats: Optional[WorldSetRunStats],
    # save_world_stats: bool = True,
) -> List[Dict[str, Any]]:
    """
    Generates a data-frame with the results of this world run

    Args:
        run_id: The ID of this run (should be unique per tournament)
        results: Results of the world run
        tournament_name: tournament name

    Returns:

        A pandas DataFrame with agent_name, agent_type, score, log_file, world, and stats_folder columns

    """
    if results is None:
        return []
    log_files, world_names_ = results.log_file_names, results.world_names
    for world_name_, log_file in zip(world_names_, log_files):
        if (
            # save_world_stats
            log_file is not None
            and pathlib.Path(log_file).exists()
        ):
            with open(log_file, "a") as f:
                f.write(
                    f"\nPART of TOURNAMENT {tournament_name}. This world run completed successfully\n"
                )
    scores = []
    log_files = [_ if _ is not None else "" for _ in log_files]
    stat_folders = ";".join(
        str(pathlib.Path(log_file_name).name) if log_file_name else ""
        for log_file_name in log_files
    )
    base_folder = str(pathlib.Path(log_files[0]).parent) if log_files[0] else ""
    total_time = sum(world_stats.agent_times.values())
    if total_time < 1e-6:
        total_time = 1
    for name_, type_, score in zip(results.names, results.types, results.scores):
        d = {
            "agent_name": name_,
            "agent_type": type_,
            "score": score,
            "log_file": ";".join(log_files),
            "world": ";".join(world_names_),
            "stats_folders": stat_folders,
            "base_stats_folder": base_folder,
            "run_id": run_id,
            "time": world_stats.agent_times.get(type_, 0) / total_time,
            "exceptions": world_stats.agent_exceptions.get(type_, []),
            "negotiator_exceptions": world_stats.negotiator_exceptions.get(type_, []),
        }
        scores.append(d)
    return scores


def _run_dask(
    scheduler_ip,
    scheduler_port,
    verbose,
    world_infos,
    world_generator,
    tournament_progress_callback,
    n_worlds,
    name,
    score_calculator,
    dry_run,
    save_world_stats,
    scores_file,
    world_stats_file,
    run_ids,
    print_exceptions,
    override_already_ran=False,
) -> None:
    """Runs the tournament on dask"""

    import distributed

    if scheduler_ip is None and scheduler_port is None:
        address = None
    else:
        if scheduler_ip is None:
            scheduler_ip = "127.0.0.1"
        if scheduler_port is None:
            scheduler_port = "8786"
        address = f"{scheduler_ip}:{scheduler_port}"
    if verbose:
        print(f"Will use DASK on {address}")
    client = distributed.Client(address=address, set_as_default=True)
    future_results = []
    for world_params in world_infos:
        run_id = _hash(world_params)
        if run_id in run_ids:
            continue
        future_results.append(
            client.submit(
                _run_worlds,
                world_params,
                world_generator,
                score_calculator,
                None,
                dry_run,
                save_world_stats,
                override_already_ran,
            )
        )
    print(f"Submitted all processes to DASK ({len(future_results)})")
    _strt = time.perf_counter()
    for i, future in enumerate(
        distributed.as_completed(future_results, with_results=True, raise_errors=False)
    ):
        try:
            run_id, score_, world_stats_ = future.result()
            if tournament_progress_callback is not None:
                tournament_progress_callback(score_, i, n_worlds)
            add_records(
                scores_file,
                process_world_run(
                    run_id,
                    score_,
                    tournament_name=name,
                    world_stats=world_stats_,
                    # save_world_stats=save_world_stats,
                ),
            )
            add_records(world_stats_file, [vars(world_stats_)])
            if verbose:
                _duration = time.perf_counter() - _strt
                print(
                    f"{i + 1:003} of {n_worlds:003} [{100 * (i + 1) / n_worlds:0.3}%] completed in "
                    f"{humanize_time(_duration)} [ETA {humanize_time(_duration * n_worlds / (i + 1))}]"
                )
        except Exception as e:
            if tournament_progress_callback is not None:
                tournament_progress_callback(None, i, n_worlds)
            if print_exceptions:
                print(traceback.format_exc())
                print(e)
    client.shutdown()


def _divide_into_sets(competitors, n_competitors_per_world):
    if len(competitors) % n_competitors_per_world == 0:
        return (
            np.array(competitors)
            .reshape(
                (len(competitors) // n_competitors_per_world, n_competitors_per_world)
            )
            .tolist()
        )
    n_div = (len(competitors) // n_competitors_per_world) * n_competitors_per_world
    divisable = competitors[:n_div]
    competitor_sets = (
        np.array(divisable)
        .reshape((len(divisable) // n_competitors_per_world, n_competitors_per_world))
        .tolist()
    )
    competitor_sets.append(
        competitors[n_div:] + ([None] * (n_competitors_per_world - n_div))
    )
    return competitor_sets


def tournament(
    competitors: Sequence[Union[str, Type[Agent]]],
    config_generator: ConfigGenerator,
    config_assigner: ConfigAssigner,
    world_generator: WorldGenerator,
    score_calculator: Callable[[List[World], Dict[str, Any], bool], WorldRunResults],
    competitor_params: Optional[Sequence[Dict[str, Any]]] = None,
    n_competitors_per_world: Optional[int] = None,
    round_robin: bool = False,
    stage_winners_fraction: float = 0.0,
    agent_names_reveal_type=False,
    n_agents_per_competitor=1,
    n_configs: int = 10,
    max_worlds_per_config: int = 100,
    n_runs_per_world: int = 5,
    max_n_configs: int = None,
    n_runs_per_config: int = None,
    tournament_path: str = None,
    total_timeout: Optional[int] = None,
    parallelism="parallel",
    scheduler_ip: Optional[str] = None,
    scheduler_port: Optional[str] = None,
    tournament_progress_callback: Callable[
        [Optional[WorldRunResults], int, int], None
    ] = None,
    world_progress_callback: Callable[[Optional[World]], None] = None,
    non_competitors: Optional[Tuple[Union[str, Any]]] = None,
    non_competitor_params: Optional[Tuple[Dict[str, Any]]] = None,
    dynamic_non_competitors: Optional[Tuple[Union[str, Any]]] = None,
    dynamic_non_competitor_params: Optional[Tuple[Dict[str, Any]]] = None,
    exclude_competitors_from_reassignment: bool = True,
    name: str = None,
    verbose: bool = False,
    configs_only: bool = False,
    compact: bool = False,
    print_exceptions: bool = True,
    metric="median",
    save_video_fraction: float = 0.001,
    forced_logs_fraction: float = 0.001,
    video_params=None,
    video_saver=None,
    **kwargs,
) -> Union[TournamentResults, PathLike]:
    """
    Runs a tournament

    Args:

        name: Tournament name
        config_generator: Used to generate unique configs that will be used to evaluate competitors
        config_assigner: Used to generate assignments of competitors to the configs created by the `config_generator`
        world_generator: A functions to generate worlds for the tournament that follows the assignments made by the
                         `config_assigner`
        score_calculator: A function for calculating the score of all agents in a world *After it finishes running*.
                          The second parameter is a dict describing any scoring context that may have been added by the
                          world config generator or assigneer.
                          The third parameter is a boolean specifying whether this is a dry_run. For dry runs, scores
                          are not expected but names and types should exist in the returned `WorldRunResults`.
        competitors: A list of class names for the competitors
        competitor_params: A list of competitor parameters (used to initialize the competitors).
        n_competitors_per_world: The number of competitors allowed in every world. It must be >= 1 and
                                 <= len(competitors) or None.

                                 - If None or len(competitors), then all competitors will exist in every world.
                                 - If 1, then each world will have one competitor

        round_robin: Only effective if 1 < n_competitors_per_world < len(competitors). if True, all
                                     combinations will be tried otherwise n_competitors_per_world must divide
                                     len(competitors) and every competitor appears only in one set.
        stage_winners_fraction: in [0, 1).  Fraction of agents to to go to the next stage at every stage. If zero, and
                                            round_robin, it becomes a single stage competition.
        agent_names_reveal_type: If true then the type of an agent should be readable in its name (most likely at its
                                 beginning).
        n_configs: The number of different world configs (up to competitor assignment) to be generated.
        max_worlds_per_config: The maximum number of worlds to run per config. If None, then all possible assignments
                             of competitors within each config will be tried (all permutations).
        n_runs_per_world: Number of runs per world. All of these world runs will have identical competitor assignment
                          and identical world configuration.
        n_agents_per_competitor: The number of agents of each competing type to be instantiated in the world.
        max_n_configs: [Depricated] The number of configs to use (it is replaced by separately setting `n_config`
                       and `max_worlds_per_config` )
        n_runs_per_config: [Depricated] The number of runs (simulation) for every config. It is replaced by
                           `n_runs_per_world`
        total_timeout: Total timeout for the complete process
        tournament_path: Path at which to store all results. A new folder with the name of the tournament will be
                         created at this path. A scores.csv file will keep the scores and logs folder will keep detailed
                         logs
        parallelism: Type of parallelism. Can be 'serial' for serial, 'parallel' for parallel and 'distributed' for
                     distributed! For parallel, you can add the fraction of CPUs to use after a colon (e.g. parallel:0.5
                     to use half of the CPU in the machine). By defaults parallel uses all CPUs in the machine
        scheduler_port: Port of the dask scheduler if parallelism is dask, dist, or distributed
        scheduler_ip:   IP Address of the dask scheduler if parallelism is dask, dist, or distributed
        world_progress_callback: A function to be called after every step of every world run (only allowed for serial
                                 and parallel evaluation and should be used with cautious).
        tournament_progress_callback: A function to be called with `WorldRunResults` after each world finished
                                      processing
        non_competitors: A list of agent types that will not be competing but will still exist in the world.
        non_competitor_params: paramters of non competitor agents
        dynamic_non_competitors: A list of non-competing agents that are assigned to the simulation dynamically during
                                 the creation of the final assignment instead when the configuration is created
        dynamic_non_competitor_params: paramters of dynamic non competitor agents
        exclude_competitors_from_reassignment: If true, competitors are excluded from the dyanamic non-competitors
        verbose: Verbosity
        configs_only: If true, a config file for each
        compact: If true, compact logs will be created and effort will be made to reduce the memory footprint
        print_exceptions: If true, print all exceptions to screen
        metric: The metric to use for evaluation
        save_video_fraction: The fraction of simulations for which to save videos
        forced_logs_fraction: The fraction of simulations for which to always save logs. Notice that this has no
                              effect except if no logs were to be saved otherwise (i.e. `no_logs` is passed as True)
        video_params: The parameters to pass to the video saving function
        video_saver: The parameters to pass to the video saving function after the world
        kwargs: Arguments to pass to the `config_generator` function

    Returns:
        `TournamentResults` The results of the tournament or a `PathLike` giving the location where configs were saved

    """
    competitors = list(competitors)
    if name is None:
        name = unique_name("", add_time=True, rand_digits=3)

    if n_competitors_per_world is None:
        n_competitors_per_world = len(competitors)

    if not round_robin and not (1 < n_competitors_per_world <= len(competitors)):
        raise ValueError(
            f"You have {len(competitors)} and you will use {n_competitors_per_world} per world but the "
            f"later does not divide the former. You have to set all_competitor_combinations to True"
        )

    if stage_winners_fraction < 0:
        stage_winners_fraction = 0
    # if not round_robin and stage_winners_fraction == 0:
    #     raise ValueError(
    #         f"elimination tournaments (i.e. ones that are not round_robin), cannot have zero stage winner"
    #         f"fraction"
    #     )

    competitor_indx = dict(
        zip(
            [
                get_class(c)._type_name() if not isinstance(c, str) else c
                for c in competitors
            ],
            range(len(competitors)),
        )
    )

    def _run_eval(competitors_, stage_name):
        final_tournament_path = create_tournament(
            competitors=competitors_,
            config_generator=config_generator,
            config_assigner=config_assigner,
            world_generator=world_generator,
            score_calculator=score_calculator,
            competitor_params=competitor_params,
            n_competitors_per_world=n_competitors_per_world,
            round_robin=round_robin,
            agent_names_reveal_type=agent_names_reveal_type,
            n_agents_per_competitor=n_agents_per_competitor,
            n_configs=n_configs,
            max_worlds_per_config=max_worlds_per_config,
            n_runs_per_world=n_runs_per_world,
            max_n_configs=max_n_configs,
            n_runs_per_config=n_runs_per_config,
            base_tournament_path=tournament_path,
            total_timeout=total_timeout,
            parallelism=parallelism,
            scheduler_ip=scheduler_ip,
            scheduler_port=scheduler_port,
            non_competitors=non_competitors,
            non_competitor_params=non_competitor_params,
            dynamic_non_competitors=dynamic_non_competitors,
            dynamic_non_competitor_params=dynamic_non_competitor_params,
            exclude_competitors_from_reassignment=exclude_competitors_from_reassignment,
            name=stage_name,
            verbose=verbose,
            compact=compact,
            save_video_fraction=save_video_fraction,
            forced_logs_fraction=forced_logs_fraction,
            video_params=video_params,
            video_saver=video_saver,
            **kwargs,
        )
        if configs_only:
            return pathlib.Path(final_tournament_path) / "configs"
        run_tournament(
            tournament_path=final_tournament_path,
            world_generator=world_generator,
            score_calculator=score_calculator,
            total_timeout=total_timeout,
            parallelism=parallelism,
            scheduler_ip=scheduler_ip,
            scheduler_port=scheduler_port,
            tournament_progress_callback=tournament_progress_callback,
            world_progress_callback=world_progress_callback,
            verbose=verbose,
            compact=compact,
            print_exceptions=print_exceptions,
        )
        return evaluate_tournament(
            tournament_path=final_tournament_path,
            verbose=verbose,
            recursive=round_robin,
            metric=metric,
        )

    def _keep_n(competitors_, results_, n):
        tscores = results_.total_scores.sort_values(by=["score"], ascending=False)
        sorted_indices = np.array(
            [competitor_indx[_] for _ in tscores["agent_type"].values]
        )[:n]
        return np.array(competitors_)[sorted_indices].tolist()

    stage = 1
    while len(competitors) > 1:
        if verbose:
            print(
                f"Stage {stage} started between ({len(competitors)} competitors): {competitors} "
            )
        stage_name = name + f"-stage-{stage:04}"
        if round_robin:
            n_winners_per_stage = min(
                max(1, int(stage_winners_fraction * len(competitors))),
                len(competitors) - 1,
            )
            results = _run_eval(competitors, stage_name)
            if n_winners_per_stage == 1:
                return results
            competitors = _keep_n(competitors, results, n_winners_per_stage)
        else:
            random.shuffle(competitors)
            competitor_sets = _divide_into_sets(competitors, n_competitors_per_world)

            next_stage_competitors = []
            results = None
            for c in competitor_sets:
                match_name_ = stage_name + _hash(c)
                n_winners_per_match = min(
                    max(1, int(stage_winners_fraction * n_competitors_per_world)),
                    len(c) - 1,
                )
                results = _run_eval(c, match_name_)
                winners_ = _keep_n(competitors, results, n_winners_per_match)
                next_stage_competitors += winners_
            competitors = next_stage_competitors
            n_competitors_per_world = min(n_competitors_per_world, len(competitors))
            if len(competitors) == 1:
                return results
        stage += 1


def _path(path: Union[str, PathLike]) -> Path:
    """Creates an absolute path from given path which can be a string"""
    if isinstance(path, str):
        if path.startswith("~"):
            path = Path.home() / ("/".join(path.split("/")[1:]))
    return pathlib.Path(path).absolute()


def is_already_run(world_params) -> bool:
    return False
    # dir_name = pathlib.Path(world_params["__dir_name"])
    # if not dir_name.exists():
    #     return False
    # if (dir_name / "stats.json").exists():
    #     return True
    # return False


def run_tournament(
    tournament_path: Union[str, PathLike],
    world_generator: WorldGenerator = None,
    score_calculator: Callable[
        [List[World], Dict[str, Any], bool], WorldRunResults
    ] = None,
    total_timeout: Optional[int] = None,
    parallelism="parallel",
    scheduler_ip: Optional[str] = None,
    scheduler_port: Optional[str] = None,
    tournament_progress_callback: Callable[
        [Optional[WorldRunResults], int, int], None
    ] = None,
    world_progress_callback: Callable[[Optional[World]], None] = None,
    verbose: bool = False,
    compact: bool = None,
    print_exceptions: bool = True,
    override_ran_worlds: bool = False,
) -> None:
    """
    Runs a tournament

    Args:
        tournament_path: Path at which configs of this tournament are stored
        world_generator: A functions to generate worlds for the tournament that follows the assignments made by the
                         `config_assigner`
        score_calculator: A function for calculating the score of all agents in a world *After it finishes running*.
                          The second parameter is a dict describing any scoring context that may have been added by the
                          world config generator or assigner.
                          The third parameter is a boolean specifying whether this is a dry_run. For dry runs, scores
                          are not expected but names and types should exist in the returned `WorldRunResults`.
        total_timeout: Total timeout for the complete process
        parallelism: Type of parallelism. Can be 'serial' for serial, 'parallel' for parallel and 'distributed' for
                     distributed! For parallel, you can add the fraction of CPUs to use after a colon (e.g. parallel:0.5
                     to use half of the CPU in the machine). By defaults parallel uses all CPUs in the machine
        scheduler_port: Port of the dask scheduler if parallelism is dask, dist, or distributed
        scheduler_ip:   IP Address of the dask scheduler if parallelism is dask, dist, or distributed
        world_progress_callback: A function to be called after every step of every world run (only allowed for serial
                                 and parallel evaluation and should be used with cautious).
        tournament_progress_callback: A function to be called with `WorldRunResults` after each world finished
                                      processing
        verbose: Verbosity
        compact: If true, compact logs will be created and effort will be made to reduce the memory footprint
        print_exceptions: If true, exceptions encountered during world simulation will be printed to stdout
        override_ran_worlds: If true worlds that are already ran will be ran again

    """
    tournament_path = _path(tournament_path)
    params = load(tournament_path / "params")
    name = params.get("name", tournament_path.name)
    if world_generator is None:
        world_generator = import_by_name(params.get("world_generator_name", None))
    if score_calculator is None:
        score_calculator = import_by_name(params.get("score_calculator_name", None))
    if total_timeout is None:
        total_timeout = params.get("total_timeout", None)
    if parallelism is None:
        parallelism = params.get("parallelism", "parallel")
    if scheduler_port is None:
        scheduler_port = params.get("scheduler_port", None)
    if scheduler_ip is None:
        scheduler_ip = params.get("scheduler_ip", None)
    if compact is None:
        compact = params.get("compact", False)

    assigned = load(tournament_path / "assigned_configs.pickle")
    random.shuffle(assigned)
    n_world_configs = len(assigned)

    if verbose:
        print(
            f"Will run {n_world_configs}  total world simulations ({parallelism})",
            flush=True,
        )

    scores_file = tournament_path / "scores.csv"
    world_stats_file = tournament_path / "world_stats.csv"
    run_ids = set()
    if scores_file.exists():
        tmp_ = pd.read_csv(scores_file)
        if "run_id" in tmp_.columns:
            run_ids = set(tmp_["run_id"].values)

    scores_file = str(scores_file)
    dask_options = ("dist", "distributed", "dask", "d")
    multiprocessing_options = ("local", "parallel", "par", "p")
    serial_options = ("none", "serial", "s")
    if parallelism is None:
        parallelism = "serial"
    assert (
        total_timeout is None or parallelism not in dask_options
    ), f"Cannot use {parallelism} with a total-timeout"
    assert world_progress_callback is None or parallelism not in dask_options, (
        f"Cannot use {parallelism} with a " f"world callback"
    )

    if parallelism in serial_options:
        strt = time.perf_counter()
        for i, worlds_params in enumerate(assigned):
            if total_timeout is not None and time.perf_counter() - strt > total_timeout:
                break
            run_id = _hash(worlds_params)
            if run_id in run_ids:
                if verbose:
                    _duration = time.perf_counter() - strt
                    print(
                        f"{i + 1:003} of {n_world_configs:003} [{(i + 1) / n_world_configs:.02%}] "
                        f'{"Skipped"} '
                        f"in {humanize_time(_duration)}"
                        f" [ETA {humanize_time(_duration * n_world_configs / (i + 1))}]"
                    )
                continue
            try:
                run_id, score_, world_stats_ = _run_worlds(
                    worlds_params=worlds_params,
                    world_generator=world_generator,
                    world_progress_callback=world_progress_callback,
                    score_calculator=score_calculator,
                    dry_run=False,
                    save_world_stats=True,
                    override_ran_worlds=override_ran_worlds,
                )
                if tournament_progress_callback is not None:
                    tournament_progress_callback(score_, i, n_world_configs)
                add_records(
                    scores_file,
                    process_world_run(
                        run_id, score_, tournament_name=name, world_stats=world_stats_,
                    ),
                )
                add_records(world_stats_file, [vars(world_stats_)])
                if verbose:
                    _duration = time.perf_counter() - strt
                    print(
                        f"{i + 1:003} of {n_world_configs:003} [{(i + 1) / n_world_configs:.02%}] "
                        f'{"completed"} '
                        f"in {humanize_time(_duration)}"
                        f" [ETA {humanize_time(_duration * n_world_configs / (i + 1))}]"
                    )
            except Exception as e:
                if tournament_progress_callback is not None:
                    tournament_progress_callback(None, i, n_world_configs)
                if print_exceptions:
                    print(traceback.format_exc())
                    print(e)
    elif any(parallelism.startswith(_) for _ in multiprocessing_options):
        fraction = None
        parallelism = parallelism.split(":")
        if len(parallelism) != 1:
            fraction = float(parallelism[-1])
        parallelism = parallelism[0]
        max_workers = (
            fraction if fraction is None else max(1, int(fraction * cpu_count()))
        )
        executor = futures.ProcessPoolExecutor(max_workers=max_workers)
        future_results = []
        for i, worlds_params in enumerate(assigned):
            run_id = _hash(worlds_params)
            if run_id in run_ids:
                continue
            future_results.append(
                executor.submit(
                    _run_worlds,
                    worlds_params,
                    world_generator,
                    score_calculator,
                    world_progress_callback,
                    False,
                    True,
                    override_ran_worlds,
                )
            )
        result_received = set()
        if verbose:
            print(
                f"Submitted all processes ({len(future_results)} of {len(assigned)})",
                end="",
            )
            if len(assigned) > 0:
                print(f"{len(future_results)/len(assigned):5.2%}")
            else:
                print("")
        n_world_configs = len(future_results)
        _strt = time.perf_counter()
        for i, future in enumerate(
            futures.as_completed(future_results, timeout=total_timeout)
        ):
            if total_timeout is not None and time.perf_counter() - strt > total_timeout:
                break
            try:
                run_id, score_, world_stats_ = future.result()
                if tournament_progress_callback is not None:
                    tournament_progress_callback(score_, i, n_world_configs)
                add_records(
                    scores_file,
                    process_world_run(
                        run_id,
                        score_,
                        tournament_name=name,
                        world_stats=world_stats_,
                        # save_world_stats=not compact,
                    ),
                )
                add_records(world_stats_file, [vars(world_stats_)]),
                result_received.add(run_id)
                if verbose:
                    _duration = time.perf_counter() - _strt
                    print(
                        f"{i + 1:003} of {n_world_configs:003} [{100 * (i + 1) / n_world_configs:0.3}%] "
                        f'{"completed"} in '
                        f"{humanize_time(_duration)}"
                        f" [ETA {humanize_time(_duration * n_world_configs / (i + 1))}]"
                    )
            except futures.TimeoutError:
                if tournament_progress_callback is not None:
                    tournament_progress_callback(None, i, n_world_configs)
                if verbose:
                    print("Tournament timed-out")
                break
            except futures.process.BrokenProcessPool as e:
                if tournament_progress_callback is not None:
                    tournament_progress_callback(None, i, n_world_configs)
                if print_exceptions:
                    print(e)
            except Exception as e:
                if tournament_progress_callback is not None:
                    tournament_progress_callback(None, i, n_world_configs)
                if print_exceptions:
                    print(traceback.format_exc())
                    print(e)
    elif parallelism in dask_options:
        _run_dask(
            scheduler_ip,
            scheduler_port,
            verbose,
            assigned,
            world_generator,
            tournament_progress_callback,
            n_world_configs,
            name,
            score_calculator,
            False,
            True,
            scores_file,
            world_stats_file,
            run_ids,
            print_exceptions,
            override_ran_worlds,
        )
    if verbose:
        print(f"Tournament completed successfully")


def create_tournament(
    competitors: Sequence[Union[str, Type[Agent]]],
    config_generator: ConfigGenerator,
    config_assigner: ConfigAssigner,
    world_generator: WorldGenerator,
    score_calculator: Callable[[List[World], Dict[str, Any], bool], WorldRunResults],
    competitor_params: Optional[Sequence[Dict[str, Any]]] = None,
    n_competitors_per_world: Optional[int] = None,
    round_robin: bool = True,
    agent_names_reveal_type=False,
    n_agents_per_competitor=1,
    n_configs: int = 10,
    max_worlds_per_config: int = 100,
    n_runs_per_world: int = 5,
    max_n_configs: int = None,
    n_runs_per_config: int = None,
    base_tournament_path: str = None,
    total_timeout: Optional[int] = None,
    parallelism="parallel",
    scheduler_ip: Optional[str] = None,
    scheduler_port: Optional[str] = None,
    non_competitors: Optional[Tuple[Union[str, Any]]] = None,
    non_competitor_params: Optional[Tuple[Dict[str, Any]]] = None,
    dynamic_non_competitors: Optional[Tuple[Union[str, Any]]] = None,
    dynamic_non_competitor_params: Optional[Tuple[Dict[str, Any]]] = None,
    exclude_competitors_from_reassignment: bool = True,
    name: str = None,
    verbose: bool = False,
    compact: bool = False,
    save_video_fraction: float = 0.0,
    forced_logs_fraction: float = 0.0,
    video_params=None,
    video_saver=None,
    **kwargs,
) -> PathLike:
    """
    Creates a tournament

    Args:

        name: Tournament name
        config_generator: Used to generate unique configs that will be used to evaluate competitors
        config_assigner: Used to generate assignments of competitors to the configs created by the `config_generator`
        world_generator: A functions to generate worlds for the tournament that follows the assignments made by the
                         `config_assigner`
        score_calculator: A function for calculating the score of all agents in a world *After it finishes running*.
                          The second parameter is a dict describing any scoring context that may have been added by the
                          world config generator or assigneer.
                          The third parameter is a boolean specifying whether this is a dry_run. For dry runs, scores
                          are not expected but names and types should exist in the returned `WorldRunResults`.
        competitors: A list of class names for the competitors
        competitor_params: A list of competitor parameters (used to initialize the competitors).
        n_competitors_per_world: The number of competitors allowed in every world. It must be >= 1 and
                                 <= len(competitors) or None.

                                 - If None or len(competitors), then all competitors will exist in every world.
                                 - If 1, then each world will have one competitor

        round_robin: Only effective if 1 < n_competitors_per_world < len(competitors). if True, all
                                     combinations will be tried otherwise n_competitors_per_world must divide
                                     len(competitors) and every competitor appears only in one set.
        agent_names_reveal_type: If true then the type of an agent should be readable in its name (most likely at its
                                 beginning).
        n_configs: The number of different world configs (up to competitor assignment) to be generated.
        max_worlds_per_config: The maximum number of worlds to run per config. If None, then all possible assignments
                             of competitors within each config will be tried (all permutations).
        n_runs_per_world: Number of runs per world. All of these world runs will have identical competitor assignment
                          and identical world configuration.
        n_agents_per_competitor: The number of agents of each competing type to be instantiated in the world.
        max_n_configs: [Depricated] The number of configs to use (it is replaced by separately setting `n_config`
                       and `max_worlds_per_config` )
        n_runs_per_config: [Depricated] The number of runs (simulation) for every config. It is replaced by
                           `n_runs_per_world`
        total_timeout: Total timeout for the complete process
        base_tournament_path: Path at which to store all results. A new folder with the name of the tournament will be
                         created at this path. A scores.csv file will keep the scores and logs folder will keep detailed
                         logs
        parallelism: Type of parallelism. Can be 'serial' for serial, 'parallel' for parallel and 'distributed' for
                     distributed! For parallel, you can add the fraction of CPUs to use after a colon (e.g. parallel:0.5
                     to use half of the CPU in the machine). By defaults parallel uses all CPUs in the machine
        scheduler_port: Port of the dask scheduler if parallelism is dask, dist, or distributed
        scheduler_ip:   IP Address of the dask scheduler if parallelism is dask, dist, or distributed
        non_competitors: A list of agent types that will not be competing but will still exist in the world.
        non_competitor_params: paramters of non competitor agents
        dynamic_non_competitors: A list of non-competing agents that are assigned to the simulation dynamically during
                                 the creation of the final assignment instead when the configuration is created
        dynamic_non_competitor_params: paramters of dynamic non competitor agents
        exclude_competitors_from_reassignment: If true, copmetitors are not included in the reassignment even
                                                          if they exist in `dynamic_non_competitors`
        verbose: Verbosity
        compact: If true, compact logs will be created and effort will be made to reduce the memory footprint
        save_video_fraction: The fraction of simulations for which to save videos
        forced_logs_fraction: The fraction of simulations for which to always save logs. Notice that this has no
                              effect except if no logs were to be saved otherwise (i.e. `no_logs` is passed as True)
        video_params: The parameters to pass to the video saving function
        video_saver: The parameters to pass to the video saving function after the world
        kwargs: Arguments to pass to the `config_generator` function

    Returns:
        The path at which tournament configs are stored

    """
    if max_n_configs is not None or n_runs_per_config is not None:
        n_runs_per_world = (
            n_runs_per_config if n_runs_per_config is not None else n_runs_per_world
        )
        n_configs = max(1, int(math.log2(max_n_configs)))
        max_worlds_per_config = int(0.5 + max_n_configs / n_configs)

        warnings.warn(
            f"max_n_configs and n_runs_per_config are deprecated and will be removed in future versions. "
            f"Use n_configs, max_worlds_per_config n_runs_per_world instead."
            f"\nWill use the following settings: n_configs ({n_configs})"
            f", max_worlds_per_config ({max_worlds_per_config})"
            f", and n_runs_per_world ({n_runs_per_world})."
        )

    if n_runs_per_world is None or n_configs is None:
        raise ValueError(
            f"Values for n_configs ({n_configs})"
            f", and n_runs_per_world ({n_runs_per_world}) must be given or possible to calculate "
            f"from max_n_configs ({max_n_configs}) and n_runs_per_config ({n_runs_per_config})"
        )

    if name is None:
        name = unique_name("", add_time=True, rand_digits=3)
    competitors = list(competitors)
    if n_competitors_per_world is None:
        n_competitors_per_world = len(competitors)

    # if not round_robin and not (len(competitors) >= n_competitors_per_world > 0):
    #     raise ValueError(
    #         f"You have {len(competitors)} and you will use {n_competitors_per_world} per world but the "
    #         f"later does not divide the former. You have to set all_competitor_combinations to True"
    #     )
    if base_tournament_path is None:
        base_tournament_path = str(pathlib.Path.home() / "negmas" / "tournaments")

    original_tournament_path = base_tournament_path
    base_tournament_path = _path(base_tournament_path)
    tournament_path = (pathlib.Path(base_tournament_path) / name).absolute()
    if tournament_path.exists() and not tournament_path.is_dir():
        raise ValueError(
            f"tournament path {str(tournament_path)} is a file. Cannot continue"
        )
    if tournament_path.exists():
        raise ValueError(
            f"tournament path {str(tournament_path)} exists. You cannot create two tournaments in the same place"
        )
    tournament_path.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"Results of Tournament {name} will be saved to {str(tournament_path)}")
    if competitor_params is None:
        competitor_params = [dict() for _ in range(len(competitors))]
    competitors = [get_full_type_name(_) for _ in competitors]
    non_competitors = (
        None
        if non_competitors is None
        else [get_full_type_name(_) for _ in non_competitors]
    )
    params = dict(
        competitors=competitors,
        competitor_params=competitor_params,
        non_competitors=non_competitors,
        non_competitor_params=non_competitor_params,
        n_agents_per_competitor=n_agents_per_competitor,
        tournament_path=str(tournament_path),
        total_timeout=total_timeout,
        parallelism=parallelism,
        scheduler_ip=scheduler_ip,
        scheduler_port=scheduler_port,
        name=name,
        n_configs=n_configs,
        n_world_per_config=max_worlds_per_config,
        n_runs_per_world=n_runs_per_world,
        n_worlds=None,
        compact=compact,
        n_competitors_per_world=n_competitors_per_world,
    )
    params.update(kwargs)
    dump(params, tournament_path / "params")

    assigned = []
    configs = [
        config_generator(
            n_competitors=n_competitors_per_world,
            n_agents_per_competitor=n_agents_per_competitor,
            agent_names_reveal_type=agent_names_reveal_type,
            non_competitors=non_competitors,
            non_competitor_params=non_competitor_params,
            compact=compact,
            **kwargs,
        )
        for _ in range(n_configs)
    ]
    dump(configs, tournament_path / "base_configs")
    if verbose:
        print(
            f"Will run {len(configs)}  different base world configurations ({parallelism})",
            flush=True,
        )

    competitor_info = list(zip(competitors, competitor_params))
    if round_robin:
        competitor_sets = itertools.combinations(
            competitor_info, n_competitors_per_world
        )
    else:
        comp_ind = list(range(len(competitor_info)))
        random.shuffle(comp_ind)
        competitor_sets = _divide_into_sets(comp_ind, n_competitors_per_world)
        competitor_sets = [[competitor_info[_] for _ in lst] for lst in competitor_sets]

    for effective_competitor_infos in competitor_sets:
        effective_competitors = [_[0] for _ in effective_competitor_infos]
        effective_params = [_[1] for _ in effective_competitor_infos]
        if verbose:
            print(f"Running {'|'.join(effective_competitors)} together")
        effective_competitors = list(effective_competitors)
        myconfigs = copy.deepcopy(configs)
        for conf in myconfigs:
            for c in conf:
                c["world_params"]["name"] += "_" + "-".join(
                    _hash(_)[:4] for _ in effective_competitors
                )
        this_assigned = list(
            itertools.chain(
                *[
                    config_assigner(
                        config=c,
                        max_n_worlds=max_worlds_per_config,
                        n_agents_per_competitor=n_agents_per_competitor,
                        competitors=effective_competitors,
                        params=effective_params,
                        dynamic_non_competitors=dynamic_non_competitors,
                        dynamic_non_competitor_params=dynamic_non_competitor_params,
                        exclude_competitors_from_reassignment=exclude_competitors_from_reassignment,
                    )
                    for c in myconfigs
                ]
            )
        )
        assigned += this_assigned

    for config in assigned:
        for c in config:
            c["world_params"].update(
                {
                    "log_folder": str(
                        (
                            tournament_path / c["world_params"].get("name", ".")
                        ).absolute()
                    ),
                    "log_to_file": not compact,
                }
            )

    score_calculator_name = (
        get_full_type_name(score_calculator)
        if not isinstance(score_calculator, str)
        else score_calculator
    )
    world_generator_name = (
        get_full_type_name(world_generator)
        if not isinstance(world_generator, str)
        else world_generator
    )

    params["n_worlds"] = len(assigned) * n_runs_per_world
    params["world_generator_name"] = world_generator_name
    params["score_calculator_name"] = score_calculator_name

    dump(params, tournament_path / "params")
    dump(assigned, tournament_path / "assigned_configs")

    if verbose:
        print(
            f"Will run {len(assigned)}  different factory/manager assignments ({parallelism})",
            flush=True,
        )

    if n_runs_per_world > 1:
        n_before_duplication = len(assigned)
        all_assigned = []
        for r in range(n_runs_per_world):
            for a_ in assigned:
                all_assigned.append([])
                for w_ in a_:
                    cpy = copy.deepcopy(w_)
                    cpy["world_params"]["name"] += f".{r+1:05}"
                    all_assigned[-1].append(cpy)
        del assigned
        assigned = all_assigned
        assert n_before_duplication * n_runs_per_world == len(assigned), (
            f"Got {len(assigned)} assigned worlds for {n_before_duplication} "
            f"initial set with {n_runs_per_world} runs/world"
        )

    for config_set in assigned:
        for config in config_set:
            dir_name = tournament_path / config["world_params"]["name"]
            config.update(
                {
                    "log_file_name": str(dir_name / "log.txt"),
                    "__dir_name": str(dir_name),
                }
            )
            config["world_params"].update(
                {"log_file_name": str("log.txt"), "log_folder": str(dir_name)}
            )
    if forced_logs_fraction > 1e-5:
        n_logged = max(1, int(len(assigned) * forced_logs_fraction))
        for cs in assigned[:n_logged]:
            for _ in cs:
                if _["world_params"].get("log_folder", None) is None:
                    _["world_params"].update(
                        {
                            "log_folder": str(
                                (
                                    tournament_path / _["world_params"].get("name", ".")
                                ).absolute()
                            ),
                            "log_to_file": True,
                            "compact": False,
                            "no_logs": False,
                        }
                    )

    if save_video_fraction > 1e-5:
        n_videos = max(1, int(len(assigned) * forced_logs_fraction))
        for cs in assigned[:n_videos]:
            for _ in cs:
                _["world_params"]["construct_graphs"] = True
                _["__save_video"] = True
                _["__video_saver"] = video_saver
                _["__video_saver_params"] = video_params
    saved_configs = []
    for cs in assigned:
        for _ in cs:
            saved_configs.append(
                {
                    k: copy.copy(v)
                    if k != "competitors"
                    else [
                        get_full_type_name(c) if not isinstance(c, str) else c
                        for c in v
                    ]
                    for k, v in _.items()
                }
            )

    for d in saved_configs:
        d["__score_calculator"] = score_calculator_name
        d["__world_generator"] = world_generator_name
        d["__tournament_name"] = name
    config_path = tournament_path / "configs"
    config_path.mkdir(exist_ok=True, parents=True)
    for i, conf in enumerate(saved_configs):
        f_name = config_path / f"{i:06}"
        dump(conf, f_name)

    dump(assigned, tournament_path / "assigned_configs.pickle")

    return tournament_path


def evaluate_tournament(
    tournament_path: Optional[Union[str, PathLike, Path]],
    scores: Optional[pd.DataFrame] = None,
    stats: Optional[pd.DataFrame] = None,
    world_stats: Optional[pd.DataFrame] = None,
    metric: Union[str, Callable[[pd.DataFrame], float]] = "mean",
    verbose: bool = False,
    recursive: bool = True,
) -> TournamentResults:
    """
    Evaluates the results of a tournament

    Args:
        tournament_path: Path to save the results to. If scores is not given, it is also used as the source of scores.
                         Pass None to avoid saving the results to disk.
        scores: Optionally the scores of all agents in all world runs. If not given they will be read from the file
                scores.csv in `tournament_path`
        stats: Optionally the stats of all world runs. If not given they will be read from the file
               stats.csv in `tournament_path`
        world_stats: Optionally the aggregate stats collected in `WorldSetRunStats` for each world set
        metric: The metric used for evaluation. Possibilities are: mean, median, std, var, sum or a callable that
                receives a pandas data-frame and returns a float.
        verbose: If true, the winners will be printed
        recursive: If true, ALL scores.csv files in all subdirectories of the given tournament_path
                   will be combined
        # independent_test: True if you want an independent t-test

    Returns:

    """
    params, world_stats = None, None
    if tournament_path is not None:
        tournament_path = _path(tournament_path)
        tournament_path = tournament_path.absolute()
        tournament_path.mkdir(parents=True, exist_ok=True)
        scores_file = str(tournament_path / "scores.csv")
        world_stats_file = tournament_path / "world_stats.csv"
        params_file = tournament_path / "params.json"
        if world_stats is None and world_stats_file.exists():
            world_stats = pd.read_csv(world_stats_file, index_col=None)
        if params_file.exists():
            params = load(str(params_file))
        if scores is None:
            if recursive:
                scores = combine_tournaments(
                    sources=[tournament_path], dest=None, verbose=verbose
                )
            else:
                scores = pd.read_csv(scores_file, index_col=None)

        if stats is None:
            stats = combine_tournament_stats(
                sources=[tournament_path], dest=None, verbose=False
            )
    if scores is not None and not isinstance(scores, pd.DataFrame):
        scores = pd.DataFrame(data=scores)
    if stats is not None and not isinstance(stats, pd.DataFrame):
        stats = pd.DataFrame(data=stats)
    if scores is None or len(scores) < 1:
        return TournamentResults(
            scores=pd.DataFrame(),
            total_scores=pd.DataFrame(),
            winners=[],
            winners_scores=np.array([]),
            ttest=pd.DataFrame(),
            kstest=pd.DataFrame(),
            stats=pd.DataFrame(),
            agg_stats=pd.DataFrame(),
            score_stats=pd.DataFrame(),
            path=str(tournament_path) if tournament_path is not None else None,
            params=params,
            world_stats=world_stats,
        )
    scores = scores.loc[~scores["agent_type"].isnull(), :]
    scores = scores.loc[scores["agent_type"].str.len() > 0, :]
    if not isinstance(metric, str):
        total_scores = (
            scores.groupby(["agent_type"])["score"]
            .apply(metric)
            .sort_values(ascending=False)
            .reset_index()
        )
    elif metric == "median":
        total_scores = (
            scores.groupby(["agent_type"])["score"]
            .median()
            .sort_values(ascending=False)
            .reset_index()
        )
    elif metric == "mean":
        total_scores = (
            scores.groupby(["agent_type"])["score"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )
    elif metric == "std":
        total_scores = (
            scores.groupby(["agent_type"])["score"]
            .std()
            .sort_values(ascending=False)
            .reset_index()
        )
    elif metric == "var":
        total_scores = (
            scores.groupby(["agent_type"])["score"]
            .var()
            .sort_values(ascending=False)
            .reset_index()
        )
    elif metric == "sum":
        total_scores = (
            scores.groupby(["agent_type"])["score"]
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )
    else:
        raise ValueError(
            f"Unknown metric: {metric}. Supported metrics include mean, median, std, var, sum or a callable"
        )
    score_stats = scores.groupby(["agent_type"])["score"].describe().reset_index()
    winner_table = total_scores.loc[
        total_scores["score"] == total_scores["score"].max(), :
    ]
    winners = winner_table["agent_type"].values.tolist()
    winner_scores = winner_table["score"].values
    types = list(scores["agent_type"].unique())

    ttest_results = []
    ks_results = []
    for i, t1 in enumerate(types):
        for j, t2 in enumerate(types[i + 1 :]):
            ascores, bscores = (
                scores.loc[scores["agent_type"] == t1, ["score", "world"]],
                scores.loc[scores["agent_type"] == t2, ["score", "world"]],
            )
            # for _ in (ascores, bscores):
            #     _["world"] = _["world"].str.split(".").str[0]
            # ascores.columns = ["score_1", "world"]
            # bscores.columns = ["score_2", "world"]
            # joined = pd.merge(ascores, bscores, on=["world"])
            # if len(joined) > 0 and not independent_test:
            #     alist, blist = joined.score_1, joined.score_2
            #     t, p = ttest_rel(alist, blist)
            # else:
            #     alist, blist = (ascores.score, bscores.score)
            #     t, p = ttest_ind(alist, blist)
            alist, blist = (ascores.score, bscores.score)
            if min(len(alist), len(blist)) >= 2:
                t, p = ttest_ind(alist, blist)

                ttest_results.append(
                    {
                        "a": t1,
                        "b": t2,
                        "t": t,
                        "p": p,
                        "n_a": len(ascores),
                        "n_b": len(bscores),
                        "n_effective": min(len(alist), len(blist)),
                    }
                )
                t, p = ks_2samp(alist, blist)
                ks_results.append(
                    {
                        "a": t1,
                        "b": t2,
                        "t": t,
                        "p": p,
                        "n_a": len(ascores),
                        "n_b": len(bscores),
                        "n_effective": min(len(alist), len(blist)),
                    }
                )
    if verbose:
        print(f"Winners: {list(zip(winners, winner_scores))}")

    agg_stats = pd.DataFrame()
    if tournament_path is not None:
        tournament_path = pathlib.Path(tournament_path)
        scores.to_csv(str(tournament_path / "scores.csv"), index_label="index")
        total_scores.to_csv(
            str(tournament_path / "total_scores.csv"), index_label="index"
        )
        winner_table.to_csv(str(tournament_path / "winners.csv"), index_label="index")
        score_stats.to_csv(str(tournament_path / "score_stats.csv"), index=False)
        ttest_results = pd.DataFrame(data=ttest_results)
        ttest_results.to_csv(str(tournament_path / "ttest.csv"), index_label="index")
        ks_results = pd.DataFrame(data=ks_results)
        ks_results.to_csv(str(tournament_path / "kstest.csv"), index_label="index")
        if stats is not None:
            stats.to_csv(str(tournament_path / "stats.csv"), index=False)
            agg_stats = _combine_stats(stats)
            agg_stats.to_csv(str(tournament_path / "agg_stats.csv"), index=False)

    if verbose:
        print(f"N. scores = {len(scores)}\tN. Worlds = {len(scores.world.unique())}")

    return TournamentResults(
        scores=scores,
        total_scores=total_scores,
        winners=winners,
        winners_scores=winner_scores,
        ttest=ttest_results,
        kstest=ks_results,
        stats=stats,
        agg_stats=agg_stats,
        score_stats=score_stats,
        path=str(tournament_path) if tournament_path is not None else None,
        params=params,
        world_stats=world_stats,
    )


def combine_tournaments(
    sources: Iterable[Union[str, PathLike]],
    dest: Union[str, PathLike] = None,
    verbose=False,
) -> pd.DataFrame:
    """Combines results of several tournament runs in the destination path."""

    scores = []
    for src in sources:
        src = _path(src)
        for filename in src.glob("**/scores.csv"):
            try:
                scores.append(pd.read_csv(filename))
                if verbose:
                    print(f"Read: {str(filename)}")
            except:
                if verbose:
                    print(f"FAILED {str(filename)}")
    if len(scores) < 1:
        if verbose:
            print("No scores found")
        return pd.DataFrame()
    scores: pd.DataFrame = pd.concat(scores, axis=0, ignore_index=True, sort=True)
    if dest is not None:
        scores.to_csv(str(_path(dest) / "scores.csv"), index=False)
    return scores


def combine_tournament_stats(
    sources: Iterable[Union[str, PathLike]],
    dest: Union[str, PathLike] = None,
    verbose=False,
) -> pd.DataFrame:
    """Combines statistical results of several tournament runs in the destination path."""

    stats = []
    for src in sources:
        src = _path(src)
        for filename in src.glob("**/stats.json"):
            # try:
            data = load(filename)
            if data is None or len(data) == 0:
                continue
            try:
                data = pd.DataFrame.from_dict(data)
            except:
                # adjust lengths. Some columns are longer than others
                min_len = min(len(_) for _ in data.values())
                for k, v in data.items():
                    if len(v) == min_len:
                        continue
                    data[k] = data[k][:min_len]
                data = pd.DataFrame.from_dict(data)
            data = data.loc[:, [c for c in data.columns if World.is_basic_stat(c)]]
            data["step"] = list(range(len(data)))
            data["world"] = filename.parent.name
            data["path"] = filename.parent.parent
            stats.append(data)
    if len(stats) < 1:
        if verbose:
            print("No stats found")
        return pd.DataFrame()
    stats: pd.DataFrame = pd.concat(stats, axis=0, ignore_index=True, sort=True)
    if dest is not None:
        stats.to_csv(str(_path(dest) / "stats.csv"), index=False)
        combined = _combine_stats(stats)
        combined.to_csv(str(_path(dest) / "agg_stats"), index=False)
    return stats


def _combine_stats(stats: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Generates aggregate stats from stats"""
    if stats is None:
        return None
    combined = (
        stats.loc[
            :,
            [c for c in stats.columns if not c.startswith("_") and c not in ("path",)],
        ]
        .groupby(["world"])
        .agg([np.mean, np.max, np.min, np.sum, np.var, np.median])
    )

    def get_last(x):
        return x.loc[x["step"] == x["step"].max(), :]

    last = stats.groupby(["world"]).apply(get_last)
    # print("IN COMBINE ---------------")
    # print(last.columns)
    # print(last.index)
    # print("IN COMBINE ---------------")
    last.columns = [
        f"{str(c)}_final" if c not in ("world", "path") else c for c in last.columns
    ]
    last.set_index("world")
    last.drop("world", axis=1, inplace=True)
    combined.columns = combined.columns.to_flat_index()
    # combined.columns = [
    #     f"{a[0]}_a{1}" if a not in ("world", "path") else a[0] for a in combined.columns
    # ]
    combined = pd.merge(combined, last, on=["world"])

    def adjust_name(s):
        if isinstance(s, tuple):
            s = "".join(s)
        return (
            s.replace("'", "")
            .replace('"', "")
            .replace(" ", "")
            .replace("(", "")
            .replace(")", "")
            .replace("amax", "_max")
            .replace("amin", "_min")
            .replace("mean", "_mean")
            .replace("var", "_var")
            .replace("median", "_median")
            .replace("sum", "_sum")
        )

    combined.columns = [adjust_name(c) for c in combined.columns]
    return combined
