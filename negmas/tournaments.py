"""
Tournament generation and management.

"""
import concurrent.futures as futures
import copy
import itertools
import math
import pathlib
import shutil
import time
import traceback
import warnings
from dataclasses import dataclass, field
from multiprocessing import cpu_count
from os import PathLike
from pathlib import Path
from typing import (
    Optional,
    List,
    Callable,
    Union,
    Type,
    Sequence,
    Dict,
    Any,
    Tuple,
    Iterable,
)

import numpy as np
import pandas as pd
import yaml
from typing_extensions import Protocol

from negmas.helpers import (
    get_class,
    unique_name,
    import_by_name,
    get_full_type_name,
    humanize_time,
    dump,
    add_records,
    load,
)
from .situated import Agent, World, save_stats
import hashlib

__all__ = [
    "tournament",
    "WorldGenerator",
    "WorldRunResults",
    "TournamentResults",
    "run_world",
    "process_world_run",
    "evaluate_tournament",
    "combine_tournaments",
    "create_tournament",
    "run_tournament",
]

PROTOCOL_CLASS_NAME_FIELD = "__mechanism_class_name"

try:
    # disable a warning in yaml 1b1 version
    yaml.warnings({"YAMLLoadWarning": False})
except:
    pass


def _hash(*args):
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
        competitors: Sequence[Type[Agent]] = (),
        params: Sequence[Dict[str, Any]] = (),
    ) -> List[List[Dict[str, Any]]]:
        ...


@dataclass
class WorldRunResults:
    """Results of a world run"""

    world_names: List[str]
    """World name"""
    log_file_names: List[str]
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


def run_world(
    world_params: dict, dry_run: bool = False, save_world_stats: bool = True
) -> Tuple[str, WorldRunResults]:
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
    world_params["log_file_name"] = world_params.get(
        "log_file_name", str(default_dir / "log.txt")
    )
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
) -> Tuple[str, WorldRunResults]:
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
            WorldRunResults(world_names=[""], log_file_names=[""]),
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
            "log_file_name", str(default_dir / "log.txt")
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
) -> Tuple[str, WorldRunResults]:
    """Runs a set of worlds (generated from a world generator) and returns stats

    Args:
        worlds_params: A list of World info dicts. See remarks for its parameters
        world_generator: World generator function.
        score_calculator: Score calculator function.
        world_progress_callback: world progress callback
        dry_run: If true, the world is not run. Its config is saved instead.
        save_world_stats: If true, saves individual world stats

    Returns:

        A tuple with the list of `WorldRunResults` for all the worlds generated using this config and the directory in
        which these results are stored

    Remarks:

        - Each `worlds_params` dict should have the following members:

            - name: world name
            - log_file_name: file name to store the world log
            - __dir_name: directory to store the world stats
            - others: values of all other keys are passed to the world generator as kwargs

    """
    worlds = []
    scoring_context = {}
    run_id = _hash(worlds_params)
    for world_params in worlds_params:
        world_params = world_params.copy()
        dir_name = world_params["__dir_name"]
        world_params.pop("__dir_name", None)
        scoring_context.update(world_params.get("scoring_context", {}))
        world = world_generator(**world_params)
        if dry_run:
            world.save_config(dir_name)
            continue
        if world_progress_callback is None:
            world.run()
        else:
            _start_time = time.monotonic()
            for _ in range(world.n_steps):
                if (
                    world.time_limit is not None
                    and (time.monotonic() - _start_time) >= world.time_limit
                ):
                    break
                if not world.step():
                    break
                world_progress_callback(world)
        worlds.append(world)
        if save_world_stats:
            save_stats(world=world, log_dir=dir_name)
    scores = score_calculator(worlds, scoring_context, dry_run)
    return run_id, scores


def process_world_run(
    run_id: str,
    results: WorldRunResults,
    tournament_name: str,
    save_world_stats: bool = True,
) -> List[Dict[str, Any]]:
    """
    Generates a data-frame with the results of this world run

    Args:
        run_id: The ID of this run (should be unique per tournament)
        results: Results of the world run
        tournament_name: tournament name
        save_world_stats: It True, it will be assumed that world stats are saved

    Returns:

        A pandas DataFrame with agent_name, agent_type, score, log_file, world, and stats_folder columns

    """
    log_files, world_names_ = results.log_file_names, results.world_names
    for world_name_, log_file in zip(world_names_, log_files):
        if (
            save_world_stats
            and log_file is not None
            and pathlib.Path(log_file).exists()
        ):
            with open(log_file, "a") as f:
                f.write(
                    f"\nPART of TOURNAMENT {tournament_name}. This world run completed successfully\n"
                )
    scores = []
    stat_folders = ";".join(
        str(pathlib.Path(log_file_name).name) for log_file_name in log_files
    )
    base_folder = str(pathlib.Path(log_files[0]).parent)
    for name_, type_, score in zip(results.names, results.types, results.scores):

        scores.append(
            {
                "agent_name": name_,
                "agent_type": type_,
                "score": score,
                "log_file": ";".join(log_files),
                "world": ";".join(world_names_),
                "stats_folders": stat_folders,
                "base_stats_folder": base_folder,
                "run_id": run_id,
            }
        )
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
    run_ids,
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
            )
        )
    print(f"Submitted all processes to DASK ({len(future_results)})")
    _strt = time.perf_counter()
    for i, future in enumerate(
        distributed.as_completed(future_results, with_results=True, raise_errors=False)
    ):
        try:
            run_id, score_ = future.result()
            if tournament_progress_callback is not None:
                tournament_progress_callback(score_, i, n_worlds)
            add_records(
                scores_file,
                process_world_run(
                    run_id,
                    score_,
                    tournament_name=name,
                    save_world_stats=save_world_stats,
                ),
            )
            if verbose:
                _duration = time.perf_counter() - _strt
                print(
                    f"{i + 1:003} of {n_worlds:003} [{100 * (i + 1) / n_worlds:0.3}%] completed in "
                    f"{humanize_time(_duration)} [ETA {humanize_time(_duration * n_worlds / (i + 1))}]"
                )
        except Exception as e:
            if tournament_progress_callback is not None:
                tournament_progress_callback(None, i, n_worlds)
            print(traceback.format_exc())
            print(e)
    client.shutdown()


def tournament(
    competitors: Sequence[Union[str, Type[Agent]]],
    config_generator: ConfigGenerator,
    config_assigner: ConfigAssigner,
    world_generator: WorldGenerator,
    score_calculator: Callable[[List[World], Dict[str, Any], bool], WorldRunResults],
    competitor_params: Optional[Sequence[Dict[str, Any]]] = None,
    agent_names_reveal_type=False,
    n_agents_per_competitor=1,
    n_configs: int = 10,
    max_worlds_per_config: int = 100,
    n_runs_per_world: int = 5,
    max_n_configs: int = None,
    n_runs_per_config: int = None,
    tournament_path: str = "./logs/tournaments",
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
    name: str = None,
    verbose: bool = False,
    configs_only: bool = False,
    compact: bool = False,
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
        non_competitors: A list of agent types that will not be competing in the sabotage competition but will exist
                         in the world
        non_competitor_params: paramters of non competitor agents
        verbose: Verbosity
        configs_only: If true, a config file for each
        compact: If true, compact logs will be created and effort will be made to reduce the memory footprint
        kwargs: Arguments to pass to the `config_generator` function

    Returns:
        `TournamentResults` The results of the tournament or a `PathLike` giving the location where configs were saved

    """
    tournament_path = create_tournament(
        competitors=competitors,
        config_generator=config_generator,
        config_assigner=config_assigner,
        world_generator=world_generator,
        score_calculator=score_calculator,
        competitor_params=competitor_params,
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
        name=name,
        verbose=verbose,
        compact=compact,
        **kwargs,
    )
    if configs_only:
        return pathlib.Path(tournament_path) / "configs"
    run_tournament(
        tournament_path=tournament_path,
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
    )

    if verbose:
        print(f"Tournament completed successfully")
        print(f"Finding winners")
    return evaluate_tournament(tournament_path=tournament_path, verbose=verbose)


def _path(path: Union[str, PathLike]) -> Path:
    """Creates an absolute path from given path which can be a string"""
    if isinstance(path, str):
        if path.startswith("~"):
            path = Path.home() / ("/".join(path.split("/")[1:]))
    return pathlib.Path(path).absolute()


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
    n_world_configs = len(assigned)

    if verbose:
        print(
            f"Will run {n_world_configs}  total world simulations ({parallelism})",
            flush=True,
        )

    scores_file = tournament_path / "scores.csv"
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
                run_id, score_ = _run_worlds(
                    worlds_params=worlds_params,
                    world_generator=world_generator,
                    world_progress_callback=world_progress_callback,
                    score_calculator=score_calculator,
                    dry_run=False,
                    save_world_stats=not compact,
                )
                if tournament_progress_callback is not None:
                    tournament_progress_callback(score_, i, n_world_configs)
                add_records(
                    scores_file,
                    process_world_run(
                        run_id,
                        score_,
                        tournament_name=name,
                        save_world_stats=not compact,
                    ),
                )
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
                    not compact,
                )
            )
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
            try:
                run_id, score_ = future.result()
                if tournament_progress_callback is not None:
                    tournament_progress_callback(score_, i, n_world_configs)
                add_records(
                    scores_file,
                    process_world_run(
                        run_id,
                        score_,
                        tournament_name=name,
                        save_world_stats=not compact,
                    ),
                )
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
                print("Tournament timed-out")
                break
            except Exception as e:
                if tournament_progress_callback is not None:
                    tournament_progress_callback(None, i, n_world_configs)
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
            not compact,
            scores_file,
            run_ids,
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
    agent_names_reveal_type=False,
    n_agents_per_competitor=1,
    n_configs: int = 10,
    max_worlds_per_config: int = 100,
    n_runs_per_world: int = 5,
    max_n_configs: int = None,
    n_runs_per_config: int = None,
    base_tournament_path: str = "./logs/tournaments",
    total_timeout: Optional[int] = None,
    parallelism="parallel",
    scheduler_ip: Optional[str] = None,
    scheduler_port: Optional[str] = None,
    non_competitors: Optional[Tuple[Union[str, Any]]] = None,
    non_competitor_params: Optional[Tuple[Dict[str, Any]]] = None,
    name: str = None,
    verbose: bool = False,
    compact: bool = False,
    **kwargs,
) -> PathLike:
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
        non_competitors: A list of agent types that will not be competing in the sabotage competition but will exist
                         in the world
        non_competitor_params: paramters of non competitor agents
        verbose: Verbosity
        compact: If true, compact logs will be created and effort will be made to reduce the memory footprint
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
        name = unique_name("", add_time=True, rand_digits=0)
    competitors = list(competitors)
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
    params = {
        "competitors": competitors,
        "competitor_params": competitor_params,
        "non_competitors": non_competitors,
        "non_competitor_params": non_competitor_params,
        "n_agents_per_competitor": n_agents_per_competitor,
        "tournament_path": str(tournament_path),
        "total_timeout": total_timeout,
        "parallelism": parallelism,
        "scheduler_ip": scheduler_ip,
        "scheduler_port": scheduler_port,
        "name": name,
        "n_configs": n_configs,
        "n_world_per_config": max_worlds_per_config,
        "n_runs_per_world": n_runs_per_world,
        "n_worlds": None,
        "compact": compact,
    }
    params.update(kwargs)
    dump(params, tournament_path / "params")

    configs = [
        config_generator(
            n_competitors=len(competitors),
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

    assigned = list(
        itertools.chain(
            *[
                config_assigner(
                    config=c,
                    max_n_worlds=max_worlds_per_config,
                    n_agents_per_competitor=n_agents_per_competitor,
                    competitors=competitors,
                    params=competitor_params,
                )
                for c in configs
            ]
        )
    )

    for config in assigned:
        for c in config:
            c["world_params"].update(
                {
                    "log_folder": str(
                        (
                            tournament_path / c["world_params"].get("name", ".")
                        ).absolute()
                    )
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

    assigned = list(itertools.chain(*([assigned] * n_runs_per_world)))

    for config_set in assigned:
        for config in config_set:
            dir_name = tournament_path / config["world_params"]["name"]
            config.update(
                {
                    "log_file_name": str(dir_name / "log.txt"),
                    "__dir_name": str(dir_name),
                }
            )
            config["world_params"].update({"log_file_name": str(dir_name / "log.txt")})

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
    tournament_path: Union[str, PathLike, Path],
    scores: Optional[pd.DataFrame] = None,
    verbose: bool = False,
):
    """
    Evaluates the results of a tournament

    Args:
        tournament_path: Path of the tournament
        scores: Optionally the scores of all agents in all world runs. If not given they will be read from the file
                scores.csv in `tournament_path`
        verbose: If true, the winners will be printed

    Returns:

    """
    if isinstance(tournament_path, str):
        if tournament_path.startswith("~"):
            tournament_path = Path.home() / ("/".join(tournament_path.split("/")[1:]))
        tournament_path = pathlib.Path(tournament_path)
    tournament_path = tournament_path.absolute()
    scores_file = str(tournament_path / "scores.csv")
    if scores is None:
        scores = pd.read_csv(scores_file, index_col=None)
    if not isinstance(scores, pd.DataFrame):
        scores = pd.DataFrame(data=scores)
    if len(scores) < 1:
        return TournamentResults(
            scores=pd.DataFrame(),
            total_scores=pd.DataFrame(),
            winners=[],
            winners_scores=np.array([]),
            ttest=pd.DataFrame(),
        )
    scores = scores.loc[~scores["agent_type"].isnull(), :]
    scores = scores.loc[scores["agent_type"].str.len() > 0, :]
    total_scores = (
        scores.groupby(["agent_type"])["score"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    winner_table = total_scores.loc[
        total_scores["score"] == total_scores["score"].max(), :
    ]
    winners = winner_table["agent_type"].values.tolist()
    winner_scores = winner_table["score"].values
    types = list(scores["agent_type"].unique())

    ttest_results = []
    for i, t1 in enumerate(types):
        for j, t2 in enumerate(types[i + 1 :]):
            from scipy.stats import ttest_ind

            t, p = ttest_ind(
                scores[scores["agent_type"] == t1].score,
                scores[scores["agent_type"] == t2].score,
            )
            ttest_results.append({"a": t1, "b": t2, "t": t, "p": p})
    if verbose:
        print(f"Winners: {list(zip(winners, winner_scores))}")

    scores.to_csv(str(tournament_path / "scores.csv"), index_label="index")
    total_scores.to_csv(str(tournament_path / "total_scores.csv"), index_label="index")
    winner_table.to_csv(str(tournament_path / "winners.csv"), index_label="index")
    ttest_results = pd.DataFrame(data=ttest_results)
    ttest_results.to_csv(str(tournament_path / "ttest.csv"), index_label="index")

    if verbose:
        print(f"N. scores = {len(scores)}\tN. Worlds = {len(scores.world.unique())}")

    return TournamentResults(
        scores=scores,
        total_scores=total_scores,
        winners=winners,
        winners_scores=winner_scores,
        ttest=ttest_results,
    )


def combine_tournaments(
    sources: Iterable[Union[str, PathLike]], dest: Union[str, PathLike]
) -> None:
    """Combines results of several tournament runs in the destination path."""

    pd.concat(
        [pd.read_csv(_path(src) / "scores.csv") for src in sources],
        axis=0,
        ignore_index=True,
    ).to_csv(str(_path(dest) / "scores.csv"))
