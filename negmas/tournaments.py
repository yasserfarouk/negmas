"""
Tournament generation and management.

"""
import concurrent.futures as futures
import copy
import hashlib
import itertools
import math
import os
import pathlib
import random
import time
import traceback
import warnings
from multiprocessing import current_process
from socket import gethostname

try:
    import distributed
except:
    ENABLE_DASK = False
else:
    ENABLE_DASK = True
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
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
    dump,
    get_class,
    get_full_type_name,
    humanize_time,
    import_by_name,
    load,
    shortest_unique_names,
    unique_name,
)
from negmas.serialization import serialize, to_flat_dict

from .situated import Agent, World, save_stats

__all__ = [
    "tournament",
    "WorldGenerator",
    "WorldRunResults",
    "TournamentResults",
    "run_world",
    "process_world_run",
    "evaluate_tournament",
    "combine_tournament_results",
    "combine_tournaments",
    "combine_tournament_stats",
    "create_tournament",
    "run_tournament",
]

PROTOCOL_CLASS_NAME_FIELD = "__mechanism_class_name"
# files created before running worlds
PARAMS_FILE = "params.json"
ASSIGNED_CONFIGS_PICKLE_FILE = "assigned_configs.pickle"
ASSIGNED_CONFIGS_JSON_FILE = "assigned_configs.json"

# File keeping final results for a single world
RESULTS_FILE = "results.json"

# files keeping track of scores and stats calculated during eval_tournament()
SCORES_FILE = "scores.csv"
STATS_FILE = "stats.csv"
TYPE_STATS_FILE = "type_stats.csv"
AGENT_STATS_FILE = "agent_stats.csv"
WORLD_STATS_FILE = "world_stats.csv"

# files containing aggregate results calculated during eval_tournament()
AGGREGATE_STATS_FILE = "agg_stats.csv"
K_STATS_FILE = "kstats.csv"
T_STATS_FILE = "tstats.csv"
SCORES_STATS_FILE = "score_stats.csv"
TOTAL_SCORES_FILE = "total_scores.csv"
WINNERS_FILE = "winners.csv"

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
        """Generates a world"""


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
    extra_scores: Dict[str, List[Dict[str, Any]]] = field(
        default_factory=dict, init=False
    )
    """The extra-scores (i.e. extra evaluation metrics). Each is a list of records"""


def score_adapter(scores_data: Dict[str, Any]) -> WorldRunResults:
    world_names = (
        [scores_data["name"]]
        if isinstance(scores_data["name"], str)
        else scores_data["world_names"]
    )
    paths = scores_data["world_paths"]
    if isinstance(paths, str):
        paths = paths.split(";")
    log_file_names = [str(pathlib.Path(_) / "log.txt") for _ in paths]
    r = WorldRunResults(world_names=world_names, log_file_names=log_file_names)
    scores = scores_data["scores"]
    r.scores = [_["score"] for _ in scores]
    r.names = [_["agent_name"] for _ in scores]
    r.ids = [_["agent_id"] for _ in scores]
    r.types = [_["agent_type"] for _ in scores]
    r.extra_scores = scores_data["extra_scores"]
    return r


@dataclass
class AgentStats:
    exceptions: Dict[str, List[Tuple[int, str]]] = field(
        default_factory=lambda: defaultdict(list)
    )
    """All exceptions thrown per agent (not including negotiator exceptions)"""
    negotiator_exceptions: Dict[str, List[Tuple[int, str]]] = field(
        default_factory=lambda: defaultdict(list)
    )
    """All exceptions thrown by negotiators of an agent"""
    times: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    """Total execution time per agent"""
    neg_requests_sent: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    """Negotiation Requests Sent"""
    neg_requests_received: Dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    """Negotiation Requests Received"""
    neg_requests_rejected: Dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    """Negotiation requests rejected"""
    negs_registered: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    """Negotiations registered"""
    negs_succeeded: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    """Negotiations succeeded"""
    negs_failed: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    """Negotiations failed"""
    negs_timedout: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    """Negotiations timedout"""
    negs_initiated: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    """Negotiations initiated"""
    contracts_concluded: Dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    """Contracts concluded"""
    contracts_signed: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    """Contracts signed"""
    contracts_dropped: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    """Contracts dropped"""
    breaches_received: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    """breaches received"""
    breaches_committed: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    """breaches committed"""
    contracts_erred: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    """Contracts erred"""
    contracts_nullified: Dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    """Contracts nullified"""
    contracts_breached: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    """Contracts breached"""
    contracts_executed: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    """Contracts executed"""

    def to_record(self, world, label="name"):
        """Converts AgentStats to a record in the form of a dict"""
        x = vars(self)
        cols = set(_ for _ in x.keys())
        rows = set()
        for d in cols:
            rows |= set(_ for _ in x[d].keys())
        cols = list(cols)
        rows = list(rows)
        results = [dict() for _ in rows]
        for i, row in enumerate(rows):
            results[i]["world"] = world
            results[i][label] = row
            for col in cols:
                results[i][col] = x[col][row]
        return results

    @classmethod
    def from_records(cls, records: List[Dict[str, Any]], label: str):
        a = cls()
        if len(records) < 1:
            return a
        worlds = ""
        for record in records:
            for k, v in record.items():
                if k == "world":
                    if len(worlds) > 0:
                        worlds += f";{v}"
                    else:
                        worlds += v
                    continue
                if k == label:
                    continue
                a.__dict__[k][record[label]] += v
        return a


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
    simulation_exceptions: List[Tuple[int, str]] = field(default_factory=list)
    """Exceptions thrown by the simulator (not including mechanism creation and contract exceptions)"""
    contract_exceptions: List[Tuple[int, str]] = field(default_factory=list)
    """Exceptions thrown by the simulator during contract execution"""
    mechanism_exceptions: List[Tuple[int, str]] = field(default_factory=list)
    """Exceptions thrown by the simulator during mechanism creation or execution"""
    other_exceptions: List[str] = field(default_factory=list)
    """Exceptions raised by tournament running code itself not any world"""

    n_agent_exceptions: int = 0
    """All exceptions thrown per agent (not including negotiator exceptions)"""
    n_negotiator_exceptions: int = 0
    """All exceptions thrown by negotiators of an agent"""
    mean_agent_time: float = 0.0
    """Average execution time per agent"""
    n_neg_requests_sent: int = 0
    """Negotiation Requests Sent"""
    n_neg_requests_received: int = 0
    """Negotiation Requests Received"""
    n_neg_requests_rejected: int = 0
    """Negotiation requests rejected"""
    n_negs_registered: int = 0
    """Negotiations registered"""
    n_negs_succeeded: int = 0
    """Negotiations succeeded"""
    n_negs_failed: int = 0
    """Negotiations failed"""
    n_negs_timedout: int = 0
    """Negotiations timedout"""
    n_negs_initiated: int = 0
    """Negotiations initiated"""
    n_contracts_concluded: int = 0
    """Contracts concluded"""
    n_contracts_signed: int = 0
    """Contracts signed"""
    n_contracts_dropped: int = 0
    """Contracts dropped"""
    n_breaches_received: int = 0
    """breaches received"""
    n_breaches_committed: int = 0
    """breaches committed"""
    n_contracts_erred: int = 0
    """Contracts erred"""
    n_contracts_nullified: int = 0
    """Contracts nullified"""
    n_contracts_breached: int = 0
    """Contracts breached"""
    n_contracts_executed: int = 0
    """Contracts executed"""

    def to_record(self, world):
        return [vars(self)]

    @classmethod
    def from_records(cls, records: List[Dict[str, Any]]):
        a = cls(name="", planned_n_steps=0, executed_n_steps=0, execution_time=0.0)
        if len(records) < 1:
            return a
        worlds = ""
        for record in records:
            for k, v in record.items():
                if k == "name":
                    if len(worlds) > 0:
                        worlds += f";{v}"
                    else:
                        worlds += v
                    continue
                a.__dict__[k] += v
        a.name = worlds
        return a


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
    type_stats: pd.DataFrame = None
    """Some statistics about each type"""
    agent_stats: pd.DataFrame = None
    """Some statistics about each agent"""
    params: Dict[str, Any] = None
    """Parameters of the tournament"""
    extra_scores: Dict[str, pd.DataFrame] = None
    """Extra scores returned from the scoring function. This can be used to have multi-dimensional scoring"""

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


def combine_partially_run_worlds(
    tournament_path: Union[str, Path],
    min_time_fraction: float = 0.0,
    min_real_time: float = 0.0,
    min_n_steps: int = 0,
    min_n_attempts: int = 0,
    dry_run: bool = False,
) -> List[Path]:
    """
    Combines partially run worlds by saving their last-saved results as final.

    Args:
        tournament_path: The path from which to read world information.
        min_time_fraction: The minimum fraction of the total world simulation time (as
                           determined by world.relative_time) for a world to be considered
                           completed.
        min_real_time: The minimum real-time spent executing a world to be considered complete.
        min_n_steps: The minimum number of steps that must be executed for a world to be
                     considered complete.
        min_n_attempts: The minimum number of attempts tried for this world to be considered
                        completed.
        dry_run: If true, the paths of the worlds to be considered complete will be returned
                 but the worlds will not actually be considered completed.

    Returns:
        A list of paths to the worlds that were completed (or to be completed for dry runs).

    Remarks:
        All conditions must be met for a world to be considered completed
    """


def run_world(
    world_params: dict,
    dry_run: bool = False,
    save_world_stats: bool = True,
    attempts_path=None,
    max_attempts=float("inf"),
    verbose=False,
) -> Tuple[
    str,
    List[str],
    Optional[WorldRunResults],
    Optional[WorldSetRunStats],
    Optional[AgentStats],
    Optional[AgentStats],
]:
    """Runs a world and returns stats. This function is designed to be used with distributed systems like dask.

    Args:
        world_params: World info dict. See remarks for its parameters
        dry_run: If true, the world will not be run. Only configs will be saved
        save_world_stats: If true, saves individual world stats
        attempts_path: The folder containing attempts information
        max_attempts: The maximum number of trials to run a world simulation

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
    world_params["log_folder"] = world_params.get("__dir_name", str(default_dir))
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
        attempts_path=attempts_path,
        max_attempts=max_attempts,
        verbose=verbose,
    )


def run_worlds(
    worlds_params: List[dict],
    dry_run: bool = False,
    save_world_stats: bool = True,
    attempts_path=None,
    max_attempts=float("inf"),
    verbose=False,
) -> Tuple[
    str,
    List[str],
    Optional[WorldRunResults],
    Optional[WorldSetRunStats],
    Optional[AgentStats],
    Optional[AgentStats],
]:
    """Runs a set of worlds and returns stats. This function is designed to be used with distributed systems like dask.

    Args:
        worlds_params: list of World info dicts. See remarks for its parameters
        dry_run: If true, the world will not be run. Only configs will be saved
        save_world_stats: If true, saves individual world stats
        attempts_path: The path containing attempts information
        max_attempts: Maximum number of trials to run a simulation

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
        attempts_path=attempts_path,
        max_attempts=max_attempts,
        verbose=verbose,
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
    attempts_path=None,
    max_attempts=float("inf"),
    verbose=False,
) -> Tuple[
    str,
    List[str],
    Optional[WorldRunResults],
    Optional[WorldSetRunStats],
    Optional[AgentStats],
    Optional[AgentStats],
]:
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
        attempts_path: The path to store attempts information.

    Returns:
        A tuple with the following components in order:

            - The run ID for this world-set
            - The paths to world folders to store results in. Note that there can be
              multiple forlders because a single score may be collected from multiple
              world simulations. Results should be duplicated in all those folders
              in such case. The run_id is unique per set of such worlds while
              world_name is unique per world.
            - The results (scores) for this world set (will be None in case of exception)
            - The stats for this world set (will not be None even in case of exception)
            - The stats for agent types
            - The stats for specific agents


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
    run_id = _run_id(worlds_params)
    video_savers, video_saver_params_list, save_videos = [], [], []
    scores: Optional[WorldRunResults] = None
    world_stats, type_stats, agent_stats = None, None, None

    simulation_exceptions = []
    mechanism_exceptions = []
    contract_exceptions = []
    other_exceptions = []
    negotiator_exceptions: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
    agent_times: Dict[str, float] = defaultdict(float)
    agent_exceptions: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
    neg_requests_sent: Dict[str, int] = defaultdict(int)
    neg_requests_received: Dict[str, int] = defaultdict(int)
    negs_registered: Dict[str, int] = defaultdict(int)
    negs_succeeded: Dict[str, int] = defaultdict(int)
    negs_failed: Dict[str, int] = defaultdict(int)
    negs_timedout: Dict[str, int] = defaultdict(int)
    negs_initiated: Dict[str, int] = defaultdict(int)
    contracts_concluded: Dict[str, int] = defaultdict(int)
    contracts_signed: Dict[str, int] = defaultdict(int)
    neg_requests_rejected: Dict[str, int] = defaultdict(int)
    contracts_dropped: Dict[str, int] = defaultdict(int)
    breaches_received: Dict[str, int] = defaultdict(int)
    breaches_committed: Dict[str, int] = defaultdict(int)
    contracts_erred: Dict[str, int] = defaultdict(int)
    contracts_nullified: Dict[str, int] = defaultdict(int)
    contracts_executed: Dict[str, int] = defaultdict(int)
    contracts_breached: Dict[str, int] = defaultdict(int)
    type_negotiator_exceptions: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
    type_agent_times: Dict[str, float] = defaultdict(float)
    type_agent_exceptions: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
    type_neg_requests_sent: Dict[str, int] = defaultdict(int)
    type_neg_requests_received: Dict[str, int] = defaultdict(int)
    type_negs_registered: Dict[str, int] = defaultdict(int)
    type_negs_succeeded: Dict[str, int] = defaultdict(int)
    type_negs_failed: Dict[str, int] = defaultdict(int)
    type_negs_timedout: Dict[str, int] = defaultdict(int)
    type_negs_initiated: Dict[str, int] = defaultdict(int)
    type_contracts_concluded: Dict[str, int] = defaultdict(int)
    type_contracts_signed: Dict[str, int] = defaultdict(int)
    type_neg_requests_rejected: Dict[str, int] = defaultdict(int)
    type_contracts_dropped: Dict[str, int] = defaultdict(int)
    type_breaches_received: Dict[str, int] = defaultdict(int)
    type_breaches_committed: Dict[str, int] = defaultdict(int)
    type_contracts_erred: Dict[str, int] = defaultdict(int)
    type_contracts_nullified: Dict[str, int] = defaultdict(int)
    type_contracts_executed: Dict[str, int] = defaultdict(int)
    type_contracts_breached: Dict[str, int] = defaultdict(int)
    n_negotiator_exceptions: int = 0
    n_agents_timed = 0
    mean_agent_time: float = 0.0
    n_agent_exceptions: int = 0
    n_neg_requests_sent: int = 0
    n_neg_requests_received: int = 0
    n_negs_registered: int = 0
    n_negs_succeeded: int = 0
    n_negs_failed: int = 0
    n_negs_timedout: int = 0
    n_negs_initiated: int = 0
    n_contracts_concluded: int = 0
    n_contracts_signed: int = 0
    n_neg_requests_rejected: int = 0
    n_contracts_dropped: int = 0
    n_breaches_received: int = 0
    n_breaches_committed: int = 0
    n_contracts_erred: int = 0
    n_contracts_nullified: int = 0
    n_contracts_executed: int = 0
    n_contracts_breached: int = 0

    attempts_file = None
    running_file = None

    already_done, results_path = False, None
    run_path = _path(worlds_params[0]["__dir_name"]).parent
    results_path = run_path / RESULTS_FILE
    if results_path.exists():
        try:
            results = load(results_path)
            scores = score_adapter(scores_data=results)
            world_stats = WorldSetRunStats.from_records(results["world_stats"])
            type_stats = AgentStats.from_records(results["type_stats"], "type")
            agent_stats = AgentStats.from_records(results["agent_stats"], "agent")
            already_done = True
        except Exception as e:
            if verbose:
                print(traceback.format_exc())
                print(
                    f"results file found at {str(results_path)} but could not be loaded, will re-run this world."
                    f"\nException: {str(e)}",
                    flush=True,
                )
            already_done = False
    if already_done:
        if verbose:
            print(f"Skipping {str(results_path)}", flush=True)
        for world_params in worlds_params:
            dir_name = world_params["__dir_name"]
            dir_names.append(dir_name)
        return run_id, dir_names, scores, world_stats, type_stats, agent_stats
    attempts_path = run_path / "attempts"
    if attempts_path and not dry_run:
        running_folder = attempts_path / "_running"
        running_folder.mkdir(parents=True, exist_ok=True)
        if len(list(running_folder.glob("run*"))) > 0:
            return run_id, dir_names, None, None, None, None
        running_file = running_folder / f"run{gethostname()}.{current_process().pid}"
        with open(running_file, "w") as rf:
            rf.write(unique_name(f"{gethostname()}.{current_process().pid}", sep="."))
        attempts_file = attempts_path / unique_name(
            f"att_{gethostname()}.{current_process().pid}", sep="."
        )
        # this should be protected and atomic but who cares. If it completely broke down we will just
        # retry unnecessarily to run some worlds.
        n_attempts = len(list(running_folder.glob("att_*")))
        if n_attempts >= max_attempts:
            try:
                os.remove(str(running_file))
            except FileNotFoundError:
                pass
            return run_id, dir_names, None, None, None, None
        n_attempts += 1
        with open(attempts_file, "w") as afile:
            afile.write(str(n_attempts))
    for world_params in worlds_params:
        world_params = world_params.copy()
        dir_name = world_params["__dir_name"]
        dir_names.append(dir_name)
        world_params.pop("__dir_name", None)
        save_videos.append(world_params.get("__save_video", None))
        video_savers.append(world_params.get("__video_saver", None))
        video_saver_params_list.append(world_params.get("__video_saver_params", dict()))
        scoring_context.update(world_params.get("scoring_context", {}))
        world_params.pop("__video_saver", None)
        world_params.pop("__video_saver_params", None)
        world_params.pop("__save_video", None)
        # results_path = _path(dir_name) / RESULTS_FILE
        # if results_path.exists():
        #     already_done = True
        #     try:
        #         results = load(results_path)
        #         scores = results.scores
        #         world_stats = results.world_stats
        #         type_stats = results.type_stats
        #         agent_stats = results.agent_stats
        #         break
        #     except:
        #         world = world_generator(**world_params)
        # else:
        #     world = world_generator(**world_params)
        world = world_generator(**world_params)
        worlds.append(world)
        if dry_run:
            world.save_config(dir_name)
            continue
    # try:
    for (
        world,
        world_params_,
        dir_name,
        save_video,
        video_saver,
        video_saver_params,
    ) in zip(
        worlds,
        worlds_params,
        dir_names,
        save_videos,
        video_savers,
        video_saver_params_list,
    ):
        for _ in range(world.n_steps):
            if not world.step():
                save_stats(world, world.log_folder, params=world_params_)
                break
            if _ % save_progress_every == 0:
                save_stats(world, world.log_folder, params=world_params_)
                # TODO reorganize the code so that the worlds are run in parallel when there are multiple of them
                if not dry_run:
                    scores_ = serialize(
                        score_calculator(worlds, scoring_context, False),
                        add_type_field=False,
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
        # if save_world_stats:
        save_stats(world=world, log_dir=dir_name)
        if save_video:
            if video_saver is None:
                video_saver = World.save_gif
            if video_saver_params is None:
                video_saver_params = {}
            video_saver(world, **video_saver_params)

    scores = score_calculator(worlds, scoring_context, dry_run)
    for w in worlds:
        for aid, agent in w.agents.items():
            atype = agent.type_name
            neg_requests_sent[aid] += w.neg_requests_sent[aid]
            neg_requests_received[aid] += w.neg_requests_received[aid]
            negs_registered[aid] += w.negs_registered[aid]
            negs_succeeded[aid] += w.negs_succeeded[aid]
            negs_failed[aid] += w.negs_failed[aid]
            negs_timedout[aid] += w.negs_timedout[aid]
            negs_initiated[aid] += w.negs_initiated[aid]
            contracts_concluded[aid] += w.contracts_concluded[aid]
            contracts_signed[aid] += w.contracts_signed[aid]
            neg_requests_rejected[aid] += w.neg_requests_rejected[aid]
            contracts_dropped[aid] += w.contracts_dropped[aid]
            breaches_received[aid] += w.breaches_received[aid]
            breaches_committed[aid] += w.breaches_committed[aid]
            contracts_erred[aid] += w.contracts_erred[aid]
            contracts_nullified[aid] += w.contracts_nullified[aid]
            contracts_executed[aid] += w.contracts_executed[aid]
            contracts_breached[aid] += w.contracts_breached[aid]
            type_neg_requests_sent[atype] += w.neg_requests_sent[aid]
            type_neg_requests_received[atype] += w.neg_requests_received[aid]
            type_negs_registered[atype] += w.negs_registered[aid]
            type_negs_succeeded[atype] += w.negs_succeeded[aid]
            type_negs_failed[atype] += w.negs_failed[aid]
            type_negs_timedout[atype] += w.negs_timedout[aid]
            type_negs_initiated[atype] += w.negs_initiated[aid]
            type_contracts_concluded[atype] += w.contracts_concluded[aid]
            type_contracts_signed[atype] += w.contracts_signed[aid]
            type_neg_requests_rejected[atype] += w.neg_requests_rejected[aid]
            type_contracts_dropped[atype] += w.contracts_dropped[aid]
            type_breaches_received[atype] += w.breaches_received[aid]
            type_breaches_committed[atype] += w.breaches_committed[aid]
            type_contracts_erred[atype] += w.contracts_erred[aid]
            type_contracts_nullified[atype] += w.contracts_nullified[aid]
            type_contracts_executed[atype] += w.contracts_executed[aid]
            type_contracts_breached[atype] += w.contracts_breached[aid]
            n_neg_requests_sent += w.neg_requests_sent[aid]
            n_neg_requests_received += w.neg_requests_received[aid]
            n_negs_registered += w.negs_registered[aid]
            n_negs_succeeded += w.negs_succeeded[aid]
            n_negs_failed += w.negs_failed[aid]
            n_negs_timedout += w.negs_timedout[aid]
            n_negs_initiated += w.negs_initiated[aid]
            n_contracts_concluded += w.contracts_concluded[aid]
            n_contracts_signed += w.contracts_signed[aid]
            n_neg_requests_rejected += w.neg_requests_rejected[aid]
            n_contracts_dropped += w.contracts_dropped[aid]
            n_breaches_received += w.breaches_received[aid]
            n_breaches_committed += w.breaches_committed[aid]
            n_contracts_erred += w.contracts_erred[aid]
            n_contracts_nullified += w.contracts_nullified[aid]
            n_contracts_executed += w.contracts_executed[aid]
            n_contracts_breached += w.contracts_breached[aid]

    for w in worlds:
        for aid, v in w.agent_exceptions.items():
            if v:
                agent_exceptions[aid] += v
                type_agent_exceptions[w.agents[aid].type_name] += v
                n_agent_exceptions += len(v)
    for w in worlds:
        for aid, v in w.negotiator_exceptions.items():
            if v:
                negotiator_exceptions[aid] += v
                type_negotiator_exceptions[w.agents[aid].type_name] += v
                n_negotiator_exceptions += len(v)

    for w in worlds:
        for k, l in w.simulation_exceptions.items():
            if l:
                simulation_exceptions.append((k, l))
    for w in worlds:
        for k, l in w.mechanism_exceptions.items():
            if l:
                mechanism_exceptions.append((k, l))
    for w in worlds:
        for k, l in w.contract_exceptions.items():
            if l:
                contract_exceptions.append((k, l))
    for w in worlds:
        for aid, _ in w.times.items():
            if _:
                agent_times[aid] += _
                type_agent_times[w.agents[aid].type_name] += _
                mean_agent_time = (mean_agent_time * n_agents_timed + _) / (
                    n_agents_timed + 1
                )
                n_agents_timed += 1
    # except Exception as e:
    #     scores = None
    #     print(traceback.format_exc())
    #     print(e)
    #     other_exceptions = [exception2str()]
    # finally:
    world_stats = WorldSetRunStats(
        name=";".join(_.name for _ in worlds),
        planned_n_steps=sum(_.n_steps for _ in worlds),
        executed_n_steps=sum(_.current_step for _ in worlds),
        execution_time=sum(_.frozen_time for _ in worlds),
        simulation_exceptions=simulation_exceptions,
        contract_exceptions=contract_exceptions,
        mechanism_exceptions=mechanism_exceptions,
        other_exceptions=other_exceptions,
        n_agent_exceptions=n_agent_exceptions,
        n_negotiator_exceptions=n_negotiator_exceptions,
        mean_agent_time=mean_agent_time,
        n_neg_requests_sent=n_neg_requests_sent,
        n_neg_requests_received=n_neg_requests_received,
        n_neg_requests_rejected=n_neg_requests_rejected,
        n_negs_registered=n_negs_registered,
        n_negs_succeeded=n_negs_succeeded,
        n_negs_failed=n_negs_failed,
        n_negs_timedout=n_negs_timedout,
        n_negs_initiated=n_negs_initiated,
        n_contracts_concluded=n_contracts_concluded,
        n_contracts_signed=n_contracts_signed,
        n_contracts_dropped=n_contracts_dropped,
        n_breaches_received=n_breaches_received,
        n_breaches_committed=n_breaches_committed,
        n_contracts_erred=n_contracts_erred,
        n_contracts_nullified=n_contracts_nullified,
        n_contracts_breached=n_contracts_breached,
        n_contracts_executed=n_contracts_executed,
    )
    agent_stats = AgentStats(
        exceptions=agent_exceptions,
        negotiator_exceptions=negotiator_exceptions,
        times=agent_times,
        neg_requests_sent=neg_requests_sent,
        neg_requests_received=neg_requests_received,
        negs_registered=negs_registered,
        negs_succeeded=negs_succeeded,
        negs_failed=negs_failed,
        negs_timedout=negs_timedout,
        negs_initiated=negs_initiated,
        contracts_concluded=contracts_concluded,
        contracts_signed=contracts_signed,
        neg_requests_rejected=neg_requests_rejected,
        contracts_dropped=contracts_dropped,
        breaches_received=breaches_received,
        breaches_committed=breaches_committed,
        contracts_erred=contracts_erred,
        contracts_nullified=contracts_nullified,
        contracts_executed=contracts_executed,
        contracts_breached=contracts_breached,
    )
    type_stats = AgentStats(
        exceptions=type_agent_exceptions,
        negotiator_exceptions=type_negotiator_exceptions,
        times=type_agent_times,
        neg_requests_sent=type_neg_requests_sent,
        neg_requests_received=type_neg_requests_received,
        negs_registered=type_negs_registered,
        negs_succeeded=type_negs_succeeded,
        negs_failed=type_negs_failed,
        negs_timedout=type_negs_timedout,
        negs_initiated=type_negs_initiated,
        contracts_concluded=type_contracts_concluded,
        contracts_signed=type_contracts_signed,
        neg_requests_rejected=type_neg_requests_rejected,
        contracts_dropped=type_contracts_dropped,
        breaches_received=type_breaches_received,
        breaches_committed=type_breaches_committed,
        contracts_erred=type_contracts_erred,
        contracts_nullified=type_contracts_nullified,
        contracts_executed=type_contracts_executed,
        contracts_breached=type_contracts_breached,
    )
    if attempts_path:
        if running_file:
            try:
                os.remove(running_file)
            except FileNotFoundError:
                pass
        # if attempts_file:
        #     try:
        #         os.remove(attempts_file)
        #     except FileNotFoundError:
        #         pass
    return run_id, dir_names, scores, world_stats, type_stats, agent_stats


def process_world_run(
    run_id: str,
    results: Optional[WorldRunResults],
    tournament_name: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    """
    Generates a data-frame with the results of this world run

    Args:
        run_id: The ID of this run (should be unique per tournament)
        results: Results of the world run
        tournament_name: tournament name

    Returns:

        A tuple of two items:

        - A list of records containing scores
        - A dict mapping extra-score types to lists of records for this type.

    Remarks:

        The score calculator returns a WorldRunResults object which must contain a scores
        element used for evaluating the agents. It can also return extra_scores that can be used
        to save additional information about agent performance. These are optional and the second
        output of this function will be the processed version of these extra scores if any.

    """
    if results is None:
        return []
    log_files, world_names_ = results.log_file_names, results.world_names
    for log_file in log_files:
        if log_file is not None and pathlib.Path(log_file).exists():
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
    for id_, name_, type_, score in zip(
        results.ids, results.names, results.types, results.scores
    ):
        d = {
            "agent_name": name_,
            "agent_id": id_,
            "agent_type": type_,
            "score": score,
            "log_file": ";".join(log_files),
            "world": ";".join(world_names_),
            "stats_folders": stat_folders,
            "base_stats_folder": base_folder,
            "run_id": run_id,
        }
        scores.append(d)
    if not results.extra_scores:
        return scores, dict()
    for _, records in results.extra_scores.items():
        for record in records:
            record.update({"world": ";".join(world_names_), "run_id": run_id})
    return scores, results.extra_scores


def _get_executor(
    method, verbose, scheduler_ip=None, scheduler_port=None, total_timeout=None
):
    """Returns an exeuctor object which has a submit method to submit calls to run worlds"""
    if method == "dask":
        if not ENABLE_DASK:
            raise RuntimeError(
                f"The library 'dask' is not installed. You can use parallel/serial tournaments but not "
                f"dask/distributed. To enable dask/distribued tournaments run:\n\t>> pip install dask[complete]"
            )
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
        print(f"Will use DASK on {address}")
        return (
            distributed.Client(address=address),
            partial(distributed.as_completed, raise_errors=True, with_results=False),
        )

    fraction = None
    parallelism = method.split(":")
    if len(parallelism) != 1:
        fraction = float(parallelism[-1])
    parallelism = parallelism[0]
    max_workers = fraction if fraction is None else max(1, int(fraction * cpu_count()))
    executor = futures.ProcessPoolExecutor(max_workers=max_workers)

    return executor, partial(futures.as_completed)


def _submit_all(
    executor,
    assigned,
    run_ids,
    world_generator,
    score_calculator,
    world_progress_callback,
    override_ran_worlds,
    attempts_path,
    verbose,
    max_attempts,
):
    """Submits all processes to be executed by the executor"""
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
                1,
                attempts_path,
                max_attempts,
                verbose,
            )
        )
    if verbose:
        print(
            f"Submitted all processes ",
            end="",
        )
        if len(assigned) > 0:
            print(f"{len(future_results)/len(assigned):5.2%}")
        else:
            print("")
    return future_results


def save_run_results(
    run_id,
    score_,
    world_stats_,
    type_stats_,
    agent_stats_,
    tournament_progress_callback,
    world_paths,
    name,
    verbose,
    _strt,
    attempts_path,
    n_world_configs,
    i,
):
    if tournament_progress_callback is not None:
        tournament_progress_callback(score_, i, n_world_configs)
    if score_ is None:
        return
    scores, extra_scores = process_world_run(run_id, score_, tournament_name=name)
    type_stats = type_stats_.to_record(run_id, "type")
    agent_stats = agent_stats_.to_record(run_id, "agent")
    world_stats = world_stats_.to_record(run_id)
    run_path = _path(world_paths[0]).parent
    for world_path in world_paths:
        world_path = _path(world_path)
        results_file = run_path / RESULTS_FILE
        all_results = dict(
            run_id=run_id,
            name=name,
            world_paths=";".join(world_paths),
            scores=scores,
            type_stats=type_stats,
            agent_stats=agent_stats,
            world_stats=world_stats,
            extra_scores=extra_scores,
        )
        dump(all_results, results_file, sort_keys=False)
    if verbose:
        _duration = time.perf_counter() - _strt
        print(
            f"{i + 1:003} of {n_world_configs:003} [{100 * (i + 1) / n_world_configs:0.3}%] "
            f'{"completed"} in '
            f"{humanize_time(_duration)}"
            f" [ETA {humanize_time(_duration * n_world_configs / (i + 1))}]"
        )
    if attempts_path:
        if (attempts_path / run_id).exists():
            try:
                if (attempts_path / run_id).exists():
                    os.remove(attempts_path / run_id)
            except Exception as e:
                print(f"Failed to remove an attempt file after completion: {e} ")


def _run_parallel(
    parallelism,
    scheduler_ip,
    scheduler_port,
    verbose,
    assigned,
    world_generator,
    tournament_progress_callback,
    world_progress_callback,
    n_worlds,
    name,
    score_calculator,
    save_world_stats,
    scores_file,
    world_stats_file,
    type_stats_file,
    agent_stats_file,
    run_ids,
    print_exceptions,
    override_ran_worlds=False,
    attempts_path=None,
    total_timeout=None,
    max_attempts=float("inf"),
) -> None:
    """Runs the tournament in parallel"""
    strt = time.perf_counter()
    executor, as_completed = _get_executor(
        parallelism,
        verbose,
        total_timeout=total_timeout,
        scheduler_ip=scheduler_ip,
        scheduler_port=scheduler_port,
    )
    future_results = _submit_all(
        executor,
        assigned,
        run_ids,
        world_generator,
        score_calculator,
        world_progress_callback,
        override_ran_worlds,
        attempts_path,
        verbose,
        max_attempts,
    )
    n_world_configs = len(future_results)
    _strt = time.perf_counter()
    for i, future in enumerate(as_completed(future_results)):
        if total_timeout is not None and time.perf_counter() - strt > total_timeout:
            break
        try:
            (
                run_id,
                world_paths,
                score_,
                world_stats_,
                type_stats_,
                agent_stats_,
            ) = future.result()
            save_run_results(
                run_id,
                score_,
                world_stats_,
                type_stats_,
                agent_stats_,
                tournament_progress_callback,
                world_paths,
                name,
                verbose,
                _strt,
                attempts_path,
                n_world_configs,
                i,
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
    if parallelism.startswith("parallel"):
        executor.shutdown()


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
    save_video_fraction: float = 0.0,
    forced_logs_fraction: float = 0.0,
    video_params=None,
    video_saver=None,
    max_attempts: int = float("inf"),
    extra_scores_to_use: Optional[str] = None,
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
        max_attempts: The maximum number of times to retry running simulations
        extra_scores_to_use: The type of extra-scores to use. If None normal scores will be used. Only effective if scores is None.
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
            max_attempts=max_attempts,
        )
        return evaluate_tournament(
            tournament_path=final_tournament_path,
            verbose=verbose,
            recursive=round_robin,
            metric=metric,
            extra_scores_to_use=extra_scores_to_use,
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
    max_attempts: int = float("inf"),
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
        max_attempts: The maximum number of attempts to run each simulation. Default is infinite

    """
    tournament_path = _path(tournament_path)
    params = load(tournament_path / PARAMS_FILE)
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

    try:
        assigned = load(tournament_path / ASSIGNED_CONFIGS_PICKLE_FILE)
        if assigned is None or len(assigned) == 0:
            assigned = load(tournament_path / ASSIGNED_CONFIGS_JSON_FILE)
    except:
        assigned = load(tournament_path / ASSIGNED_CONFIGS_JSON_FILE)
    random.shuffle(assigned)

    scores_file = tournament_path / SCORES_FILE
    world_stats_file = tournament_path / WORLD_STATS_FILE
    type_stats_file = tournament_path / TYPE_STATS_FILE
    agent_stats_file = tournament_path / AGENT_STATS_FILE
    run_ids = set()
    # if scores_file.exists():
    #     try
    #         tmp_ = pd.read_csv(scores_file)
    #         if "run_id" in tmp_.columns:
    #             run_ids = set(tmp_["run_id"].values)
    #     except:
    #         pass
    world_paths_ = get_world_paths(assignments=assigned)
    for dir_name_ in world_paths_:
        if not dir_name_:
            continue
        if not (dir_name_.parent / RESULTS_FILE).exists():
            continue
        try:
            results_ = load(dir_name_.parent / RESULTS_FILE)
            run_ids.add(results_["run_id"])
        except:
            continue

    # save and check attempts
    attempts_path = tournament_path / "attempts"
    attempts_path.mkdir(exist_ok=True, parents=True)
    attempts = defaultdict(int)
    files_to_remove = []
    for afile in attempts_path.glob("*"):
        if afile.is_dir():
            continue
        fname = afile.name
        if fname in run_ids:
            files_to_remove.append(afile)
            continue
        try:
            with open(afile, "r") as f:
                try:
                    n_attempts = int(f.read())
                except Exception:
                    n_attempts = 0
        except:
            # This means that the file was there then was removed
            # This happens when another process runs this world. I should
            # just ignore this file and update the run_ids
            for dir_name_ in world_paths_:
                if not dir_name_:
                    continue
                if not (dir_name_.parent / RESULTS_FILE).exists():
                    continue
                try:
                    results_ = load(dir_name_.parent / RESULTS_FILE)
                    run_ids.add(results_["run_id"])
                except:
                    continue
            if fname not in run_ids:
                n_attempts = 0
            else:
                attempts[fname] = n_attempts
        if n_attempts > max_attempts:
            run_ids.add(fname)

    for afile in files_to_remove:
        try:
            os.remove(afile)
        except:
            print(f"Failed to remove {str(afile)}")

    scores_file = str(scores_file)
    dask_options = ("dist", "distributed", "dask", "d")
    multiprocessing_options = ("local", "parallel", "par", "p")
    serial_options = ("none", "serial", "s")
    # serial_timeout_options = ("serial-timeout", "serial_timeout", "t")
    if parallelism is None:
        parallelism = "serial"
    assert (
        total_timeout is None or parallelism not in dask_options
    ), f"Cannot use {parallelism} with a total-timeout"
    assert world_progress_callback is None or parallelism not in dask_options, (
        f"Cannot use {parallelism} with a " f"world callback"
    )

    n_world_configs = len(assigned)
    n_already_done = len(run_ids)
    n_to_run = n_world_configs - n_already_done

    if verbose:
        print(
            f"Will run {n_to_run} of {n_world_configs} "
            f" ({(n_to_run) / n_world_configs if n_world_configs else 0.0})"
            f" simulations ({parallelism})",
            flush=True,
        )
    if n_to_run == 0:
        if verbose:
            print(f"Nothing to run. Returning!!", flush=True)
        return

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
                (
                    run_id,
                    world_paths,
                    score_,
                    world_stats_,
                    type_stats_,
                    agent_stats_,
                ) = _run_worlds(
                    worlds_params=worlds_params,
                    world_generator=world_generator,
                    world_progress_callback=world_progress_callback,
                    score_calculator=score_calculator,
                    dry_run=False,
                    save_world_stats=True,
                    override_ran_worlds=override_ran_worlds,
                    save_progress_every=1,
                    attempts_path=attempts_path,
                    max_attempts=max_attempts,
                    verbose=verbose,
                )
                save_run_results(
                    run_id,
                    score_,
                    world_stats_,
                    type_stats_,
                    agent_stats_,
                    tournament_progress_callback,
                    world_paths,
                    name,
                    verbose,
                    strt,
                    attempts_path,
                    n_world_configs,
                    i,
                )
            except Exception as e:
                if tournament_progress_callback is not None:
                    tournament_progress_callback(None, i, n_world_configs)
                if print_exceptions:
                    print(traceback.format_exc())
                    print(e)
    elif any(parallelism.startswith(_) for _ in multiprocessing_options) or (
        parallelism in dask_options
    ):
        _run_parallel(
            parallelism,
            scheduler_ip,
            scheduler_port,
            verbose,
            assigned,
            world_generator,
            tournament_progress_callback,
            world_progress_callback,
            n_world_configs,
            name,
            score_calculator,
            True,
            scores_file,
            world_stats_file,
            type_stats_file,
            agent_stats_file,
            run_ids,
            print_exceptions,
            override_ran_worlds,
            attempts_path,
            total_timeout,
            max_attempts,
        )
    if verbose:
        print(f"Tournament completed successfully")


def _run_id(config_set):
    names = [c["world_params"]["name"] for c in config_set]
    if len(names) == 1:
        return names[0] + _hash(config_set)[:6]
    return names[0] + _hash(names[1:])[:8] + _hash(config_set)[:6]


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

    # original_tournament_path = base_tournament_path
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
    dump(params, tournament_path / PARAMS_FILE)

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
    for i, cs in enumerate(configs):
        for c in cs:
            c["config_id"] = f"{i:04d}" + unique_name(
                "", add_time=False, sep="", rand_digits=2
            )
            c["world_params"]["name"] = c["config_id"]
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
        effective_names = [
            a + _hash(b)[:4] if b else a for a, b in effective_competitor_infos
        ]
        effective_names = shortest_unique_names(effective_names, max_compression=True)
        if verbose:
            print(
                f"Running {'|'.join(effective_competitors)} together ({'|'.join(effective_names)})"
            )
        myconfigs = copy.deepcopy(configs)
        for conf in myconfigs:
            for c in conf:
                c["world_params"]["name"] += (
                    "_"
                    + "-".join(effective_names)
                    + unique_name("", add_time=False, rand_digits=4, sep="")
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
        for i, config_set in enumerate(this_assigned):
            for c in config_set:
                c["world_params"]["name"] += f".{i:02d}"
        assigned += this_assigned

    for config_set in assigned:
        run_id = _run_id(config_set)
        for c in config_set:
            c["world_params"].update(
                {
                    "log_folder": str(
                        (
                            tournament_path / run_id / c["world_params"]["name"]
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

    dump(params, tournament_path / PARAMS_FILE)

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
                    cpy["world_params"]["name"] += f"_{r+1}"
                    if cpy["world_params"]["log_folder"]:
                        cpy["world_params"]["log_folder"] += f"_{r+1}"
                    all_assigned[-1].append(cpy)
        del assigned
        assigned = all_assigned
        assert n_before_duplication * n_runs_per_world == len(assigned), (
            f"Got {len(assigned)} assigned worlds for {n_before_duplication} "
            f"initial set with {n_runs_per_world} runs/world"
        )

    for config_set in assigned:
        run_id = _run_id(config_set)
        for config in config_set:
            dir_name = tournament_path / run_id / config["world_params"]["name"]
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
            run_id = _run_id(cs)
            for _ in cs:
                for subkey in ("world_params",):
                    if subkey not in _.keys():
                        continue
                    _[subkey].update(
                        dict(
                            compact=False,
                            log_negotiations=True,
                            log_to_file=True,
                            no_logs=False,
                        )
                    )
                    if _[subkey].get("log_folder", None) is None:
                        _[subkey].update(
                            dict(
                                log_folder=str(
                                    (
                                        tournament_path / run_id / _[subkey]["name"]
                                    ).absolute()
                                ),
                            )
                        )

                _.update(
                    dict(
                        compact=False,
                        no_logs=False,
                        log_negotiations=True,
                        log_to_file=True,
                    )
                )
                if _.get("log_folder", None) is None:
                    _.update(
                        dict(
                            log_folder=str(
                                (
                                    tournament_path / run_id / _["world_params"]["name"]
                                ).absolute()
                            ),
                        )
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

    dump(assigned, tournament_path / "assigned_configs")
    dump(assigned, tournament_path / ASSIGNED_CONFIGS_PICKLE_FILE)

    return tournament_path


def compile_results(
    path: Union[str, PathLike, Path],
):
    path = _path(path)
    if not path.exists():
        return
    scores, world_stats, agent_stats, type_stats = [], [], [], []
    extra_scores = defaultdict(list)
    paths = set(get_world_paths(tournament_path=path))
    for d in paths:
        if not d.is_dir():
            continue
        if d.name in ("configs", "attempts"):
            continue
        results_path = d.parent / RESULTS_FILE
        if not results_path.exists():
            continue
        try:
            results = load(results_path)
        except:
            continue
        scores += results["scores"]
        world_stats += results["world_stats"]
        type_stats += results["type_stats"]
        agent_stats += results["agent_stats"]
        for k, v in results["extra_scores"].items():
            extra_scores[k] += v
    combine_tournament_stats(paths, path)
    pd.DataFrame.from_records(scores).to_csv(path / SCORES_FILE, index=False)
    pd.DataFrame.from_records(world_stats).to_csv(path / WORLD_STATS_FILE, index=False)
    pd.DataFrame.from_records(agent_stats).to_csv(path / AGENT_STATS_FILE, index=False)
    pd.DataFrame.from_records(type_stats).to_csv(path / TYPE_STATS_FILE, index=False)
    for k, v in extra_scores.items():
        pd.DataFrame.from_records(v).to_csv(path / f"{k}.csv", index=False)


def get_world_paths(*, assignments=None, tournament_path=None):
    """Gets all world paths from a tournament path

    Args:
        assignments: A list of list of world configs
        tournament_path: A path from which to get the assignments.

    Remarks:

        - You must pass assignments xor tournament_path.

    """
    world_paths = set()
    if assignments is None:
        try:
            assignments = load(tournament_path / ASSIGNED_CONFIGS_PICKLE_FILE)
            if assignments is None or len(assignments) == 0:
                assignments = load(tournament_path / ASSIGNED_CONFIGS_JSON_FILE)
        except:
            assignments = load(tournament_path / ASSIGNED_CONFIGS_JSON_FILE)
    for a in assignments:
        for w in a:
            # dir_name = w["world_params"]["log_folder"]
            dir_name = w["__dir_name"]
            world_paths.add(_path(dir_name))
    return world_paths


def evaluate_tournament(
    tournament_path: Optional[Union[str, PathLike, Path]],
    scores: Optional[pd.DataFrame] = None,
    stats: Optional[pd.DataFrame] = None,
    world_stats: Optional[pd.DataFrame] = None,
    type_stats: Optional[pd.DataFrame] = None,
    agent_stats: Optional[pd.DataFrame] = None,
    metric: Union[str, Callable[[pd.DataFrame], float]] = "mean",
    verbose: bool = False,
    recursive: bool = True,
    extra_scores_to_use: Optional[str] = None,
    compile: bool = True,
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
        type_stats: Optionally the aggregate stats collected in `AgentStats` for each agent type
        agent_stats: Optionally the aggregate stats collected in `AgentStats` for each agent instance
        metric: The metric used for evaluation. Possibilities are: mean, median, std, var, sum or a callable that
                receives a pandas data-frame and returns a float.
        verbose: If true, the winners will be printed
        recursive: If true, ALL scores.csv files in all subdirectories of the given tournament_path
                   will be combined
        extra_scores_to_use: The type of extra-scores to use. If None normal scores will be used. Only effective if scores is None.
        compile: Takes effect only if `tournament_path` is not None. If true, the results will be recompiled
                         from individual world results. This is accurate but slow. If false, it will be assumed that
                         all results are already compiled.
        # independent_test: True if you want an independent t-test

    Returns:

    """
    params, world_stats = None, None
    if tournament_path is not None:
        tournament_path = _path(tournament_path)
        tournament_path = tournament_path.absolute()
        tournament_path.mkdir(parents=True, exist_ok=True)
        if compile:
            if verbose:
                print("Compiling results from individual world runs")
            compile_results(tournament_path)
        scores_file = str(
            tournament_path / SCORES_FILE
            if extra_scores_to_use is None
            else f"{extra_scores_to_use}.csv"
        )
        world_stats_file = tournament_path / WORLD_STATS_FILE
        type_stats_file = tournament_path / TYPE_STATS_FILE
        agent_stats_file = tournament_path / AGENT_STATS_FILE
        params_file = tournament_path / PARAMS_FILE
        if world_stats is None and world_stats_file.exists():
            world_stats = pd.read_csv(world_stats_file, index_col=None)
        if type_stats is None and type_stats_file.exists():
            type_stats = pd.read_csv(type_stats_file, index_col=None)
        if agent_stats is None and agent_stats_file.exists():
            agent_stats = pd.read_csv(agent_stats_file, index_col=None)
        if params_file.exists():
            params = load(str(params_file))
        if scores is None:
            if recursive:
                scores = combine_tournament_results(
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
            type_stats=type_stats,
            agent_stats=agent_stats,
        )
    if verbose:
        print("Calculating Scores")
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

    if verbose:
        print("Running statistical tests")
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
        print(f"Saving results")

    agg_stats = pd.DataFrame()
    if tournament_path is not None:
        tournament_path = pathlib.Path(tournament_path)
        scores.to_csv(str(tournament_path / SCORES_FILE), index_label="index")
        total_scores.to_csv(
            str(tournament_path / TOTAL_SCORES_FILE), index_label="index"
        )
        winner_table.to_csv(str(tournament_path / WINNERS_FILE), index_label="index")
        score_stats.to_csv(str(tournament_path / SCORES_STATS_FILE), index=False)
        ttest_results = pd.DataFrame(data=ttest_results)
        ttest_results.to_csv(str(tournament_path / T_STATS_FILE), index_label="index")
        ks_results = pd.DataFrame(data=ks_results)
        ks_results.to_csv(str(tournament_path / K_STATS_FILE), index_label="index")
        if stats is not None and len(stats) > 0:
            stats.to_csv(str(tournament_path / STATS_FILE), index=False)
            agg_stats = _combine_stats(stats)
            agg_stats.to_csv(str(tournament_path / AGGREGATE_STATS_FILE), index=False)

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
        agent_stats=agent_stats,
        type_stats=type_stats,
    )


def combine_tournaments(
    sources: Iterable[Union[str, PathLike]],
    dest: Union[str, PathLike] = None,
    verbose=False,
) -> Tuple[int, int]:
    """
    Combines contents of several tournament runs in the destination path
    allowing for continuation of the tournament

    Returns:
        Tuple[int, int] The number of base configs and assigned configs combined
    """
    assignments = []
    configs = []
    for src in sources:
        src = _path(src)
        for filename in src.glob("**/assigned_configs.pickle"):
            try:
                if verbose:
                    print(f"{filename.parent} ", end="")
                a, c = load(filename), load(filename.parent / "base_configs.json")
            except:
                if verbose:
                    print("FAILED.")
                continue
            else:
                assignments += a
                configs += c
                if verbose:
                    print(f"=> {len(c)} base, {len(a)} assigned configs.")
    if len(configs) == 0:
        return len(configs), len(assignments)
    dest = _path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    dump(configs, dest / "base_configs.json")
    dump(assignments, dest / ASSIGNED_CONFIGS_PICKLE_FILE)
    dump(assignments, dest / ASSIGNED_CONFIGS_JSON_FILE)
    if verbose:
        print(f"=> {len(configs)} base, {len(assignments)} assigned configs.")
    return len(configs), len(assignments)


def combine_tournament_results(
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
        scores.to_csv(str(_path(dest) / SCORES_FILE), index=False)
    return scores


def extract_basic_stats(filename):
    """Adjusts world statistics collected during world execution"""
    data = load(filename)
    if data is None or len(data) == 0:
        return None
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
    return data


def combine_tournament_stats(
    sources: Iterable[Union[str, PathLike]],
    dest: Union[str, PathLike] = None,
    verbose=False,
) -> pd.DataFrame:
    """Combines statistical results of several tournament runs in the destination path."""
    stats = []
    for src in sources:
        src = _path(src)
        for filename in src.glob(f"**/{STATS_FILE}"):
            # try:
            data = extract_basic_stats(filename)
            if data is None:
                continue
            stats.append(data)
    if len(stats) < 1:
        if verbose:
            print("No stats found")
        return pd.DataFrame()
    stats: pd.DataFrame = pd.concat(stats, axis=0, ignore_index=True, sort=True)
    if dest is not None:
        stats.to_csv(str(_path(dest) / STATS_FILE), index=False)
        combined = _combine_stats(stats)
        combined.to_csv(str(_path(dest) / AGGREGATE_STATS_FILE), index=False)
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
