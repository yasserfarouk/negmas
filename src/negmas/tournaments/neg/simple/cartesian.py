"""
Negotiation tournaments module.
"""

from __future__ import annotations
from rich import print
import sys
import shutil
import datetime
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
import traceback
from concurrent.futures.process import BrokenProcessPool
from itertools import product
from math import exp, log, isinf
from os import cpu_count
from pathlib import Path
from random import randint, random, shuffle
from time import perf_counter
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import pandas as pd
from attr import asdict, define
from rich.progress import track
from negmas.common import TraceElement

from negmas.helpers import unique_name
from negmas.helpers.inout import dump, has_needed_files, load
from negmas.helpers.strings import humanize_time, shortest_unique_names
from negmas.helpers.types import get_class, get_full_type_name
from negmas.inout import Scenario, scenario_size
from negmas.mechanisms import Mechanism, Traceable
from negmas.negotiators import Negotiator
from negmas.plots.util import plot_offline_run
from negmas.preferences.ops import (
    ScenarioStats,
    calc_outcome_distances,
    calc_outcome_optimality,
    calc_scenario_stats,
    estimate_max_dist,
)
from negmas.sao.common import SAOState
from negmas.sao.mechanism import SAOMechanism
from negmas.serialization import serialize, to_flat_dict, PYTHON_CLASS_IDENTIFIER
import signal
import os
import time

__all__ = [
    "run_negotiation",
    "cartesian_tournament",
    "SimpleTournamentResults",
    "combine_tournaments",
]
MAX_TASKS_PER_CHILD = 10
LOG_UNIFORM_LIMIT = 10
TERMINATION_WAIT_TIME = 10.0

EXTENSION = ".csv"
ALL_SCORES_FILE_NAME = f"all_scores{EXTENSION}"
ALL_RESULTS_FILE_NAME = f"details{EXTENSION}"
TYPE_SCORES_FILE_NAME = f"type_scores{EXTENSION}"
FINAL_SCORES_FILE_NAME = f"scores{EXTENSION}"
NEGOTIATOR_BEHAVIOR_DIR_NAME = "negotiator_behavior"
SCENARIOS_DIR_NAME = "scenarios"
PLOTS_DIR_NAME = "plots"
NEGOTIATIONS_DIR_NAME = "negotiations"
RESULTS_DIR_NAME = "results"
TOURNAMENT_COL_NAME = "tournament"
OPTIONAL_COLS = (TOURNAMENT_COL_NAME,)
OPTIMALITY_COLS = (
    "nash_optimality",
    "kalai_optimality",
    "ks_optimality",
    "max_welfare_optimality",
    "pareto_optimality",
)

TOURNAMENT_DIRS = [
    SCENARIOS_DIR_NAME,
    NEGOTIATOR_BEHAVIOR_DIR_NAME,
    PLOTS_DIR_NAME,
    NEGOTIATIONS_DIR_NAME,
    RESULTS_DIR_NAME,
]
TOURNAMENT_FILES = [
    ALL_SCORES_FILE_NAME,
    ALL_RESULTS_FILE_NAME,
    TYPE_SCORES_FILE_NAME,
    FINAL_SCORES_FILE_NAME,
]
MECHANISM_FILE_NAME = "mechanism.json"


@define
class SimpleTournamentResults:
    scores: pd.DataFrame
    """All scores per negotiator"""
    details: pd.DataFrame
    """All negotiation results"""
    scores_summary: pd.DataFrame
    """All score statistics summarized per type"""
    final_scores: pd.DataFrame
    """A list of negotiators and their final scores sorted from highest (winner) to lowest score"""
    path: Path | None = None
    """Location at which the logs are stored"""

    @classmethod
    def from_records(
        cls,
        scores: list[dict[str, Any]] | pd.DataFrame | None = None,
        results: list[dict[str, Any]] | pd.DataFrame | None = None,
        type_scores: pd.DataFrame | None = None,
        final_scores: pd.DataFrame | None = None,
        final_score_stat: tuple[str, str] = ("advantage", "mean"),
        path: Path | None = None,
    ) -> "SimpleTournamentResults":
        """Creates SimpleTournamentResults from records of results

        Args:
            scores: The scores of negotiators in all negotiations (If not given, `results` can be used to calculate it).
            results: Results of all negotiations (If not given, the resulting SimpleTournamentResults object will lack details)
            type_scores: Optionally, type-scores. If not given, it will be calculated from scores
            final_scores: Optionally, final scores. If not given, `final_scoer_stat` will be used to calculate them
            final_score_stat: A tuple of the measure used and the statistic applied to it for calculating final score. See `cartesian_tournament` for more details
            path: The path in which the data for this tournament is stored.

        Raises:
            ValueError: If no scores or results are given

        Returns:
            A new SimpleTournamentResults with the given data
        """
        if scores is None and results is None:
            raise ValueError("Cannot pass both scoers and results as None")
        if scores is None or (
            len(scores) == 0 and results is not None and len(results) > 0
        ):
            rd = (
                results.to_dict("records")
                if isinstance(results, pd.DataFrame)
                else results
            )
            assert rd is not None
            scores = pd.DataFrame.from_records([make_scores(_) for _ in (rd)])
        if results is None:
            results = pd.DataFrame()

        if not isinstance(scores, pd.DataFrame):
            scores_df = pd.DataFrame.from_records(scores)
        else:
            scores_df = scores
        if len(scores_df) > 0:
            if type_scores is None:
                type_scores = pd.DataFrame()
                if scores_df is not None and len(scores_df) > 0:
                    cols = [
                        _
                        for _ in scores_df.columns
                        if _ not in ("scenario", "partners")
                    ]
                    type_scores = (
                        scores_df.loc[:, cols]
                        .groupby("strategy")
                        .describe()
                        .sort_values(final_score_stat, ascending=False)
                    )
            if final_scores is None:
                final = pd.DataFrame()
                if type_scores is not None and len(type_scores) > 0:
                    final = type_scores[final_score_stat]
                    final.name = "score"
                    final = final.reset_index()
            else:
                final = final_scores
            if not isinstance(results, pd.DataFrame):
                details_df = pd.DataFrame.from_records(results)
            else:
                details_df = results
        else:
            details_df = type_scores = final = pd.DataFrame()
        return SimpleTournamentResults(
            scores=scores_df,
            details=details_df,
            scores_summary=type_scores,  # type: ignore
            final_scores=final,
            path=path,
        )

    @classmethod
    def from_result_records(
        cls,
        path: Path,
        verbosity: int = 1,
        final_score_stat: tuple[str, str] = ("advantage", "mean"),
    ) -> "SimpleTournamentResults":
        return cls.combine(
            [Path(path)],
            recursive=False,
            recalc_details=True,
            recalc_scores=True,
            must_have_details=True,
            verbosity=verbosity,
            final_score_stat=final_score_stat,
        )[0]

    @classmethod
    def combine(
        cls,
        paths: Path | Iterable[Path],
        recursive: bool = True,
        recalc_details: bool = True,
        recalc_scores: bool = False,
        must_have_details: bool = False,
        verbosity: int = 1,
        final_score_stat: tuple[str, str] = ("advantage", "mean"),
        add_tournament_column: bool = True,
        complete_only: bool = True,
    ) -> tuple["SimpleTournamentResults", list[Path]]:
        """Combines the results of multiple tournaments stored on disk

        Args:
            paths: Paths to look for results within
            recursive: Check children of given paths recursively
            recalc_details: Recalculate detailed results from the `negotiations` folder
            recalc_scores: Recalculate scores from detailed negotiation results
            must_have_details: Raise an exception if detailed negotiation results cannot be found
            verbosity: Verbosity level
            final_score_stat: Used to calculate the final scores. See `cartesian_tournament` for details.
            add_tournament_column: Add a column called tournament with tournament name in detailed and scores.
            complete_only: If given, only a completed tournament will be used in the combination. The rest are ignored.

        Raises:
            FileNotFoundError: If a needed file is not found

        Returns:
            A newly constructed SimpleTournamentResults with the combined results of all tournaments
        """
        """Loads results from the given paths (recursively if given)"""
        if isinstance(paths, Path):
            paths = [paths]
        assert isinstance(paths, Iterable)
        if complete_only:
            recalc_details = False
            recalc_scores = False
            must_have_details = True

        needed_files: list[tuple[str, str] | str] = []
        if complete_only:
            needed_files += [
                ALL_RESULTS_FILE_NAME,
                ALL_SCORES_FILE_NAME,
                FINAL_SCORES_FILE_NAME,
            ]
        else:
            if recalc_details:
                needed_files.append(RESULTS_DIR_NAME)
            elif must_have_details:
                needed_files.append((ALL_RESULTS_FILE_NAME, RESULTS_DIR_NAME))
            if recalc_scores:
                needed_files.append((ALL_RESULTS_FILE_NAME, RESULTS_DIR_NAME))

        if recursive:
            known_dirs = set(TOURNAMENT_DIRS)
            found_dirs = set()
            for path in track(paths, "Walking Tree") if verbosity else paths:
                for base, dirs, _ in os.walk(path):
                    p = Path(base).absolute()
                    if has_needed_files(p, needed_files):
                        found_dirs.add(p)
                    remaining_dirs = []
                    for d in dirs:
                        if d in known_dirs:
                            continue
                        p = (Path(base) / d).absolute()
                        if has_needed_files(p, needed_files):
                            found_dirs.add(p)
                        else:
                            remaining_dirs.append(d)
                    dirs = remaining_dirs
            paths = list(found_dirs)
        else:
            paths = [_.absolute() for _ in paths if has_needed_files(_, needed_files)]
        if not paths:
            raise FileNotFoundError(
                "None of the given paths has the needed files to reconstruct the results of a tournament"
            )
        results, scores = [], []
        loaded_paths = set()
        path_names = shortest_unique_names([str(_) for _ in paths], sep=os.sep)
        for path, pname in (
            track(zip(paths, path_names), "Reading ... ", total=len(paths))
            if verbosity
            else zip(paths, path_names)
        ):
            if verbosity > 1:
                print(f"Reading {path}")
            if recalc_details or not (path / ALL_RESULTS_FILE_NAME).exists():
                src = path / RESULTS_DIR_NAME
                d = pd.DataFrame.from_records([load(_) for _ in src.glob("*.json")])
            else:
                d = pd.read_csv(path / ALL_RESULTS_FILE_NAME, index_col=0)
            if add_tournament_column:
                d[TOURNAMENT_COL_NAME] = pname
            if must_have_details and len(d) < 1:
                print(
                    f"Cannot find detailed results in {path / ALL_RESULTS_FILE_NAME} and you specified `must_have_details` ... Will ignore it"
                )
                continue
            if recalc_scores or not (path / ALL_SCORES_FILE_NAME).exists():
                if len(d) <= 0:
                    if verbosity:
                        print(
                            f"Failed to calculate scores for {path / ALL_SCORES_FILE_NAME} ... Will ignore it"
                        )
                    continue
                s = pd.DataFrame.from_records(
                    [make_scores(_) for _ in d.to_dict("records")]
                )
            else:
                s = pd.read_csv(path / ALL_SCORES_FILE_NAME, index_col=0)
            if add_tournament_column:
                s[TOURNAMENT_COL_NAME] = pname
            if len(d) > 0:
                loaded_paths.add(path)
                results.append(d)
            if len(s) > 0:
                loaded_paths.add(path)
                scores.append(s)
        if len(scores) < 1:
            raise FileNotFoundError("Cannot find any records or details to use")
        return cls.from_records(
            scores=pd.concat(scores, ignore_index=True),
            results=pd.concat(results, ignore_index=True),
            final_score_stat=final_score_stat,
        ), list(loaded_paths)

    @classmethod
    def load(
        cls, path: Path, must_have_details: bool = False
    ) -> "SimpleTournamentResults":
        """Loads results from the given path"""
        kwargs = dict()
        for k, name, required, header, index_col in (
            ("scores", ALL_SCORES_FILE_NAME, must_have_details, 0, 0),
            ("details", ALL_RESULTS_FILE_NAME, must_have_details, 0, 0),
            ("scores_summary", TYPE_SCORES_FILE_NAME, must_have_details, [0, 1], 0),
            ("final_scores", FINAL_SCORES_FILE_NAME, True, 0, 0),
        ):
            p = path / name
            if p.exists():
                df = pd.read_csv(p, header=header, index_col=index_col)
                # if name == TYPE_SCORES_FILE_NAME:
                #     df = df.reset_index()
                #     df = df.rename(columns=(dict(index="agent_type")))
                kwargs[k] = df

            elif required:
                raise FileNotFoundError(f"{name} not found in {path}")
        return SimpleTournamentResults(**kwargs)

    def save(self, path: Path | None, exist_ok: bool = True) -> None:
        """Save all results to the given path"""
        if path is None:
            path = self.path
        if path is None:
            raise FileNotFoundError(
                "You must pass path to save or have a path in the tournament to save it"
            )
        path = Path(path).absolute()
        path.mkdir(exist_ok=exist_ok, parents=True)
        for df, fname in (
            (self.scores, ALL_SCORES_FILE_NAME),
            (self.details, ALL_RESULTS_FILE_NAME),
            (self.scores_summary, TYPE_SCORES_FILE_NAME),
            (self.final_scores, FINAL_SCORES_FILE_NAME),
        ):
            if df is not None and len(df) > 0:
                df.to_csv(path / fname, index_label="index")


def combine_tournaments(
    srcs: Path | Iterable[Path],
    dst: Path | None = None,
    *,
    recursive: bool = True,
    recalc_details: bool = True,
    recalc_scores: bool = False,
    must_have_details: bool = False,
    verbosity: int = 1,
    final_score_stat: tuple[str, str] = ("advantage", "mean"),
    copy: bool = False,
    rename_scenarios: bool = True,
    rename_short: bool = True,
    add_tournament_folders: bool = True,
    override_existing: bool = False,
    add_tournament_column: bool = True,
    complete_only: bool = False,
) -> SimpleTournamentResults:
    results, paths = SimpleTournamentResults.combine(
        srcs,
        recursive,
        recalc_details,
        recalc_scores,
        must_have_details,
        verbosity,
        final_score_stat,
        add_tournament_column,
        complete_only=complete_only,
    )
    if results and dst is not None:
        results.save(dst)
    if verbosity:
        print("[green]Done Combining Results[/green]")
    if copy and dst is not None:
        dst = Path(dst).absolute()
        if paths:
            for current in TOURNAMENT_DIRS:
                (dst / current).mkdir(exist_ok=True, parents=True)
        for i, path in (
            enumerate(track(paths, "Copying ... ")) if verbosity else enumerate(paths)
        ):
            path = Path(path).absolute()
            tname = path.name
            prefix = f"{tname}_" if not rename_short else f"N{i}"
            for current in TOURNAMENT_DIRS:
                if not (path / current).exists():
                    if verbosity:
                        print(f"[yellow]{current}[/yellow] not found in {path}")
                        continue
                if add_tournament_folders:
                    this_dst = dst / current / tname
                    this_dst.mkdir(exist_ok=True, parents=True)
                else:
                    this_dst = dst / current
                if verbosity > 1:
                    print(f"Copying {this_dst.relative_to(dst)} from {path} ")
                for x in (path / current).glob("*"):
                    try:
                        if x.is_dir():
                            shutil.copytree(
                                x, this_dst / x.name, dirs_exist_ok=override_existing
                            )
                        else:
                            shutil.copy(x, this_dst / x.name)
                    except Exception as e:
                        if verbosity:
                            print(
                                f"[red]Copy Error:[/red] {x} -> {this_dst} Failed ({e})"
                            )

                if rename_scenarios:
                    files = list(this_dst.glob("*"))
                    for p in files:
                        p = p.absolute()
                        pnew = p.parent / (f"{prefix}{p.name}")
                        try:
                            os.rename(p, pnew)
                        except Exception:
                            if verbosity:
                                print(
                                    f"[red]Rename Failed:[/red]{p.parent}: {p.name} -> {pnew.name}"
                                )
                        # shutil.move(p, p.parent / f"{tname}{p.name}")
                    for df in (results.scores, results.details):
                        for col in df.columns:
                            if isinstance(col, int) or "scenario" not in col:
                                continue
                            df[col] = prefix + df[col].astype(str)
                    results.save(dst)

    return results


def oneinint(x: int | tuple[int, int] | None, log_uniform=None) -> int | None:
    """Returns x or a random sample within its values.

    Args:
        x: The value or 2-valued tuple to sample from
        log_uniform: If true samples using a log-uniform distribution instead of a uniform distribution.
                     If `None`, uses a log-uniform distribution if min > 0 and max/min >= 10

    """
    if isinstance(x, tuple):
        if log_uniform is None:
            log_uniform = x[0] > 0 and x[1] / x[0] >= LOG_UNIFORM_LIMIT
        if x[0] == x[-1]:
            return x[0]
        if log_uniform:
            L = [log(_) for _ in x]
            return min(x[1], max(x[0], int(exp(random() * (L[1] - L[0]) + L[0]))))
        return randint(*x)
    return x


def oneinfloat(x: float | tuple[float, float] | None) -> float | None:
    """Returns x or a random sample within its values"""
    if isinstance(x, tuple):
        if x[0] == x[-1]:
            return x[0]
        return x[0] + random() * (x[1] - x[0])
    return x


def _make_mechanism(
    s: Scenario,
    partners: tuple[type[Negotiator]],
    partner_names: tuple[str] | None = None,
    partner_params: tuple[dict[str, Any]] | None = None,
    rep: int = 0,
    path: Path | None = None,
    mechanism_type: type[Mechanism] = SAOMechanism,
    mechanism_params: dict[str, Any] | None = None,
    full_names: bool = True,
    verbosity: int = 0,
    run_id: int | str | None = None,
    annotation: dict[str, Any] | None = None,
    private_infos: tuple[dict[str, Any] | None] | None = None,
    id_reveals_type: bool = False,
    name_reveals_type: bool = True,
    mask_scenario_name: bool = True,
    ignore_exceptions: bool = False,
) -> tuple[Mechanism, dict, Scenario, str | None]:
    """
    Run a single negotiation with fully specified parameters

    Args:
        s: The `Scenario` representing the negotiation (outcome space and preferences).
        partners: The partners running the negotiation in order of addition to the mechanism.
        real_scenario_name: The real name of the scenario (used when saving logs).
        partner_names: Names of partners. Either `None` for defaults or a tuple of the same length as `partners`
        partner_params: Parameters used to create the partners. Either `None` for defaults or a tuple of the same length as `partners`
        rep: The repetition number for this run of the negotiation
        path: A folder to save the logs into. If not given, no logs will be saved.
        mechanism_type: the type of the `Mechanism` to use for this negotiation
        mechanism_params: The parameters used to create the `Mechanism` or `None` for defaults
        full_names: Use full names for partner names (only used if `partner_names` is None)
        verbosity: Verbosity level as an integer
        plot: If true, save a plot of the negotiation (only if `path` is given)
        plot_params: Parameters to pass to the plotting function
        run_id: A unique ID for this run. If not given one is generated based on date and time
        stats: statistics of the scenario. If not given or `path` is `None`, statistics are not saved
        annotation: Common information saved in the mechanism's annotation (accessible by negotiators using `self.nmi.annotation`). `None` for nothing
        private_infos: Private information saved in the negotiator's `private_info` attribute (accessible by negotiators as `self.private_info`). `None` for nothing
        id_reveals_type: Each negotiator ID will reveal its type.
        name_reveals_type: Each negotiator name will reveal its type.


    Returns:
        A dictionary of negotiation results that contains the final state of the negotiation alongside other information
    """
    if path:
        path = Path(path)
        for name in (NEGOTIATIONS_DIR_NAME, PLOTS_DIR_NAME, RESULTS_DIR_NAME):
            (path / name).mkdir(exist_ok=True, parents=True)
    s = copy.deepcopy(s)
    assert s.outcome_space is not None
    real_scenario_name = s.outcome_space.name
    if not run_id:
        run_id = unique_name("run", add_time=False, sep=".")
    run_id = str(run_id)
    effective_scenario_name = real_scenario_name
    if mask_scenario_name:
        effective_scenario_name = run_id
        new_os = type(s.outcome_space)(
            issues=s.outcome_space.issues, name=effective_scenario_name
        )
        for u in s.ufuns:
            s.outcome_space = new_os
        s = Scenario(outcome_space=new_os, ufuns=s.ufuns)

    if mechanism_params is None:
        mechanism_params = dict()
    if annotation is None:
        annotation = dict(rep=rep)
    else:
        annotation["rep"] = rep
    if partner_params is None:
        partner_params = tuple(dict() for _ in partners)  # type: ignore
    if private_infos is None:
        private_infos = tuple(dict() for _ in partners)  # type: ignore
    assert mechanism_params is not None
    assert all(_ is not None for _ in partner_params)  # type: ignore

    def _name(a: Negotiator) -> str:
        name = a.short_type_name if not full_names else a.type_name
        if name is None:
            name = get_full_type_name(type(a))
        return name

    mechanism_params["name"] = effective_scenario_name
    mechanism_params["verbosity"] = verbosity - 1
    mechanism_params["annotation"] = annotation

    m = mechanism_type(outcome_space=s.outcome_space, **mechanism_params)
    complete_names, negotiators, failures = [], [], dict()
    for type_, p, pinfo in zip(partners, partner_params, private_infos):  # type: ignore
        try:
            negotiator = type_(**p, private_info=copy.deepcopy(pinfo))
            name = _name(negotiator)
            complete_names.append(name)
            negotiators.append(negotiator)
            if p:
                name += str(hash(str(p)))
        except Exception as e:
            if ignore_exceptions:
                failures = dict(
                    erred_negotiator=get_full_type_name(type_),
                    error_details=str(e),
                    has_error=True,
                )
                break
            else:
                raise (e)
    if failures:
        return m, failures, s, real_scenario_name

    if not partner_names:
        partner_names = tuple(complete_names)  # type: ignore
    for L_, (negotiator, name, u) in enumerate(
        zip(negotiators, partner_names, s.ufuns)
    ):
        if id_reveals_type:
            negotiator.id = f"{name}@{L_}"
        else:
            negotiator.id = unique_name("n", add_time=False, sep="")
        if name_reveals_type:
            negotiator.name = f"{name}@{L_}"
        else:
            negotiator.name = unique_name("n", add_time=False, sep="")
        complete_names.append(name)
        m.add(negotiator, ufun=copy.deepcopy(u))

    return (m, failures, s, real_scenario_name)


def _make_failure_record(
    state: SAOState,
    s: Scenario,
    param_dump,
    partner_names,
    run_id,
    execution_time,
    real_scenario_name,
    stats,
    mechanism_type,
    mechanism_params,
    partners,
):
    if not partner_names:
        partner_names = [get_full_type_name(_) for _ in partners]
    if all(_ is None for _ in param_dump):
        param_dump = None
    agreement_utils = tuple(u(state.agreement) for u in s.ufuns)
    reservations = tuple(u.reserved_value for u in s.ufuns)
    max_utils = [_.max() for _ in s.ufuns]
    run_record = asdict(state)
    run_record["utilities"] = agreement_utils
    run_record["max_utils"] = max_utils
    run_record["reserved_values"] = reservations
    run_record["partners"] = partner_names
    run_record["params"] = param_dump
    run_record["run_id"] = run_id
    run_record["execution_time"] = execution_time
    run_record["negotiator_names"] = partner_names
    run_record["negotiator_ids"] = partner_names
    run_record["negotiator_types"] = partner_names
    run_record["negotiator_times"] = [float("nan") for _ in partners]
    run_record["n_steps"] = mechanism_params.get("n_steps", float("inf"))
    run_record["time_limit"] = mechanism_params.get("time_limit", float("inf"))
    run_record["pend"] = mechanism_params.get("pend", float("inf"))
    run_record["pend_per_second"] = mechanism_params.get(
        "pend_per_second", float("inf")
    )
    run_record["step_time_limit"] = mechanism_params.get(
        "step_time_limit", float("inf")
    )
    run_record["negotiator_time_limit"] = mechanism_params.get(
        "negotiator_time_limit", float("inf")
    )
    run_record["annotation"] = mechanism_params.get("annotation", dict())
    run_record["scenario"] = real_scenario_name
    run_record["mechanism_name"] = "Unknown"
    run_record["mechanism_type"] = get_full_type_name(mechanism_type)
    run_record["effective_scenario_name"] = s.outcome_space.name
    run_record["running"] = state.running
    run_record["waiting"] = state.waiting
    run_record["started"] = state.started
    run_record["last_step"] = state.step
    run_record["last_time"] = state.time
    run_record["relative_time"] = state.relative_time
    run_record["broken"] = state.broken
    run_record["timedout"] = state.timedout
    run_record["agreement"] = state.agreement
    run_record[RESULTS_DIR_NAME] = state.results
    run_record["n_negotiators"] = state.n_negotiators
    run_record["has_error"] = state.has_error
    run_record["erred_negotiator"] = state.erred_negotiator
    run_record["error_details"] = state.error_details

    if stats is not None:
        dists = calc_outcome_distances(agreement_utils, stats)
        run_record.update(
            to_flat_dict(
                calc_outcome_optimality(dists, stats, estimate_max_dist(s.ufuns))
            )
        )

    return run_record


def _make_record(
    m: Mechanism,
    s: Scenario,
    param_dump,
    partner_names,
    run_id,
    execution_time,
    real_scenario_name,
    stats,
):
    state = m.state
    if all(_ is None for _ in param_dump):
        param_dump = None
    agreement_utils = tuple(u(state.agreement) for u in s.ufuns)
    reservations = tuple(u.reserved_value for u in s.ufuns)
    max_utils = [_.max() for _ in s.ufuns]
    run_record = asdict(state)
    run_record["utilities"] = agreement_utils
    run_record["max_utils"] = max_utils
    run_record["reserved_values"] = reservations
    run_record["partners"] = partner_names
    run_record["params"] = param_dump
    run_record["run_id"] = run_id
    run_record["execution_time"] = execution_time
    run_record["negotiator_names"] = m.negotiator_names
    run_record["negotiator_ids"] = m.negotiator_ids
    run_record["negotiator_types"] = [_.type_name for _ in m.negotiators]
    run_record["negotiator_times"] = [m.negotiator_times[_] for _ in m.negotiator_ids]
    run_record["n_steps"] = m.nmi.n_steps
    run_record["time_limit"] = m.nmi.time_limit
    run_record["pend"] = m.nmi.pend
    run_record["pend_per_second"] = m.nmi.pend_per_second
    run_record["step_time_limit"] = m.nmi.step_time_limit
    run_record["negotiator_time_limit"] = m.nmi.negotiator_time_limit
    run_record["annotation"] = m.nmi.annotation
    run_record["scenario"] = real_scenario_name
    run_record["mechanism_name"] = m.name
    run_record["mechanism_type"] = m.type_name
    run_record["effective_scenario_name"] = s.outcome_space.name
    run_record["running"] = state.running
    run_record["waiting"] = state.waiting
    run_record["started"] = state.started
    run_record["last_step"] = state.step
    run_record["last_time"] = state.time
    run_record["relative_time"] = state.relative_time
    run_record["broken"] = state.broken
    run_record["timedout"] = state.timedout
    run_record["agreement"] = state.agreement
    run_record[RESULTS_DIR_NAME] = state.results
    run_record["n_negotiators"] = state.n_negotiators
    run_record["has_error"] = state.has_error
    run_record["erred_negotiator"] = state.erred_negotiator
    run_record["error_details"] = state.error_details

    if m.nmi.annotation:
        run_record.update(m.nmi.annotation)
    if stats is not None:
        dists = calc_outcome_distances(agreement_utils, stats)
        run_record.update(
            to_flat_dict(
                calc_outcome_optimality(dists, stats, estimate_max_dist(s.ufuns))
            )
        )

    return run_record


def _save_record(
    run_record,
    m: Mechanism,
    partner_names,
    real_scenario_name,
    rep,
    run_id,
    path,
    python_class_identifier=PYTHON_CLASS_IDENTIFIER,
):
    file_name = f"{real_scenario_name}_{'_'.join(partner_names)}_{rep}_{run_id}"
    if not path:
        return

    def save_as_df(data: list[TraceElement] | list[tuple], names, file_name):
        pd.DataFrame(data=data, columns=names).to_csv(file_name, index=False)

    for k, v in m._negotiator_logs.items():
        if not v:
            continue
        if k in m.negotiator_ids:
            k = m._negotiator_map[k].name
        neg_name = path / "logs" / file_name / f"{k}.csv"
        if neg_name.exists():
            print(f"[yellow]{neg_name} already found[/yellow]")
            neg_name = (
                path
                / "logs"
                / file_name
                / unique_name("{k}.csv", sep="", add_time=True)
            )
        neg_name.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame.from_records(v).to_csv(neg_name, index=True, index_label="index")

    full_name = path / NEGOTIATIONS_DIR_NAME / f"{file_name}.csv"
    if full_name.exists():
        print(f"[yellow]{full_name} already found[/yellow]")
        full_name = (
            path / unique_name(NEGOTIATIONS_DIR_NAME, sep="") / f"{file_name}.csv"
        )

    if isinstance(m, Traceable):
        assert hasattr(m, "full_trace")
        save_as_df(
            m.full_trace,
            (
                "time",
                "relative_time",
                "step",
                "negotiator",
                "offer",
                "responses",
                "state",
            ),
            full_name,
        )  # type: ignore
        for i, negotiator in enumerate(m.negotiators):
            neg_name = (
                path
                / NEGOTIATOR_BEHAVIOR_DIR_NAME
                / file_name
                / f"{negotiator.name}_at{i}.csv"
            )
            if neg_name.exists():
                print(f"[yellow]{neg_name} already found[/yellow]")
                neg_name = (
                    path
                    / NEGOTIATOR_BEHAVIOR_DIR_NAME
                    / file_name
                    / unique_name(f"{negotiator.name}_at{i}.csv", sep="")
                )
            neg_name.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(m, Traceable):
                save_as_df(
                    m.negotiator_full_trace(negotiator.id),
                    ("time", "relative_time", "step", "offer", "response"),
                    neg_name,
                )  # type: ignore
    else:
        pd.DataFrame.from_records(
            serialize(m.history, python_class_identifier=python_class_identifier)
        ).to_csv(full_name, index=False)
    full_name = path / RESULTS_DIR_NAME / f"{file_name}.json"
    if full_name.exists():
        print(f"[yellow]{full_name} already found[/yellow]")
        full_name = path / RESULTS_DIR_NAME / unique_name(f"{file_name}.json", sep="")
    dump(run_record, full_name)


def _plot_run(
    m, partner_names, real_scenario_name, rep, run_id, path, plot, plot_params
):
    file_name = f"{real_scenario_name}_{'_'.join(partner_names)}_{rep}_{run_id}"
    if not path or not plot:
        return
    if plot_params is None:
        plot_params = dict()
    plot_params["save_fig"] = (True,)
    full_name = path / PLOTS_DIR_NAME / f"{file_name}.png"
    m.plot(path=path, fig_name=full_name, **plot_params)
    try:
        plt.close(plt.gcf())
    except Exception:
        pass


def run_negotiation(
    s: Scenario,
    partners: tuple[type[Negotiator]],
    partner_names: tuple[str] | None = None,
    partner_params: tuple[dict[str, Any]] | None = None,
    rep: int = 0,
    path: Path | None = None,
    mechanism_type: type[Mechanism] = SAOMechanism,
    mechanism_params: dict[str, Any] | None = None,
    full_names: bool = True,
    verbosity: int = 0,
    plot=False,
    plot_params: dict[str, Any] | None = None,
    run_id: int | str | None = None,
    stats: ScenarioStats | None = None,
    annotation: dict[str, Any] | None = None,
    private_infos: tuple[dict[str, Any] | None] | None = None,
    id_reveals_type: bool = False,
    name_reveals_type: bool = True,
    mask_scenario_name: bool = True,
    ignore_exceptions: bool = False,
) -> dict[str, Any]:
    """
    Run a single negotiation with fully specified parameters

    Args:
        s: The `Scenario` representing the negotiation (outcome space and preferences).
        partners: The partners running the negotiation in order of addition to the mechanism.
        real_scenario_name: The real name of the scenario (used when saving logs).
        partner_names: Names of partners. Either `None` for defaults or a tuple of the same length as `partners`
        partner_params: Parameters used to create the partners. Either `None` for defaults or a tuple of the same length as `partners`
        rep: The repetition number for this run of the negotiation
        path: A folder to save the logs into. If not given, no logs will be saved.
        mechanism_type: the type of the `Mechanism` to use for this negotiation
        mechanism_params: The parameters used to create the `Mechanism` or `None` for defaults
        full_names: Use full names for partner names (only used if `partner_names` is None)
        verbosity: Verbosity level as an integer
        plot: If true, save a plot of the negotiation (only if `path` is given)
        plot_params: Parameters to pass to the plotting function
        run_id: A unique ID for this run. If not given one is generated based on date and time
        stats: statistics of the scenario. If not given or `path` is `None`, statistics are not saved
        annotation: Common information saved in the mechanism's annotation (accessible by negotiators using `self.nmi.annotation`). `None` for nothing
        private_infos: Private information saved in the negotiator's `private_info` attribute (accessible by negotiators as `self.private_info`). `None` for nothing
        id_reveals_type: Each negotiator ID will reveal its type.
        name_reveals_type: Each negotiator name will reveal its type.


    Returns:
        A dictionary of negotiation results that contains the final state of the negotiation alongside other information
    """
    m, failures, s, real_scenario_name = _make_mechanism(
        s=s,
        partners=partners,
        partner_names=partner_names,
        partner_params=partner_params,
        rep=rep,
        path=path,
        mechanism_type=mechanism_type,
        mechanism_params=mechanism_params,
        full_names=full_names,
        run_id=run_id,
        annotation=annotation,
        private_infos=private_infos,
        id_reveals_type=id_reveals_type,
        name_reveals_type=name_reveals_type,
        mask_scenario_name=mask_scenario_name,
        ignore_exceptions=ignore_exceptions,
    )
    reservations = tuple(u.reserved_value for u in s.ufuns)
    if partner_params is None:
        partner_params = tuple(dict() for _ in partners)  # type: ignore
    param_dump = tuple(str(to_flat_dict(_)) if _ else None for _ in partner_params)  # type: ignore
    if failures:
        agreement_utils = reservations
        execution_time = 0.0
        state = SAOState(
            has_error=True,
            error_details=failures["error_details"],
            erred_negotiator=failures["erred_negotiator"],
        )
    else:
        if verbosity > 0:
            print(
                f"{datetime.datetime.now()} {partner_names} on {real_scenario_name} (rep: {rep}): [magenta]started[/magenta]",
                flush=True,
            )
        strt = perf_counter()
        try:
            state = m.run()
        except Exception as e:
            if not ignore_exceptions:
                raise e
            else:
                state = m.state
                state.has_error = True
                state.error_details = str(e)
        execution_time = perf_counter() - strt
        if verbosity > 0:
            agreement_utils = tuple(u(state.agreement) for u in s.ufuns)
            advs = tuple(round(a - b, 3) for a, b in zip(agreement_utils, reservations))
            print(
                f"{datetime.datetime.now()} {partner_names} on {real_scenario_name} (rep: {rep}): {state.agreement} in "
                f"{state.relative_time:4.2%} of allowed steps/time with advantages: "
                f"{advs} "
                f"[green]done[/green] in {humanize_time(execution_time)}",
                flush=True,
            )

    run_record = _make_record(
        m=m,
        s=s,
        param_dump=param_dump,
        partner_names=partner_names,
        run_id=run_id,
        execution_time=execution_time,
        real_scenario_name=real_scenario_name,
        stats=stats,
    )
    _save_record(run_record, m, partner_names, real_scenario_name, rep, run_id, path)
    _plot_run(
        m, partner_names, real_scenario_name, rep, run_id, path, plot, plot_params
    )
    return run_record


def failed_run_record(
    s: Scenario,
    partners: tuple[type[Negotiator]],
    timeout: float,
    partner_names: tuple[str] | None = None,
    partner_params: tuple[dict[str, Any]] | None = None,
    error: str | None = None,
    rep: int = 0,
    path: Path | None = None,
    mechanism_type: type[Mechanism] = SAOMechanism,
    mechanism_params: dict[str, Any] | None = None,
    full_names: bool = True,
    run_id: int | str | None = None,
    annotation: dict[str, Any] | None = None,
    private_infos: tuple[dict[str, Any] | None] | None = None,
    id_reveals_type: bool = False,
    name_reveals_type: bool = True,
    mask_scenario_name: bool = True,
    ignore_exceptions: bool = False,
    stats: ScenarioStats | None = None,
):
    if partner_params is None:
        partner_params = tuple(dict() for _ in partners)  # type: ignore
    param_dump = tuple(str(to_flat_dict(_)) if _ else None for _ in partner_params)  # type: ignore
    execution_time = timeout
    try:
        m, _, s, real_scenario_name = _make_mechanism(
            s=s,
            partners=partners,
            partner_names=partner_names,
            partner_params=partner_params,
            rep=rep,
            path=path,
            mechanism_type=mechanism_type,
            mechanism_params=mechanism_params,
            full_names=full_names,
            run_id=run_id,
            annotation=annotation,
            private_infos=private_infos,
            id_reveals_type=id_reveals_type,
            name_reveals_type=name_reveals_type,
            mask_scenario_name=mask_scenario_name,
            ignore_exceptions=ignore_exceptions,
        )
        state = m.state
        state.has_error = True
        state.timedout = True
        state.started = True
        state.error_details = f"Timedout after {timeout} with error {error}"

        run_record = _make_record(
            m=m,
            s=s,
            param_dump=param_dump,
            partner_names=partner_names,
            run_id=run_id,
            execution_time=execution_time,
            real_scenario_name=real_scenario_name,
            stats=stats,
        )
    except Exception as e:
        real_scenario_name = s.outcome_space.name
        m = SAOMechanism()
        state = SAOState()
        state.has_error = True
        state.timedout = True
        state.started = True
        state.error_details = (
            f"Timedout after {timeout} with exception {error} then Raised {e}"
        )
        run_record = _make_failure_record(
            state=state,
            s=s,
            param_dump=param_dump,
            partner_names=partner_names,
            run_id=run_id,
            execution_time=execution_time,
            real_scenario_name=real_scenario_name,
            stats=stats,
            mechanism_type=mechanism_type,
            mechanism_params=mechanism_params,
            partners=partners,
        )
    _save_record(run_record, m, partner_names, real_scenario_name, rep, run_id, path)
    return run_record


# def _stop_process_pool(executor):
#     try:
#         if executor and executor._processes:
#             for _, process in executor._processes.items():
#                 process.terminate()
#         if executor:
#             executor.shutdown(wait=False)
#             executor.cancel_pending_futures()
#             executor.shutdown(wait=False)
#     except Exception:
#         pass


def make_scores(record: dict[str, Any]) -> list[dict[str, float]]:
    utils, partners = record["utilities"], record["partners"]
    reserved_values = record["reserved_values"]
    negids = record["negotiator_ids"]
    max_utils, times = (
        record["max_utils"],
        record.get("negotiator_times", [None] * len(utils)),
    )
    has_error = record["has_error"]
    erred_negotiator = record["erred_negotiator"]
    error_details = record["error_details"]
    mech_error = has_error and not erred_negotiator
    scores = []
    for i, (u, r, a, m, t, nid) in enumerate(
        zip(utils, reserved_values, partners, max_utils, times, negids)
    ):
        n_p = len(partners)
        bilateral = n_p == 2
        basic = dict(
            strategy=a,
            utility=u,
            reserved_value=r,
            advantage=(u - r) / (m - r),
            partner_welfare=sum(_ for j, _ in enumerate(utils) if j != i) / (n_p - 1),
            welfare=sum(_ for _ in utils) / n_p,
            scenario=record["scenario"],
            partners=(partners[_] for _ in range(len(partners)) if _ != i)
            if not bilateral
            else partners[1 - i],
            time=t,
            negotiator_id=nid,
            has_error=has_error,
            self_error=has_error and not mech_error and (erred_negotiator == nid),
            mechanism_error=mech_error,
            error_details=error_details,
            mechanism_name=record.get("mechanism_name", ""),
        )
        for col in OPTIONAL_COLS:
            if col in record:
                basic[col] = record[col]
        for c in OPTIMALITY_COLS:
            if c in record:
                basic[c] = record[c]
        scores.append(basic)
    return scores


def cartesian_tournament(
    competitors: list[type[Negotiator] | str] | tuple[type[Negotiator] | str, ...],
    scenarios: list[Scenario] | tuple[Scenario, ...],
    private_infos: list[None | tuple[dict, ...]] | None = None,
    competitor_params: Sequence[dict | None] | None = None,
    rotate_ufuns: bool = True,
    rotate_private_infos: bool = True,
    n_repetitions: int = 1,
    path: Path | None = None,
    njobs: int = 0,
    mechanism_type: type[Mechanism] = SAOMechanism,
    mechanism_params: dict[str, Any] | None = None,
    n_steps: int | tuple[int, int] | None = 100,
    time_limit: float | tuple[float, float] | None = None,
    pend: float | tuple[float, float] = 0.0,
    pend_per_second: float | tuple[float, float] = 0.0,
    step_time_limit: float | tuple[float, float] | None = None,
    negotiator_time_limit: float | tuple[float, float] | None = None,
    hidden_time_limit: float | tuple[float, float] | None = None,
    external_timeout: int | None = None,
    # full_names: bool = True,
    plot_fraction: float = 0.0,
    plot_params: dict[str, Any] | None = None,
    verbosity: int = 1,
    self_play: bool = True,
    randomize_runs: bool = True,
    sort_runs: bool = False,
    save_every: int = 0,
    save_stats: bool = True,
    save_scenario_figs: bool = True,
    final_score: tuple[str, str] = ("advantage", "mean"),
    id_reveals_type: bool = False,
    name_reveals_type: bool = True,
    shorten_names: bool = True,
    raise_exceptions: bool = True,
    mask_scenario_names: bool = True,
    only_failures_on_self_play: bool = False,
    python_class_identifier=PYTHON_CLASS_IDENTIFIER,
) -> SimpleTournamentResults:
    """A simplified version of Cartesian tournaments not using the internal machinay of NegMAS  tournaments

    Args:
        competitors: A tuple of the competing negotiator types.
        scenarios: A tuple of base scenarios to use for the tournament.
        competitor_params: Either None for no-parameters or a tuple of dictionaries with parameters to initialize the competitors (in order).
        private_infos: If given, a list of the same length as scenarios. Each item is a tuple giving the private information to be passed to every negotiator in every scenario.
        rotate_ufuns: If `True`, the ufuns will be rotated over negotiator positions (for bilateral negotiation this leads to two scenarios for each input scenario with reversed ufun order).
        rotate_private_infos: If `True` and `rotate_ufuns` is also `True`, private information will be rotated with the utility functions.
        n_repetitions: Number of times to repeat each scenario/partner combination
        path: Path on disk to save the results and details of this tournament. Pass None to disable logging
        n_jobs: Number of parallel jobs to run. -1 means running serially (useful for debugging) and 0 means using all cores.
        mechanism_type: The mechanism (protocol) used for all negotiations.
        n_steps: Number of steps/rounds allowed for the each negotiation (None for no-limit and a 2-valued tuple for sampling from a range)
        time_limit: Number of seconds allowed for the each negotiation (None for no-limit and a 2-valued tuple for sampling from a range)
        pend: Probability of ending the negotiation every step/round (None for no-limit and a 2-valued tuple for sampling from a range)
        pend_per_second: Probability of ending the negotiation every second (None for no-limit and a 2-valued tuple for sampling from a range)
        step_time_limit: Time limit for every negotiation step (None for no-limit and a 2-valued tuple for sampling from a range)
        negotiator_time_limit: Time limit for all actions of every negotiator (None for no-limit and a 2-valued tuple for sampling from a range)
        hidden_time_limit: Time limit for negotiations that is not known to the negotiators
        external_timeout: A timeout applied directly to reception of results from negotiations in parallel runs only.
        mechanism_params: Parameters of the mechanism (protocol). Usually you need to pass one or more of the following:
                          time_limit (in seconds), n_steps (in rounds), p_ending (probability of ending the negotiation every step).
        plot_fraction: fraction of negotiations for which plots are to be saved (only if `path` is not `None`)
        plot_params: Parameters to pass to the plotting function
        verbosity: Verbosity level (minimum is 0)
        self_play: Allow negotiations in which all partners are of the same type
        only_failures_on_self_play: If given, self-play runs will only be recorded if they fail to reach agreement. This is useful if you want to keep self-play but still penalize strategies for
                                    failing to reach agreements in self-play
        randomize_runs: If `True` negotiations will be run in random order, otherwise each scenario/partner combination will be finished before starting on the next
        save_every: Number of negotiations after which we dump details and scores
        save_stats: Whether to calculate and save extra statistics like pareto_optimality, nash_optimality, kalai-smorodinsky optimality (ks_optimality), kalai_optimality, etc
        save_scenario_figs: Whether to save a png of the scenario represented in the utility domain for every scenario.
        final_score: A tuple of two strings giving the metric used for ordering the negotiators for the final score:
                     First string can be one of the following (advantage, utility,
                     partner_welfare, welfare) or any statistic from the set calculated if `save_stats` is `True`.
                     The second string can be mean, median, min, max, or std. The default is ('advantage', 'mean')
        id_reveals_type: Each negotiator ID will reveal its type.
        name_reveals_type: Each negotiator name will reveal its type.
        shorten_names: If True, shorter versions of names will be used for results
        raise_exceptions: When given, negotiators and mechanisms are allowed to raise exceptions stopping the tournament
        mask_scenario_names: If given, scenario names will be masked so that the negotiators do not know the original scenario name

    Returns:
        A pandas DataFrame with all negotiation results.
    """
    if mechanism_params is None:
        mechanism_params = dict()
    mechanism_params["ignore_negotiator_exceptions"] = not raise_exceptions

    competitors = [get_class(_) for _ in competitors]
    if competitor_params is None:
        competitor_params = [dict() for _ in competitors]
    if private_infos is None:
        private_infos = [tuple(dict() for _ in s.ufuns) for s in scenarios]

    runs = []
    scenarios_path = path if path is None else Path(path) / SCENARIOS_DIR_NAME
    if scenarios_path is not None:
        scenarios_path.mkdir(exist_ok=True, parents=True)
    stats = None

    def shorten(name):
        #        for s in ("Negotiator", "Agent"):
        #            x = name.replace(s, "")
        #            if not x:
        #                return name
        #            name = x
        return name

    if shorten_names:
        competitor_names = [
            shorten(_)
            for _ in shortest_unique_names([get_full_type_name(_) for _ in competitors])
        ]
    else:
        competitor_names = [get_full_type_name(_) for _ in competitors]
    competitor_info = list(
        zip(competitors, competitor_params, competitor_names, strict=True)
    )
    for s, pinfo in zip(scenarios, private_infos):
        pinfolst = list(pinfo) if pinfo else [dict() for _ in s.ufuns]
        n = len(s.ufuns)
        partners_list = list(product(*tuple([competitor_info] * n)))
        if not self_play:
            partners_list = [
                _
                for _ in partners_list
                if len(
                    {
                        str(
                            serialize(
                                p, python_class_identifier=python_class_identifier
                            )
                        )
                        for p in _
                    }
                )
                > 1
            ]

        ufun_sets = [[copy.deepcopy(_) for _ in s.ufuns]]
        pinfo_sets = [pinfo]
        for i, u in enumerate(s.ufuns):
            u.name = f"{i}_{u.name}"
        if rotate_ufuns:
            for _ in range(len(ufun_sets)):
                ufuns = ufun_sets[-1]
                ufun_sets.append([ufuns[-1]] + ufuns[:-1])
                if rotate_private_infos and pinfolst:
                    pinfo_sets.append(tuple([pinfolst[-1]] + pinfolst[:-1]))
                else:
                    pinfo_sets.append(pinfo)

        original_name = s.outcome_space.name
        # original_ufun_names = [_.name for _ in s.ufuns]
        for i, (ufuns, pinfo_tuple) in enumerate(zip(ufun_sets, pinfo_sets)):
            if len(ufun_sets) > 1:
                for j, u in enumerate(ufuns):
                    n = "_".join(u.name.split("_")[1:])
                    u.name = f"{j}_{n}"
                scenario = Scenario(
                    type(s.outcome_space)(
                        issues=s.outcome_space.issues,
                        name=f"{original_name}-{i}" if i else original_name,
                    ),
                    tuple(ufuns),
                )
            else:
                scenario = s
            this_path = None
            if scenarios_path:
                this_path = scenarios_path / str(scenario.outcome_space.name)
                scenario.to_yaml(this_path)
                if save_scenario_figs:
                    plot_offline_run(
                        trace=[],
                        ids=["First", "Second"],
                        ufuns=s.ufuns,  # type: ignore
                        agreement=None,
                        timedout=False,
                        broken=False,
                        has_error=False,
                        names=["First", "Second"],
                        save_fig=True,
                        path=str(this_path),
                        fig_name="fig.png",
                        only2d=True,
                        show_annotations=False,
                        show_agreement=False,
                        show_pareto_distance=False,
                        show_nash_distance=False,
                        show_kalai_distance=False,
                        show_ks_distance=False,
                        show_max_welfare_distance=False,
                        show_max_relative_welfare_distance=False,
                        show_end_reason=False,
                        show_reserved=True,
                        show_total_time=False,
                        show_relative_time=False,
                        show_n_steps=False,
                    )
            plt.close()
            if save_stats:
                stats = calc_scenario_stats(scenario.ufuns)
                if this_path:
                    dump(
                        serialize(
                            stats, python_class_identifier=python_class_identifier
                        ),
                        this_path / "stats.json",
                    )

            mparams = copy.deepcopy(mechanism_params)
            mparams.update(
                dict(
                    n_steps=oneinint(n_steps),
                    time_limit=oneinfloat(time_limit),
                    pend=oneinfloat(pend),
                    pend_per_second=oneinfloat(pend_per_second),
                    negotiator_time_limit=oneinfloat(negotiator_time_limit),
                    step_time_limit=oneinfloat(step_time_limit),
                    hidden_time_limit=oneinfloat(hidden_time_limit),
                )
            )
            if scenarios_path:
                params_path = (
                    scenarios_path
                    / str(scenario.outcome_space.name)
                    / MECHANISM_FILE_NAME
                )
                pdict = dict(type=get_full_type_name(mechanism_type)) | mparams
                dump(pdict, params_path)
            for partners in partners_list:
                runs += [
                    dict(
                        s=scenario,
                        partners=[_[0] for _ in partners],
                        partner_names=[_[2] for _ in partners],
                        partner_params=[_[1] for _ in partners],
                        rep=i,
                        annotation=dict(rep=i, n_repetitions=n_repetitions),
                        path=path if path else None,
                        mechanism_type=mechanism_type,
                        mechanism_params=mparams,
                        full_names=True,
                        verbosity=verbosity - 1,
                        plot=random() < plot_fraction,
                        stats=stats,
                        id_reveals_type=id_reveals_type,
                        name_reveals_type=name_reveals_type,
                        plot_params=plot_params,
                        mask_scenario_name=mask_scenario_names,
                        private_infos=pinfo_tuple,
                    )
                    for i in range(n_repetitions)
                ]
    if randomize_runs:
        shuffle(runs)
    if sort_runs:
        runs = sorted(runs, key=lambda x: scenario_size(x["s"]))
    if verbosity > 0:
        print(
            f"Will run {len(runs)} negotiations on {len(scenarios)} scenarios between {len(competitors)} competitors",
            flush=True,
        )
    results, scores = [], []
    results_path = path if not path else path / ALL_RESULTS_FILE_NAME
    scores_path = path if not path else path / ALL_SCORES_FILE_NAME

    def process_record(record, results=results, scores=scores):
        if self_play and only_failures_on_self_play:
            is_self_play = len(set(record["partners"])) == 1
            if is_self_play and record["agreement"] is not None:
                return results, scores
        results.append(record)
        scores += make_scores(record)
        if results_path and save_every and i % save_every == 0:
            pd.DataFrame.from_records(results).to_csv(results_path, index_label="index")
            pd.DataFrame.from_records(scores).to_csv(scores_path, index_label="index")
        return results, scores

    def get_run_id(info):
        return hash(
            str(serialize(info, python_class_identifier=python_class_identifier))
        )

    if njobs < 0:
        for i, info in enumerate(
            track(runs, total=len(runs), description=NEGOTIATIONS_DIR_NAME)
        ):
            process_record(run_negotiation(**info, run_id=get_run_id(info)))

    else:
        timeout = external_timeout if external_timeout else float("inf")

        def _safe_max(x) -> float:
            if x is None:
                return float("inf")
            if isinstance(x, tuple):
                return x[-1]
            return x

        tparams = dict(
            time_limit=mechanism_params.get("time_limit", float("inf")),
            negotiator_time_limit=mechanism_params.get(
                "negotiator_time_limit", float("inf")
            ),
            step_time_limit=mechanism_params.get("step_time_limit", float("inf")),
            hidden_time_limit=mechanism_params.get("hidden_time_limit", float("inf")),
        ) | dict(
            time_limit=_safe_max(time_limit),
            negotiator_time_limit=_safe_max(negotiator_time_limit),
            step_time_limit=_safe_max(step_time_limit),
            hidden_time_limit=_safe_max(hidden_time_limit),
        )
        touts = [_ * 1.05 for _ in tparams.values() if _ is not None and not isinf(_)]
        timeout = min(max(touts) if touts else float("inf"), timeout)
        if isinf(timeout):
            timeout = None
        if timeout is not None and verbosity > 0:
            print(
                f"[magenta]Will use {timeout} as a timeout when receiving results[/magenta]"
            )

        futures = dict()
        n_cores = cpu_count()
        if n_cores is None:
            n_cores = 4
        cpus = min(n_cores, njobs) if njobs else cpu_count()
        kwargs_ = dict(max_workers=cpus)
        version = sys.version_info
        if version.major > 3 or version.minor > 10:
            kwargs_.update(max_tasks_per_child=MAX_TASKS_PER_CHILD)

        with ProcessPoolExecutor(**kwargs_) as pool:  # type: ignore
            for info in runs:
                futures[
                    pool.submit(run_negotiation, **info, run_id=get_run_id(info))
                ] = info
            for i, f in enumerate(
                track(
                    as_completed(futures),
                    total=len(futures),
                    description=NEGOTIATIONS_DIR_NAME,
                )
            ):
                try:
                    result = f.result(timeout=timeout)
                    process_record(result)
                except TimeoutError:
                    info = futures.get(f, dict(partners=["Unknown", "Unknown"]))
                    print(
                        f"[red]Negotiation between {info['partners']} [bold]timedout[/bold] [red] after {timeout} seconds ...\n\tKilling the process",
                        end="",
                    )
                    if len(info) > 1:
                        result = failed_run_record(**info)
                        process_record(result)

                    f.cancel()
                    try:
                        if os.name == "nt":  # Check if running on Windows
                            pool._processes[f._process_ident].terminate()
                        else:
                            os.kill(
                                f._process_ident,  # type: ignore
                                signal.SIGTERM,
                            )  # Default to SIGTERM
                            time.sleep(
                                TERMINATION_WAIT_TIME
                            )  # Allow brief time for termination
                            if not pool._processes[f._process_ident].is_alive():  # type: ignore
                                os.kill(
                                    f._process_ident,  # type: ignore
                                    signal.SIGKILL,
                                )  # Forceful if needed
                        print("[yellow]SUCCEEDED[/yellow]")
                    except Exception as e:
                        print(f"[red]FAILED[/red] with exception {e}")

                except BrokenProcessPool as e:
                    if verbosity > 1:
                        print("[red]Broken Pool[/red]")
                        print(e)
                    break
                except Exception as e:
                    if verbosity > 1:
                        print("[red]Exception[/red]")
                        if verbosity > 2:
                            print(traceback.format_exc())
                        print(e)
            pool.shutdown(wait=False)
            # _stop_process_pool(pool)

    tresults = SimpleTournamentResults.from_records(
        scores, results, final_score_stat=final_score, path=path
    )
    if verbosity > 0:
        print(tresults.final_scores)
    if path:
        tresults.save(path)
    return tresults


if __name__ == "__main__":
    from random import randint, random

    from negmas.helpers.misc import intin
    from negmas.outcomes import make_issue, make_os
    from negmas.preferences.crisp.linear import LinearAdditiveUtilityFunction as U
    from negmas.preferences.generators import generate_utility_values
    from negmas.preferences.value_fun import TableFun
    from negmas.sao.negotiators import (
        AspirationNegotiator,
        MiCRONegotiator,
        NaiveTitForTatNegotiator,
    )

    n_scenarios, n_outcomes = 5, (10, 100)
    ufun_sets = []

    for i in range(n_scenarios):
        r = random()
        n = intin(n_outcomes, log_uniform=True)
        name = "S"
        if r < 0.3:
            n_pareto = n
            name = "DivideThePieGen"
        else:
            n_pareto = randint(min(5, n // 2), n // 2)
        if r < 0.05:
            vals = generate_utility_values(
                n_pareto, n, n_ufuns=2, pareto_first=False, pareto_generator="zero_sum"
            )
            name = "DivideThePie"
        else:
            vals = generate_utility_values(
                n_pareto,
                n,
                n_ufuns=2,
                pareto_first=False,
                pareto_generator="curve" if random() < 0.5 else "piecewise_linear",
            )

        issues = (make_issue([f"{i}_{n-1 - i}" for i in range(n)], "portions"),)
        ufuns = tuple(
            U(
                values=(
                    TableFun(
                        {_: float(vals[i][k]) for i, _ in enumerate(issues[0].all)}
                    ),
                ),
                name=f"{uname}{i}",
                reserved_value=0.0,
                outcome_space=make_os(issues, name=f"{name}{i}"),
            )
            for k, uname in enumerate(("First", "Second"))
        )
        ufun_sets.append(ufuns)

    scenarios = [
        Scenario(
            outcome_space=ufuns[0].outcome_space,  # type: ignore We are sure this is not None
            ufuns=ufuns,
        )
        for ufuns in ufun_sets
    ]

    cartesian_tournament(
        competitors=(AspirationNegotiator, NaiveTitForTatNegotiator, MiCRONegotiator),
        scenarios=scenarios,
        n_repetitions=1,
    )
