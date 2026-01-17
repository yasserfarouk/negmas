#!/usr/bin/env DOCS_ZIP
"""The NegMAS universal command line tool"""

from __future__ import annotations
from datetime import datetime
import http.server
import json
import os
import pathlib
import socketserver
import sys
import urllib.request
import zipfile
from functools import partial
from pathlib import Path
from time import perf_counter

import click
import click_config_file
import yaml
from rich import print
from tabulate import tabulate

import negmas
from negmas.config import negmas_config
from negmas.genius.common import DEFAULT_JAVA_PORT
from negmas.helpers import humanize_time, unique_name
from negmas.helpers.inout import load as load_file
from negmas.tournaments import (
    combine_tournament_results,
    combine_tournament_stats,
    combine_tournaments,
    create_tournament,
    evaluate_tournament,
    run_tournament,
)

try:
    from .vendor.quick.quick import gui_option  # type: ignore
except Exception:

    def gui_option(x):
        return x


try:
    # disable a warning in yaml 1b1 version
    yaml.warnings({"YAMLLoadWarning": False})
except Exception:
    pass

n_completed = 0
n_total = 0

BASE_WEBSITE = "https://yasserfarouk.github.io/files/"
GENIUS_JAR_NAME = "geniusbridge.jar"
DOCS_ZIP = "negmas_docs.zip"

DEFAULT_NEGOTIATOR = "negmas.sao.AspirationNegotiator"


def default_log_path():
    """Default location for all logs"""

    return Path(negmas_config("log_base", Path.home() / "negmas" / "logs"))  # type: ignore


def default_tournament_path():
    """The default path to store tournament run info"""

    return default_log_path() / "tournaments"


def print_progress(_, i, n) -> None:
    """Prints the progress of a tournament"""
    global n_completed, n_total
    n_completed = i + 1
    n_total = n
    print(
        f"{datetime.now()} {n_completed:04} of {n:04} worlds completed ({n_completed / n:0.2%})",
        flush=True,
    )


def print_world_progress(world) -> None:
    """Prints the progress of a world"""
    step = world.current_step + 1
    s = (
        f"{datetime.now()} World# {n_completed:04}: {step:04}  of {world.n_steps:04} "
        f"steps completed ({step / world.n_steps:0.2f}) "
    )

    if n_total > 0:
        s += f"TOTAL: ({n_completed + step / world.n_steps / n_total:0.2f})"
    print(s, flush=True)


click.option = partial(click.option, show_default=True)


@gui_option
@click.group()
def cli():
    pass


@cli.group(chain=True, invoke_without_command=True)
@click.pass_context
@click.option(
    "--ignore-warnings/--show-warnings",
    default=False,
    help="Ignore/show runtime warnings",
)
def tournament(ctx, ignore_warnings):
    if ignore_warnings:
        import warnings

        warnings.filterwarnings("ignore")
    ctx.obj = {}


@tournament.command(help="Creates a tournament")
@click.option(
    "--name",
    "-n",
    default="random",
    help='The name of the tournament. The special value "random" will result in a random name',
)
@click.option(
    "--timeout",
    "-t",
    default=0,
    type=int,
    help="Timeout the whole tournament after the given number of seconds (0 for infinite)",
)
@click.option(
    "--configs",
    default=5,
    type=int,
    help="Number of unique configurations to generate.",
)
@click.option("--runs", default=2, help="Number of runs for each configuration")
@click.option(
    "--max-runs",
    default=-1,
    type=int,
    help="Maximum total number of runs. Zero or negative numbers mean no limit",
)
@click.option(
    "--steps-min",
    default=50,
    type=int,
    help="Minimum number of steps (only used if --steps was not passed",
)
@click.option(
    "--steps-max",
    default=100,
    type=int,
    help="Maximum number of steps (only used if --steps was not passed",
)
@click.option("--agents", default=3, type=int, help="Number of agents per competitor")
@click.option(
    "--competitors",
    help="A semicolon (;) separated list of agent types to use for the competition.",
)
@click.option(
    "--non-competitors",
    help="A semicolon (;) separated list of agent types to exist in the worlds as non-competitors "
    "(their scores will not be calculated).",
)
@click.option(
    "--log",
    "-l",
    type=click.Path(dir_okay=True, file_okay=False),
    default=default_tournament_path(),
    help="Default location to save logs (A folder will be created under it)",
)
@click.option(
    "--world-config",
    type=click.Path(dir_okay=False, file_okay=True),
    default=tuple(),
    multiple=True,
    help="A file to load extra configuration parameters for world simulations from.",
)
@click.option(
    "--verbosity",
    default=1,
    type=int,
    help="verbosity level (from 0 == silent to 1 == world progress)",
)
@click.option(
    "--log-ufuns/--no-ufun-logs",
    default=False,
    help="Log ufuns into their own CSV file. Only effective if --debug is given",
)
@click.option(
    "--log-negs/--no-neg-logs",
    default=False,
    help="Log all negotiations. Only effective if --debug is given",
)
@click.option(
    "--compact/--debug",
    default=True,
    help="If True, effort is exerted to reduce the memory footprint which"
    "includes reducing logs dramatically.",
)
@click.option(
    "--raise-exceptions/--ignore-exceptions",
    default=True,
    help="Whether to ignore agent exceptions",
)
@click.option(
    "--path",
    default="",
    help="A path to be added to PYTHONPATH in which all competitors are stored. You can path a : separated list of "
    "paths on linux/mac and a ; separated list in windows",
)
@click.option(
    "--config-generator",
    default="",
    help="The full path to a configuration generator function that is used to generate"
    " all configs for the "
    "tournament. MUST be specified",
)
@click.option(
    "--world-generator",
    default="",
    help="The full path to a world generator function that is used to generate all worlds (given the assigned "
    "configs for the tournament. MUST be specified",
)
@click.option(
    "--assigner",
    default="",
    help="The full path to an assigner function that assigns competitors to different configurations",
)
@click.option("--scorer", default="", help="The full path to a scoring function")
@click.option(
    "--cw",
    default=None,
    type=int,
    help="Number of competitors to run at every world simulation. It must "
    "either be left at default or be a number > 1 and < the number "
    "of competitors passed using --competitors",
)
@click_config_file.configuration_option()
@click.pass_context
def create(
    ctx,
    name,
    timeout,
    log,
    verbosity,
    reveal_names,
    runs,
    configs,
    max_runs,
    competitors,
    world_config,
    non_competitors,
    compact,
    agents,
    log_ufuns,
    log_negs,
    raise_exceptions,
    steps_min,
    steps_max,
    path,
    cw,
    world_generator,
    config_generator,
    assigner,
    scorer,
):
    if len(config_generator is None or config_generator.strip()) == 0:  # type: ignore
        print(
            "ERROR: You did not specify a config generator. Use --config-generator to specify one and see the "
            "documentation of the create_tournament method in negmas.situated for details about it."
            "\nThe following must be explicitly specified to create a tournament: a world-generator, "
            "an assigner, a scorer, and a config-generator."
        )

        return -4

    if len(world_generator is None or world_generator.strip()) == 0:  # type: ignore
        print(
            "ERROR: You did not specify a world generator. Use --world-generator to specify one and see the "
            "documentation of the create_tournament method in negmas.situated for details about it."
            "\nThe following must be explicitly specified to create a tournament: a world-generator, "
            "an assigner, a scorer, and a config-generator."
        )

        return -3

    if len(assigner is None or assigner.strip()) == 0:  # type: ignore
        print(
            "ERROR: You did not specify an assigner. Use --assigner to specify one and see the documentation"
            " of the create_tournament method in negmas.situated for details about it."
            "\nThe following must be explicitly specified to create a tournament: a world-generator, "
            "an assigner, a scorer, and a config-generator."
        )

        return -2

    if len(scorer is None or scorer.strip()) == 0:  # type: ignore
        print(
            "ERROR: You did not specify a scorer. Use --scorer to specify one and see the documentation of the "
            "create_tournament method in negmas.situated for details about it."
            "\nThe following must be explicitly specified to create a tournament: a world-generator, "
            "an assigner, a scorer, and a config-generator."
        )

        return -1

    if len(path) > 0:
        sys.path.append(path)
    kwargs = {}

    if world_config is not None and len(world_config) > 0:
        for wc in world_config:
            kwargs.update(load_file(wc))
    warning_n_runs = 2000

    if timeout <= 0:
        timeout = None

    if name == "random":
        name = unique_name(base="", rand_digits=0)
    ctx.obj["tournament_name"] = name

    if max_runs <= 0:
        max_runs = None

    if compact:
        log_ufuns = False

    if not compact:
        if not reveal_names:
            print(
                "You are running the tournament with --debug. Will reveal "
                "agent types in their names"
            )
        reveal_names = True
        verbosity = max(1, verbosity)

    worlds_per_config = (
        None if max_runs is None else int(round(max_runs / (configs * runs)))
    )

    all_competitors = competitors.split(";")
    all_competitors_params = [dict() for _ in range(len(all_competitors))]

    permutation_size = 1
    recommended = runs * configs * permutation_size

    if worlds_per_config is not None and worlds_per_config < 1:
        print(
            f"You need at least {(configs * runs)} runs even with a single permutation of managers."
            f".\n\nSet --max-runs to at least {(configs * runs)} (Recommended {recommended})"
        )

        return

    if max_runs is not None and max_runs < recommended:
        print(
            f"You are running {max_runs} worlds only but it is recommended to set {max_runs} to at least "
            f"{recommended}. Will continue"
        )

    steps = (steps_min, steps_max)

    if worlds_per_config is None:
        len(all_competitors)
        n_worlds = permutation_size * runs * configs

        if n_worlds > warning_n_runs:
            print(
                f"You are running the maximum possible number of permutations for each configuration. This is roughly"
                f" {n_worlds} simulations (each for {steps} steps). That will take a VERY long time."
                f"\n\nYou can reduce the number of simulations by setting --configs>=1 (currently {configs}) or "
                f"--runs>= 1 (currently {runs}) to a lower value. "
                f"\nFinally, you can limit the maximum number of worlds to run by setting --max-runs=integer."
            )
            # if (
            #     not input(f"Are you sure you want to run {n_worlds} simulations?")
            #     .lower()
            #     .startswith("y")
            # ):
            #     exit(0)
            max_runs = int(
                input(
                    f"Input the maximum number of simulations to run. Zero to run all of the {n_worlds} "
                    f"simulations. ^C or a negative number to exit [0 : {n_worlds}]:"
                )
            )

            if max_runs == 0:
                max_runs = None

            if max_runs is not None and max_runs < 0:
                exit(0)
            worlds_per_config = (
                None if max_runs is None else int(round(max_runs / (configs * runs)))
            )

    non_competitor_params = None

    if len(non_competitors) < 1:
        non_competitors = None
    else:
        non_competitors = non_competitors.split(";")

    print(f"Tournament will be run between {len(all_competitors)} agents: ")
    print(all_competitors)
    print("Non-competitors are: ")
    print(non_competitors)
    results = create_tournament(
        competitors=all_competitors,
        competitor_params=all_competitors_params,
        non_competitors=non_competitors,
        non_competitor_params=non_competitor_params,
        agent_names_reveal_type=reveal_names,
        n_competitors_per_world=cw,
        n_configs=configs,
        n_runs_per_world=runs,
        max_worlds_per_config=worlds_per_config,
        base_tournament_path=log,
        total_timeout=timeout,
        name=name,
        verbose=verbosity > 0,
        n_agents_per_competitor=agents,
        world_generator=world_generator,
        config_generator=config_generator,
        config_assigner=assigner,
        score_calculator=scorer,
        compact=compact,
        n_steps=steps,
        log_ufuns=log_ufuns,
        log_negotiations=log_negs,
        ignore_agent_exceptions=not raise_exceptions,
        ignore_contract_execution_exceptions=not raise_exceptions,
        **kwargs,
    )
    results = Path(results)
    ctx.obj["tournament_name"] = results.name
    ctx.obj["tournament_log_folder"] = log
    ctx.obj["compact"] = compact
    print(f"Saved all configs to {str(results)}\nTournament name is {results.name}")


def display_results(results, metric, significance):
    if metric == "truncated_mean":
        print(
            tabulate(
                results.total_scores.sort_values(by="score", ascending=False),
                headers="keys",
                tablefmt="psql",
            )
        )
        print(
            tabulate(
                results.score_stats.sort_values(by="median", ascending=False),
                headers="keys",
                tablefmt="psql",
            )
        )
    else:
        viewmetric = ["50%" if metric == "median" else metric]
        print(
            tabulate(
                results.score_stats.sort_values(by=viewmetric, ascending=False),
                headers="keys",
                tablefmt="psql",
            )
        )

    if significance:
        if metric in ("mean", "sum", "tuncated_mean"):
            print(tabulate(results.ttest, headers="keys", tablefmt="psql"))
        else:
            print(tabulate(results.kstest, headers="keys", tablefmt="psql"))

    try:
        agg_stats = results.agg_stats.loc[
            :,
            [
                "n_negotiations_sum",
                "n_contracts_concluded_sum",
                "n_contracts_signed_sum",
                "n_contracts_executed_sum",
                "activity_level_sum",
            ],
        ]
        agg_stats.columns = [
            "negotiated",
            "concluded",
            "signed",
            "executed",
            "business",
        ]
        print(tabulate(agg_stats.describe(), headers="keys", tablefmt="psql"))
    except Exception:
        pass


@tournament.command(help="Runs/continues a tournament")
@click.option(
    "--name",
    "-n",
    default="",
    help="The name of the tournament. When invoked after create, there is no need to pass it",
)
@click.option(
    "--log",
    "-l",
    type=click.Path(dir_okay=True, file_okay=False),
    default=default_tournament_path(),
    help="Default location to save logs",
)
@click.option(
    "--verbosity",
    default=1,
    type=int,
    help="verbosity level (from 0 == silent to 1 == world progress)",
)
@click.option(
    "--parallel/--serial",
    default=True,
    help="Run a parallel/serial tournament on a single machine",
)
@click.option(
    "--distributed/--single-machine",
    default=False,
    help="Run a distributed tournament using dask",
)
@click.option(
    "--ip",
    default="127.0.0.1",
    help="The IP address for a dask scheduler to run the distributed tournament."
    " Effective only if --distributed",
)
@click.option(
    "--port",
    default=8786,
    type=int,
    help="The IP port number a dask scheduler to run the distributed tournament."
    " Effective only if --distributed",
)
@click.option(
    "--compact/--debug",
    default=True,
    help="If True, effort is exerted to reduce the memory footprint which"
    "includes reducing logs dramatically.",
)
@click.option(
    "--path",
    default="",
    help="A path to be added to PYTHONPATH in which all competitors are stored. You can path a : separated list of "
    "paths on linux/mac and a ; separated list in windows",
)
@click.option(
    "--metric",
    default="truncated_mean",
    type=str,
    help="The statistical metric used for choosing the winners. Possibilities are mean, median, std, var, sum, truncated_mean",
)
@click.option(
    "--significance/--no-significance",
    default=False,
    help="Whether to show significance table",
)
@click.option(
    "--eval/--no-eval",
    default=False,
    help="Whether evaluate and show results after the tournament is run",
)
@click_config_file.configuration_option()
@click.pass_context
def run(
    ctx,
    name,
    verbosity,
    parallel,
    distributed,
    ip,
    port,
    compact,
    path,
    log,
    metric,
    significance,
    eval,
):
    if len(name) == 0:
        name = ctx.obj.get("tournament_name", "")

    if len(name) == 0:
        print(
            "Name is not given to run command and was not stored during a create command call"
        )
        exit(1)

    if len(path) > 0:
        sys.path.append(path)
    saved_log_folder = ctx.obj.get("tournament_log_folder", None)

    if saved_log_folder is not None:
        log = saved_log_folder
    parallelism = "distributed" if distributed else "parallel" if parallel else "serial"
    prog_callback = print_world_progress if verbosity > 1 and not distributed else None
    tpath = str(pathlib.Path(log) / name)
    start = perf_counter()
    run_tournament(
        tournament_path=tpath,
        verbose=verbosity > 0,
        compact=compact,
        world_progress_callback=prog_callback,
        parallelism=parallelism,
        scheduler_ip=ip,
        scheduler_port=port,
        print_exceptions=verbosity > 1,
    )
    end_time = humanize_time(perf_counter() - start)
    if eval:
        results = evaluate_tournament(
            tournament_path=tpath, verbose=verbosity > 0, metric=metric, compile=True
        )
        display_results(results, metric, significance)
    print(f"Finished in {end_time}")


@tournament.command(help="Evaluates a tournament and returns the results")
@click.argument("path", type=click.Path(dir_okay=True, file_okay=False))
@click.option(
    "--metric",
    default="truncated_mean",
    type=str,
    help="The statistical metric used for choosing the winners. Possibilities are mean, median, std, var, sum, truncated_mean",
)
@click.option(
    "--significance/--no-significance",
    default=False,
    help="Whether to show significance table",
)
@click.option(
    "--compile/--show",
    default=True,
    help="Whether to recompile results from individual world runs or just show the already-compiled results",
)
@click.option("--verbose/--silent", default=True, help="Whether to be verbose")
@click_config_file.configuration_option()
@click.pass_context
def eval(ctx, path, metric, significance, compile, verbose):
    results = evaluate_tournament(
        tournament_path=path, metric=metric, compile=compile, verbose=verbose
    )
    display_results(results, metric, significance)


@tournament.command(
    help="Finds winners of a tournament or a set of tournaments sharing a root"
)
@click.option(
    "--name",
    "-n",
    default="",
    help="The name of the tournament. When invoked after create, there is no need to pass it",
)
@click.option(
    "--log",
    "-l",
    type=click.Path(dir_okay=True, file_okay=False),
    default=default_tournament_path(),
    help="Default location to save logs",
)
@click.option(
    "--recursive/--no-recursive",
    default=True,
    help="Whether to recursively look for tournament results. --name should not be given if --recursive",
)
@click.option(
    "--metric",
    default="truncated_mean",
    type=str,
    help="The statistical metric used for choosing the winners. Possibilities are mean, median, std, var, sum, truncated_mean",
)
@click.option("--verbose/--silent", default=True, help="Whether to be verbose")
@click.option(
    "--significance/--no-significance",
    default=False,
    help="Whether to show significance table",
)
@click.option(
    "--compile/--show",
    default=True,
    help="Whether to recompile results from individual world runs or just show the already-compiled results",
)
@click_config_file.configuration_option()
@click.pass_context
def winners(ctx, name, log, recursive, metric, significance, compile, verbose):
    if len(name) == 0:
        if not recursive:
            name = ctx.obj.get("tournament_name", "")
        else:
            name = None

    if (name is None or len(name) == 0) and not recursive:
        print(
            "Name is not given to run command and was not stored during a create command call"
        )
        exit(1)
    saved_log_folder = ctx.obj.get("tournament_log_folder", None)

    if saved_log_folder is not None:
        log = saved_log_folder

    if name is not None and len(name) > 0:
        tpath = str(pathlib.Path(log) / name)
    else:
        tpath = str(pathlib.Path(log))
    results = evaluate_tournament(
        tournament_path=tpath,
        verbose=verbose,
        recursive=recursive,
        metric=metric,
        compile=compile,
    )
    display_results(results, metric, significance)


def _path(path) -> Path:
    """Creates an absolute path from given path which can be a string"""

    if isinstance(path, Path):
        return path.absolute()
    path.replace("/", os.sep)

    if isinstance(path, str):
        if path.startswith("~"):
            path = Path.home() / (os.sep.join(path.split(os.sep)[1:]))

    return Path(path).absolute()


@tournament.command(help="Combine multiple tournaments at the given base path(s)")
@click.argument("path", type=click.Path(dir_okay=True, file_okay=False), nargs=-1)
@click.option("--verbose/--silent", default=True, help="Whether to be verbose")
@click.option(
    "--dest",
    "-d",
    type=click.Path(dir_okay=True, file_okay=False),
    help="The location to save the results",
    default=None,
)
@click_config_file.configuration_option()
def combine(path, dest, verbose):
    if dest is None:
        print("Must specify the destination using --dest/-d option")
        return
    tpath = [_path(_) for _ in path]

    if len(tpath) < 1:
        print("No paths are given to combine")
    combine_tournaments(sources=tpath, dest=dest, verbose=verbose)


@tournament.command(help="Combine results from multiple tournaments")
@click.argument("path", type=click.Path(dir_okay=True, file_okay=False), nargs=-1)
@click.option(
    "--dest",
    "-d",
    type=click.Path(dir_okay=True, file_okay=False),
    help="The location to save the results",
    default=None,
)
@click.option(
    "--metric",
    default="truncated_mean",
    type=str,
    help="The statistical metric used for choosing the winners. Possibilities are mean, median, std, var, sum, truncated_mean",
)
@click.option(
    "--max-sources",
    default=None,
    type=int,
    help="Maximum number of sources to use. Default to all available",
)
@click.option(
    "--significance/--no-significance",
    default=False,
    help="Whether to show significance table",
)
@click.option("--verbose/--silent", default=True, help="Whether to be verbose")
# @click.option(
#     "--compile/--show",
#     default=True,
#     help="Whether to recompile results from individual world runs or just show the already-compiled results",
# )
@click_config_file.configuration_option()
def combine_results(path, dest, metric, max_sources, significance, verbose):
    if max_sources is not None and max_sources == 0:
        max_sources = None
    tpath = [_path(_) for _ in path]

    if len(tpath) < 1:
        print("No paths are given to combine")
    scores = combine_tournament_results(
        sources=tpath, dest=dest, verbose=verbose, max_sources=max_sources
    )
    stats = combine_tournament_stats(
        sources=tpath, dest=dest, verbose=verbose, max_sources=max_sources
    )
    print(f"Collected {len(scores)} scores and {len(stats)} stats")
    results = evaluate_tournament(
        dest, scores=scores, stats=stats, verbose=verbose, metric=metric, compile=False
    )
    try:
        display_results(results, metric, significance)
    except Exception as e:
        print(
            f"Cannot display results: {e}\n{metric=}, {significance=}, {len(results.total_scores)=}"
        )
        print(results.total_scores)


@cli.command(help="Start the bridge to genius (to use GeniusNegotiator)")
@click.option(
    "--path",
    "-p",
    default="auto",
    help='Path to geniusbridge.jar with embedded NegLoader. Use "auto" to '
    "read the path from ~/negmas/config.json"
    "\n\tConfig key is genius_bridge_jar"
    "\nYou can download this jar from: "
    f"{BASE_WEBSITE}geniusbridge.jar",
)
@click.option(
    "--port",
    "-r",
    default=DEFAULT_JAVA_PORT,
    help="Port to run the NegLoader on. Pass 0 for the default value",
)
@click.option(
    "--debug/--no-debug",
    default=False,
    help="Run the bridge in debug mode if --debug else silently",
)
@click.option("--verbose/--silent", default=True, help="Verbose output")
@click.option("--save-logs/--no-logs", default=False, help="Save logs")
@click.option(
    "--die-on-exit/--no-die-on-exit",
    type=bool,
    default=False,
    help="Whether to kill the bridge on exit. For future use. Currently it does nothing.",
)
# @click.option(
#     "--capture-output/--no-capture-output",
#     type=bool,
#     default=False,
#     help="Whether to capture the output of the bridge or not",
# )
@click.option(
    "--use-shell/--no-shell",
    type=bool,
    default=False,
    help="Whether to start the new process in a shell",
)
@click.option(
    "--force-timeout/--no-forced-timeout",
    type=bool,
    default=True,
    help="Whether to force a timeout on the bridge",
)
@click.option(
    "--timeout",
    default=0,
    type=float,
    help="The timeout to pass. Zero or negative numbers to disable and use the bridge's global timeout.",
)
@click.option(
    "--log-path",
    default=None,
    type=click.Path(file_okay=False),
    help="Directory to save logs within. Only used if --save-logs. If not given, ~/negmas/logs will be used",
)
def genius(
    path,
    port,
    debug,
    timeout,
    force_timeout: bool = True,
    save_logs: bool = False,
    log_path: os.PathLike | None = None,
    die_on_exit: bool = False,
    use_shell: bool = False,
    verbose: bool = False,
    allow_agent_print: bool = False,
    # capture_output: bool = False,
):
    if port and negmas.genius_bridge_is_running(port):
        print(f"Genius Bridge is already running on port {port} ... exiting")
        sys.exit()
    port = negmas.init_genius_bridge(
        path=path if path != "auto" else None,
        port=port,
        debug=debug,
        timeout=timeout,
        force_timeout=force_timeout,
        save_logs=save_logs,
        log_path=log_path,
        die_on_exit=die_on_exit,
        use_shell=use_shell,
        verbose=verbose,
        allow_agent_print=allow_agent_print,
        # capture_output=capture_output,
    )
    if port > 0:
        print(f"Started on port {port}")
    elif port == 0:
        print(
            "Failed to start. Try running 'java -jar $HOME/negmas/files/geniusbridge.jar' directly."
        )
        exit(-1)
    else:
        print(f"A bridge is already running on {port}")
        exit(0)
    while True:
        pass


def download_and_set(key, url, file_name, extract=False):
    """
    Downloads a file and sets the corresponding key in ~/negmas/config.json

    Args:
        key: Key name in config.json
        url: URL to download from
        file_name: file name
        extract: If extract, the file is extracted into a folder (assuming it is a zip)

    Returns:

    """
    config_path = Path.home() / "negmas"
    file_path = config_path
    if not extract:
        file_path /= "files"
    file_path.mkdir(parents=True, exist_ok=True)
    file_path = file_path / file_name
    urllib.request.urlretrieve(url, file_path)
    config = {}
    config_file = config_path / "config.json"

    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)
    if extract:
        folder_name = file_name.replace(".zip", "").replace("negmas_", "")
        extracted_path = file_path.parent / folder_name
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(extracted_path)
        try:
            os.unlink(file_path)
        except Exception:
            pass
        file_path = extracted_path
    config[key] = str(file_path)
    with open(config_file, "w") as f:
        json.dump(config, fp=f, sort_keys=True, indent=4)
    return file_path


@cli.command(help="Downloads and installs docs to ~/negmas/docs")
def docs_setup():
    url = f"{BASE_WEBSITE}{DOCS_ZIP}"
    print(f"Downloading and extracting: {url}", end="", flush=True)
    path = download_and_set(key="docs", url=url, file_name=DOCS_ZIP, extract=True)
    print(
        f" done successfully.\nYou can open the docs by going to: file://{Path(path).absolute()}/index.html"
    )


@cli.command(
    help="Opens negmas docs in the browser. Make sure to install the docs first using negmas docs-setup"
)
def docs():
    path = Path.home() / "negmas" / "docs"
    if not path.exists():
        print(
            f"Cannot find docs in {path}.\nRun `negmas docs-setup` first then run `negmas docs` again."
        )
        return
    # webbrowser.open(str(path))
    PORT = 9970
    DIRECTORY = str(path)

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=DIRECTORY, **kwargs)

    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print("serving at port", PORT)
        httpd.serve_forever()


@cli.command(help="Downloads the genius bridge and updates your settings")
def genius_setup():
    url = f"{BASE_WEBSITE}{GENIUS_JAR_NAME}"
    print(f"Downloading: {url}", end="", flush=True)
    download_and_set(key="genius_bridge_jar", url=url, file_name=GENIUS_JAR_NAME)
    print(" done successfully")


@cli.command(help="Prints NegMAS version")
def version():
    print(negmas.__version__)


# ============================================================================
# Registry commands
# ============================================================================


@cli.group()
def registry():
    """Query the NegMAS registry for mechanisms, negotiators, and components."""
    pass


def _get_registry(registry_type: str):
    """Get the appropriate registry based on type."""
    from negmas import (
        mechanism_registry,
        negotiator_registry,
        component_registry,
        scenario_registry,
    )

    registries = {
        "mechanism": mechanism_registry,
        "mechanisms": mechanism_registry,
        "negotiator": negotiator_registry,
        "negotiators": negotiator_registry,
        "component": component_registry,
        "components": component_registry,
        "scenario": scenario_registry,
        "scenarios": scenario_registry,
    }
    return registries.get(registry_type.lower())


def _format_info(
    name: str, info, verbose: bool = False, include_tags: bool = False
) -> dict:
    """Format registry info for display."""
    from negmas.registry import ScenarioInfo

    # Handle ScenarioInfo differently (no full_type_name or cls)
    if isinstance(info, ScenarioInfo):
        # Determine format from tags
        fmt = "unknown"
        for tag in ("xml", "json", "yaml"):
            if tag in info.tags:
                fmt = tag
                break
        result = {
            "name": name,
            "path": str(info.path),
            "format": fmt,
            "file": "file" in info.tags,
        }
        if info.n_negotiators is not None:
            result["n_negotiators"] = info.n_negotiators
        if info.n_outcomes is not None:
            result["n_outcomes"] = info.n_outcomes
        if info.opposition_level is not None:
            result["opposition_level"] = round(info.opposition_level, 3)
        # Derive normalized and anac from tags
        if "normalized" in info.tags:
            result["normalized"] = True
        if "anac" in info.tags:
            result["anac"] = True
        if include_tags and info.tags:
            result["tags"] = ",".join(sorted(info.tags))
        if verbose and info.extra:
            result["extra"] = info.extra
        return result

    # Handle regular RegistryInfo subclasses
    result = {"name": name, "type": info.full_type_name}

    # Add type-specific fields derived from tags
    tags = info.tags if info.tags else set()

    # MechanismInfo fields (from tags)
    if "requires-deadline" in tags:
        result["requires_deadline"] = True

    # NegotiatorInfo fields (from tags)
    if "bilateral-only" in tags:
        result["bilateral_only"] = True
    if "requires-opponent-ufun" in tags:
        result["requires_opponent_ufun"] = True
    if "learning" in tags:
        result["learns"] = True
    # Check for anac-YYYY tags
    for tag in tags:
        if tag.startswith("anac-") and tag[5:].isdigit():
            result["anac_year"] = int(tag[5:])
            break
    if "supports-uncertainty" in tags:
        result["supports_uncertainty"] = True
    if "supports-discounting" in tags:
        result["supports_discounting"] = True

    # ComponentInfo fields
    if hasattr(info, "component_type"):
        result["component_type"] = info.component_type

    if include_tags and info.tags:
        result["tags"] = ",".join(sorted(info.tags))

    if verbose and info.extra:
        result["extra"] = info.extra

    return result


def _parse_tags(tag_str: str) -> set[str]:
    """Parse a comma-separated tag string into a set of tags."""
    if not tag_str:
        return set()
    return {t.strip() for t in tag_str.split(",") if t.strip()}


def _format_output(
    items: list[dict] | dict, output_format: str, headers: list[str] | None = None
) -> None:
    """Format and print output in the requested format.

    Args:
        items: List of dicts (for tables) or dict (for key-value display)
        output_format: One of 'free', 'txt', 'json'
        headers: Optional headers for table format
    """
    if output_format == "json":
        print(json.dumps(items, indent=2, default=str))
    elif output_format == "txt":
        # Plain text format suitable for piping to fzf, grep, etc.
        if isinstance(items, dict):
            for key, value in items.items():
                print(f"{key}\t{value}")
        elif isinstance(items, list):
            for item in items:
                if isinstance(item, dict):
                    # Tab-separated values for easy parsing
                    print("\t".join(str(v) for v in item.values()))
                else:
                    print(item)
    else:  # free - nicely formatted output (default)
        if isinstance(items, dict):
            for key, value in items.items():
                print(f"{key}: {value}")
        elif isinstance(items, list) and items:
            if isinstance(items[0], dict):
                print(tabulate(items, headers="keys", tablefmt="psql"))
            else:
                for item in items:
                    print(item)


def _parse_filter(filter_str: str) -> dict:
    """Parse filter string into a dictionary.

    Accepts formats like:
        - "anac_year=2019"
        - "component_type=acceptance"
        - "bilateral_only=true"
        - "requires_deadline=false"
    """
    filters = {}
    if not filter_str:
        return filters

    for part in filter_str.split(","):
        part = part.strip()
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        key = key.strip()
        value = value.strip()

        # Convert value to appropriate type
        if value.lower() == "true":
            value = True
        elif value.lower() == "false":
            value = False
        elif value.lower() in ("none", "null"):
            value = None
        else:
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass  # Keep as string

        filters[key] = value

    return filters


def _parse_numeric_range(value_str: str) -> int | float | tuple | None:
    """Parse a numeric value or range string.

    Accepts formats like:
        - "100" (exact value)
        - "100:500" (min:max range)
        - ":100" (max only)
        - "100:" (min only)
        - "0.3:0.7" (float range)

    Returns:
        int, float, or tuple of (min, max) where None means unbounded
    """
    if not value_str:
        return None

    if ":" in value_str:
        parts = value_str.split(":", 1)
        try:
            min_val = float(parts[0]) if parts[0] else None
            max_val = float(parts[1]) if parts[1] else None
            # Convert to int if possible
            if min_val is not None and min_val == int(min_val):
                min_val = int(min_val)
            if max_val is not None and max_val == int(max_val):
                max_val = int(max_val)
            return (min_val, max_val)
        except ValueError:
            return None
    else:
        try:
            val = float(value_str)
            if val == int(val):
                return int(val)
            return val
        except ValueError:
            return None


@registry.command(name="list", help="List all registered items of a given type")
@click.argument(
    "registry_type",
    type=click.Choice(
        ["mechanisms", "negotiators", "components", "scenarios"], case_sensitive=False
    ),
)
@click.option(
    "--tag",
    "-t",
    "tags_str",
    default="",
    help="Filter by tags - items must have ALL specified tags (comma-separated, e.g., 'genius,anac-2019')",
)
@click.option(
    "--any-tag",
    "-a",
    "any_tags_str",
    default="",
    help="Filter by tags - items must have ANY of the specified tags (comma-separated)",
)
@click.option(
    "--exclude-tag",
    "-x",
    "exclude_tags_str",
    default="",
    help="Exclude items with ANY of these tags (comma-separated)",
)
@click.option(
    "--n-outcomes",
    default="",
    help="Filter scenarios by n_outcomes (e.g., '100', '100:500', ':1000')",
)
@click.option(
    "--n-negotiators",
    default="",
    help="Filter scenarios by n_negotiators (e.g., '2', '2:4')",
)
@click.option(
    "--opposition-level",
    default="",
    help="Filter scenarios by opposition level (e.g., '0.3:0.7')",
)
@click.option(
    "--format",
    "-o",
    "output_format",
    type=click.Choice(["free", "txt", "json"], case_sensitive=False),
    default="free",
    help="Output format: 'free' (nicely formatted), 'txt' (plain text for piping), 'json'",
)
@click.option("--verbose", "-v", is_flag=True, help="Show extra details")
@click.option("--show-tags", is_flag=True, help="Include tags in output")
@click.option("--count", "-c", is_flag=True, help="Only show count of matching items")
def list_registry(
    registry_type,
    tags_str,
    any_tags_str,
    exclude_tags_str,
    n_outcomes,
    n_negotiators,
    opposition_level,
    output_format,
    verbose,
    show_tags,
    count,
):
    """List all registered mechanisms, negotiators, or components."""
    reg = _get_registry(registry_type)
    if reg is None:
        print(f"Unknown registry type: {registry_type}")
        return

    # Parse tag filters
    tags = _parse_tags(tags_str)
    any_tags = _parse_tags(any_tags_str)
    exclude_tags = _parse_tags(exclude_tags_str)

    # Apply query with all filters
    query_kwargs = {}
    if tags:
        query_kwargs["tags"] = tags
    if any_tags:
        query_kwargs["any_tags"] = any_tags
    if exclude_tags:
        query_kwargs["exclude_tags"] = exclude_tags

    # Add numeric filters for scenarios
    if registry_type.lower() == "scenarios":
        if n_outcomes:
            parsed = _parse_numeric_range(n_outcomes)
            if parsed is not None:
                query_kwargs["n_outcomes"] = parsed
        if n_negotiators:
            parsed = _parse_numeric_range(n_negotiators)
            if parsed is not None:
                query_kwargs["n_negotiators"] = parsed
        if opposition_level:
            parsed = _parse_numeric_range(opposition_level)
            if parsed is not None:
                query_kwargs["opposition_level"] = parsed

    if query_kwargs:
        items = reg.query(**query_kwargs)
    else:
        items = dict(reg)

    if count:
        print(len(items))
        return

    if not items:
        print(f"No {registry_type} found matching the criteria.")
        return

    if output_format == "txt":
        # Plain text - just names, one per line (ideal for piping to fzf)
        for name in sorted(items.keys()):
            print(name)
    elif output_format == "json":
        result = {
            name: _format_info(name, info, verbose, show_tags)
            for name, info in items.items()
        }
        print(json.dumps(result, indent=2, default=str))
    else:  # free - nicely formatted table
        rows = [
            _format_info(name, info, verbose, show_tags)
            for name, info in sorted(items.items())
        ]
        print(tabulate(rows, headers="keys", tablefmt="psql"))


@registry.command(help="Get details about a specific registered item")
@click.argument("name")
@click.option(
    "--type",
    "-t",
    "registry_type",
    type=click.Choice(
        ["mechanism", "negotiator", "component", "scenario", "auto"],
        case_sensitive=False,
    ),
    default="auto",
    help="Registry type to search (default: auto-detect)",
)
@click.option(
    "--format",
    "-o",
    "output_format",
    type=click.Choice(["free", "txt", "json"], case_sensitive=False),
    default="free",
    help="Output format: 'free' (nicely formatted), 'txt' (plain text for piping), 'json'",
)
def get(name, registry_type, output_format):
    """Get details about a specific registered item by name."""
    from negmas import (
        mechanism_registry,
        negotiator_registry,
        component_registry,
        scenario_registry,
    )

    registries = [
        ("mechanism", mechanism_registry),
        ("negotiator", negotiator_registry),
        ("component", component_registry),
        ("scenario", scenario_registry),
    ]

    if registry_type != "auto":
        registries = [(registry_type, _get_registry(registry_type))]

    found = None
    found_type = None
    for reg_type, reg in registries:
        if reg is None:
            continue
        info = reg.get(name)
        if info is not None:
            found = info
            found_type = reg_type
            break

    if found is None:
        print(f"'{name}' not found in any registry.")
        return

    result = _format_info(name, found, verbose=True, include_tags=True)
    result["registry"] = found_type

    if output_format == "json":
        print(json.dumps(result, indent=2, default=str))
    elif output_format == "txt":
        # Tab-separated key-value pairs for easy parsing
        for key, value in result.items():
            print(f"{key}\t{value}")
    else:  # free - nicely formatted
        for key, value in result.items():
            print(f"{key}: {value}")


@registry.command(help="Search for registered items by name pattern")
@click.argument("pattern")
@click.option(
    "--type",
    "-t",
    "registry_type",
    type=click.Choice(
        ["mechanisms", "negotiators", "components", "scenarios", "all"],
        case_sensitive=False,
    ),
    default="all",
    help="Registry type to search",
)
@click.option(
    "--tag",
    "tags_str",
    default="",
    help="Filter by tags - items must have ALL specified tags (comma-separated)",
)
@click.option(
    "--any-tag",
    "-a",
    "any_tags_str",
    default="",
    help="Filter by tags - items must have ANY of the specified tags (comma-separated)",
)
@click.option(
    "--exclude-tag",
    "-x",
    "exclude_tags_str",
    default="",
    help="Exclude items with ANY of these tags (comma-separated)",
)
@click.option(
    "--format",
    "-o",
    "output_format",
    type=click.Choice(["free", "txt", "json"], case_sensitive=False),
    default="free",
    help="Output format: 'free' (nicely formatted), 'txt' (plain text for piping), 'json'",
)
@click.option("--case-sensitive", "-s", is_flag=True, help="Case-sensitive search")
@click.option("--show-tags", is_flag=True, help="Include tags in output")
def search(
    pattern,
    registry_type,
    tags_str,
    any_tags_str,
    exclude_tags_str,
    output_format,
    case_sensitive,
    show_tags,
):
    """Search for registered items by name pattern (supports wildcards)."""
    import fnmatch
    from negmas import (
        mechanism_registry,
        negotiator_registry,
        component_registry,
        scenario_registry,
    )

    # Parse tag filters
    tags = _parse_tags(tags_str)
    any_tags = _parse_tags(any_tags_str)
    exclude_tags = _parse_tags(exclude_tags_str)

    registries = []
    if registry_type == "all":
        registries = [
            ("mechanism", mechanism_registry),
            ("negotiator", negotiator_registry),
            ("component", component_registry),
            ("scenario", scenario_registry),
        ]
    else:
        reg = _get_registry(registry_type)
        if reg:
            registries = [(registry_type.rstrip("s"), reg)]

    results = []
    for reg_type, reg in registries:
        for name, info in reg.items():
            # For scenarios, match against the scenario name, not the path
            if reg_type == "scenario":
                match_name = info.name if case_sensitive else info.name.lower()
            else:
                match_name = name if case_sensitive else name.lower()
            match_pattern = pattern if case_sensitive else pattern.lower()
            if not fnmatch.fnmatch(match_name, match_pattern):
                continue

            # Check tag filters
            if tags and not info.has_all_tags(tags):
                continue
            if any_tags and not info.has_any_tag(any_tags):
                continue
            if exclude_tags and info.has_any_tag(exclude_tags):
                continue

            result = _format_info(name, info, include_tags=show_tags)
            result["registry"] = reg_type
            results.append(result)

    if not results:
        print(f"No items found matching pattern '{pattern}'")
        return

    if output_format == "txt":
        # Plain text - just names, one per line
        for r in sorted(results, key=lambda x: x["name"]):
            print(r["name"])
    elif output_format == "json":
        print(json.dumps(results, indent=2, default=str))
    else:  # free - nicely formatted table
        print(tabulate(results, headers="keys", tablefmt="psql"))


@registry.command(help="Show summary statistics for all registries")
@click.option(
    "--tag",
    "-t",
    "tags_str",
    default="",
    help="Filter by tags - only count items with ALL specified tags (comma-separated)",
)
@click.option(
    "--any-tag",
    "-a",
    "any_tags_str",
    default="",
    help="Filter by tags - only count items with ANY of the specified tags (comma-separated)",
)
@click.option(
    "--exclude-tag",
    "-x",
    "exclude_tags_str",
    default="",
    help="Exclude items with ANY of these tags (comma-separated)",
)
@click.option(
    "--format",
    "-o",
    "output_format",
    type=click.Choice(["free", "txt", "json"], case_sensitive=False),
    default="free",
    help="Output format: 'free' (nicely formatted), 'txt' (plain text), 'json'",
)
def stats(tags_str, any_tags_str, exclude_tags_str, output_format):
    """Show summary statistics for all registries."""
    from negmas import (
        mechanism_registry,
        negotiator_registry,
        component_registry,
        scenario_registry,
    )

    # Parse tag filters
    tags = _parse_tags(tags_str)
    any_tags = _parse_tags(any_tags_str)
    exclude_tags = _parse_tags(exclude_tags_str)

    def filter_items(registry):
        """Apply tag filters to a registry and return matching items."""
        query_kwargs = {}
        if tags:
            query_kwargs["tags"] = tags
        if any_tags:
            query_kwargs["any_tags"] = any_tags
        if exclude_tags:
            query_kwargs["exclude_tags"] = exclude_tags

        if query_kwargs:
            return registry.query(**query_kwargs)
        return dict(registry)

    # Get filtered items for each registry
    mechanisms = filter_items(mechanism_registry)
    negotiators = filter_items(negotiator_registry)
    components = filter_items(component_registry)
    scenarios = filter_items(scenario_registry)

    # Compute statistics
    stats_data = {
        "mechanisms": {
            "total": len(mechanisms),
            "requires_deadline": sum(
                1 for info in mechanisms.values() if info.requires_deadline
            ),
            "no_deadline_required": sum(
                1 for info in mechanisms.values() if not info.requires_deadline
            ),
        },
        "negotiators": {"total": len(negotiators), "by_anac_year": {}, "non_anac": 0},
        "components": {"total": len(components), "by_type": {}},
        "scenarios": {"total": len(scenarios), "by_format": {}, "by_n_negotiators": {}},
    }

    # Count negotiators by ANAC year
    for info in negotiators.values():
        if info.anac_year:
            year_key = f"anac_{info.anac_year}"
            stats_data["negotiators"]["by_anac_year"][year_key] = (
                stats_data["negotiators"]["by_anac_year"].get(year_key, 0) + 1
            )
        else:
            stats_data["negotiators"]["non_anac"] += 1

    # Count components by type
    for info in components.values():
        ct = info.component_type
        stats_data["components"]["by_type"][ct] = (
            stats_data["components"]["by_type"].get(ct, 0) + 1
        )

    # Count scenarios by format and n_negotiators
    for info in scenarios.values():
        fmt = info.format
        stats_data["scenarios"]["by_format"][fmt] = (
            stats_data["scenarios"]["by_format"].get(fmt, 0) + 1
        )
        if info.n_negotiators is not None:
            n_key = f"{info.n_negotiators}_party"
            stats_data["scenarios"]["by_n_negotiators"][n_key] = (
                stats_data["scenarios"]["by_n_negotiators"].get(n_key, 0) + 1
            )

    # Output based on format
    if output_format == "json":
        print(json.dumps(stats_data, indent=2))
    elif output_format == "txt":
        # Tab-separated format for easy parsing
        print(f"mechanisms_total\t{stats_data['mechanisms']['total']}")
        print(
            f"mechanisms_requires_deadline\t{stats_data['mechanisms']['requires_deadline']}"
        )
        print(
            f"mechanisms_no_deadline\t{stats_data['mechanisms']['no_deadline_required']}"
        )
        print(f"negotiators_total\t{stats_data['negotiators']['total']}")
        print(f"negotiators_non_anac\t{stats_data['negotiators']['non_anac']}")
        for year, count in sorted(stats_data["negotiators"]["by_anac_year"].items()):
            print(f"negotiators_{year}\t{count}")
        print(f"components_total\t{stats_data['components']['total']}")
        for ct, count in sorted(stats_data["components"]["by_type"].items()):
            print(f"components_{ct}\t{count}")
        print(f"scenarios_total\t{stats_data['scenarios']['total']}")
        for fmt, count in sorted(stats_data["scenarios"]["by_format"].items()):
            print(f"scenarios_{fmt}\t{count}")
        for n_key, count in sorted(stats_data["scenarios"]["by_n_negotiators"].items()):
            print(f"scenarios_{n_key}\t{count}")
    else:  # free - nicely formatted output
        filter_desc = ""
        if tags or any_tags or exclude_tags:
            parts = []
            if tags:
                parts.append(f"all of \\[{', '.join(tags)}]")
            if any_tags:
                parts.append(f"any of \\[{', '.join(any_tags)}]")
            if exclude_tags:
                parts.append(f"excluding \\[{', '.join(exclude_tags)}]")
            filter_desc = f" (filtered: {', '.join(parts)})"

        print(f"Registry Statistics{filter_desc}")
        print("=" * 40)

        # Mechanisms
        print(f"\nMechanisms: {stats_data['mechanisms']['total']}")
        print(f"  - Requires deadline: {stats_data['mechanisms']['requires_deadline']}")
        print(
            f"  - No deadline required: {stats_data['mechanisms']['no_deadline_required']}"
        )

        # Negotiators
        print(f"\nNegotiators: {stats_data['negotiators']['total']}")
        print(f"  - Non-ANAC: {stats_data['negotiators']['non_anac']}")
        for year_key in sorted(stats_data["negotiators"]["by_anac_year"].keys()):
            year = year_key.replace("anac_", "ANAC ")
            print(f"  - {year}: {stats_data['negotiators']['by_anac_year'][year_key]}")

        # Components
        print(f"\nComponents: {stats_data['components']['total']}")
        for ct in sorted(stats_data["components"]["by_type"].keys()):
            print(f"  - {ct}: {stats_data['components']['by_type'][ct]}")

        # Scenarios
        print(f"\nScenarios: {stats_data['scenarios']['total']}")
        for fmt in sorted(stats_data["scenarios"]["by_format"].keys()):
            print(f"  - {fmt}: {stats_data['scenarios']['by_format'][fmt]}")
        for n_key in sorted(stats_data["scenarios"]["by_n_negotiators"].keys()):
            print(f"  - {n_key}: {stats_data['scenarios']['by_n_negotiators'][n_key]}")


@registry.command(help="List all tags used in the registries")
@click.option(
    "--type",
    "-t",
    "registry_type",
    type=click.Choice(
        [
            "all",
            "any",
            "mechanisms",
            "negotiators",
            "components",
            "scenarios",
            "acceptance",
            "offering",
            "model",
        ],
        case_sensitive=False,
    ),
    default="all",
    help="Registry/item type to list tags from. 'all'/'any' for all registries, or filter by specific type including component subtypes (acceptance, offering, model)",
)
@click.option(
    "--format",
    "-o",
    "output_format",
    type=click.Choice(["free", "txt", "json"], case_sensitive=False),
    default="free",
    help="Output format: 'free' (nicely formatted), 'txt' (plain text), 'json'",
)
@click.option("--count", "-c", is_flag=True, help="Show count of items per tag")
def tags(registry_type, output_format, count):
    """List all tags used in the registries."""
    from negmas import (
        mechanism_registry,
        negotiator_registry,
        component_registry,
        scenario_registry,
    )

    # Component subtypes filter components by component_type
    component_subtypes = {"acceptance", "offering", "model"}

    registries = []
    component_filter = None  # For filtering component subtypes

    if registry_type in ("all", "any"):
        registries = [
            ("mechanisms", mechanism_registry),
            ("negotiators", negotiator_registry),
            ("components", component_registry),
            ("scenarios", scenario_registry),
        ]
    elif registry_type in component_subtypes:
        # Filter components by component_type
        registries = [(registry_type, component_registry)]
        component_filter = registry_type
    else:
        reg = _get_registry(registry_type)
        if reg:
            registries = [(registry_type, reg)]

    if count:
        # Count items per tag, grouped by registry
        tag_counts: dict[str, dict[str, int]] = {}
        for reg_name, reg in registries:
            for info in reg.values():
                # Apply component subtype filter if needed
                if component_filter is not None:
                    if (
                        not hasattr(info, "component_type")
                        or info.component_type != component_filter
                    ):
                        continue
                for tag in info.tags:
                    if tag not in tag_counts:
                        tag_counts[tag] = {}
                    tag_counts[tag][reg_name] = tag_counts[tag].get(reg_name, 0) + 1

        if output_format == "json":
            print(json.dumps(tag_counts, indent=2))
        elif output_format == "txt":
            # Tab-separated: tag, registry, count
            for tag in sorted(tag_counts.keys()):
                total = sum(tag_counts[tag].values())
                print(f"{tag}\t{total}")
        else:  # free
            rows = []
            # Determine which columns to show based on registry types
            if registry_type in ("all", "any"):
                columns = ["mechanisms", "negotiators", "components", "scenarios"]
            elif registry_type in component_subtypes:
                columns = [registry_type]
            else:
                columns = [registry_type]

            for tag in sorted(tag_counts.keys()):
                row = {"tag": tag}
                total = 0
                for col in columns:
                    c = tag_counts[tag].get(col, 0)
                    row[col] = c if c > 0 else ""
                    total += c
                row["total"] = total
                rows.append(row)
            print(tabulate(rows, headers="keys", tablefmt="psql"))
    else:
        # Just list unique tags
        all_tags: set[str] = set()
        for _, reg in registries:
            if component_filter is not None:
                # Filter by component subtype
                for info in reg.values():
                    if (
                        hasattr(info, "component_type")
                        and info.component_type == component_filter
                    ):
                        all_tags |= info.tags
            else:
                all_tags |= reg.list_tags()

        if output_format == "json":
            print(json.dumps(sorted(all_tags), indent=2))
        elif output_format == "txt":
            for tag in sorted(all_tags):
                print(tag)
        else:  # free
            print(f"Tags ({len(all_tags)} total):")
            for tag in sorted(all_tags):
                print(f"  - {tag}")


@registry.command(help="List available filter attributes for a registry type")
@click.argument(
    "registry_type",
    type=click.Choice(["mechanism", "negotiator", "component"], case_sensitive=False),
)
def filters(registry_type):
    """List available filter attributes for a registry type."""
    if registry_type == "mechanism":
        print("Available filters for mechanisms:")
        print("  - requires_deadline (bool): Whether the mechanism requires a deadline")
        print(
            "\nExample: negmas registry list mechanisms --filter 'requires_deadline=false'"
        )
    elif registry_type == "negotiator":
        print("Available filters for negotiators:")
        print("  - bilateral_only (bool): Only works in bilateral negotiations")
        print("  - requires_opponent_ufun (bool): Needs opponent's utility function")
        print("  - learns (bool): Learns from repeated negotiations")
        print("  - anac_year (int): ANAC competition year (e.g., 2019)")
        print("  - supports_uncertainty (bool): Supports uncertain preferences")
        print("  - supports_discounting (bool): Supports time-discounted utilities")
        print("\nExample: negmas registry list negotiators --filter 'anac_year=2019'")
        print(
            "Example: negmas registry list negotiators --filter 'learns=true,bilateral_only=false'"
        )
    elif registry_type == "component":
        print("Available filters for components:")
        print(
            "  - component_type (str): Type of component (acceptance, offering, model)"
        )
        print(
            "\nExample: negmas registry list components --filter 'component_type=acceptance'"
        )


if __name__ == "__main__":
    cli()
