#!/usr/bin/env python
"""The NegMAS universal command line tool"""
import json
import os
import pathlib
import sys
import traceback
import urllib.request
import warnings
from functools import partial
from pathlib import Path
from pprint import pformat, pprint
from time import perf_counter
import numpy as np

import click
import click_config_file
import pandas as pd
import progressbar
import yaml
from tabulate import tabulate

import negmas
from negmas import save_stats
from negmas.apps.scml import *
from negmas.apps.scml.utils import (
    anac2019_sabotage,
    anac2019_world_generator,
    anac2019_config_generator,
    anac2019_assigner,
    anac2019_sabotage_config_generator,
    anac2019_sabotage_assigner,
    sabotage_effectiveness,
)
from negmas.helpers import humanize_time, unique_name, camel_case, load
from negmas.java import init_jnegmas_bridge, jnegmas_bridge_is_running
from negmas.tournaments import (
    create_tournament,
    run_tournament,
    evaluate_tournament,
    combine_tournaments,
    combine_tournament_stats,
)

try:
    # disable a warning in yaml 1b1 version
    yaml.warnings({"YAMLLoadWarning": False})
except:
    pass

n_completed = 0
n_total = 0

JNEGMAS_JAR_NAME = "jnegmas-0.2.6-all.jar"
GENIUS_JAR_NAME = "genius-8.0.4-bridge.jar"


def print_progress(_, i, n) -> None:
    """Prints the progress of a tournament"""
    global n_completed, n_total
    n_completed = i + 1
    n_total = n
    print(
        f"{n_completed:04} of {n:04} worlds completed ({n_completed / n:0.2%})",
        flush=True,
    )


def print_world_progress(world) -> None:
    """Prints the progress of a world"""
    step = world.current_step + 1
    s = (
        f"World# {n_completed:04}: {step:04}  of {world.n_steps:04} "
        f"steps completed ({step / world.n_steps:0.2f}) "
    )
    if n_total > 0:
        s += f"TOTAL: ({n_completed + step / world.n_steps / n_total:0.2f})"
    print(s, flush=True)


click.option = partial(click.option, show_default=True)


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


# @cli.group(invoke_without_command=True)
# @click.option(
#     "--name",
#     "-n",
#     default="random",
#     help='The name of the tournament. The special value "random" will result in a random name',
# )
# @click.option("--steps", "-s", default=100, help="Number of steps.")
# @click.option(
#     "--ttype",
#     "--tournament-type",
#     "--tournament",
#     default="anac2019collusion",
#     type=click.Choice(["anac2019collusion", "anac2019std", "anac2019sabotage"]),
#     help="The config to use. Default is ANAC 2019. Options supported are anac2019std, anac2019collusion, "
#     "anac2019sabotage",
# )
# @click.option(
#     "--timeout",
#     "-t",
#     default=0,
#     type=int,
#     help="Timeout the whole tournament after the given number of seconds (0 for infinite)",
# )
# @click.option(
#     "--configs",
#     default=5,
#     type=int,
#     help="Number of unique configurations to generate.",
# )
# @click.option("--runs", default=2, help="Number of runs for each configuration")
# @click.option(
#     "--max-runs",
#     default=-1,
#     type=int,
#     help="Maximum total number of runs. Zero or negative numbers mean no limit",
# )
# @click.option(
#     "--agents",
#     default=5,
#     type=int,
#     help="Number of agents per competitor (not used for anac2019std in which this is preset to 1).",
# )
# @click.option(
#     "--factories",
#     default=5,
#     type=int,
#     help="Minimum numbers of factories to have per level.",
# )
# @click.option(
#     "--competitors",
#     default="DoNothingFactoryManager;GreedyFactoryManager",
#     help="A semicolon (;) separated list of agent types to use for the competition.",
# )
# @click.option(
#     "--jcompetitors",
#     "--java-competitors",
#     default="",
#     help="A semicolon (;) separated list of agent types to use for the competition.",
# )
# @click.option(
#     "--non-competitors",
#     default="",
#     help="A semicolon (;) separated list of agent types to exist in the worlds as non-competitors "
#     "(their scores will not be calculated).",
# )
# @click.option(
#     "--parallel/--serial",
#     default=True,
#     help="Run a parallel/serial tournament on a single machine",
# )
# @click.option(
#     "--distributed/--single-machine",
#     default=False,
#     help="Run a distributed tournament using dask",
# )
# @click.option(
#     "--log",
#     "-l",
#     type=click.Path(dir_okay=True, file_okay=False),
#     default="~/negmas/logs/tournaments",
#     help="Default location to save logs (A folder will be created under it)",
# )
# @click.option(
#     "--verbosity",
#     default=1,
#     type=int,
#     help="verbosity level (from 0 == silent to 1 == world progress)",
# )
# @click.option("--configs-only/--run", default=False, help="configs_only")
# @click.option(
#     "--reveal-names/--hidden-names",
#     default=True,
#     help="Reveal agent names (should be used only for " "debugging)",
# )
# @click.option(
#     "--ip",
#     default="127.0.0.1",
#     help="The IP address for a dask scheduler to run the distributed tournament."
#     " Effective only if --distributed",
# )
# @click.option(
#     "--port",
#     default=8786,
#     type=int,
#     help="The IP port number a dask scheduler to run the distributed tournament."
#     " Effective only if --distributed",
# )
# @click.option(
#     "--log-ufuns/--no-ufun-logs",
#     default=False,
#     help="Log ufuns into their own CSV file. Only effective if --debug is given",
# )
# @click.option(
#     "--log-negs/--no-neg-logs",
#     default=False,
#     help="Log all negotiations. Only effective if --debug is given",
# )
# @click.option(
#     "--compact/--debug",
#     default=True,
#     help="If True, effort is exerted to reduce the memory footprint which"
#     "includes reducing logs dramatically.",
# )
# @click.option(
#     "--raise-exceptions/--ignore-exceptions",
#     default=True,
#     help="Whether to ignore agent exceptions",
# )
# @click.option(
#     "--path",
#     default="",
#     help="A path to be added to PYTHONPATH in which all competitors are stored. You can path a : separated list of "
#     "paths on linux/mac and a ; separated list in windows",
# )
# @click_config_file.configuration_option()
# def tournament(
#     name,
#     steps,
#     parallel,
#     distributed,
#     ttype,
#     timeout,
#     log,
#     verbosity,
#     configs_only,
#     reveal_names,
#     ip,
#     port,
#     runs,
#     configs,
#     max_runs,
#     competitors,
#     jcompetitors,
#     non_competitors,
#     compact,
#     factories,
#     agents,
#     log_ufuns,
#     log_negs,
#     path,
#     raise_exceptions,
# ):
#     if name == "random":
#         name = unique_name("", rand_digits=0)
#     create(
#         name,
#         steps,
#         parallel,
#         distributed,
#         ttype,
#         timeout,
#         log,
#         verbosity,
#         reveal_names,
#         ip,
#         port,
#         runs,
#         configs,
#         max_runs,
#         competitors,
#         jcompetitors,
#         non_competitors,
#         compact,
#         factories,
#         agents,
#         log_ufuns,
#         log_negs,
#         raise_exceptions,
#     )
#     if not configs_only:
#         run(name, verbosity, distributed, compact, path, log)


@tournament.command(help="Creates a tournament")
@click.option(
    "--name",
    "-n",
    default="random",
    help='The name of the tournament. The special value "random" will result in a random name',
)
@click.option(
    "--steps",
    "-s",
    default=None,
    type=int,
    help="Number of steps. If passed then --steps-min and --steps-max are " "ignored",
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
@click.option(
    "--ttype",
    "--tournament-type",
    "--tournament",
    default="anac2019collusion",
    type=click.Choice(["anac2019collusion", "anac2019std", "anac2019sabotage"]),
    help="The config to use. Default is ANAC 2019. Options supported are anac2019std, anac2019collusion, "
    "anac2019sabotage",
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
    "--agents",
    default=3,
    type=int,
    help="Number of agents per competitor (not used for anac2019std in which this is preset to 1).",
)
@click.option(
    "--factories",
    default=5,
    type=int,
    help="Minimum numbers of factories to have per level.",
)
@click.option(
    "--competitors",
    default="DoNothingFactoryManager;GreedyFactoryManager",
    help="A semicolon (;) separated list of agent types to use for the competition.",
)
@click.option(
    "--jcompetitors",
    "--java-competitors",
    default="",
    help="A semicolon (;) separated list of agent types to use for the competition.",
)
@click.option(
    "--non-competitors",
    default="",
    help="A semicolon (;) separated list of agent types to exist in the worlds as non-competitors "
    "(their scores will not be calculated).",
)
@click.option(
    "--log",
    "-l",
    type=click.Path(dir_okay=True, file_okay=False),
    default="~/negmas/logs/tournaments",
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
    "--reveal-names/--hidden-names",
    default=True,
    help="Reveal agent names (should be used only for " "debugging)",
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
    steps,
    ttype,
    timeout,
    log,
    verbosity,
    reveal_names,
    runs,
    configs,
    max_runs,
    competitors,
    world_config,
    jcompetitors,
    non_competitors,
    compact,
    factories,
    agents,
    log_ufuns,
    log_negs,
    raise_exceptions,
    steps_min,
    steps_max,
    path,
    cw,
):
    if len(path) > 0:
        sys.path.append(path)
    kwargs = {}
    if world_config is not None and len(world_config) > 0:
        for wc in world_config:
            kwargs.update(load(wc))
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
                "You are running the tournament with --debug. Will reveal agent types in their names"
            )
        reveal_names = True
        verbosity = max(1, verbosity)

    worlds_per_config = (
        None if max_runs is None else int(round(max_runs / (configs * runs)))
    )

    all_competitors = competitors.split(";")
    for i, cp in enumerate(all_competitors):
        if "." not in cp:
            all_competitors[i] = "negmas.apps.scml.factory_managers." + cp
    all_competitors_params = [dict() for _ in range(len(all_competitors))]
    if jcompetitors is not None and len(jcompetitors) > 0:
        jcompetitor_params = [{"java_class_name": _} for _ in jcompetitors.split(";")]
        for jp in jcompetitor_params:
            if "." not in jp["java_class_name"]:
                jp["java_class_name"] = (
                    "jnegmas.apps.scml.factory_managers." + jp["java_class_name"]
                )
        jcompetitors = ["negmas.apps.scml.JavaFactoryManager"] * len(jcompetitor_params)
        all_competitors += jcompetitors
        all_competitors_params += jcompetitor_params
        print("You are using some Java agents. The tournament MUST run serially")
        if not jnegmas_bridge_is_running():
            print(
                "Error: You are using java competitors but jnegmas bridge is not running\n\nTo correct this issue"
                " run the following command IN A DIFFERENT TERMINAL because it will block:\n\n"
                "$ negmas jnegmas"
            )
            exit(1)

    # if ttype.lower() == "anac2019std":
    #     if (
    #         "negmas.apps.scml.factory_managers.GreedyFactoryManager"
    #         not in all_competitors
    #     ):
    #         all_competitors.append(
    #             "negmas.apps.scml.factory_managers.GreedyFactoryManager"
    #         )
    #         all_competitors_params.append({})

    permutation_size = len(all_competitors) if "sabotage" not in ttype else 1
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

    if ttype == "anac2019std":
        agents = 1

    if steps is None:
        steps = (steps_min, steps_max)

    if worlds_per_config is None:
        n_comp = len(all_competitors) if ttype != "anac2019sabotage" else 2
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

    if len(jcompetitors) > 0:
        print("You are using java-competitors. The tournament will be run serially")
        parallelism = "serial"

    non_competitor_params = None
    if len(non_competitors) < 1:
        non_competitors = None
    else:
        non_competitors = non_competitors.split(";")
        for i, cp in enumerate(non_competitors):
            if "." not in cp:
                non_competitors[i] = "negmas.apps.scml.factory_managers." + cp

    if ttype.lower() == "anac2019std":
        if non_competitors is None:
            non_competitors = (DefaultGreedyManager,)
            non_competitor_params = ({},)
        print(f"Tournament will be run between {len(all_competitors)} agents: ")
        pprint(all_competitors)
        print("Non-competitors are: ")
        pprint(non_competitors)
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
            n_agents_per_competitor=1,
            world_generator=anac2019_world_generator,
            config_generator=anac2019_config_generator,
            config_assigner=anac2019_assigner,
            score_calculator=balance_calculator,
            min_factories_per_level=factories,
            compact=compact,
            n_steps=steps,
            log_ufuns=log_ufuns,
            log_negotiations=log_negs,
            ignore_agent_exceptions=not raise_exceptions,
            ignore_contract_execution_exceptions=not raise_exceptions,
            **kwargs,
        )
    elif ttype.lower() in ("anac2019collusion", "anac2019"):
        print(f"Tournament will be run between {len(all_competitors)} agents: ")
        pprint(all_competitors)
        print("Non-competitors are: ")
        pprint(non_competitors)
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
            world_generator=anac2019_world_generator,
            config_generator=anac2019_config_generator,
            config_assigner=anac2019_assigner,
            score_calculator=balance_calculator,
            min_factories_per_level=factories,
            compact=compact,
            n_steps=steps,
            log_ufuns=log_ufuns,
            log_negotiations=log_negs,
            ignore_agent_exceptions=not raise_exceptions,
            ignore_contract_execution_exceptions=not raise_exceptions,
            **kwargs,
        )
    elif ttype.lower() == "anac2019sabotage":
        print(f"Tournament will be run between {len(all_competitors)} agents: ")
        pprint(all_competitors)
        print("Non-competitors are: ")
        pprint(non_competitors)
        results = create_tournament(
            competitors=all_competitors,
            competitor_params=all_competitors_params,
            agent_names_reveal_type=reveal_names,
            n_agents_per_competitor=agents,
            base_tournament_path=log,
            total_timeout=timeout,
            name=name,
            verbose=verbosity > 0,
            n_runs_per_world=runs,
            n_configs=configs,
            max_worlds_per_config=worlds_per_config,
            non_competitors=non_competitors,
            min_factories_per_level=factories,
            n_steps=steps,
            compact=compact,
            log_ufuns=log_ufuns,
            log_negotiations=log_negs,
            ignore_agent_exceptions=not raise_exceptions,
            ignore_contract_execution_exceptions=not raise_exceptions,
            non_competitor_params=non_competitor_params,
            world_generator=anac2019_world_generator,
            config_generator=anac2019_sabotage_config_generator,
            config_assigner=anac2019_sabotage_assigner,
            score_calculator=sabotage_effectiveness,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown tournament type {ttype}")
    ctx.obj["tournament_name"] = results.name
    ctx.obj["tournament_log_folder"] = log
    ctx.obj["compact"] = compact
    print(f"Saved all configs to {str(results)}\nTournament name is {results.name}")


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
    default="~/negmas/logs/tournaments",
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
    default="mean",
    type=str,
    help="The statistical metric used for choosing the winners. Possibilities are mean, median, std, var, sum",
)
@click_config_file.configuration_option()
@click.pass_context
def run(
    ctx, name, verbosity, parallel, distributed, ip, port, compact, path, log, metric
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
    results = evaluate_tournament(
        tournament_path=tpath, verbose=verbosity > 0, metric=metric
    )
    display_results(results, metric)
    print(f"Finished in {end_time}")


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
    default="~/negmas/logs/tournaments",
    help="Default location to save logs",
)
@click.option(
    "--recursive/--no-recursive",
    default=True,
    help="Whether to recursively look for tournament results. --name should not be given if --recursive",
)
@click.option(
    "--metric",
    default="mean",
    type=str,
    help="The statistical metric used for choosing the winners. Possibilities are mean, median, std, var, sum",
)
@click_config_file.configuration_option()
@click.pass_context
def winners(ctx, name, log, recursive, metric):
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
        tournament_path=tpath, verbose=True, recursive=recursive, metric=metric
    )
    display_results(results, metric)


def _path(path) -> Path:
    """Creates an absolute path from given path which can be a string"""
    if isinstance(path, str):
        if path.startswith("~"):
            path = Path.home() / ("/".join(path.split("/")[1:]))
    return pathlib.Path(path).absolute()


def display_results(results, metric):
    viewmetric = ["50%" if metric == "median" else metric]
    print(
        tabulate(
            results.score_stats.sort_values(by=viewmetric, ascending=False),
            headers="keys",
            tablefmt="psql",
        )
    )
    if metric in ("mean", "sum"):
        print(tabulate(results.ttest, headers="keys", tablefmt="psql"))
    else:
        print(tabulate(results.kstest, headers="keys", tablefmt="psql"))

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
    agg_stats.columns = ["negotiated", "concluded", "signed", "executed", "business"]
    print(tabulate(agg_stats.describe(), headers="keys", tablefmt="psql"))


@tournament.command(help="Finds winners of an arbitrary set of tournaments")
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
    default="mean",
    type=str,
    help="The statistical metric used for choosing the winners. Possibilities are mean, median, std, var, sum",
)
@click_config_file.configuration_option()
def combine(path, dest, metric):
    tpath = [_path(_) for _ in path]
    if len(tpath) < 1:
        print("No paths are given to combine")
    scores = combine_tournaments(sources=tpath, dest=None, verbose=True)
    stats = combine_tournament_stats(sources=tpath, dest=None, verbose=True)
    results = evaluate_tournament(dest, scores, stats, verbose=True, metric=metric)
    display_results(results, metric)


@cli.command(help="Run an SCML world simulation")
@click.option("--steps", default=100, type=int, help="Number of steps.")
@click.option(
    "--levels",
    default=3,
    type=int,
    help="Number of intermediate production levels (processes). "
    "-1 means a single product and no factories.",
)
@click.option("--neg-speedup", default=21, help="Negotiation Speedup.")
@click.option(
    "--negotiator",
    default=DEFAULT_NEGOTIATOR,
    help="Negotiator type to use for builtin agents.",
)
@click.option(
    "--min-consumption",
    default=3,
    type=int,
    help="The minimum number of units consumed by each consumer at every " "time-step.",
)
@click.option(
    "--max-consumption",
    default=5,
    type=int,
    help="The maximum number of units consumed by each consumer at every " "time-step.",
)
@click.option(
    "--agents",
    default=5,
    type=int,
    help="Number of agents (miners/negmas.consumers) per production level",
)
@click.option("--horizon", default=15, type=int, help="Consumption horizon.")
@click.option("--transport", default=0, type=int, help="Transportation Delay.")
@click.option("--time", default=7200, type=int, help="Total time limit.")
@click.option(
    "--neg-time", default=120, type=int, help="Time limit per single negotiation"
)
@click.option(
    "--neg-steps", default=20, type=int, help="Number of rounds per single negotiation"
)
@click.option(
    "--sign",
    default=1,
    type=int,
    help="The default delay between contract conclusion and signing",
)
@click.option(
    "--guaranteed",
    default=False,
    help="Whether to only sign contracts that are guaranteed not to cause " "breaches",
)
@click.option("--lines", default=10, help="The number of lines per factory")
@click.option(
    "--retrials",
    default=2,
    type=int,
    help="The number of times an agent re-tries on failed negotiations",
)
@click.option(
    "--use-consumer/--no-consumer",
    default=True,
    help="Use internal consumer object in factory managers",
)
@click.option(
    "--max-insurance",
    default="inf",
    type=float,
    help="Use insurance against partner in factory managers up to this premium. Pass zero for never buying insurance"
    " and a 'inf' (without quotes) for infinity.",
)
@click.option(
    "--riskiness", default=0.0, help="How risky is the default factory manager"
)
@click.option(
    "--competitors",
    default="GreedyFactoryManager",
    help="A semicolon (;) separated list of agent types to use for the competition.",
)
@click.option(
    "--jcompetitors",
    "--java-competitors",
    default="",
    help="A semicolon (;) separated list of agent types to use for the competition.",
)
@click.option(
    "--log",
    type=click.Path(file_okay=False, dir_okay=True),
    default="~/negmas/logs",
    help="Default location to save logs (A folder will be created under it)",
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
    default=False,
    help="If True, effort is exerted to reduce the memory footprint which"
    "includes reducing logs dramatically.",
)
@click.option(
    "--shared-profile/--multi-profile",
    default=True,
    help="If True, all lines in the same factory will have the same cost.",
)
@click.option(
    "--reserved-value",
    default="-inf",
    type=float,
    help="The reserved value used by GreedyFactoryManager",
)
@click.option(
    "--raise-exceptions/--ignore-exceptions",
    default=True,
    help="Whether to ignore agent exceptions",
)
@click.option(
    "--balance", default="1000.0", type=float, help="Initial balance of all factories"
)
@click.option(
    "--path",
    default="",
    help="A path to be added to PYTHONPATH in which all competitors are stored. You can path a : separated list of "
    "paths on linux/mac and a ; separated list in windows",
)
@click.option(
    "--world-config",
    type=click.Path(dir_okay=False, file_okay=True),
    default=tuple(),
    multiple=True,
    help="A file to load extra configuration parameters for world simulations from.",
)
@click_config_file.configuration_option()
def scml(
    steps,
    levels,
    neg_speedup,
    negotiator,
    agents,
    horizon,
    min_consumption,
    max_consumption,
    transport,
    time,
    neg_time,
    neg_steps,
    sign,
    guaranteed,
    lines,
    retrials,
    use_consumer,
    max_insurance,
    riskiness,
    competitors,
    jcompetitors,
    log,
    compact,
    log_ufuns,
    log_negs,
    reserved_value,
    balance,
    shared_profile,
    raise_exceptions,
    path,
    world_config,
):
    kwargs = dict(
        no_bank=True,
        no_insurance=False,
        prevent_cfp_tampering=True,
        ignore_negotiated_penalties=False,
        neg_step_time_limit=10,
        breach_penalty_society=0.02,
        premium=0.03,
        premium_time_increment=0.1,
        premium_breach_increment=0.001,
        max_allowed_breach_level=None,
        breach_penalty_society_min=0.0,
        breach_penalty_victim=0.0,
        breach_move_max_product=True,
        transfer_delay=0,
        start_negotiations_immediately=False,
        catalog_profit=0.15,
        financial_reports_period=10,
        default_price_for_products_without_one=1,
        compensation_fraction=0.5,
    )
    if world_config is not None and len(world_config) > 0:
        for wc in world_config:
            kwargs.update(load(wc))
    if len(path) > 0:
        sys.path.append(path)
    if max_insurance < 0:
        warnings.warn(
            f"Negative max insurance ({max_insurance}) is deprecated. Set --max-insurance=inf for always "
            f"buying and --max-insurance=0.0 for never buying. Will continue assuming --max-insurance=inf"
        )
        max_insurance = float("inf")

    if "." not in negotiator:
        negotiator = "negmas.sao." + negotiator

    params = {
        "steps": steps,
        "levels": levels,
        "neg_speedup": neg_speedup,
        "negotiator": negotiator,
        "agents": agents,
        "horizon": horizon,
        "min_consumption": min_consumption,
        "max_consumption": max_consumption,
        "transport": transport,
        "time": time,
        "neg_time": neg_time,
        "neg_steps": neg_steps,
        "sign": sign,
        "guaranteed": guaranteed,
        "lines": lines,
        "retrials": retrials,
        "use_consumer": use_consumer,
        "max_insurance": max_insurance,
        "riskiness": riskiness,
    }
    if compact:
        log_ufuns = False
        log_negs = False
    neg_speedup = neg_speedup if neg_speedup is not None and neg_speedup > 0 else None
    if min_consumption == max_consumption:
        consumption = min_consumption
    else:
        consumption = (min_consumption, max_consumption)
    customer_kwargs = {"negotiator_type": negotiator, "consumption_horizon": horizon}
    miner_kwargs = {"negotiator_type": negotiator, "n_retrials": retrials}
    factory_kwargs = {
        "negotiator_type": negotiator,
        "n_retrials": retrials,
        "sign_only_guaranteed_contracts": guaranteed,
        "use_consumer": use_consumer,
        "riskiness": riskiness,
        "max_insurance_premium": max_insurance,
        "reserved_value": reserved_value,
    }
    if log.startswith("~/"):
        log_dir = Path.home() / log[2:]
    else:
        log_dir = Path(log)
    world_name = unique_name(base="scml", add_time=True, rand_digits=0)
    log_dir = log_dir / world_name
    log_dir = log_dir.absolute()
    os.makedirs(log_dir, exist_ok=True)

    exception = None

    def _no_default(s):
        return not (
            s.startswith("negmas.apps.scml") and s.endswith("GreedyFactoryManager")
        )

    all_competitors = competitors.split(";")
    for i, cp in enumerate(all_competitors):
        if "." not in cp:
            all_competitors[i] = "negmas.apps.scml.factory_managers." + cp
    all_competitors_params = [
        dict() if _no_default(_) else factory_kwargs for _ in all_competitors
    ]
    if jcompetitors is not None and len(jcompetitors) > 0:
        jcompetitor_params = [{"java_class_name": _} for _ in jcompetitors.split(";")]
        for jp in jcompetitor_params:
            if "." not in jp["java_class_name"]:
                jp["java_class_name"] = (
                    "jnegmas.apps.scml.factory_managers." + jp["java_class_name"]
                )
        jcompetitors = ["negmas.apps.scml.JavaFactoryManager"] * len(jcompetitor_params)
        all_competitors += jcompetitors
        all_competitors_params += jcompetitor_params
        print("You are using some Java agents. The tournament MUST run serially")
        parallelism = "serial"
        if not jnegmas_bridge_is_running():
            print(
                "Error: You are using java competitors but jnegmas bridge is not running\n\nTo correct this issue"
                " run the following command IN A DIFFERENT TERMINAL because it will block:\n\n"
                "$ negmas jnegmas"
            )
            exit(1)

    world = SCMLWorld.chain_world(
        n_steps=steps,
        negotiation_speed=neg_speedup,
        n_intermediate_levels=levels,
        n_miners=agents,
        n_consumers=agents,
        n_factories_per_level=agents,
        consumption=consumption,
        consumer_kwargs=customer_kwargs,
        miner_kwargs=miner_kwargs,
        default_manager_params=factory_kwargs,
        transportation_delay=transport,
        time_limit=time,
        neg_time_limit=neg_time,
        neg_n_steps=neg_steps,
        default_signing_delay=sign,
        n_lines_per_factory=lines,
        compact=compact,
        agent_names_reveal_type=True,
        log_ufuns=log_ufuns,
        manager_types=all_competitors,
        manager_params=all_competitors_params,
        log_negotiations=log_negs,
        log_folder=log_dir,
        name=world_name,
        shared_profile_per_factory=shared_profile,
        initial_wallet_balances=balance,
        ignore_agent_exceptions=not raise_exceptions,
        ignore_contract_execution_exceptions=not raise_exceptions,
        **kwargs,
    )
    failed = False
    strt = perf_counter()
    try:
        for i in progressbar.progressbar(range(world.n_steps), max_value=world.n_steps):
            elapsed = perf_counter() - strt
            if world.time_limit is not None and elapsed >= world.time_limit:
                break
            if not world.step():
                break
    except Exception:
        exception = traceback.format_exc()
        failed = True
    elapsed = perf_counter() - strt

    def print_and_log(s):
        world.logdebug(s)
        print(s)

    world.logdebug(f"{pformat(world.stats, compact=True)}")
    world.logdebug(
        f"=================================================\n"
        f"steps: {steps}, horizon: {horizon}, time: {time}, levels: {levels}, agents_per_level: "
        f"{agents}, lines: {lines}, guaranteed: {guaranteed}, negotiator: {negotiator}\n"
        f"consumption: {consumption}"
        f", transport_to: {transport}, sign: {sign}, speedup: {neg_speedup}, neg_steps: {neg_steps}"
        f", retrials: {retrials}"
        f", neg_time: {neg_time}\n"
        f"=================================================="
    )

    save_stats(world=world, log_dir=log_dir, params=params)

    if len(world.saved_contracts) > 0:
        data = pd.DataFrame(world.saved_contracts)
        data = data.sort_values(["delivery_time"])
        data = data.loc[
            data.signed_at >= 0,
            [
                "seller_type",
                "buyer_type",
                "seller_name",
                "buyer_name",
                "delivery_time",
                "unit_price",
                "quantity",
                "product_name",
                "n_neg_steps",
                "signed_at",
            ],
        ]
        data.columns = [
            "seller_type",
            "buyer_type",
            "seller",
            "buyer",
            "t",
            "price",
            "q",
            "product",
            "steps",
            "signed",
        ]
        print_and_log(tabulate(data, headers="keys", tablefmt="psql"))

        data["product_id"] = np.array([_.id for _ in data["product"].values])
        d2 = (
            data.loc[(~(data["signed"].isnull())) & (data["signed"] > -1), :]
            .groupby(["product_id"])
            .apply(
                lambda x: pd.DataFrame(
                    [
                        {
                            "uprice": np.sum(x["price"] * x["q"]) / np.sum(x["q"]),
                            "quantity": np.sum(x["q"]),
                        }
                    ]
                )
            )
        )
        d2 = d2.reset_index().sort_values(["product_id"])
        products = dict(zip([_.id for _ in world.products], world.products))
        d2["Product"] = np.array([products[_] for _ in d2["product_id"].values])
        d2 = d2.loc[:, ["Product", "uprice", "quantity"]]
        d2.columns = ["Product", "Avg. Unit Price", "Total Quantity"]
        print_and_log(tabulate(d2, headers="keys", tablefmt="psql"))

        n_executed = sum(world.stats["n_contracts_executed"])
        n_negs = sum(world.stats["n_negotiations"])
        n_contracts = len(world.saved_contracts)
        try:
            agent_scores = sorted(
                [
                    [_.name, world.a2f[_.id].total_balance]
                    for _ in world.agents.values()
                    if isinstance(_, FactoryManager)
                ],
                key=lambda x: x[1],
                reverse=True,
            )
            agent_scores = pd.DataFrame(
                data=np.array(agent_scores), columns=["Agent", "Final Balance"]
            )
            print_and_log(tabulate(agent_scores, headers="keys", tablefmt="psql"))
        except:
            pass
        winners = [
            f"{_.name} gaining {world.a2f[_.id].total_balance / world.a2f[_.id].initial_balance - 1.0:0.0%}"
            for _ in world.winners
        ]
        print_and_log(
            f"{n_contracts} contracts :-) [N. Negotiations: {n_negs}, Agreement Rate: "
            f"{world.agreement_rate:0.0%}]"
            f" (rounds/successful negotiation: {world.n_negotiation_rounds_successful:5.2f}, "
            f"rounds/broken negotiation: {world.n_negotiation_rounds_failed:5.2f})"
        )
        print_and_log(
            f"Cancelled: {world.cancellation_rate:0.0%}, Executed: {world.contract_execution_fraction:0.0%}"
            f", Breached: {world.breach_rate:0.0%}, N. Executed: {n_executed}, Business size: "
            f"{world.business_size}\n"
            f"Winners: {winners}\n"
            f"Running Time {humanize_time(elapsed)}"
        )
    else:
        print_and_log("No contracts! :-(")
        print_and_log(f"Running Time {humanize_time(elapsed)}")

    if failed:
        print(exception)
        world.logdebug(exception)
        print(f"FAILED at step {world.current_step} of {world.n_steps}\n")


@cli.command(help="Start the bridge to genius (to use GeniusNegotiator)")
@click.option(
    "--path",
    "-p",
    default="auto",
    help='Path to genius-8.0.4.jar with embedded NegLoader. Use "auto" to '
    "read the path from ~/negmas/config.json"
    "\n\tConfig key is genius_bridge_jar"
    "\nYou can download this jar from: "
    "http://www.yasserm.com/scml/genius-8.0.4-bridge.jar",
)
@click.option(
    "--port",
    "-r",
    default=0,
    help="Port to run the NegLoader on. Pass 0 for the default value",
)
@click.option(
    "--force/--no-force",
    default=False,
    help="Force trial even if an earlier instance exists",
)
def genius(path, port, force):
    negmas.init_genius_bridge(
        path=path if path != "auto" else None, port=port, force=force
    )


@cli.command(help="Start the bridge to JNegMAS (to use Java agents in worlds)")
@click.option(
    "--path",
    "-p",
    default="auto",
    help='Path to jnegmas*.jar with. Use "auto" to '
    "read the path from ~/negmas/config.json."
    "\n\tConfig key is jnegmas_jar"
    "\nYou can download the latest version of this jar from: "
    "http://www.yasserm.com/scml/jnegmas-all.jar",
)
@click.option(
    "--port",
    "-r",
    default=0,
    help="Port to run the jnegmas on. Pass 0 for the default value",
)
@click_config_file.configuration_option()
def jnegmas(path, port):
    init_jnegmas_bridge(path=path if path != "auto" else None, port=port)
    input(
        "Press C^c to quit. You may also need to kill any remaining java processes manually"
    )


def download_and_set(key, url, file_name):
    """
    Downloads a file and sets the corresponding key in ~/negmas/config.json

    Args:
        key: Key name in config.json
        url: URL to download from
        file_name: file name

    Returns:

    """
    config_path = Path.home() / "negmas"
    jar_path = config_path / "files"
    jar_path.mkdir(parents=True, exist_ok=True)
    jar_path = jar_path / file_name
    urllib.request.urlretrieve(url, jar_path)
    config = {}
    config_file = config_path / "config.json"
    if config_file.exists():
        with open(config_file, "r") as f:
            config = json.load(f)
    config[key] = str(jar_path)
    with open(config_file, "w") as f:
        json.dump(config, fp=f, sort_keys=True, indent=4)


@cli.command(help="Downloads jnegmas and updates your settings")
def jnegmas_setup():
    url = f"http://www.yasserm.com/scml/{JNEGMAS_JAR_NAME}"
    print(f"Downloading: {url}", end="", flush=True)
    download_and_set(key="jnegmas_jar", url=url, file_name="jnegmas.jar")
    print(" done successfully")


@cli.command(help="Downloads the genius bridge and updates your settings")
def genius_setup():
    url = f"http://www.yasserm.com/scml/{GENIUS_JAR_NAME}"
    print(f"Downloading: {url}", end="", flush=True)
    download_and_set(key="genius_bridge_jar", url=url, file_name="jnegmas.jar")
    print(" done successfully")


@cli.command(help="Prints NegMAS version")
def version():
    print(negmas.__version__)


if __name__ == "__main__":
    cli()
