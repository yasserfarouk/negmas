#!/usr/bin/env python

import multiprocessing
import os
import traceback
from functools import partial
from pathlib import Path
from pprint import pformat
from time import perf_counter, sleep

import click
import pandas as pd
import pkg_resources
import progressbar
import yaml
from tabulate import tabulate

import negmas
from negmas import save_stats, tournaments
from negmas.apps.scml import *
from negmas.apps.scml.utils import anac2019_world, balance_calculator
from negmas.helpers import humanize_time, unique_name
from negmas.java import init_jnegmas_bridge

try:
    # disable a warning in yaml 1b1 version
    yaml.warnings({'YAMLLoadWarning': False})
except:
    pass

n_completed = 0
n_total = 0
external_path = pkg_resources.resource_filename('negmas', resource_name='external/genius-8.0.4.jar')


def print_progress(_, i, n) -> None:
    global n_completed, n_total
    n_completed = i + 1
    n_total = n
    print(f'{n_completed:04} of {n:04} worlds completed ({n_completed / n:0.2%})', flush=True)


def print_world_progress(world) -> None:
    step = world.current_step + 1
    s = f'World# {n_completed:04}: {step:04}  of {world.n_steps:04} ' \
        f'steps completed ({step / world.n_steps:0.2f}) '
    if n_total > 0:
        s += f'TOTAL: ({n_completed + step / world.n_steps / n_total:0.2f})'
    print(s, flush=True)


click.option = partial(click.option, show_default=True)


@click.group()
def cli():
    pass


@cli.command(help='Run a tournament between candidate agent types')
@click.option('--name', '-n', default='random',
              help='The name of the tournament. The special value "random" will result in a random name')
@click.option('--steps', '-s', default=60, help='Number of steps.')
@click.option('--ttype', '--tournament-type', '--tournament', default='anac2019'
    , help='The config to use. Default is ANAC 2019')
@click.option('--timeout', '-t', default=0, help='Timeout after the given number of seconds (0 for infinite)')
@click.option('--runs', default=5, help='Number of runs for each configuration')
@click.option('--max-runs', default=-1, help='Maximum total number of runs. Zero or negative numbers mean no limit')
@click.option('--randomize/--permutations', default=False, help='Random worlds or try all permutations up to max-runs')
@click.option('--competitors'
    , default='negmas.apps.scml.DoNothingFactoryManager;negmas.apps.scml.GreedyFactoryManager'
    , help='A semicolon (;) separated list of agent types to use for the competition.')
@click.option('--parallel/--serial', default=True, help='Run a parallel/serial tournament on a single machine')
@click.option('--distributed/--single-machine', default=False, help='Run a distributed tournament using dask')
@click.option('--log', '-l', default='~/negmas/logs/tournaments',
              help='Default location to save logs (A folder will be created under it)')
@click.option('--verbosity', default=1, help='verbosity level (from 0 == silent to 1 == world progress)')
@click.option('--configs-only/--run', default=False, help='configs_only')
@click.option('--reveal-names/--hidden-names', default=False, help='Reveal agent names (should be used only for '
                                                                   'debugging)')
@click.option('--ip', default='127.0.0.1', help='The IP address for a dask scheduler to run the distributed tournament.'
                                                ' Effective only if --distributed')
@click.option('--port', default=8786, help='The IP port number a dask scheduler to run the distributed tournament.'
                                           ' Effective only if --distributed')
def tournament(name, steps, parallel, distributed, ttype, timeout, log, verbosity, configs_only,
               reveal_names, ip, port, runs, max_runs, randomize, competitors
               ):
    if timeout <= 0:
        timeout = None
    if name == 'random':
        name = None
    if max_runs <= 0:
        max_runs = None
    parallelism = 'distributed' if distributed else 'parallel' if parallel else 'serial'
    start = perf_counter()
    if ttype.lower() == 'anac2019':
        results = tournaments.tournament(competitors=competitors.split(';'), agent_names_reveal_type=reveal_names
                             , tournament_path=log, total_timeout=timeout
                             , parallelism=parallelism, scheduler_ip=ip, scheduler_port=port
                             , world_progress_callback=print_world_progress if verbosity > 1 and not distributed else None
                             , name=name, verbose=verbosity > 0, n_runs_per_config=runs, max_n_runs=max_runs
                             , world_generator=anac2019_world, score_calculator=balance_calculator
                             , configs_only=configs_only, randomize=randomize
                             , n_steps=steps)
    else:
        print('Only anac2019 tournament type is supported')
        exit(1)
    if configs_only:
        print(f'Saved all configs to {str(results)}')
        print(f'Finished in {humanize_time(perf_counter() - start)} [config-only]')
        return
    print(tabulate(results.total_scores, headers='keys', tablefmt='psql'))
    print(f'Finished in {humanize_time(perf_counter() - start)} [{parallelism}]')


@cli.command(help='Run an SCML world simulation')
@click.option('--steps', default=120, help='Number of steps.')
@click.option('--levels', default=3, help='Number of intermediate production levels (processes). '
                                          '-1 means a single product and no factories.')
@click.option('--neg-speedup', default=21, help='Negotiation Speedup.')
@click.option('--negotiator', default='negmas.sao.AspirationNegotiator',
              help='Negotiator type to use for builtin agents.')
@click.option('--min-consumption', default=3, help='The minimum number of units consumed by each consumer at every '
                                                   'time-step.')
@click.option('--max-consumption', default=5, help='The maximum number of units consumed by each consumer at every '
                                                   'time-step.')
@click.option('--agents', default=5, help='Number of agents (miners/negmas.consumers) per production level')
@click.option('--horizon', default=20, help='Consumption horizon.')
@click.option('--transport', default=0, help='Transportation Delay.')
@click.option('--time', default=60 * 90, help='Total time limit.')
@click.option('--neg-time', default=60 * 4, help='Time limit per single negotiation')
@click.option('--neg-steps', default=20, help='Number of rounds per single negotiation')
@click.option('--sign', default=1, help='The default delay between contract conclusion and signing')
@click.option('--guaranteed', default=False, help='Whether to only sign contracts that are guaranteed not to cause '
                                                  'breaches')
@click.option('--lines', default=10, help='The number of lines per factory')
@click.option('--retrials', default=5, help='The number of times an agent re-tries on failed negotiations')
@click.option('--use-consumer', default=True, help='Use internal consumer object in factory managers')
@click.option('--max-insurance', default=100, help='Use insurance against partner in factory managers up to this '
                                                   'premium')
@click.option('--riskiness', default=0.0, help='How risky is the default factory manager')
@click.option('--log', default='~/negmas/logs',
              help='Default location to save logs (A folder will be created under it)')
def scml(steps, levels, neg_speedup, negotiator, agents, horizon, min_consumption, max_consumption
         , transport, time, neg_time
         , neg_steps, sign, guaranteed, lines, retrials, use_consumer, max_insurance, riskiness, log):
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
        "riskiness": riskiness
    }
    neg_speedup = neg_speedup if neg_speedup is not None and neg_speedup > 0 else None
    if min_consumption == max_consumption:
        consumption = min_consumption
    else:
        consumption = (min_consumption, max_consumption)
    customer_kwargs = {'negotiator_type': negotiator, 'consumption_horizon': horizon}
    miner_kwargs = {'negotiator_type': negotiator, 'n_retrials': retrials}
    factory_kwargs = {'negotiator_type': negotiator, 'n_retrials': retrials
        , 'sign_only_guaranteed_contracts': guaranteed, 'use_consumer': use_consumer
        , 'riskiness': riskiness, 'max_insurance_premium': max_insurance}
    if log.startswith('~/'):
        log_dir = Path.home() / log[2:]
    else:
        log_dir = Path(log)
    log_dir = log_dir / unique_name(base='scml', add_time=True, rand_digits=0)
    log_dir = log_dir.absolute()
    os.makedirs(log_dir, exist_ok=True)
    log_file_name = str(log_dir / 'log.txt')
    stats_file_name = str(log_dir / 'stats.json')
    params_file_name = str(log_dir / 'params.json')
    world = SCMLWorld.single_path_world(log_file_name=log_file_name, n_steps=steps
                                        , negotiation_speed=neg_speedup
                                        , n_intermediate_levels=levels
                                        , n_miners=agents
                                        , n_consumers=agents
                                        , n_factories_per_level=agents
                                        , consumption=consumption
                                        , consumer_kwargs=customer_kwargs
                                        , miner_kwargs=miner_kwargs
                                        , manager_kwargs=factory_kwargs
                                        , transportation_delay=transport, time_limit=time, neg_time_limit=neg_time
                                        , neg_n_steps=neg_steps, default_signing_delay=sign
                                        , n_lines_per_factory=lines)
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

    world.logdebug(f'{pformat(world.stats, compact=True)}')
    world.logdebug(f'=================================================\n'
                   f'steps: {steps}, horizon: {horizon}, time: {time}, levels: {levels}, agents_per_level: '
                   f'{agents}, lines: {lines}, guaranteed: {guaranteed}, negotiator: {negotiator}\n'
                   f'consumption: {consumption}'
                   f', transport_to: {transport}, sign: {sign}, speedup: {neg_speedup}, neg_steps: {neg_steps}'
                   f', retrials: {retrials}'
                   f', neg_time: {neg_time}\n'
                   f'==================================================')

    save_stats(world=world, log_dir=log_dir, params=params)

    if len(world.saved_contracts) > 0:
        data = pd.DataFrame(world.saved_contracts)
        data = data.sort_values(['delivery_time'])
        data = data.loc[:, ['seller_type', 'buyer_type', 'seller_name', 'buyer_name', 'delivery_time', 'unit_price'
                               , 'quantity', 'product_name', 'n_neg_steps', 'signed_at', 'concluded_at', 'cfp']]
        print_and_log(tabulate(data, headers='keys', tablefmt='psql'))
        n_executed = sum(world.stats['n_contracts_executed'])
        n_negs = sum(world.stats["n_negotiations"])
        n_contracts = len(world.saved_contracts)
        winners = [f'{_.name} gaining {world.a2f[_.id].balance / world.a2f[_.id].initial_balance - 1.0:0.0%}'
                   for _ in world.winners]
        print_and_log(f'{n_contracts} contracts :-) [N. Negotiations: {n_negs}'
                      f', Agreement Rate: {world.agreement_rate:0.0%}]')
        print_and_log(f'Executed: {world.contract_execution_fraction:0.0%}'
                      f', Breached: {world.breach_rate:0.0%}'
                      f', N. Executed: {n_executed}'
                      f', Business size: {world.business_size}\n'
                      f'Winners: {winners}\n'
                      f'Running Time {humanize_time(elapsed)}')
    else:
        print_and_log('No contracts! :-(')
        print_and_log(f'Running Time {humanize_time(elapsed)}')

    if failed:
        print(exception)
        world.logdebug(exception)
        print(f'FAILED at step {world.current_step} of {world.n_steps}\n')


@cli.command(help='Start the bridge to genius (to use GeniusNegotiator)')
@click.option('--path', '-p', default=external_path, help='Path to genius-8.0.4.jar with embedded NegLoader')
@click.option('--port', '-r', default=0, help='Port to run the NegLoader on. Pass 0 for the default value')
@click.option('--force/--no-force', default=False, help='Force trial even if an earlier instance exists')
def genius(path, port, force):
    negmas.init_genius_bridge(path=path, port=port, force=force)


@cli.command(help='Start the bridge to JNegMAS (to use Java agents in worlds)')
@click.option('--path', '-p', default=None, help='Path to jnegmas library. If not given, an internal version '
                                                          'will be used')
@click.option('--port', '-r', default=0, help='Port to run the jnegmas on. Pass 0 for the default value')
def jnegmas(path, port):
    init_jnegmas_bridge(path=path, port=port)


if __name__ == '__main__':
    cli()
