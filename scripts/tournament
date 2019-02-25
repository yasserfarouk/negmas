#!/usr/bin/env python
from functools import partial
from time import perf_counter

import click
from tabulate import tabulate

from negmas.apps.scml import GreedyFactoryManager, DoNothingFactoryManager
from negmas.apps.scml.utils import anac2019_tournament
from negmas.helpers import humanize_time

n_completed = 0
n_total = 0


def print_progress(_, i, n) -> None:
    global n_completed, n_total
    n_completed = i + 1
    n_total = n
    print(f'{i:04} of {n:04} worlds completed ({i / n:0.2%})', flush=True)


def print_world_progress(world) -> None:
    step = world.current_step + 1
    s = f'World# {n_completed:04}: {step:04}  of {world.n_steps:04} ' \
        f'steps completed ({step / world.n_steps:0.2f}) '
    if n_total > 0:
        s += f'TOTAL: ({n_completed + step / world.n_steps / n_total:0.2f})'
    print(s, flush=True)


click.option = partial(click.option, show_default=True)


@click.command()
@click.option('--steps', '-s', default=60, help='Number of steps.')
@click.option('--config', '-f', default='anac2019', help='The config to use. Default is ANAC 2019')
@click.option('--worlds', '-w', default=50, help='Number of worlds to run.')
@click.option('--competitors', '-c'
              , default='negmas.apps.scml.DoNothingFactoryManager;negmas.apps.scml.GreedyFactoryManager'
              , help='A semicolon (;) separated list of agent types to use for the competition.')
@click.option('--parallel/--serial', default=True, help='Run a distributed tournament using dask')
@click.option('--log', '-l', default='~/negmas/logs/tournaments',
              help='Default location to save logs (A folder will be created under it)')
@click.option('--verbose/--silent', default=False, help='verbosity')
def cli(steps, worlds, parallel, config, log, competitors, verbose):
    if config.lower() != 'anac2019':
        print('Only anac2019 config is supported')
        exit(1)
    parallelism = 'parallel' if parallel else 'serial'
    start = perf_counter()
    results = anac2019_tournament(competitors=competitors.split(';'),
                                  tournament_path=log
                                  , agent_names_reveal_type=False
                                  , n_runs=worlds, n_steps=steps, parallelism=parallelism
                                  , verbose=verbose
                                  , tournament_progress_callback=print_progress if verbose else None
                                  , world_progress_callback=print_world_progress if parallelism != 'dask' and verbose else None
                                  )
    print(tabulate(results.total_scores, headers='keys', tablefmt='psql'))
    print(f'Finished in {humanize_time(perf_counter() - start)} [{parallelism}]')


if __name__ == '__main__':
    cli()
