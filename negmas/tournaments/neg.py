"""
Negotiation tournaments module.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from functools import partial
from itertools import combinations, cycle, permutations
from os import PathLike
from random import randint
from typing import Any, Generator, Sequence

from negmas.helpers import get_class, unique_name
from negmas.negotiators import Negotiator
from negmas.outcomes import Issue, make_issue
from negmas.preferences.crisp.linear import LinearUtilityFunction
from negmas.serialization import deserialize, serialize
from negmas.situated import Agent
from negmas.situated.neg import NegAgent, NegDomain, NegWorld  # , _wrap_in_agents
from negmas.tournaments.tournaments import (
    TournamentResults,
    WorldRunResults,
    create_tournament,
    tournament,
)

__all__ = [
    "create_neg_tournament",
    "neg_tournament",
    "random_discrete_domains",
    "domains_from_list",
]


def neg_config_generator(
    n_competitors: int,
    domains: Generator[NegDomain, None, None],
    n_agents_per_competitor: int = 1,
    agent_names_reveal_type: bool = False,
    non_competitors: tuple[str | NegAgent] | None = None,
    non_competitor_params: tuple[dict[str, Any]] | None = None,
    compact: bool = False,
    **kwargs,
) -> list[dict[str, Any]]:
    """
    Used internally to generate the configuration of a world.

    Args:
        n_competitors: How many competitors will run?
        domains: A generator used to extract `NegDomain` objects (one config per `NegDomain`)
        n_agents_per_competitor: Must be 1
        agent_names_reveal_type: If given, the agent name will be the same as its class name
        non_competitors: Must be None
        non_competitor_params: Must be None
        compact: If given, the system tries to minimize the memory footprint and secondary storage
        kwargs: Extra arguments passed directly to the world constructor

    Returns:
        A list containing one world configuration
    """
    domain = next(domains)
    if non_competitors:
        raise ValueError(
            f"Non-competitors are not supported for negotiation tournaments. (You provided {non_competitors})"
        )
    if n_agents_per_competitor != 1:
        raise ValueError(
            f"n_agents_per_competitor must be 1 ({n_agents_per_competitor} given)"
        )

    world_name = unique_name("", add_time=True, rand_digits=4)
    no_logs = compact
    world_params = dict(
        name=world_name,
        domain=serialize(domain),
        compact=compact,
        no_logs=no_logs,
    )
    world_params.update(kwargs)
    config = {
        "world_params": world_params,
        "scoring_context": {},
        "non_competitors": None,
        "non_competitor_params": None,
        "is_default": [False] * n_competitors + [True] * (len(domain.ufuns) - 1),
    }
    return [config]


def neg_config_assigner(
    config: list[dict[str, Any]],
    max_n_worlds: int = 1,
    n_agents_per_competitor: int = 1,
    fair: bool = True,
    competitors: Sequence[type[Negotiator] | type[Agent]] = (),
    params: Sequence[dict[str, Any]] = (),
    dynamic_non_competitors: list[type[Agent]] | None = None,
    dynamic_non_competitor_params: list[dict[str, Any]] | None = None,
    exclude_competitors_from_reassignment: bool = True,
) -> list[list[dict[str, Any]]]:
    """
    Assigns agents to the world configuration.

    In negotiation tournaments, a single assignment is generated per configuration in
    which all agents exist in the world.

    All parameters other than `competitors` and `params` are ignored
    """
    competitors, params = deepcopy(competitors), deepcopy(params)
    # competitors, params = _wrap_in_agents(competitors, params, NegAgent)
    config[0]["world_params"]["types"] = competitors
    config[0]["world_params"]["params"] = params
    return [config]


def neg_world_generator(**kwargs):
    """
    Generates the world
    """
    config = kwargs.pop("world_params", dict())
    config["types"], config["params"] = deepcopy(config["types"]), deepcopy(
        config["params"]
    )
    config["types"] = [get_class(_) for _ in config["types"]]
    config["domain"] = deserialize(config["domain"])
    return NegWorld(**config)


def neg_score_calculator(
    worlds: list[NegWorld],
    scoring_context: dict[str, Any],
    dry_run: bool,
    scoring_method: str = "received_utility",
) -> WorldRunResults:
    """A scoring function that scores agents based on their performance.

    Args:

        worlds: The world which is assumed to be run up to the point at which the scores are to be calculated.
        scoring_context:  A dict of context parameters passed by the world generator or assigner.
        dry_run: A boolean specifying whether this is a dry_run. For dry runs, only names and types are expected in
                 the returned `WorldRunResults`
        scoring_method: the method used for scoring. Can be received_utility, partner_utility, received_advantage, partner_advantage

    Returns:
        WorldRunResults giving the names, scores, and types of factory managers.

    """
    if scoring_context is not None:
        scoring_method = scoring_context.get("scoring_method", scoring_method)
    assert len(worlds) == 1
    world = worlds[0]

    fun = dict(
        received_utility=world.received_utility,
        received_advantage=world.received_advantage,
        partner_utility=world.partner_utility,
        partner_advantage=world.partner_advantage,
    )

    result = WorldRunResults(
        world_names=[world.name], log_file_names=[world.log_file_name]
    )
    extra = defaultdict(list)
    for aid, agent in world.competitors.items():
        agent_type = agent.type_name
        result.names.append(agent.name)
        result.ids.append(agent.id)
        result.types.append(agent_type)
        if dry_run:
            result.scores.append(None)
            continue
        result.scores.append(fun[scoring_method](aid))
        for k, f in fun.items():
            if k == scoring_method:
                continue
            extra[k].append(dict(type=k, score=f(aid)))
    for k in fun.keys():
        result.extra_scores[k] = extra[k]
    return result


def _update_kwargs(kwargs, domains, competitors):
    kwargs["config_generator"] = partial(neg_config_generator, domains=domains)
    kwargs["config_assigner"] = neg_config_assigner
    kwargs["world_generator"] = neg_world_generator
    kwargs["score_calculator"] = neg_score_calculator
    kwargs["n_competitors_per_world"] = len(competitors)
    kwargs["n_agents_per_competitor"] = 1
    kwargs["max_worlds_per_config"] = 1
    kwargs["non_competitors"] = None
    kwargs["non_competitor_params"] = None
    kwargs["dynamic_non_competitors"] = False
    kwargs["dynamic_non_competitor_params"] = False
    kwargs["exclude_competitors_from_reassignment"] = False
    return kwargs


def create_neg_tournament(
    competitors: Sequence[str | type[Agent]],
    domains: Generator[NegDomain, None, None],
    competitor_params: Sequence[dict | None] | None = None,
    **kwargs,
) -> PathLike:
    """
    Creates a tournament

    Args:
        competitors: A list of class names for the competitors
        domains:A generator that yields `NegDomain` objects specifying negotiation domains upon request
        name: Tournament name
        competitor_params: A list of competitor parameters (used to initialize the competitors).
        agent_names_reveal_type: If true then the type of an agent should be readable in its name (most likely at its
                                 beginning).
        n_configs: The number of different world configs (up to competitor assignment) to be generated.
        n_runs_per_world: Number of runs per world. All of these world runs will have identical competitor assignment
                          and identical world configuration.
        total_timeout: Total timeout for the complete process
        base_tournament_path: Path at which to store all results. A new folder with the name of the tournament will be
                         created at this path. A scores.csv file will keep the scores and logs folder will keep detailed
                         logs
        parallelism: Type of parallelism. Can be 'serial' for serial, 'parallel' for parallel and 'distributed' for
                     distributed! For parallel, you can add the fraction of CPUs to use after a colon (e.g. parallel:0.5
                     to use half of the CPU in the machine). By defaults parallel uses all CPUs in the machine
        scheduler_port: Port of the dask scheduler if parallelism is dask, dist, or distributed
        scheduler_ip:   IP Address of the dask scheduler if parallelism is dask, dist, or distributed
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
    return create_tournament(
        competitors=competitors,
        competitor_params=competitor_params,
        **_update_kwargs(kwargs, domains, competitors),
    )


def neg_tournament(
    competitors: Sequence[str | type[Agent] | type[Negotiator]],
    domains: Generator[NegDomain, None, None],
    competitor_params: Sequence[dict | None] | None = None,
    **kwargs,
) -> TournamentResults | PathLike:
    """
    Runs a tournament

    Args:

        competitors: A list of class names for the competitors
        domains:A generator that yields `NegDomain` objects specifying negotiation domains upon request
        name: Tournament name
        competitor_params: A list of competitor parameters (used to initialize the competitors).
        stage_winners_fraction: in [0, 1).  Fraction of agents to to go to the next stage at every stage. If zero, and
                                            round_robin, it becomes a single stage competition.
        agent_names_reveal_type: If true then the type of an agent should be readable in its name (most likely at its
                                 beginning).
        n_configs: The number of different world configs (up to competitor assignment) to be generated.
        n_runs_per_world: Number of runs per world. All of these world runs will have identical competitor assignment
                          and identical world configuration.
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
    return tournament(
        competitors=competitors,
        competitor_params=competitor_params,
        **_update_kwargs(kwargs, domains, competitors),
    )


def random_discrete_domains(
    issues: list[Issue | int | tuple[int, int]],
    partners: list[Negotiator],
    n_negotiators=2,
    positions: int | tuple[int, int] = None,
    normalized=True,
    ufun_type=LinearUtilityFunction,
    roles: list[str] | None = None,
    partner_extraction_method="round-robin",
) -> Generator[NegDomain, None, None]:
    """
    Generates an infinite sequence of random discrete domains

    Args:
        issues: A list defining the issue space. Each element can be an `Issue`
                object, an integer (defining the number of outcomes) or a tuple
                of two integers (defining the minimum and maximum number of outcomes
                for the issue).
        partners: A list of `Negotiator` types from which partners are extracted.
                  The system will extract `n_negotiators` - 1 partners for each
                  domain.
        n_negotiators: The number of negotiators in each negotiation.
        positions: The positions at which the competitors will be added in all
                   negotiations.
        normalized: Will the ufuns generated by normalized
        ufun_type: Type of the utility function to use.
        roles: The roles of the `n_negotiators` (including the competitor) in negotiations
        partner_extraction_method: The method used to create partners for negotaitions
                                   from the given `partners` list:

                                   - round-robin: will extract overalapping `n_negotiators` - 1 sets
                                   - permutations: Will use all `n_negotaitors` - 1 permutations
                                   - random: Will sample randm sets of `n_negotiators` - 1 partners
                                   - compinations: Will use all `n_negotiators` - 1 combinations
    """
    if positions is None:
        positions = (0, n_negotiators)
    elif isinstance(positions, int):
        positions = (positions, positions)

    while len(partners) < n_negotiators - 1:
        partners += [_ for _ in partners]

    def intin(i):
        if isinstance(i, int):
            return i
        return randint(i[0], i[1])

    while True:
        current_issues = [
            _ if isinstance(_, Issue) else make_issue(values=intin(_), name=f"i{i}")
            for i, _ in enumerate(issues)
        ]
        ufuns = [
            ufun_type.random(
                current_issues, reserved_value=(0.0, 0.2), normalized=normalized
            )
            for _ in range(n_negotiators)
        ]

        def partners_generator():
            if partner_extraction_method.startswith("permutation"):
                yield from permutations(partners, n_negotiators - 1)
            if partner_extraction_method.startswith("combination"):
                yield from combinations(partners, n_negotiators - 1)
            else:
                is_random = partner_extraction_method.startswith("random")
                n = len(partners) - n_negotiators + 1
                for j in range(n):
                    if is_random:
                        yield partners[randint(0, n) : j + n_negotiators - 1]
                    else:
                        yield partners[j : j + n_negotiators - 1]

        for u in permutations(ufuns):
            for index in range(*positions):
                for p in partners_generator():
                    assert len(u) == len(p) + 1
                    yield NegDomain(
                        name="d0",
                        ufuns=u,
                        issues=current_issues,
                        partner_types=p,
                        index=index,
                        roles=roles,
                    )


def domains_from_list(domains: list[NegDomain]):
    """
    Creats an appropriate `NegDomain` generator from a list/tuple of domains
    """
    return cycle(domains)


if __name__ == "__main__":
    from negmas.sao import AspirationNegotiator, NaiveTitForTatNegotiator

    domains = random_discrete_domains(
        issues=[5, 4, (3, 5)],
        partners=[AspirationNegotiator, NaiveTitForTatNegotiator],
    )
    print(
        neg_tournament(
            n_configs=2 * 2 * 2 * 4,
            domains=domains,
            competitors=[AspirationNegotiator, NaiveTitForTatNegotiator],
            n_steps=1,
            neg_n_steps=10,
            neg_time_limit=None,
            name="neg-tour-test",
        )
    )
