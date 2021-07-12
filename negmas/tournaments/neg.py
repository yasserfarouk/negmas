from itertools import cycle, permutations
from random import randint
from functools import partial
from os import PathLike
from typing import Any, Dict, Generator, List, Optional, Sequence, Tuple, Type, Union

from negmas.helpers import unique_name, get_class
from negmas.negotiators import Negotiator
from negmas.outcomes import Issue
from negmas.serialization import deserialize, serialize
from negmas.situated import Agent
from negmas.tournaments.tournaments import (
    TournamentResults,
    WorldRunResults,
    create_tournament,
    tournament,
)
from negmas.situated.neg import NegDomain, NegWorld, NegAgent, _wrap_in_agents
from copy import deepcopy

__all__ = [
    "create_neg_tournament",
    "neg_tournament",
    "random_discrete_domains",
    "domains_from_list"
]


def neg_config_generator(
    n_competitors: int,
    domains: Generator[NegDomain, None, None],
    n_agents_per_competitor: int = 1,
    agent_names_reveal_type: bool = False,
    non_competitors: Optional[Tuple[Union[str, NegAgent]]] = None,
    non_competitor_params: Optional[Tuple[Dict[str, Any]]] = None,
    compact: bool = False,
    **kwargs,
) -> List[Dict[str, Any]]:
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
    }
    config.update(kwargs)
    return [config]


def neg_config_assigner(
    config: List[Dict[str, Any]],
    max_n_worlds: int = 1,
    n_agents_per_competitor: int = 1,
    fair: bool = True,
    competitors: Sequence[Type[Agent]] = (),
    params: Sequence[Dict[str, Any]] = (),
    dynamic_non_competitors: Optional[List[Type[Agent]]] = None,
    dynamic_non_competitor_params: Optional[List[Dict[str, Any]]] = None,
    exclude_competitors_from_reassignment: bool = True,
) -> List[List[Dict[str, Any]]]:
    competitors, params = deepcopy(competitors), deepcopy(params)
    competitors, params = _wrap_in_agents(competitors, params, NegAgent)
    config[0]["world_params"]["types"] = competitors
    config[0]["world_params"]["params"] = params
    return [config]


def neg_world_generator(**kwargs):
    config = kwargs.pop("world_params", dict())
    config["types"], config["params"] = deepcopy(config["types"]), deepcopy(
        config["params"]
    )
    config["types"] = [get_class(_) for _ in config["types"]]
    config["domain"] = deserialize(config["domain"])
    return NegWorld(**config)


def neg_score_calculator(
    worlds: List[NegWorld],
    scoring_context: Dict[str, Any],
    dry_run: bool,
    scoring_method: str = "received_utility",
) -> WorldRunResults:
    """A scoring function that scores factory managers' performance by the final balance only ignoring whatever still
    in their inventory.

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
    extra = []
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
            extra.append(dict(type=k, score=f(aid)))
    for k in fun.keys():
        result.extra_scores[k] = extra
    return result


def create_neg_tournament(
    competitors: Sequence[Union[str, Type[Agent]]],
    domains: Generator[NegDomain, None, None],
    competitor_params: Optional[Sequence[Optional[dict]]] = None,
    **kwargs,
) -> PathLike:
    """
    Creates a tournament

    Args:
        competitors: A list of class names for the competitors
        domains:A generator that yields `NegDomain` objects specifying negotiation domains upon request
        name: Tournament name
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
    kwargs["config_generator"] = partial(neg_config_generator, domains=domains)
    kwargs["config_assigner"] = neg_config_assigner
    kwargs["world_generator"] = neg_world_generator
    kwargs["score_calculator"] = neg_score_calculator
    kwargs["n_agents_per_competitor"] = 1
    return create_tournament(
        competitors=competitors, competitor_params=competitor_params, **kwargs
    )


def neg_tournament(
    competitors: Sequence[Union[str, Type[Agent], Type[Negotiator]]],
    domains: Generator[NegDomain, None, None],
    competitor_params: Optional[Sequence[Optional[dict]]] = None,
    **kwargs,
) -> Union[TournamentResults, PathLike]:
    """
    Runs a tournament

    Args:

        competitors: A list of class names for the competitors
        domains:A generator that yields `NegDomain` objects specifying negotiation domains upon request
        name: Tournament name
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
    kwargs["config_generator"] = partial(neg_config_generator, domains=domains)
    kwargs["config_assigner"] = neg_config_assigner
    kwargs["world_generator"] = neg_world_generator
    kwargs["score_calculator"] = neg_score_calculator
    kwargs["n_agents_per_competitor"] = 1
    return tournament(
        competitors=competitors, competitor_params=competitor_params, **kwargs
    )


def random_discrete_domains(
    issues: List[Union[Issue, Union[int, Tuple[int, int]]]],
    partners: List[Negotiator],
    n_negotiators=2,
    positions: Tuple[int, int] = None,
    normalized=True,
) -> Generator[NegDomain, None, None]:
    from negmas.utilities import LinearUtilityAggregationFunction as U
    if positions is None:
        positions = (0, n_negotiators)

    while len(partners) < n_negotiators:
        partners += [_ for _ in partners]
        if len(partners) > n_negotiators:
            partners = partners[:n_negotiators]

    def partner_generator():
        n = len(partners)
        for i in range(n - n_negotiators):
            yield partners[i : i + n_negotiators]

    def intin(i):
        if isinstance(i, int):
            return i
        return randint(i[0], i[1])

    while True:
        current_issues = [
            _ if isinstance(_, Issue) else Issue(values=intin(_), name=f"i{i}")
            for i, _ in enumerate(issues)
        ]
        ufuns = [
            U.random(current_issues, reserved_value=(0.0, 0.2), normalized=normalized)
            for _ in range(n_negotiators)
        ]
        for u in permutations(ufuns):
            for index in range(*positions):
                for p in partner_generator():
                    assert len(u) == len(p)
                    yield NegDomain(
                        name="d0",
                        ufuns=u,
                        issues=current_issues,
                        partner_types=p,
                        index=index,
                    )


def domains_from_list(domains: List[NegDomain]) -> Generator[NegDomain, None, None]:
    return cycle(domains)

if __name__ == "__main__":
    from negmas.sao import AspirationNegotiator, NaiveTitForTatNegotiator
    from negmas.utilities import LinearUtilityFunction as U
    from negmas.genius import genius_bridge_is_running
    from negmas.genius import Atlas3, NiceTitForTat

    issues = [Issue(10, "quantity"), Issue(5, "price")]
    competitors = [AspirationNegotiator, NaiveTitForTatNegotiator]
    if genius_bridge_is_running():
        competitors += [Atlas3, NiceTitForTat]

    domains = []
    for index in range(2):
        for partner in competitors:
            domains.append(
                NegDomain(
                    name="d0",
                    issues=issues,
                    ufuns=[
                        U.random(issues, reserved_value=(0.0, 0.2), normalized=False),
                        U.random(issues, reserved_value=(0.0, 0.2), normalized=False),
                    ],
                    partner_types=[partner],
                    index=index,
                )
            )

    print(
        neg_tournament(
            n_configs=2 * 2,
            domains=domains_from_list(domains),
            competitors=competitors,
            n_steps=2,
            neg_n_steps=10,
            neg_time_limit=None,
            parallelism="serial",
        )
    )
