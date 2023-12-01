"""
Negotiation tournaments created and manged as standard situated tournaments.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from functools import partial
from itertools import chain, combinations, cycle, permutations, repeat
from os import PathLike
from random import randint
from typing import Any, Generator, Sequence

from negmas.helpers import get_class, unique_name
from negmas.inout import Scenario
from negmas.negotiators import Negotiator
from negmas.outcomes import Issue, make_issue
from negmas.preferences.crisp.linear import LinearUtilityFunction
from negmas.serialization import deserialize, serialize
from negmas.situated import Agent
from negmas.situated.neg import Condition, NegAgent, NegWorld  # , _wrap_in_agents
from negmas.tournaments.tournaments import (
    TournamentResults,
    WorldRunResults,
    create_tournament,
    tournament,
)

__all__ = [
    "random_discrete_scenarios",
    "scenarios_from_list",
    "create_neg_tournament",
    "create_cartesian_neg_tournament",
    "neg_tournament",
    "cartesian_neg_tournement",
]


AgentType = str | type[Agent] | type[Negotiator]


def neg_config_generator(
    n_competitors: int,
    scenarios: Generator[Condition, None, None],
    n_agents_per_competitor: int = 1,
    agent_names_reveal_type: bool = False,
    non_competitors: tuple[str | NegAgent] | None = None,
    non_competitor_params: tuple[dict[str, Any]] | None = None,
    compact: bool = False,
    n_repetitions: int = 1,
    **kwargs,
) -> list[dict[str, Any]]:
    """
    Used internally to generate the configuration of a world.

    Args:
        n_competitors: How many competitors will run?
        scenarios: A generator used to extract `NegScenario` objects (one config per `NegScenario`)
        n_agents_per_competitor: Must be 1
        agent_names_reveal_type: If given, the agent name will be the same as its class name
        non_competitors: Must be None
        non_competitor_params: Must be None
        compact: If given, the system tries to minimize the memory footprint and secondary storage
        kwargs: Extra arguments passed directly to the world constructor

    Returns:
        A list containing one world configuration
    """
    scenario = next(scenarios)
    if non_competitors:
        raise ValueError(
            f"Non-competitors are not supported for negotiation tournaments. (You provided {non_competitors})"
        )
    if n_agents_per_competitor != 1:
        raise ValueError(
            f"n_agents_per_competitor must be 1 ({n_agents_per_competitor} given)"
        )

    world_name = unique_name(f"{scenario.name}", add_time=False, rand_digits=1, sep=".")
    no_logs = compact
    world_params = dict(
        name=world_name,
        scenario=serialize(scenario),
        compact=compact,
        no_logs=no_logs,
    )
    world_params.update(kwargs)
    config = {
        "world_params": world_params,
        "scoring_context": {},
        "non_competitors": None,
        "non_competitor_params": None,
        "is_default": [False] * n_competitors + [True] * (len(scenario.ufuns) - 1),
        "n_repetitions": n_repetitions,
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
    n_reps = config[0].pop("n_repetitions", 1)
    if n_reps == 1:
        return [config]
    configs = []
    for i in range(n_reps):
        c = deepcopy(config[0])
        c["world_params"]["name"] += f"_{i}"
        configs.append([c])
    return configs


def neg_world_generator(**kwargs):
    """
    Generates the world
    """
    config = kwargs.pop("world_params", dict())
    config["types"], config["params"] = deepcopy(config["types"]), deepcopy(
        config["params"]
    )
    config["types"] = [get_class(_) for _ in config["types"]]
    config["scenario"] = deserialize(config["scenario"])
    # name = config["scenario"].name
    # config["name"] = name if name is not None else unique_name("world", sep="_")
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
    assert len(worlds) >= 1
    result = WorldRunResults(
        world_names=[world.name for world in worlds],
        log_file_names=[world.log_file_name for world in worlds]  # type: ignore
        if worlds[0].log_file_name
        else [],
    )

    for world in worlds:

        def sumdict(a, b):
            return lambda x: 0.5 * (a(x) + b(x))

        # world = worlds
        scoring_map = dict(
            received_utility=world.received_utility,
            received_advantage=world.received_advantage,
            partner_utility=world.partner_utility,
            partner_advantage=world.partner_advantage,
            welfare=sumdict(world.received_utility, world.partner_utility),
            total_advantage=sumdict(world.received_advantage, world.partner_advantage),
        )

        extra = defaultdict(list)
        for aid, agent in world.competitors.items():
            agent_type = agent.type_name
            result.names.append(agent.name)
            result.ids.append(agent.id)
            result.types.append(agent_type)
            if dry_run:
                result.scores.append(float("nan"))
                continue
            result.scores.append(scoring_map[scoring_method](aid))  # type: ignore
            for k, f in scoring_map.items():
                # if k == scoring_method:
                #     continue
                extra[k].append(
                    dict(
                        type=agent_type,
                        score=f(aid),
                        world=world.name,
                        run_id=world.name,
                    )
                )
        for k in scoring_map.keys():
            result.extra_scores[k] = extra[k]
    return result


def _update_kwargs(kwargs, scenarios, competitors, n_repetitions):
    kwargs["config_generator"] = partial(
        neg_config_generator, scenarios=scenarios, n_repetitions=1
    )
    kwargs["n_runs_per_world"] = n_repetitions
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
    scenarios: Generator[Condition, None, None],
    competitor_params: Sequence[dict[str, Any]] | None = None,
    n_repetitions: int = 1,
    **kwargs,
) -> PathLike:
    """
    Creates a tournament

    Args:
        competitors: A list of class names for the competitors
        scenarios:A generator that yields `NegScenario` objects specifying negotiation scenarios upon request
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
        **_update_kwargs(kwargs, scenarios, competitors, n_repetitions=n_repetitions),
    )


def neg_tournament(
    competitors: list[AgentType] | tuple[AgentType, ...],
    scenarios: Generator[Condition, None, None],
    competitor_params: Sequence[dict | None] | None = None,
    n_repetitions: int = 1,
    **kwargs,
) -> TournamentResults | PathLike:
    """
    Runs a tournament

    Args:

        competitors: A list of class names for the competitors
        scenarios:A generator that yields `NegScenario` objects specifying negotiation scenarios upon request
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
        competitors=competitors,  # type: ignore
        competitor_params=competitor_params,  # type: ignore
        **_update_kwargs(kwargs, scenarios, competitors, n_repetitions=n_repetitions),
    )


def random_discrete_scenarios(
    issues: list[Issue | int | tuple[int, int]],
    partners: list[type[Negotiator]] | tuple[type[Negotiator], ...],
    n_negotiators=2,
    positions: int | tuple[int, int] | None = None,
    normalized=True,
    ufun_type=LinearUtilityFunction,
    roles: tuple[str, ...] | None = None,
    partner_extraction_method="round-robin",
) -> Generator[Condition, None, None]:
    """
    Generates an infinite sequence of random discrete scenarios

    Args:
        issues: A list defining the issue space. Each element can be an `Issue`
                object, an integer (defining the number of outcomes) or a tuple
                of two integers (defining the minimum and maximum number of outcomes
                for the issue).
        partners: A list of `Negotiator` types from which partners are extracted.
                  The system will extract `n_negotiators` - 1 partners for each
                  scenario.
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
    partners = list(partners)
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
                    yield Condition(
                        name="d0",
                        ufuns=u,
                        issues=current_issues,  # type: ignore
                        partner_types=tuple(p),
                        index=index,
                        roles=roles,
                    )


def scenarios_from_list(
    scenarios: list[Condition],
) -> Generator[Condition, None, None]:
    """
    Creates an appropriate `NegScenario` generator from a list/tuple of scenarios
    """
    while True:
        yield from cycle(scenarios)


def _make_negs(
    scenarios,
    competitors,
    competitor_params,
    non_competitors,
    non_competitor_params,
    rotate_ufuns=False,
):
    competitors = list(competitors)
    if competitor_params is None:
        competitor_params = [dict() for _ in range(len(competitors))]
    else:
        competitor_params = list(competitor_params)
        assert len(competitor_params) == len(competitors)

    if non_competitors is None:
        non_competitors = tuple()
    non_competitors = list(non_competitors)
    if non_competitor_params is None:
        non_competitor_params = [dict() for _ in range(len(non_competitors))]
    else:
        non_competitor_params = list(non_competitor_params)
        assert len(non_competitor_params) == len(non_competitors)

    try:
        intersection = set(non_competitors).intersection(set(competitors))
        assert (
            not intersection
        ), f"Non-competitors and competitors must be disjoint. This is the intersection between them now: {intersection}"
    except:
        pass

    def make_neg_scenarios(
        scenarios: tuple[Scenario, ...] | list[Scenario]
    ) -> list[Condition]:
        negs = []
        for s in scenarios:
            k = 0
            assert (
                len(s.ufuns) == 2
            ), f"Only supporting bilateral negotiations: Scenario {s.outcome_space.name} has {len(s.ufuns)} ufuns"
            for typ, params, score in chain(
                zip(competitors, competitor_params, repeat(True)),
                zip(non_competitors, competitor_params, repeat(False)),
            ):
                ufuns = [list(s.ufuns)]
                if rotate_ufuns:
                    for _ in range(len(ufuns) - 1):
                        u = ufuns[-1]
                        ufuns.append([u[-1]] + u[:-1])
                indices = [0, 1] if not score else [0]
                for indx in indices:
                    for u in ufuns:
                        negs.append(
                            Condition(
                                name=f"{s.outcome_space.name}_{k}"
                                if s.outcome_space.name
                                else unique_name("s"),
                                issues=s.outcome_space.issues,
                                ufuns=tuple(u),
                                partner_types=(get_class(typ),),
                                partner_params=(params,),
                                scored_indices=tuple(range(len(s.ufuns)))
                                if score
                                else None,
                                index=indx,
                            )
                        )
                        k += 1

        return negs

    neg_scenarios = make_neg_scenarios(scenarios)
    return (
        neg_scenarios,
        competitors,
        competitor_params,
        non_competitors,
        non_competitor_params,
    )


def create_cartesian_neg_tournament(
    competitors: list[AgentType] | tuple[AgentType, ...],
    scenarios: list[Scenario] | tuple[Scenario, ...],
    competitor_params: Sequence[dict[str, None] | None] | None = None,
    non_competitors: list[AgentType] | tuple[AgentType, ...] = tuple(),
    non_competitor_params: Sequence[dict | None] | None = None,
    rotate_ufuns: bool = False,
    n_repetitions: int = 1,
    n_steps: int = 2,
    n_rounds: int | None = None,
    timelimit: float | None = None,
    **kwargs,
) -> PathLike:
    """
    Creates a Cartesian tournament (every competitor against every other competitor)

    Args:

        competitors: A list of class names for the competitors.
        non_competitors: A list of class names for agents that will run against the competitors but never be evaluated themselves.
        scenarios:A generator that yields `NegScenario` objects specifying negotiation scenarios upon request
        name: Tournament name
        competitor_params: A list of competitor parameters (used to initialize the competitors).
        non_competitor_params: A list of non-competitor parameters (used to initialize the non-competitors).
        rotate_ufuns: If `True`, all N rotations of the N ufuns in every scenario will be tried.
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
        forced_logs_fraction: The fraction of simulations for which to always save logs. Notice that this has no
                              effect except if no logs were to be saved otherwise (i.e. `no_logs` is passed as True)
        save_video_fraction: The fraction of simulations for which to save videos
        video_params: The parameters to pass to the video saving function
        video_saver: The parameters to pass to the video saving function after the world
        max_attempts: The maximum number of times to retry running simulations
        extra_scores_to_use: The type of extra-scores to use. If None normal scores will be used. Only effective if scores is None.
        kwargs: Arguments to pass to the `config_generator` function

    Returns:
        `TournamentResults` The results of the tournament or a `PathLike` giving the location where configs were saved

    """
    kwargs["n_steps"] = n_steps
    if n_rounds is not None:
        kwargs["neg_n_steps"] = n_rounds
    if timelimit is not None:
        kwargs["neg_time_limit"] = n_rounds
    (
        neg_scenarios,
        competitors,
        competitor_params,
        non_competitors,
        non_competitor_params,
    ) = _make_negs(
        scenarios,
        competitors,
        competitor_params,
        non_competitors,
        non_competitor_params,
        rotate_ufuns=rotate_ufuns,
    )
    return create_neg_tournament(
        competitors=competitors,  # type: ignore
        scenarios=scenarios_from_list(neg_scenarios),
        competitor_params=competitor_params,  # type: ignore
        n_repetitions=n_repetitions,
        **kwargs,
    )


def cartesian_neg_tournement(
    competitors: list[AgentType] | tuple[AgentType, ...],
    scenarios: list[Scenario] | tuple[Scenario, ...],
    competitor_params: Sequence[dict | None] | None = None,
    non_competitors: list[AgentType] | tuple[AgentType, ...] = tuple(),
    non_competitor_params: Sequence[dict | None] | None = None,
    rotate_ufuns=False,
    n_repetitions: int = 1,
    n_steps: int = 2,
    n_rounds: int | None = None,
    timelimit: float | None = None,
    **kwargs,
) -> TournamentResults | PathLike:
    """
    Runs a Cartesian tournament between `competitors` optionally rotating preferences.

    Args:

        competitors: A list of class names for the competitors.
        non_competitors: A list of class names for agents that will run against the competitors but never be evaluated themselves.
        scenarios:A generator that yields `NegScenario` objects specifying negotiation scenarios upon request
        n_repetitions: Number of time each world with a single negotiation scenario is repeated
        timelimit: Time limit per negotiation
        n_rounds: Number of rounds per negotiation
        name: Tournament name
        competitor_params: A list of competitor parameters (used to initialize the competitors).
        non_competitor_params: A list of non-competitor parameters (used to initialize the non-competitors).
        stage_winners_fraction: in [0, 1).  Fraction of agents to to go to the next stage at every stage. If zero, and
                                            round_robin, it becomes a single stage competition.
        agent_names_reveal_type: If true then the type of an agent should be readable in its name (most likely at its
                                 beginning).
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
        forced_logs_fraction: The fraction of simulations for which to always save logs. Notice that this has no
                              effect except if no logs were to be saved otherwise (i.e. `no_logs` is passed as True)
        save_video_fraction: The fraction of simulations for which to save videos
        video_params: The parameters to pass to the video saving function
        video_saver: The parameters to pass to the video saving function after the world
        max_attempts: The maximum number of times to retry running simulations
        extra_scores_to_use: The type of extra-scores to use. If None normal scores will be used. Only effective if scores is None.
        kwargs: Arguments to pass to the `config_generator` function

    Returns:
        `TournamentResults` The results of the tournament or a `PathLike` giving the location where configs were saved

    """
    kwargs["n_steps"] = n_steps
    if n_rounds is not None:
        kwargs["neg_n_steps"] = n_rounds
    if timelimit is not None:
        kwargs["neg_time_limit"] = n_rounds
    (
        neg_scenarios,
        competitors,
        competitor_params,
        non_competitors,
        non_competitor_params,
    ) = _make_negs(
        scenarios,
        competitors,
        competitor_params,
        non_competitors,
        non_competitor_params,
        rotate_ufuns=rotate_ufuns,
    )
    kwargs["n_configs"] = len(neg_scenarios)
    return neg_tournament(
        competitors=competitors,
        scenarios=scenarios_from_list(neg_scenarios),
        competitor_params=competitor_params,
        n_repetitions=n_repetitions,
        **kwargs,
    )


if __name__ == "__main__":
    from negmas.sao import AspirationNegotiator, NaiveTitForTatNegotiator

    scenarios = random_discrete_scenarios(
        issues=[5, 4, (3, 5)],
        partners=[AspirationNegotiator, NaiveTitForTatNegotiator],
    )
    print(
        neg_tournament(
            n_configs=2 * 2 * 2 * 4,
            scenarios=scenarios,
            competitors=[AspirationNegotiator, NaiveTitForTatNegotiator],
            n_steps=1,
            neg_n_steps=10,
            neg_time_limit=None,
            name="neg-tour-test",
        )
    )
