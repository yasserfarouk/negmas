from __future__ import annotations

import pytest

from negmas.genius import genius_bridge_is_running
from negmas.inout import Scenario
from negmas.outcomes import make_issue
from negmas.outcomes.outcome_space import make_os
from negmas.preferences import LinearAdditiveUtilityFunction as U
from negmas.sao import (
    AspirationNegotiator,
    NaiveTitForTatNegotiator,
    RandomAlwaysAcceptingNegotiator,
    RandomNegotiator,
)
from negmas.sao.mechanism import SAOMechanism
from negmas.situated.neg import Condition
from negmas.tournaments.neg import (
    cartesian_neg_tournement,
    neg_tournament,
    scenarios_from_list,
)
from tests.switches import NEGMAS_FASTRUN

SKIP_CONDITION = NEGMAS_FASTRUN


@pytest.mark.skipif(
    condition=SKIP_CONDITION,
    reason="skipped for fast runs",
)
def test_can_run_cartesian_tournament_n_reps():
    n = 2
    issues = (
        make_issue([f"q{i}" for i in range(10)], "quantity"),
        make_issue([f"p{i}" for i in range(5)], "price"),
    )
    ufuns = [
        (
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=False),
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=False),
        )
        for _ in range(n)
    ]
    scenarios = [
        Scenario(
            outcome_space=make_os(issues, name=f"S{i}"),
            ufuns=u,
            mechanism_type=SAOMechanism,  # type: ignore
            mechanism_params=dict(),
        )
        for i, u in enumerate(ufuns)
    ]
    results = cartesian_neg_tournement(
        competitors=[RandomNegotiator, AspirationNegotiator],
        non_competitors=[NaiveTitForTatNegotiator],
        scenarios=scenarios,
        neg_time_limit=10,
        neg_n_steps=None,
        n_repetitions=4,
        verbose=True,
    )
    print(results)


@pytest.mark.skipif(
    condition=SKIP_CONDITION,
    reason="skipped for fast runs",
)
def test_can_run_cartesian_tournament():
    n = 2
    issues = (
        make_issue([f"q{i}" for i in range(10)], "quantity"),
        make_issue([f"p{i}" for i in range(5)], "price"),
    )
    ufuns = [
        (
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=False),
            U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=False),
        )
        for _ in range(n)
    ]
    scenarios = [
        Scenario(
            outcome_space=make_os(issues, name=f"S{i}"),
            ufuns=u,
            mechanism_type=SAOMechanism,  # type: ignore
            mechanism_params=dict(),
        )
        for i, u in enumerate(ufuns)
    ]
    results = cartesian_neg_tournement(
        competitors=[RandomNegotiator, AspirationNegotiator],
        non_competitors=[NaiveTitForTatNegotiator],
        scenarios=scenarios,
        neg_time_limit=10,
        neg_n_steps=None,
        n_steps=4,
        verbose=True,
    )
    print(results)


@pytest.mark.skipif(
    condition=SKIP_CONDITION,
    reason="skipped for fast runs",
)
def test_can_run_world_repeated_runs():
    issues = (make_issue(10, "quantity"), make_issue(5, "price"))
    competitors = [AspirationNegotiator, NaiveTitForTatNegotiator]
    if genius_bridge_is_running():
        competitors += [RandomNegotiator, RandomAlwaysAcceptingNegotiator]

    ufuns = (
        U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=False),
        U.random(issues=issues, reserved_value=(0.0, 0.2), normalized=False),
    )

    scenarios = []

    for index in range(2):
        for partner in competitors:
            scenarios.append(
                Condition(
                    name="d0",
                    issues=issues,
                    ufuns=ufuns,
                    partner_types=(partner,),
                    index=index,
                )
            )

    print(
        neg_tournament(
            n_configs=2 * 2,
            scenarios=scenarios_from_list(scenarios),
            competitors=competitors,
            n_steps=2,
            neg_n_steps=10,
            neg_time_limit=None,
            parallelism="serial",
        )
    )


@pytest.mark.skip(reason="no way of currently testing this")
def test_can_run_tournament():
    issues = (make_issue(10, "quantity"), make_issue(5, "price"))
    competitors = [AspirationNegotiator, NaiveTitForTatNegotiator]
    if genius_bridge_is_running():
        competitors += [RandomNegotiator, RandomAlwaysAcceptingNegotiator]

    domains = []
    for index in range(2):
        for partner in competitors:
            domains.append(
                Condition(
                    name="d0",
                    issues=issues,
                    ufuns=(
                        U.random(issues, reserved_value=(0.0, 0.2), normalized=False),  # type: ignore
                        U.random(issues, reserved_value=(0.0, 0.2), normalized=False),  # type: ignore
                    ),
                    partner_types=(partner,),
                    index=index,
                )
            )

    neg_tournament(
        n_configs=len(domains),
        scenarios=scenarios_from_list(domains),
        competitors=competitors,
        n_steps=1,
        neg_n_steps=10,
        neg_time_limit=None,
        compact=True,
    )


@pytest.mark.skip(reason="no way of currently testing this")
def test_can_run_tournament_from_generator():
    from negmas.tournaments.neg import random_discrete_scenarios

    n_configs = 1
    n_repetitions = 1
    competitors = [AspirationNegotiator, NaiveTitForTatNegotiator]
    if genius_bridge_is_running():
        competitors += [RandomNegotiator, RandomAlwaysAcceptingNegotiator]

    domains = random_discrete_scenarios(issues=[5, 4, (3, 5)], partners=competitors)

    neg_tournament(
        n_configs=len(competitors) * n_configs,
        scenarios=domains,
        competitors=competitors,
        n_steps=n_repetitions,
        neg_n_steps=10,
        neg_time_limit=None,
        compact=True,
    )
