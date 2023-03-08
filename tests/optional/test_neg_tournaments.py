from __future__ import annotations

import pytest

from negmas.genius import Atlas3, NiceTitForTat, genius_bridge_is_running
from negmas.outcomes import make_issue
from negmas.preferences import LinearUtilityFunction as U
from negmas.sao import AspirationNegotiator, NaiveTitForTatNegotiator
from negmas.situated.neg import NegDomain
from negmas.tournaments.neg import domains_from_list, neg_tournament


def test_can_run_world():
    issues = (make_issue(10, "quantity"), make_issue(5, "price"))
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
                    ufuns=(
                        U.random(
                            issues=issues, reserved_value=(0.0, 0.2), normalized=False
                        ),
                        U.random(
                            issues=issues, reserved_value=(0.0, 0.2), normalized=False
                        ),
                    ),
                    partner_types=(partner,),
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


@pytest.mark.skip(reason="no way of currently testing this")
def test_can_run_tournament():
    issues = (make_issue(10, "quantity"), make_issue(5, "price"))
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
                    ufuns=(
                        U.random(issues, reserved_value=(0.0, 0.2), normalized=False),
                        U.random(issues, reserved_value=(0.0, 0.2), normalized=False),
                    ),
                    partner_types=(partner,),
                    index=index,
                )
            )

    neg_tournament(
        n_configs=len(domains),
        domains=domains_from_list(domains),
        competitors=competitors,
        n_steps=1,
        neg_n_steps=10,
        neg_time_limit=None,
        compact=True,
    )


@pytest.mark.skip(reason="no way of currently testing this")
def test_can_run_tournament_from_generator():
    from negmas.tournaments.neg import random_discrete_domains

    n_configs = 1
    n_repetitions = 1
    competitors = [AspirationNegotiator, NaiveTitForTatNegotiator]
    if genius_bridge_is_running():
        competitors += [Atlas3, NiceTitForTat]

    domains = random_discrete_domains(issues=[5, 4, (3, 5)], partners=competitors)

    neg_tournament(
        n_configs=len(competitors) * n_configs,
        domains=domains,
        competitors=competitors,
        n_steps=n_repetitions,
        neg_n_steps=10,
        neg_time_limit=None,
        compact=True,
    )
