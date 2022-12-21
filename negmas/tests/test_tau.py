from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable

import hypothesis.strategies as st
import pkg_resources
import pytest
from hypothesis import Verbosity, example, given, settings

from negmas.gb.evaluators.tau import INFINITE
from negmas.gb.mechanisms.mechanisms import GAOMechanism, TAUMechanism
from negmas.gb.negotiators.escs import ESCSNegotiator
from negmas.gb.negotiators.scs import SCSNegotiator
from negmas.gb.negotiators.timebased import AspirationNegotiator
from negmas.genius.gnegotiators import Atlas3
from negmas.inout import Scenario
from negmas.mechanisms import Mechanism
from negmas.outcomes.base_issue import make_issue
from negmas.outcomes.categorical_issue import CategoricalIssue
from negmas.outcomes.common import Outcome
from negmas.outcomes.outcome_space import DiscreteCartesianOutcomeSpace, make_os
from negmas.preferences.crisp import MappingUtilityFunction as U
from negmas.preferences.crisp.linear import LinearAdditiveUtilityFunction as LU
from negmas.preferences.crisp.mapping import MappingUtilityFunction
from negmas.preferences.crisp_ufun import UtilityFunction
from negmas.preferences.ops import pareto_frontier
from negmas.preferences.value_fun import AffineFun, IdentityFun, LinearFun, TableFun
from negmas.sao.mechanism import SAOMechanism
from tests.switches import NEGMAS_RUN_GENIUS

SHOW_PLOTS = False
SHOW_ALL_PLOTS = False
FORCE_PLOT = False


@pytest.fixture
def scenarios_folder():
    return pkg_resources.resource_filename(
        "negmas", resource_name="tests/data/scenarios"
    )


def _plot(p, err=False, force=False):
    if not force and err and not SHOW_PLOTS:
        return
    if not force and not err and not SHOW_ALL_PLOTS:
        return
    import matplotlib.pyplot as plt

    p.plot(xdim="step")
    plt.show()
    plt.savefig("fig.png")


def test_small_tau_session2_escs():
    n = 2
    u1 = [1.0, 0.0]
    u2 = [0.0, 1.0]
    r1, r2 = float("-inf"), float("-inf")
    force_plot = False
    os: DiscreteCartesianOutcomeSpace = make_os([make_issue(n)])  # type: ignore
    p = TAUMechanism(outcome_space=os)
    ufuns = [
        U({(_,): u[_] for _ in range(n)}, reserved_value=r, outcome_space=os)
        for u, r in ((u1, r1), (u2, r2))
    ]
    for i, u in enumerate(ufuns):
        p.add(
            ESCSNegotiator(name=f"RCS{i}", id=f"RCS{i}"),
            preferences=u,
        )
    front_utils, front_outcomes = p.pareto_frontier()
    no_valid_outcomes = all(u1 <= r1 or u2 <= r2 for u1, u2 in front_utils)
    p.run()
    assert len(p.history) > 0, f"{p.state}"
    assert (
        p.agreement is not None or no_valid_outcomes
    ), f"{p.history}{_plot(p, True, force=force_plot)}"
    assert p.agreement in front_outcomes or p.agreement is None
    assert all((_,) in p.negotiator_offers(f"RCS{_}") for _ in range(n))
    _plot(p, False, force=force_plot)


def test_small_tau_session2():
    n = 2
    u1 = [1.0, 0.0]
    u2 = [0.0, 1.0]
    r1, r2 = float("-inf"), float("-inf")
    force_plot = False
    os: DiscreteCartesianOutcomeSpace = make_os([make_issue(n)])  # type: ignore
    p = TAUMechanism(outcome_space=os)
    ufuns = [
        U({(_,): u[_] for _ in range(n)}, reserved_value=r, outcome_space=os)
        for u, r in ((u1, r1), (u2, r2))
    ]
    for i, u in enumerate(ufuns):
        p.add(
            SCSNegotiator(name=f"RCS{i}", id=f"RCS{i}"),
            preferences=u,
        )
    front_utils, front_outcomes = p.pareto_frontier()
    no_valid_outcomes = all(u1 <= r1 or u2 <= r2 for u1, u2 in front_utils)
    p.run()
    assert len(p.history) > 0, f"{p.state}"
    assert (
        p.agreement is not None or no_valid_outcomes
    ), f"{p.history}{_plot(p, True, force=force_plot)}"
    assert p.agreement in front_outcomes or p.agreement is None
    assert all((_,) in p.negotiator_offers(f"RCS{_}") for _ in range(n))
    _plot(p, False, force=force_plot)


def test_a_tau_session_example():
    for _ in range(100):
        r1, r2, n1, n2 = 0.2, 0.3, 5, 10
        eps = 1e-3
        time.perf_counter()
        os: DiscreteCartesianOutcomeSpace = make_os([make_issue(n1), make_issue(n2)])  # type: ignore
        p = TAUMechanism(outcome_space=os)
        ufuns = [LU.random(os, reserved_value=r1), LU.random(os, reserved_value=r2)]
        for i, u in enumerate(ufuns):
            p.add(
                SCSNegotiator(name=f"RCS{i}", id=f"RCS{i}"),
                preferences=u,
            )
        front_utils, front_outcomes = p.pareto_frontier()
        no_valid_outcomes = all(
            u1 <= r1 + eps or u2 <= r2 + eps for u1, u2 in front_utils
        )
        p.run()
        assert len(p.history) > 0, f"{p.state}"
        assert (
            p.agreement is not None or no_valid_outcomes
        ), f"{p.history}{_plot(p, True)}"
        assert p.agreement in front_outcomes or p.agreement is None


def run_adversarial_case(
    buyer,
    seller,
    normalized=False,
    seller_reserved=0.0,
    buyer_reserved=0.0,
    cardinality=INFINITE,
    mechanism_type: type[Mechanism] = TAUMechanism,
    force_plot=False,
    do_asserts=True,
    n_steps=10 * 10 * 3,
):
    n1, n2 = 100, 11
    # create negotiation agenda (issues)
    issues = [
        make_issue(name="issue", values=n1 + n2),
    ]
    os = make_os(issues)

    # create the mechanism
    if mechanism_type == TAUMechanism:
        session = mechanism_type(outcome_space=os, cardinality=cardinality)  # type: ignore
    else:
        session = mechanism_type(outcome_space=os, n_steps=n_steps)

    # define buyer and seller utilities
    seller_utility = MappingUtilityFunction(
        dict(zip(os.enumerate(), [i for i in range(n1 + n2)])),  # type: ignore
        outcome_space=os,
    )
    buyer_utility = MappingUtilityFunction(
        dict(
            zip(
                os.enumerate(),  # type: ignore
                [n1 + n2 for _ in range(n1)] + [n1 + n2 - 1 - i for i in range(n2)],
            )
        ),
        outcome_space=os,
    )
    if normalized:
        seller_utility = seller_utility.scale_max(1.0)
        buyer_utility = buyer_utility.scale_max(1.0)
    seller_utility.reserved_value = seller_reserved
    buyer_utility.reserved_value = buyer_reserved

    # create and add buyer and seller negotiators
    b = buyer(name="buyer", id="buyer")
    s = seller(name="seller", id="seller")
    s.name += f"{s.short_type_name}"
    b.name += f"{b.short_type_name}"
    session.add(b, ufun=buyer_utility)
    session.add(s, ufun=seller_utility)

    front_utils, front_outcomes = session.pareto_frontier()
    eps = 1e-3
    no_valid_outcomes = all(
        u1 <= buyer_reserved + eps or u2 <= seller_reserved + eps
        for u1, u2 in front_utils
    )
    session.run()
    assert len(session.history) > 0, f"{session.state}"
    if do_asserts:
        assert (
            session.agreement is not None or no_valid_outcomes
        ), f"{session.history}{_plot(session, True, force=force_plot)}"
        assert session.agreement in front_outcomes or session.agreement is None
    _plot(session, force=force_plot)
    return session


def run_buyer_seller(
    buyer,
    seller,
    normalized=False,
    seller_reserved=0.0,
    buyer_reserved=0.0,
    cardinality=INFINITE,
    min_unique=0,
    mechanism_type: type[Mechanism] = TAUMechanism,
    force_plot=False,
    do_asserts=True,
    n_steps=10 * 10 * 3,
):
    # create negotiation agenda (issues)
    issues = [
        make_issue(name="price", values=10),
        make_issue(name="quantity", values=(1, 11)),
        make_issue(name="delivery_time", values=["today", "tomorrow", "nextweek"]),
    ]

    # create the mechanism
    if mechanism_type == TAUMechanism:
        session = mechanism_type(issues=issues, cardinality=cardinality, min_unique=min_unique)  # type: ignore
    else:
        session = mechanism_type(issues=issues, n_steps=n_steps)

    # define buyer and seller utilities
    seller_utility = LU(
        values=[  # type: ignore
            IdentityFun(),
            LinearFun(0.2),
            TableFun(dict(today=1.0, tomorrow=0.2, nextweek=0.0)),
        ],
        outcome_space=session.outcome_space,
    )

    buyer_utility = LU(
        values={  # type: ignore
            "price": AffineFun(-1, bias=9.0),
            "quantity": LinearFun(0.2),
            "delivery_time": TableFun(dict(today=0, tomorrow=0.7, nextweek=1.0)),
        },
        outcome_space=session.outcome_space,
    )
    if normalized:
        seller_utility = seller_utility.scale_max(1.0)
        buyer_utility = buyer_utility.scale_max(1.0)
    seller_utility.reserved_value = seller_reserved
    buyer_utility.reserved_value = buyer_reserved

    # create and add buyer and seller negotiators
    b = buyer(name="buyer", id="buyer")
    s = seller(name="seller", id="seller")
    s.name += f"{s.short_type_name}"
    b.name += f"{b.short_type_name}"
    session.add(b, ufun=buyer_utility)
    session.add(s, ufun=seller_utility)

    session.run()

    front_utils, front_outcomes = session.pareto_frontier_bf(sort_by_welfare=True)
    eps = 1e-3
    no_valid_outcomes = all(
        u1 <= buyer_reserved + eps or u2 <= seller_reserved + eps
        for u1, u2 in front_utils
    )
    assert len(session.history) > 0, f"{session.state}"
    if do_asserts:
        assert (
            session.agreement is not None or no_valid_outcomes
        ), f"{session.state}{_plot(session, True, force=force_plot)}"
        assert (
            session.agreement in front_outcomes or session.agreement is None
        ), f"{session.state}{_plot(session, True, force=force_plot)}"
    _plot(session, force=force_plot)
    return session


def remove_under_line(
    os: DiscreteCartesianOutcomeSpace,
    ufuns: Iterable[UtilityFunction],
    limits=((0.0, 0.8), (0.0, None)),
):
    mxs = [_.max() for _ in ufuns]

    limits = list(limits)
    for i, (x, y) in enumerate(limits):
        if y is None:
            limits[i] = (x, mxs[i])
    x1, x2 = limits[0]
    y1, y2 = limits[1]

    outcomes = list(os.enumerate())
    _, frontier = pareto_frontier(ufuns, outcomes)
    frontier = set(frontier)

    accepted: list[Outcome] = []
    for outcome in outcomes:
        if outcome in frontier:
            accepted.append(outcome)
            continue
        continue
        xA, yA = (_(outcome) for _ in ufuns)
        v1 = (x2 - x1, y2 - y1)  # Vector 1
        v2 = (x2 - xA, y2 - yA)  # Vector 2
        xp = v1[0] * v2[1] - v1[1] * v2[0]  # Cross product
        if xp > 0:
            # s = (y2-y1) / (x2 - x1)
            # assert yA < s * xA + y1 - s * x1
            continue
        accepted.append(outcome)
    return accepted


def run_anac_example(
    first_type,
    second_type,
    mechanism_type: type[Mechanism] = TAUMechanism,
    force_plot=False,
    do_asserts=True,
    domain_name="cameradomain",
    mechanism_params=dict(),
    single_issue=False,
    remove_under=False,
):
    src = pkg_resources.resource_filename(
        "negmas", resource_name=f"tests/data/{domain_name}"
    )
    base_folder = src
    domain = Scenario.from_genius_folder(Path(base_folder))
    assert domain is not None
    # create the mechanism

    a1 = first_type(
        name="a1",
        id="a1",
    )
    a2 = second_type(
        name="a2",
        id="a2",
    )
    a1.name += f"{a1.short_type_name}"
    a2.name += f"{a2.short_type_name}"
    domain.normalize()
    if single_issue or remove_under:
        domain = domain.to_single_issue(randomize=True)
        assert len(domain.issues) == 1
    if remove_under:
        outcomes = remove_under_line(domain.agenda, domain.ufuns)  # type: ignore
        issue = CategoricalIssue(
            values=[_[0] for _ in outcomes], name=domain.agenda.issues[0].name
        )
        domain.agenda = make_os([issue])

        # domain.ufuns = [ MappingUtilityFunction(dict(zip(outcomes, [u(_) for _ in outcomes])), outcome_space=domain.agenda) for u in domain.ufuns ]
    domain.mechanism_type = mechanism_type
    domain.mechanism_params = mechanism_params
    neg = domain.make_session([a1, a2], n_steps=domain.agenda.cardinality)
    if neg is None:
        raise ValueError(f"Failed to lead domain from {base_folder}")

    _, front_outcomes = neg.pareto_frontier()
    neg.run()
    assert len(neg.history) > 0, f"{neg.state}"
    if do_asserts:
        assert (
            neg.agreement is not None
        ), f"{neg.history}{_plot(neg, True, force=force_plot)}"
        assert neg.agreement in front_outcomes or neg.agreement is None
    _plot(neg, force=force_plot)
    return neg


def test_buyer_seller_easy():
    run_buyer_seller(
        SCSNegotiator,
        SCSNegotiator,
        normalized=True,
        seller_reserved=0.1,
        buyer_reserved=0.1,
        force_plot=FORCE_PLOT,
    )


@pytest.mark.skipif(not NEGMAS_RUN_GENIUS, reason="Skipping genius tests")
def test_buyer_seller_sao_easy_genius():
    run_buyer_seller(
        Atlas3,
        Atlas3,
        normalized=True,
        seller_reserved=0.1,
        buyer_reserved=0.1,
        force_plot=FORCE_PLOT,
        mechanism_type=SAOMechanism,
        do_asserts=False,
    )


def test_buyer_seller_sao_easy():
    run_buyer_seller(
        AspirationNegotiator,
        AspirationNegotiator,
        normalized=True,
        seller_reserved=0.1,
        buyer_reserved=0.1,
        force_plot=FORCE_PLOT,
        mechanism_type=GAOMechanism,
        do_asserts=False,
    )


def test_buyer_seller_sao():
    run_buyer_seller(
        AspirationNegotiator,
        AspirationNegotiator,
        normalized=True,
        seller_reserved=0.5,
        buyer_reserved=0.6,
        force_plot=FORCE_PLOT,
        mechanism_type=GAOMechanism,
        do_asserts=False,
    )


@pytest.mark.skipif(not NEGMAS_RUN_GENIUS, reason="Skipping genius tests")
def test_buyer_seller_sao_genius():
    run_buyer_seller(
        Atlas3,
        Atlas3,
        normalized=True,
        seller_reserved=0.5,
        buyer_reserved=0.6,
        force_plot=FORCE_PLOT,
        mechanism_type=SAOMechanism,
        do_asserts=False,
    )


def test_buyer_seller_alphainf():
    run_buyer_seller(
        SCSNegotiator,
        SCSNegotiator,
        normalized=True,
        seller_reserved=0.5,
        buyer_reserved=0.6,
        force_plot=FORCE_PLOT,
        min_unique=0,
        cardinality=INFINITE,
    )


def test_buyer_seller_alpha0():
    run_buyer_seller(
        SCSNegotiator,
        SCSNegotiator,
        normalized=True,
        seller_reserved=0.5,
        buyer_reserved=0.6,
        force_plot=FORCE_PLOT,
        min_unique=0,
        cardinality=0,
    )


def test_buyer_seller_betainf():
    with pytest.raises(AssertionError):
        run_buyer_seller(
            SCSNegotiator,
            SCSNegotiator,
            normalized=True,
            seller_reserved=0.5,
            buyer_reserved=0.6,
            force_plot=FORCE_PLOT,
            min_unique=INFINITE,
        )


def test_buyer_seller_beta0():
    run_buyer_seller(
        SCSNegotiator,
        SCSNegotiator,
        normalized=True,
        seller_reserved=0.5,
        buyer_reserved=0.6,
        force_plot=FORCE_PLOT,
        min_unique=0,
    )


def test_anac_scenario_example_single():
    run_anac_example(
        SCSNegotiator,
        SCSNegotiator,
        force_plot=FORCE_PLOT,
        single_issue=True,
        remove_under=False,
    )


def test_anac_scenario_example_sao_single():
    run_anac_example(
        AspirationNegotiator,
        AspirationNegotiator,
        mechanism_type=GAOMechanism,
        force_plot=FORCE_PLOT,
        single_issue=True,
        remove_under=False,
    )


def test_anac_scenario_example_sao_single2():
    run_anac_example(
        AspirationNegotiator,
        AspirationNegotiator,
        mechanism_type=SAOMechanism,
        force_plot=FORCE_PLOT,
        single_issue=True,
        remove_under=False,
    )


@pytest.mark.skipif(not NEGMAS_RUN_GENIUS, reason="Skipping genius tests")
def test_anac_scenario_example_genius2_single():
    run_anac_example(
        Atlas3,
        Atlas3,
        mechanism_type=GAOMechanism,
        force_plot=FORCE_PLOT,
        single_issue=True,
        remove_under=False,
    )


@pytest.mark.skipif(not NEGMAS_RUN_GENIUS, reason="Skipping genius tests")
def test_anac_scenario_example_genius_single():
    run_anac_example(
        Atlas3,
        Atlas3,
        mechanism_type=SAOMechanism,
        force_plot=FORCE_PLOT,
        single_issue=True,
        remove_under=False,
    )


def test_anac_scenario_example():
    run_anac_example(SCSNegotiator, SCSNegotiator, force_plot=FORCE_PLOT)


def test_anac_scenario_example_sao():
    run_anac_example(
        AspirationNegotiator,
        AspirationNegotiator,
        mechanism_type=GAOMechanism,
        force_plot=FORCE_PLOT,
    )


@pytest.mark.skipif(not NEGMAS_RUN_GENIUS, reason="Skipping genius tests")
def test_anac_scenario_example_genius():
    run_anac_example(
        Atlas3,
        Atlas3,
        mechanism_type=SAOMechanism,
        force_plot=FORCE_PLOT,
    )


def test_adversarial_case_easy():
    run_adversarial_case(
        SCSNegotiator,
        SCSNegotiator,
        force_plot=FORCE_PLOT,
    )


@pytest.mark.skipif(not NEGMAS_RUN_GENIUS, reason="Skipping genius tests")
def test_adversarial_case_sao_easy_genius():
    run_adversarial_case(
        Atlas3,
        Atlas3,
        force_plot=FORCE_PLOT,
        mechanism_type=SAOMechanism,
        do_asserts=False,
    )


def test_adversarial_case_sao_easy():
    run_adversarial_case(
        AspirationNegotiator,
        AspirationNegotiator,
        force_plot=FORCE_PLOT,
        mechanism_type=GAOMechanism,
        do_asserts=False,
    )


@given(
    r1=st.floats(0, 1),
    r2=st.floats(0, 1),
    n1=st.integers(3, 5),
    n2=st.integers(3, 10),
    U1=st.sampled_from(
        [
            LU,
        ]
    ),
    U2=st.sampled_from(
        [
            LU,
        ]
    ),
)
@settings(deadline=10_000, verbosity=Verbosity.verbose, max_examples=10)
def test_a_tau_session(r1, r2, n1, n2, U1, U2):
    eps = 1e-3
    time.perf_counter()
    os: DiscreteCartesianOutcomeSpace = make_os([make_issue(n1), make_issue(n2)])  # type: ignore
    p = TAUMechanism(outcome_space=os)
    ufuns = [U1.random(os, reserved_value=r1), U2.random(os, reserved_value=r2)]
    for i, u in enumerate(ufuns):
        p.add(
            SCSNegotiator(name=f"SCS{i}"),
            preferences=u,
        )
    p.run()
    front_utils, front_outcomes = p.pareto_frontier()
    no_valid_outcomes = all(u1 <= r1 + eps or u2 <= r2 + eps for u1, u2 in front_utils)
    assert len(p.history) > 0, f"{p.state}"
    assert p.agreement is not None or no_valid_outcomes, f"{p.history}"
    assert p.agreement in front_outcomes or p.agreement is None
