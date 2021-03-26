import os
from typing import Union

import numpy as np
import pkg_resources
import pytest

from negmas import load_genius_domain_from_folder
from negmas.elicitation import (
    DummyElicitor,
    EStrategy,
    FullKnowledgeElicitor,
    PandoraElicitor,
    SAOElicitingMechanism,
    User,
    next_query,
    possible_queries,
)
from negmas.helpers import instantiate
from negmas.sao import AspirationNegotiator, LimitedOutcomesNegotiator, SAOMechanism
from negmas.utilities import IPUtilityFunction, MappingUtilityFunction, pareto_frontier

try:
    from blist import sortedlist

    BLIST_AVAILABLE = True
except ImportError:
    BLIST_AVAILABLE = False
    print("blist is not avialable. This is a known issue with python 3.9. Use python 3.8 if you are testing VOI")

n_outcomes = 5
cost = 0.02
utility = 0.17682
accepted = [(0,), (2,)]
ufun = MappingUtilityFunction(
    dict(zip([(_,) for _ in range(n_outcomes)], [utility] * n_outcomes)),
    reserved_value=0.0,
)

import negmas.elicitation as elicitation


all_countable_queries_elicitor_types = [
    _
    for _ in elicitation.__all__
    if _.endswith("Elicitor") and not _.startswith("Base") and not "VOIOptimal" in _
]
if not BLIST_AVAILABLE:
    all_countable_queries_elicitor_types = [
        _ for _ in all_countable_queries_elicitor_types if "voi" not in _.lower()
    ]


@pytest.fixture
def neg() -> SAOMechanism:
    return SAOMechanism(outcomes=[(_,) for _ in range(n_outcomes)])


@pytest.fixture
def strategy(neg: SAOMechanism) -> EStrategy:
    s = EStrategy(strategy="exact")
    s.on_enter(ami=neg.ami)
    return s


@pytest.fixture
def user() -> User:
    return User(ufun=ufun, cost=cost)


@pytest.fixture
def true_utilities():
    return list(np.random.rand(n_outcomes).tolist())


@pytest.fixture
def master(true_utilities, strategy_name="titration-0.05"):
    user = User(
        ufun=MappingUtilityFunction(
            dict(zip([(_,) for _ in range(n_outcomes)], true_utilities)),
            reserved_value=0.0,
        ),
        cost=cost,
    )
    strategy = EStrategy(strategy=strategy_name)
    return user, strategy


class TestCountableOutcomesUser(object):
    def test_countable_outcmoes_user_initializable(self):
        user = User(ufun=ufun, cost=cost)
        assert user.total_cost == 0.0, "total cost is not initialized to zero"

    def test_countable_outcmoes_user_can_enter(self, user):
        true_utils = list(user.ufun.mapping.values())
        assert len(true_utils) == n_outcomes, "incorrect number of utilities"
        assert user.total_cost == 0.0
        assert user.ufun((0,)) == utility
        assert user.cost_of_asking() == cost

    def test_countable_outcmoes_user_can_elicit_exact(self, strategy, user):
        u, _ = strategy.apply(user=user, outcome=(0,))
        assert isinstance(u, float), "exact elicitation gets u a non float"
        assert u == utility, "exact elicitation gets u an incorrect value"

    def test_countable_outcmoes_user_can_elicit_bisection(self, neg, user):
        strategy = EStrategy(strategy="bisection", resolution=1e-4)
        strategy.on_enter(neg.ami)
        elicited = []
        estimated = []
        total_cost = 0.0
        while True:
            e = strategy.utility_estimate((0,))
            # assert user.total_cost <= total_cost + cost, 'incorrect total cost'
            u, reply = strategy.apply(user=user, outcome=(0,))
            if isinstance(u, float) or u.scale < cost:
                assert abs(u - utility) < 1e-2
                break
            total_cost += cost
            elicited.append(u)
            estimated.append(e)

        assert elicited[0].loc == 0.0, "first loc is incorrect in elicitation"
        assert elicited[0].scale == 0.5, "first scale is incorrect in elicitation"
        assert estimated[0].loc == 0.0, "first loc is incorrect in estimation"
        assert estimated[0].scale == 1.0, "first scale is incorrect in estimation"
        for u, e in zip(elicited, estimated):
            assert u.scale == 0.5 * e.scale, "uncertainty is not decreasing as expected"
        for i in range(len(elicited) - 1):
            assert (
                elicited[i + 1].scale == 0.5 * elicited[i].scale
            ), "uncertainty is not decreasing"
        for i in range(len(estimated) - 1):
            assert (
                estimated[i + 1].scale == 0.5 * estimated[i].scale
            ), "uncertainty is not decreasing"

    def test_countable_outcmoes_user_can_elicit_titration_up(self, neg, user):
        step = 0.05
        strategy = EStrategy(strategy=f"titration+{step}", resolution=1e-4)
        strategy.on_enter(neg.ami)
        elicited = []
        estimated = []
        total_cost = 0.0
        while True:
            e = strategy.utility_estimate((0,))
            assert user.total_cost == total_cost, "incorrect total cost"
            u, reply = strategy.apply(user=user, outcome=(0,))
            if isinstance(u, float) or u.scale < cost:
                assert abs(u - utility) < step * 2
                break
            total_cost += cost
            elicited.append(u)
            estimated.append(e)

        assert elicited[0].loc == step, "first loc is incorrect in elicitation"
        assert (
            elicited[0].scale == 1.0 - step
        ), "first scale is incorrect in elicitation"
        assert estimated[0].loc == 0.0, "first loc is incorrect in estimation"
        assert estimated[0].scale == 1.0, "first scale is incorrect in estimation"
        for u, e in zip(elicited[:-1], estimated[:-1]):
            assert (
                abs(u.scale - e.scale + step) < 1e-3
            ), "uncertainty is not decreasing as expected"
        for i in range(len(elicited) - 2):
            assert (
                abs(elicited[i + 1].scale - elicited[i].scale + step) < 1e-3
            ), "uncertainty is not decreasing"
        for i in range(len(estimated) - 2):
            assert (
                estimated[i + 1].scale - estimated[i].scale + step
            ) < 1e-3, "uncertainty is not decreasing"

    def test_countable_outcmoes_user_can_elicit_titration_up_no_step(self, neg, user):
        step = 0.01
        strategy = EStrategy(strategy=f"titration", resolution=step)
        strategy.on_enter(neg.ami)
        elicited = []
        estimated = []
        total_cost = 0.0
        for _ in range(int(0.5 + 1.0 / step) + 2):
            e = strategy.utility_estimate((0,))
            # assert user.total_cost <= total_cost + cost, 'incorrect total cost'
            u, reply = strategy.apply(user=user, outcome=(0,))
            if isinstance(u, float) or u.scale < cost:
                u = float(u)
                assert abs(u - utility) < 2 * step
                break
            total_cost += cost
            elicited.append(u)
            estimated.append(e)
        else:
            # print(elicited)
            assert False, "did not end in expected time"

        assert elicited[0].loc == step, "first loc is incorrect in elicitation"
        assert (
            elicited[0].scale == 1.0 - step
        ), "first scale is incorrect in elicitation"
        assert estimated[0].loc == 0.0, "first loc is incorrect in estimation"
        assert estimated[0].scale == 1.0, "first scale is incorrect in estimation"
        for u, e in zip(elicited[:-1], estimated[:-1]):
            assert (
                abs(u.scale - e.scale + step) < 1e-3
            ), "uncertainty is not decreasing as expected"
        for i in range(len(elicited) - 2):
            assert (
                abs(elicited[i + 1].scale - elicited[i].scale + step) < 1e-3
            ), "uncertainty is not decreasing"
        for i in range(len(estimated) - 2):
            assert (
                estimated[i + 1].scale - estimated[i].scale + step
            ) < 1e-3, "uncertainty is not decreasing"

    def test_countable_outcmoes_user_can_elicit_titration_down(self, neg, user):
        step = -0.05
        strategy = EStrategy(strategy=f"titration{step}", resolution=1e-4)
        strategy.on_enter(neg.ami)
        elicited = []
        estimated = []
        total_cost = 0.0
        while True:
            e = strategy.utility_estimate((0,))
            # assert user.total_cost <= total_cost + cost, 'incorrect total cost'
            u, reply = strategy.apply(user=user, outcome=(0,))
            if isinstance(u, float) or u.scale < cost:
                assert abs(u - utility) < -step * 2
                break
            total_cost += cost
            elicited.append(u)
            estimated.append(e)

        step = -step
        assert (
            elicited[0].loc + elicited[0].scale == 1.0 - step
        ), "first loc is incorrect in elicitation"
        assert (
            elicited[0].scale == 1.0 - step
        ), "first scale is incorrect in elicitation"
        assert estimated[0].loc == 0.0, "first loc is incorrect in estimation"
        assert estimated[0].scale == 1.0, "first scale is incorrect in estimation"

    def test_countable_outcmoes_user_can_elicit_dtitration_up(self, neg, user):
        step = 0.05
        strategy = EStrategy(strategy=f"dtitration+{step}", resolution=1e-4)
        strategy.on_enter(neg.ami)
        elicited = []
        estimated = []
        total_cost = 0.0
        while True:
            e = strategy.utility_estimate((0,))
            # assert user.total_cost <= total_cost + cost, 'incorrect total cost'
            u, reply = strategy.apply(user=user, outcome=(0,))
            if isinstance(u, float) or u.scale < cost:
                assert abs(u - utility) < 1e-2
                break
            total_cost += cost
            elicited.append(u)
            estimated.append(e)
        assert elicited[0].loc == step, "first loc is incorrect in elicitation"
        assert (
            elicited[0].scale == 1.0 - step
        ), "first scale is incorrect in elicitation"
        assert estimated[0].loc == 0.0, "first loc is incorrect in estimation"
        assert estimated[0].scale == 1.0, "first scale is incorrect in estimation"

    def test_countable_outcmoes_user_can_elicit_dtitration_down(self, neg, user):
        step = -0.05
        strategy = EStrategy(strategy=f"dtitration{step}", resolution=1e-4)
        strategy.on_enter(neg.ami)
        elicited = []
        estimated = []
        total_cost = 0.0
        while True:
            e = strategy.utility_estimate((0,))
            # assert user.total_cost <= total_cost + cost, 'incorrect total cost'
            u, reply = strategy.apply(user=user, outcome=(0,))
            if isinstance(u, float) or u.scale < cost:
                assert abs(u - utility) < -step * 2
                break
            total_cost += cost
            elicited.append(u)
            estimated.append(e)

        step = -step
        assert (
            elicited[0].loc + elicited[0].scale == 1.0 - step
        ), "first loc is incorrect in elicitation"
        assert (
            elicited[0].scale == 1.0 - step
        ), "first scale is incorrect in elicitation"
        assert estimated[0].loc == 0.0, "first loc is incorrect in estimation"
        assert estimated[0].scale == 1.0, "first scale is incorrect in estimation"

    def test_countable_outcmoes_user_stops_eliciting_at_cost(self, neg, user):
        strategy = EStrategy(strategy="bisection", resolution=1e-4, stop_at_cost=True)
        strategy.on_enter(neg.ami)
        elicited = []
        estimated = []
        total_cost = 0.0
        while True:
            e = strategy.utility_estimate((0,))
            # assert user.total_cost <= total_cost + cost, 'incorrect total cost'
            u, reply = strategy.apply(user=user, outcome=(0,))
            if isinstance(u, float) or u.scale < cost:
                assert abs(u - utility) < cost
                break
            total_cost += cost
            elicited.append(u)
            estimated.append(e)

        assert elicited[0].loc == 0.0, "first loc is incorrect in elicitation"
        assert elicited[0].scale == 0.5, "first scale is incorrect in elicitation"
        assert estimated[0].loc == 0.0, "first loc is incorrect in estimation"
        assert estimated[0].scale == 1.0, "first scale is incorrect in estimation"
        for u, e in zip(elicited, estimated):
            assert u.scale == 0.5 * e.scale, "uncertainty is not decreasing as expected"
        for i in range(len(elicited) - 1):
            assert (
                elicited[i + 1].scale == 0.5 * elicited[i].scale
            ), "uncertainty is not decreasing"
        for i in range(len(estimated) - 1):
            assert (
                estimated[i + 1].scale == 0.5 * estimated[i].scale
            ), "uncertainty is not decreasing"
        assert estimated[-1].scale >= cost

    def test_countable_outcmoes_user_elicits_non_exact(self, neg, user):
        strategy = EStrategy(strategy="bisection", resolution=1e-2)
        strategy.on_enter(neg.ami)
        elicited = []
        estimated = []
        total_cost = 0.0
        while True:
            e = strategy.utility_estimate((0,))
            # assert user.total_cost <= total_cost + cost, 'incorrect total cost'
            u, reply = strategy.apply(user=user, outcome=(0,))
            if isinstance(u, float) or u.scale < cost:
                assert abs(u - utility) < 1e-2 and abs(u - utility) > 1e-3
                break
            total_cost += cost
            elicited.append(u)
            estimated.append(e)

        assert elicited[0].loc == 0.0, "first loc is incorrect in elicitation"
        assert elicited[0].scale == 0.5, "first scale is incorrect in elicitation"
        assert estimated[0].loc == 0.0, "first loc is incorrect in estimation"
        assert estimated[0].scale == 1.0, "first scale is incorrect in estimation"
        for u, e in zip(elicited, estimated):
            assert u.scale == 0.5 * e.scale, "uncertainty is not decreasing as expected"
        for i in range(len(elicited) - 1):
            assert (
                elicited[i + 1].scale == 0.5 * elicited[i].scale
            ), "uncertainty is not decreasing"
        for i in range(len(estimated) - 1):
            assert (
                estimated[i + 1].scale == 0.5 * estimated[i].scale
            ), "uncertainty is not decreasing"

    def test_countable_outcomes_user_can_return_all_queries(self, neg):
        for s in (
            "exact",
            "bisection",
            "titration+0.05",
            "titration-0.5",
            "dtitration+0.5",
            "dtitration-0.05",
            "pingpong0.5",
            "dpingpong0.5",
        ):
            user = User(ufun=ufun, cost=cost)
            strategy = EStrategy(strategy=s)
            strategy.on_enter(neg.ami)
            q = possible_queries(ami=neg.ami, strategy=strategy, user=user)
            assert (
                len(q) > 0 and s != "exact" or len(q) == 0 and s == "exact"
            ), "returns some queries"

    def test_countable_outcomes_user_can_return_next_queries(self, neg):
        for s in (
            "exact",
            "bisection",
            "titration+0.05",
            "titration-0.5",
            "dtitration+0.5",
            "dtitration-0.05",
        ):
            strategy = EStrategy(strategy=s)
            user = User(ufun=ufun, cost=cost)
            strategy.on_enter(neg.ami)
            q = next_query(strategy=strategy, user=user)
            # print(f'{strategy} Strategy:\n---------------')
            # pprint.pprint(q)
            assert len(q) > 0, "returns some queries"

    def test_elicit_until(self, neg):
        for s in ("bisection", "titration+0.05", "titration-0.05", "pingpong"):
            user = User(ufun=ufun, cost=cost)
            stretegy = EStrategy(strategy=s, resolution=1e-3)
            stretegy.on_enter(neg.ami)
            outcome, query, qcost = next_query(
                strategy=stretegy, outcome=(0,), user=user
            )[0]
            stretegy.until(
                user=user, outcome=(0,), dist=query.answers[1].constraint.marginal((0,))
            )


def u0(neg: SAOMechanism, reserved_value=0.0):
    return IPUtilityFunction(outcomes=neg.outcomes, reserved_value=reserved_value)


@pytest.fixture
def data_folder():
    return pkg_resources.resource_filename("negmas", resource_name="tests/data")


class TestCountableOutcomesElicitor(object):
    def test_dummy(self, master, true_utilities):
        user, strategy = master
        neg = SAOMechanism(outcomes=[(_,) for _ in range(n_outcomes)], n_steps=10)
        accepted = [(0,), (2,)]
        opponent = LimitedOutcomesNegotiator(
            acceptable_outcomes=accepted,
            acceptance_probabilities=[1.0] * len(accepted),
        )
        elicitor = DummyElicitor(user=user)
        neg.add(opponent)
        neg.add(elicitor, ufun=u0(neg))
        neg.run()
        # print(
        #     f'Got {elicitor.ufun(neg.agreement)} with elicitation cost {elicitor.elicitation_cost}'
        #     f' for {elicitor} using 0 elicited_queries')

        assert len(neg.history) > 0
        assert neg.agreement is None or neg.agreement in accepted
        # assert len(set([_[0] for _ in elicitor.offers])) > 1 or len(elicitor.offers) < 2
        assert elicitor.elicitation_cost == 0.0

    def test_full_knowledge(self, master):
        user, strategy = master
        neg = SAOMechanism(outcomes=[(_,) for _ in range(n_outcomes)], n_steps=10)
        opponent = LimitedOutcomesNegotiator(
            acceptable_outcomes=accepted,
            acceptance_probabilities=[1.0] * len(accepted),
        )
        elicitor = FullKnowledgeElicitor(user=user)
        neg.add(opponent)
        neg.add(elicitor, ufun=u0(neg))
        neg.run()
        # print(
        #     f'Got {elicitor.ufun(neg.agreement)} with elicitation cost {elicitor.elicitation_cost}'
        #     f' for {elicitor} using 0 elicited_queries')

        assert len(neg.history) > 0
        assert neg.agreement is None or neg.agreement in accepted
        # assert len(set([_[0] for _ in elicitor.offers])) > 1 or len(elicitor.offers) < 2
        assert elicitor.elicitation_cost == 0.0

    @pytest.mark.parametrize("elicitor", all_countable_queries_elicitor_types)
    def test_elicitor_runs(
        self, elicitor: Union[str, "BaseElicitor"], master, true_utilities, **kwargs
    ):
        neg = SAOMechanism(outcomes=n_outcomes, n_steps=10)
        user, strategy = master
        opponent = LimitedOutcomesNegotiator(
            acceptable_outcomes=accepted,
            acceptance_probabilities=[1.0] * len(accepted),
        )
        strategy.on_enter(ami=neg.ami)
        if isinstance(elicitor, str):
            elicitor = f"negmas.elicitation.{elicitor}"
            if "VOI" in elicitor:
                kwargs["dynamic_query_set"] = True
            elicitor = instantiate(elicitor, strategy=strategy, user=user, **kwargs)
        neg.add(opponent)
        neg.add(elicitor)
        assert elicitor.elicitation_cost == 0.0
        neg.run()
        queries = list(elicitor.user.elicited_queries())
        assert len(neg.history) > 0
        assert neg.agreement is None or neg.agreement in accepted
        assert (
            elicitor.elicitation_cost > 0.0
            or cost == 0.0
            or elicitor.strategy is None
            or neg.state.step < 2
        )
        if neg.agreement is not None:
            assert (
                elicitor.user_ufun(neg.agreement)
                == true_utilities[neg.agreement[0]] - elicitor.elicitation_cost
            )
        if hasattr(elicitor, "each_outcome_once") and elicitor.each_outcome_once:
            assert len(set([_[0] for _ in elicitor.offers])) == len(elicitor.offers)
        # print(
        #     f"Got {elicitor.ufun(neg.agreement)} with elicitation cost {elicitor.elicitation_cost} "
        #     f"for {elicitor} using {len(queries)} elicited_queries"
        # )

    # def test_pareto_frontier_in_mechanism(self, master, true_utilities):
    #     neg, elicitor, opponent = self._run_optimal_test(
    #         "FullElicitor", master, true_utilities
    #     )
    #     front, _ = neg.pareto_frontier()
    #     # print(front)
    #     assert len(front) > 0

    def test_pareto_frontier_2(self):
        n_outcomes = 10
        strategy = "titration-0.5"
        cost = 0.01
        reserved_value = 0.1
        outcomes = [(_,) for _ in range(n_outcomes)]
        accepted = [(2,), (3,), (4,), (5,)]
        elicitor_utilities = [
            0.5337723805661662,
            0.8532272031479199,
            0.4781281413197942,
            0.7242899747791032,
            0.3461879818432919,
            0.2608677043479706,
            0.9419131964655383,
            0.29368079952747694,
            0.6093201983562316,
            0.7066918086398718,
        ]
        # list(np.random.rand(n_outcomes).tolist())
        opponent_utilities = [
            1.0 if (_,) in accepted else 0.0 for _ in range(n_outcomes)
        ]
        frontier, frontier_locs = pareto_frontier(
            [
                MappingUtilityFunction(
                    lambda o: elicitor_utilities[o[0]],
                    reserved_value=reserved_value,
                    outcome_type=tuple,
                ),
                MappingUtilityFunction(
                    lambda o: opponent_utilities[o[0]],
                    reserved_value=reserved_value,
                    outcome_type=tuple,
                ),
            ],
            outcomes=outcomes,
            sort_by_welfare=True,
        )
        welfare = (
            np.asarray(elicitor_utilities) + np.asarray(opponent_utilities)
        ).tolist()
        # print(f'frontier: {frontier}\nmax. welfare: {max(welfare)} at outcome: ({welfare.index(max(welfare))},)')
        # print(f'frontier_locs: frontier_locs')
        neg = SAOMechanism(outcomes=n_outcomes, n_steps=10, outcome_type=tuple)
        opponent = LimitedOutcomesNegotiator(
            acceptable_outcomes=accepted,
            acceptance_probabilities=[1.0] * len(accepted),
        )
        eufun = MappingUtilityFunction(
            dict(zip(outcomes, elicitor_utilities)),
            reserved_value=reserved_value,
            outcome_type=tuple,
        )
        user = User(ufun=eufun, cost=cost)
        strategy = EStrategy(strategy=strategy)
        strategy.on_enter(ami=neg.ami)
        elicitor = FullKnowledgeElicitor(strategy=strategy, user=user)
        neg.add(opponent)
        neg.add(elicitor)
        neg.run()
        f2, f2_outcomes = neg.pareto_frontier(sort_by_welfare=True)
        assert len(frontier) == len(f2)
        assert all([_1 == _2] for _1, _2 in zip(frontier, f2))
        assert [_[0] for _ in f2_outcomes] == frontier_locs

    def test_loading_laptop(self, data_folder):
        domain, agents_info, issues = load_genius_domain_from_folder(
            os.path.join(data_folder, "Laptop"),
            force_single_issue=True,
            keep_issue_names=True,
            keep_value_names=True,
            agent_factories=lambda: AspirationNegotiator(),
            normalize_utilities=True,
        )
        # [domain.add(LimitedOutcomesNegotiator(outcomes=n_outcomes)
        #            , ufun=_['ufun']) for _ in agents_info]
        front, locs = domain.pareto_frontier(sort_by_welfare=True)
        assert front == [
            (0.7715533992081258, 0.8450562871935449),
            (0.5775524426410947, 1.0),
            (1.0, 0.5136317604069089),
            (0.8059990434329689, 0.6685754732133642),
        ]

    def test_loading_laptop_no_names(self, data_folder):
        domain, agents_info, issues = load_genius_domain_from_folder(
            os.path.join(data_folder, "Laptop"),
            force_single_issue=True,
            keep_issue_names=False,
            keep_value_names=False,
            agent_factories=lambda: AspirationNegotiator(),
            normalize_utilities=True,
        )
        # [domain.add(LimitedOutcomesNegotiator(outcomes=n_outcomes)
        #            , ufun=_['ufun']) for _ in agents_info]
        front, _ = domain.pareto_frontier(sort_by_welfare=True)
        assert front == [
            (0.7715533992081258, 0.8450562871935449),
            (0.5775524426410947, 1.0),
            (1.0, 0.5136317604069089),
            (0.8059990434329689, 0.6685754732133642),
        ]

    def test_elicitor_can_get_frontier(self, data_folder):
        domain, agents_info, issues = load_genius_domain_from_folder(
            os.path.join(data_folder, "Laptop"),
            force_single_issue=True,
            keep_issue_names=False,
            keep_value_names=False,
            normalize_utilities=True,
        )
        assert len(issues) == 1
        assert len(agents_info) == 2
        domain.add(LimitedOutcomesNegotiator(), ufun=agents_info[0]["ufun"])
        user = User(ufun=agents_info[0]["ufun"], cost=cost)
        strategy = EStrategy(strategy="titration-0.5")
        strategy.on_enter(ami=domain.ami)
        elicitor = FullKnowledgeElicitor(strategy=strategy, user=user)
        domain.add(elicitor)
        front, _ = domain.pareto_frontier()
        assert front == [(1.0, 1.0)]

    def test_elicitor_can_run_from_genius_domain(self, data_folder):
        domain, agents_info, issues = load_genius_domain_from_folder(
            os.path.join(data_folder, "Laptop"),
            force_single_issue=True,
            keep_issue_names=False,
            keep_value_names=False,
            normalize_utilities=True,
        )
        domain.add(LimitedOutcomesNegotiator(), ufun=agents_info[0]["ufun"])
        # domain.n_steps = 10
        user = User(ufun=agents_info[0]["ufun"], cost=0.2)
        strategy = EStrategy(strategy="titration-0.5")
        strategy.on_enter(ami=domain.ami)
        elicitor = PandoraElicitor(strategy=strategy, user=user)
        domain.add(elicitor)
        front, _ = domain.pareto_frontier()
        domain.run()
        assert len(domain.history) > 0

    def test_voi_baarslag_example(self):
        outcomes = [(0,), (1,)]
        alphas = [0.0, 1.0]
        opponent = LimitedOutcomesNegotiator(
            acceptable_outcomes=outcomes,
            acceptance_probabilities=alphas,
        )


def test_elicitation_run_with_no_conflict():
    n_outcomes = 50
    n_steps = 100
    config = SAOElicitingMechanism.generate_config(
        cost=0.05,
        n_outcomes=n_outcomes,
        conflict=0.0,
        winwin=1.0,
        n_steps=n_steps,
        own_reserved_value=0.1,
        opponent_type="tough",
        opponent_model_uncertainty=0.0,
        own_utility_uncertainty=0.0,
    )
    neg = SAOElicitingMechanism(**config, elicitor_type="full_knowledge")
    frontier, frontier_outcomes = neg.pareto_frontier(sort_by_welfare=True)
    assert len(frontier) > 0
    neg.run()
    # assert neg.agreement is None or neg.agreement in frontier_outcomes


def test_alternating_offers_eliciting_mechanism():
    for strategy, dynamic in [("pingpong", False), ("bisection", True)]:
        config = SAOElicitingMechanism.generate_config(
            cost=0.001,
            n_outcomes=10,
            opponent_model_adaptive=False,
            opponent_type="limited_outcomes",
            conflict=1.0,
            n_steps=500,
            time_limit=100000.0,
            own_utility_uncertainty=0.1,
            own_reserved_value=0.1,
        )
        p = SAOElicitingMechanism(
            **config,
            elicitation_strategy=strategy,
            elicitor_type="balanced",
            dynamic_queries=dynamic,
        )
        p.run()
        assert len(p.history) > 0
        # pprint(p.elicitation_state)
        # print(p.agents[1].reserved_value)
        assert p.elicitation_state["elicitation_cost"] >= 0.0
        assert (
            p.elicitation_state["elicitor_utility"] >= p.negotiators[1].reserved_value
        )


def test_alternating_offers_eliciting_mechanism_voi():
    for strategy, dynamic in [(None, False), (None, True)]:
        config = SAOElicitingMechanism.generate_config(
            cost=0.001,
            n_outcomes=10,
            opponent_model_adaptive=False,
            opponent_type="limited_outcomes",
            conflict=1.0,
            n_steps=500,
            time_limit=100000.0,
            own_utility_uncertainty=0.1,
            own_reserved_value=0.1,
        )
        p = SAOElicitingMechanism(
            **config,
            elicitation_strategy=strategy,
            elicitor_type="voi",
            dynamic_queries=dynamic,
        )
        p.run()
        assert len(p.history) > 0
        # pprint(p.elicitation_state)
        # print(p.agents[1].reserved_value)
        assert p.elicitation_state["elicitation_cost"] >= 0.0
        assert (
            p.elicitation_state["elicitor_utility"] >= p.negotiators[1].reserved_value
        )


@pytest.mark.skipif(
    condition=not BLIST_AVAILABLE,
    reason="Blist is not available. Ignoring this test",
)
def test_alternating_offers_eliciting_mechanism_voi_optimal():
    config = SAOElicitingMechanism.generate_config(
        cost=0.001,
        n_outcomes=10,
        opponent_model_adaptive=False,
        opponent_type="limited_outcomes",
        conflict=1.0,
        n_steps=500,
        time_limit=100000.0,
        own_utility_uncertainty=0.5,
        own_reserved_value=0.1,
    )
    p = SAOElicitingMechanism(**config, elicitor_type="voi_optimal")
    p.run()
    assert len(p.history) > 0
    # pprint(p.elicitation_state)
    # print(p.agents[1].reserved_value)
    assert p.elicitation_state["elicitation_cost"] >= 0.0
    assert (
        p.elicitation_state["elicitor_utility"]
        + p.elicitation_state["elicitation_cost"]
        >= p.negotiators[1].reserved_value
    )
    assert (
        p.elicitation_state["total_voi"] is not None
        and p.elicitation_state["total_voi"] >= 0
    )


def test_alternating_offers_eliciting_mechanism_full_knowledge():
    config = SAOElicitingMechanism.generate_config(
        cost=0.001,
        n_outcomes=10,
        opponent_model_adaptive=False,
        opponent_type="limited_outcomes",
        conflict=1.0,
        n_steps=500,
        time_limit=100000.0,
        own_utility_uncertainty=0.1,
        own_reserved_value=0.1,
    )
    p = SAOElicitingMechanism(
        **config, elicitation_strategy="bisection", elicitor_type="full_knowledge"
    )
    p.run()
    assert len(p.history) > 0
    # pprint(p.elicitation_state)
    # print(p.agents[1].reserved_value)
    assert p.elicitation_state["elicitation_cost"] == 0.0
    assert p.elicitation_state["elicitor_utility"] >= p.negotiators[1].reserved_value


def test_a_small_elicitation_session():
    import os

    config = SAOElicitingMechanism.generate_config(cost=0.2, n_outcomes=5, n_steps=10)
    p = SAOElicitingMechanism(
        **config,
        history_file_name=f"{os.path.expanduser('~/logs/negmas/tmp/elicit.log')}'",
    )
    p.run()
    # pprint(p.history)


def test_a_typical_elicitation_session():
    import random
    import os

    n_outcomes = 10
    accepted_outcomes = [int(random.random() <= 0.5) for _ in range(n_outcomes)]
    config = SAOElicitingMechanism.generate_config(
        cost=0.2, n_outcomes=n_outcomes, n_steps=100, own_reserved_value=0.25
    )
    p = SAOElicitingMechanism(
        **config,
        history_file_name=f"{os.path.expanduser('~/logs/negmas/tmp/elicit.log')}'",
    )
    p.run()
    # pprint(p.elicitation_state)
