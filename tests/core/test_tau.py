# from __future__ import annotations
#
# import random
# import time
# from pathlib import Path
# from typing import Iterable
#
# import hypothesis.strategies as st
# import pkg_resources
# import pytest
# from hypothesis import Verbosity, example, given, settings
# from matplotlib.axes import itertools
#
# from negmas.gb.evaluators.tau import INFINITE
# from negmas.gb.mechanisms.mechanisms import (
#     GAOMechanism,
#     GeneralizedTAUMechanism,
#     TAUMechanism,
# )
# from negmas.gb.negotiators.cab import CABNegotiator, CANNegotiator, CARNegotiator
# from negmas.gb.negotiators.timebased import AspirationNegotiator
# from negmas.gb.negotiators.war import WABNegotiator, WANNegotiator, WARNegotiator
# from negmas.inout import Scenario
# from negmas.mechanisms import Mechanism
# from negmas.outcomes.base_issue import make_issue
# from negmas.outcomes.categorical_issue import CategoricalIssue
# from negmas.outcomes.common import Outcome
# from negmas.outcomes.outcome_space import DiscreteCartesianOutcomeSpace, make_os
# from negmas.preferences.crisp.linear import LinearAdditiveUtilityFunction as LU
# from negmas.preferences.crisp.mapping import MappingUtilityFunction
# from negmas.preferences.crisp_ufun import UtilityFunction
# from negmas.preferences.ops import dominating_points, pareto_frontier
# from negmas.preferences.value_fun import AffineFun, IdentityFun, LinearFun, TableFun
# from negmas.sao.mechanism import SAOMechanism
# from tests.switches import NEGMAS_FASTRUN, NEGMAS_RUN_GENIUS
#
# SHOW_PLOTS = False
# SHOW_ALL_PLOTS = False
# FORCE_PLOT = False
# SHOW_HISTORY = False
# SHOW_ALL_HISTORIES = False
# FORCE_HISTORY = False
# SAONEGOTIATORS = [
#     AspirationNegotiator,
# ]
# MECHS = (TAUMechanism,) if NEGMAS_FASTRUN else (TAUMechanism, GeneralizedTAUMechanism)
# # TODO: resolve the infinite loop in WAN and WAB
# NEGOTIATORS = [
#     WARNegotiator,
#     CABNegotiator,
#     CANNegotiator,
#     CARNegotiator,
#     WANNegotiator,
#     WABNegotiator,
# ]
#
# PROPOSED = [
#     WARNegotiator,
#     CABNegotiator,
# ]
#
# RATIONAL_REJECTOR = [
#     CANNegotiator,
#     WANNegotiator,
#     CABNegotiator,
#     WABNegotiator,
# ]
#
# BEST_ACCEPTOR = [
#     CABNegotiator,
#     WABNegotiator,
# ]
# # all proposed TAU strategies are rational
# RATIONAL = list(itertools.product(NEGOTIATORS, NEGOTIATORS))
# IRRATIONAL = [
#     _ for _ in itertools.product(NEGOTIATORS, NEGOTIATORS) if _ not in RATIONAL
# ]
#
#
# # If one side is WAR and the other side may reject a rational outcome, the negotiation may be incomplete
# INCOMPLETE = list(
#     set(
#         list(
#             itertools.chain(
#                 *(
#                     [
#                         itertools.permutations(_)
#                         for _ in zip(itertools.repeat(WANNegotiator), RATIONAL_REJECTOR)
#                     ]
#                     + [
#                         itertools.permutations(_)
#                         for _ in zip(itertools.repeat(WARNegotiator), RATIONAL_REJECTOR)
#                     ]
#                     + [
#                         itertools.permutations(_)
#                         for _ in zip(itertools.repeat(WABNegotiator), RATIONAL_REJECTOR)
#                     ]
#                 )
#             )
#         )
#     )
# )
# COMPLETE = [
#     _ for _ in itertools.product(NEGOTIATORS, NEGOTIATORS) if _ not in INCOMPLETE
# ]
#
# # An optimal strategy profile is one for which both sides are from the proposed set
# OPTIMAL = list(itertools.product(BEST_ACCEPTOR, BEST_ACCEPTOR))
# NONOPTIMAL = [
#     _ for _ in itertools.product(NEGOTIATORS, NEGOTIATORS) if _ not in OPTIMAL
# ]
#
#
# EQUILIBRIUM = [
#     (WARNegotiator, WARNegotiator),
# ]
# NONEQUILIBRIUM = [
#     _ for _ in itertools.product(NEGOTIATORS, NEGOTIATORS) if _ not in EQUILIBRIUM
# ]
#
#
# NORAISE = [
#     (WARNegotiator, WARNegotiator),
#     (WARNegotiator, CANNegotiator),
#     (WARNegotiator, WANNegotiator),
#     (WARNegotiator, CABNegotiator),
#     (WARNegotiator, WABNegotiator),
#     (CANNegotiator, WARNegotiator),
#     (CANNegotiator, WANNegotiator),
#     (CANNegotiator, WABNegotiator),
#     (WANNegotiator, WARNegotiator),
#     (WANNegotiator, CANNegotiator),
#     (WANNegotiator, WANNegotiator),
#     (WANNegotiator, CABNegotiator),
#     (WANNegotiator, WABNegotiator),
#     (CABNegotiator, WARNegotiator),
#     (CABNegotiator, WANNegotiator),
#     (CABNegotiator, WABNegotiator),
#     (WABNegotiator, WARNegotiator),
#     (WABNegotiator, CANNegotiator),
#     (WABNegotiator, WANNegotiator),
#     (WABNegotiator, CABNegotiator),
#     (WABNegotiator, WABNegotiator),
# ]
#
#
# @pytest.fixture
# def scenarios_folder():
#     return pkg_resources.resource_filename(
#         "negmas", resource_name="tests/data/scenarios"
#     )
#
#
# def _plot(p, err=False, force=False):
#     if not force and err and not SHOW_PLOTS:
#         return
#     if not force and not err and not SHOW_ALL_PLOTS:
#         return
#     import matplotlib.pyplot as plt
#
#     p.plot(xdim="step")
#     plt.show()
#     plt.savefig("fig.png")
#
#
# def _history(p: Mechanism, err=False, force=False):
#     if not force and err and not SHOW_HISTORY:
#         return ""
#     if not force and not err and not SHOW_ALL_HISTORIES:
#         return ""
#     return p.trace  # type: ignore
#
#
# @pytest.mark.parametrize("neg", NEGOTIATORS)
# def test_a_tau_session_example(neg):
#     for _ in range(100):
#         r1, r2, n1, n2 = 0.2, 0.3, 5, 10
#         eps = 1e-3
#         time.perf_counter()
#         os: DiscreteCartesianOutcomeSpace = make_os([make_issue(n1), make_issue(n2)])  # type: ignore
#         p = TAUMechanism(outcome_space=os)
#         ufuns = [LU.random(os, reserved_value=r1), LU.random(os, reserved_value=r2)]
#         for i, u in enumerate(ufuns):
#             p.add(
#                 neg(name=f"RCS{i}", id=f"RCS{i}"),
#                 preferences=u,
#             )
#         front_utils, front_outcomes = p.pareto_frontier()
#         no_valid_outcomes = all(
#             u1 <= r1 + eps or u2 <= r2 + eps for u1, u2 in front_utils
#         )
#         p.run()
#         assert len(p.history) > 0, f"{p.state}"
#         if (neg, neg) in COMPLETE:
#             assert (
#                 p.agreement is not None or no_valid_outcomes
#             ), f"No agreement in a supposedly complete profile {_history(p)}{_plot(p, True)}"
#         if (neg, neg) in OPTIMAL:
#             assert (
#                 p.agreement in front_outcomes or p.agreement is None
#             ), f"Suboptimal agreement in a supposedly optimal profile {_history(p)}{_plot(p, True)}"
#
#
# def run_adversarial_case(
#     buyer,
#     seller,
#     normalized=False,
#     seller_reserved=0.0,
#     buyer_reserved=0.0,
#     cardinality=INFINITE,
#     mechanism_type: type[Mechanism] = TAUMechanism,
#     force_plot=False,
#     do_asserts=True,
#     n_steps=10 * 10 * 3,
# ):
#     n1, n2 = 100, 11
#     # create negotiation agenda (issues)
#     issues = [
#         make_issue(name="issue", values=n1 + n2),
#     ]
#     os = make_os(issues)
#
#     # create the mechanism
#     if mechanism_type == GeneralizedTAUMechanism:
#         session = mechanism_type(outcome_space=os, cardinality=cardinality)  # type: ignore
#     elif mechanism_type == TAUMechanism:
#         session = mechanism_type(outcome_space=os)
#     else:
#         session = mechanism_type(outcome_space=os, n_steps=n_steps)
#
#     # define buyer and seller utilities
#     seller_utility = MappingUtilityFunction(
#         dict(zip(os.enumerate(), [i for i in range(n1 + n2)])),  # type: ignore
#         outcome_space=os,
#     )
#     buyer_utility = MappingUtilityFunction(
#         dict(
#             zip(
#                 os.enumerate(),  # type: ignore
#                 [n1 + n2 for _ in range(n1)] + [n1 + n2 - 1 - i for i in range(n2)],
#             )
#         ),
#         outcome_space=os,
#     )
#     if normalized:
#         seller_utility = seller_utility.scale_max(1.0)
#         buyer_utility = buyer_utility.scale_max(1.0)
#     seller_utility.reserved_value = seller_reserved
#     buyer_utility.reserved_value = buyer_reserved
#
#     # create and add buyer and seller negotiators
#     b = buyer(name="buyer", id="buyer")
#     s = seller(name="seller", id="seller")
#     s.name += f"{s.short_type_name}"
#     b.name += f"{b.short_type_name}"
#     session.add(b, ufun=buyer_utility)
#     session.add(s, ufun=seller_utility)
#
#     front_utils, front_outcomes = session.pareto_frontier()
#     eps = 1e-3
#     no_valid_outcomes = all(
#         u1 <= buyer_reserved + eps or u2 <= seller_reserved + eps
#         for u1, u2 in front_utils
#     )
#     session.run()
#     assert len(session.history) > 0, f"{session.state}"
#     if do_asserts:
#         if (buyer, seller) in COMPLETE:
#             assert (
#                 session.agreement is not None or no_valid_outcomes
#             ), f"No agreement in a supposedly complete profile\n{_history(session)}{_plot(session, True, force=force_plot)}"
#         if (buyer, seller) in OPTIMAL:
#             assert (
#                 session.agreement in front_outcomes or session.agreement is None
#             ), f"Suboptimal agreement in a supposedly optimal profile\n{_history(session)}{_plot(session, True, force=force_plot)}"
#     _plot(session, force=force_plot)
#     return session
#
#
# def run_buyer_seller(
#     buyer,
#     seller,
#     normalized=False,
#     seller_reserved=0.0,
#     buyer_reserved=0.0,
#     cardinality=INFINITE,
#     min_unique=0,
#     mechanism_type: type[Mechanism] = TAUMechanism,
#     force_plot=False,
#     do_asserts=True,
#     n_steps=10 * 10 * 3,
#     mechanism_params: dict | None = None,
# ):
#     if not mechanism_params:
#         mechanism_params = dict()
#     # create negotiation agenda (issues)
#     issues = (
#         make_issue(name="price", values=10),
#         make_issue(name="quantity", values=(1, 11)),
#         make_issue(name="delivery_time", values=["today", "tomorrow", "nextweek"]),
#     )
#
#     # create the mechanism
#     if mechanism_type == GeneralizedTAUMechanism:
#         mechanism_params.update(dict(cardinality=cardinality, min_unique=min_unique))
#     elif mechanism_type == TAUMechanism:
#         pass
#     else:
#         mechanism_params.update(dict(n_steps=n_steps))
#     session = mechanism_type(issues=issues, **mechanism_params)  # type: ignore
#
#     # define buyer and seller utilities
#     seller_utility = LU(
#         values=[  # type: ignore
#             IdentityFun(),
#             LinearFun(0.2),
#             TableFun(dict(today=1.0, tomorrow=0.2, nextweek=0.0)),
#         ],
#         outcome_space=session.outcome_space,
#     )
#
#     buyer_utility = LU(
#         values={  # type: ignore
#             "price": AffineFun(-1, bias=9.0),
#             "quantity": LinearFun(0.2),
#             "delivery_time": TableFun(dict(today=0, tomorrow=0.7, nextweek=1.0)),
#         },
#         outcome_space=session.outcome_space,
#     )
#     if normalized:
#         seller_utility = seller_utility.scale_max(1.0)
#         buyer_utility = buyer_utility.scale_max(1.0)
#     seller_utility.reserved_value = seller_reserved
#     buyer_utility.reserved_value = buyer_reserved
#
#     # create and add buyer and seller negotiators
#     b = buyer(name="buyer", id="buyer")
#     s = seller(name="seller", id="seller")
#     s.name += f"{s.short_type_name}"
#     b.name += f"{b.short_type_name}"
#     session.add(b, ufun=buyer_utility)
#     session.add(s, ufun=seller_utility)
#
#     agreement = session.run().agreement
#
#     front_utils, front_outcomes = session.pareto_frontier(sort_by_welfare=True)
#     eps = 1e-3
#     no_valid_outcomes = all(
#         u1 <= buyer_reserved + eps or u2 <= seller_reserved + eps
#         for u1, u2 in front_utils
#     )
#     assert len(session.history) > 0, f"{session.state}"
#     if do_asserts:
#         if (buyer, seller) in COMPLETE:
#             assert (
#                 agreement is not None or no_valid_outcomes
#             ), f"No agreement in a supposedly complete profile\n{_history(session)}{_plot(session, True, force=force_plot)}"
#         if (buyer, seller) in OPTIMAL:
#             x = tuple(float(u(agreement)) for u in [buyer_utility, seller_utility])
#             dominating = [
#                 (front_utils[_], front_outcomes[_])
#                 for _ in dominating_points(x, front_utils)
#             ]
#             assert (
#                 len(dominating) == 0 or agreement is None
#             ), f"Suboptimal agreement in a supposedly optimal profile\nAgreement:{agreement} (u={x}) is dominated by {dominating}\n{_history(session)}{_plot(session, True, force=force_plot)}"
#             assert (
#                 agreement in front_outcomes or agreement is None
#             ), f"Suboptimal agreement in a supposedly optimal profile\n{_history(session)}{_plot(session, True, force=force_plot)}"
#     _plot(session, force=force_plot)
#     return session
#
#
# def remove_under_line(
#     os: DiscreteCartesianOutcomeSpace,
#     ufuns: Iterable[UtilityFunction],
#     limits=((0.0, 0.8), (0.0, None)),
# ):
#     mxs = [_.max() for _ in ufuns]
#
#     limits = list(limits)
#     for i, (x, y) in enumerate(limits):
#         if y is None:
#             limits[i] = (x, mxs[i])
#     x1, x2 = limits[0]
#     y1, y2 = limits[1]
#
#     outcomes = list(os.enumerate())
#     _, frontier = pareto_frontier(list(ufuns), outcomes)
#     frontier = set(frontier)
#
#     accepted: list[Outcome] = []
#     for outcome in outcomes:
#         if outcome in frontier:
#             accepted.append(outcome)
#             continue
#         xA, yA = (_(outcome) for _ in ufuns)
#         v1 = (x2 - x1, y2 - y1)  # Vector 1
#         v2 = (x2 - xA, y2 - yA)  # Vector 2
#         xp = v1[0] * v2[1] - v1[1] * v2[0]  # Cross product
#         if xp > 0:
#             # s = (y2-y1) / (x2 - x1)
#             # assert yA < s * xA + y1 - s * x1
#             continue
#         accepted.append(outcome)
#     return accepted
#
#
# def run_anac_example(
#     first_type,
#     second_type,
#     mechanism_type: type[Mechanism] = TAUMechanism,
#     force_plot=False,
#     do_asserts=True,
#     domain_name="cameradomain",
#     mechanism_params=dict(),
#     single_issue=False,
#     remove_under=False,
# ):
#     src = pkg_resources.resource_filename(
#         "negmas", resource_name=f"tests/data/{domain_name}"
#     )
#     base_folder = src
#     domain = Scenario.from_genius_folder(Path(base_folder))
#     assert domain is not None
#     # create the mechanism
#
#     a1 = first_type(
#         name="a1",
#         id="a1",
#     )
#     a2 = second_type(
#         name="a2",
#         id="a2",
#     )
#     a1.name += f"{a1.short_type_name}"
#     a2.name += f"{a2.short_type_name}"
#     domain.normalize()
#     if single_issue or remove_under:
#         domain = domain.to_single_issue(randomize=True)
#         assert len(domain.issues) == 1
#     if remove_under:
#         outcomes = remove_under_line(domain.outcome_space, domain.ufuns)  # type: ignore
#         issue = CategoricalIssue(
#             values=[_[0] for _ in outcomes], name=domain.outcome_space.issues[0].name
#         )
#         domain.outcome_space = make_os([issue])
#
#         # domain.ufuns = [ MappingUtilityFunction(dict(zip(outcomes, [u(_) for _ in outcomes])), outcome_space=domain.outcome_space) for u in domain.ufuns ]
#     domain.mechanism_type = mechanism_type
#     domain.mechanism_params = mechanism_params
#     neg = domain.make_session([a1, a2], n_steps=domain.outcome_space.cardinality)
#     if neg is None:
#         raise ValueError(f"Failed to lead domain from {base_folder}")
#
#     _, front_outcomes = neg.pareto_frontier()
#     neg.run()
#     assert len(neg.history) > 0, f"{neg.state}"
#     if do_asserts:
#         assert (
#             neg.agreement is not None
#         ), f"{_history(neg)}{_plot(neg, True, force=force_plot)}"
#         assert neg.agreement in front_outcomes or neg.agreement is None
#     _plot(neg, force=force_plot)
#     return neg
#
#
# @pytest.mark.parametrize(
#     ["neg1", "neg2"], list(itertools.product(NEGOTIATORS, NEGOTIATORS))
# )
# def test_buyer_seller_easy(neg1, neg2):
#     run_buyer_seller(
#         neg1,
#         neg2,
#         normalized=True,
#         seller_reserved=0.1,
#         buyer_reserved=0.1,
#         force_plot=FORCE_PLOT,
#     )
#
#
# # @pytest.mark.skip(reason="Known failure. Enters an infinite loop")
# def test_buyer_seller_alphainf_war_cab():
#     run_buyer_seller(
#         WARNegotiator,
#         CABNegotiator,
#         mechanism_type=TAUMechanism,
#         normalized=True,
#         seller_reserved=0.5,
#         buyer_reserved=0.6,
#         force_plot=FORCE_PLOT,
#         min_unique=0,
#         cardinality=INFINITE,
#     )
#
#
# # @pytest.mark.skip(reason="Known failure. Enters an infinite loop")
# def test_buyer_seller_easy_wab():
#     run_buyer_seller(
#         WABNegotiator,
#         WABNegotiator,
#         normalized=True,
#         seller_reserved=0.1,
#         buyer_reserved=0.1,
#         force_plot=FORCE_PLOT,
#     )
#
#
# # @pytest.mark.skip(reason="Known failure. Enters an infinite loop")
# def test_buyer_seller_easy_wan():
#     run_buyer_seller(
#         WANNegotiator,
#         WANNegotiator,
#         normalized=True,
#         seller_reserved=0.1,
#         buyer_reserved=0.1,
#         force_plot=FORCE_PLOT,
#     )
#
#
# def test_buyer_seller_easy_cab():
#     run_buyer_seller(
#         CABNegotiator,
#         CABNegotiator,
#         normalized=True,
#         seller_reserved=0.1,
#         buyer_reserved=0.1,
#         force_plot=FORCE_PLOT,
#     )
#
#
# def test_buyer_seller_gao_easy():
#     run_buyer_seller(
#         AspirationNegotiator,
#         AspirationNegotiator,
#         normalized=True,
#         seller_reserved=0.1,
#         buyer_reserved=0.1,
#         force_plot=FORCE_PLOT,
#         mechanism_type=GAOMechanism,
#         do_asserts=False,
#     )
#
#
# def test_buyer_seller_sao():
#     run_buyer_seller(
#         AspirationNegotiator,
#         AspirationNegotiator,
#         normalized=True,
#         seller_reserved=0.5,
#         buyer_reserved=0.6,
#         force_plot=FORCE_PLOT,
#         mechanism_type=GAOMechanism,
#         do_asserts=False,
#     )
#
#
# @pytest.mark.parametrize(
#     ["neg1", "neg2", "mechanism"],
#     list(
#         itertools.product(
#             NEGOTIATORS, NEGOTIATORS, (TAUMechanism, GeneralizedTAUMechanism)
#         )
#     ),
# )
# def test_buyer_seller_alphainf(neg1, neg2, mechanism):
#     run_buyer_seller(
#         neg1,
#         neg2,
#         mechanism_type=mechanism,
#         normalized=True,
#         seller_reserved=0.5,
#         buyer_reserved=0.6,
#         force_plot=FORCE_PLOT,
#         min_unique=0,
#         cardinality=INFINITE,
#     )
#
#
# @pytest.mark.parametrize(
#     ["neg1", "neg2"], list(itertools.product(NEGOTIATORS, NEGOTIATORS))
# )
# def test_buyer_seller_alpha0(neg1, neg2):
#     run_buyer_seller(
#         neg1,
#         neg2,
#         mechanism_type=GeneralizedTAUMechanism,
#         normalized=True,
#         seller_reserved=0.5,
#         buyer_reserved=0.6,
#         force_plot=FORCE_PLOT,
#         min_unique=0,
#         cardinality=0,
#     )
#
#
# @pytest.mark.parametrize(
#     ["neg1", "neg2"],
#     [
#         _
#         for _ in itertools.product(NEGOTIATORS, NEGOTIATORS)
#         if (_[0], _[1]) not in NORAISE
#     ],
# )
# def test_buyer_seller_betainf(neg1, neg2):
#     with pytest.raises(AssertionError):
#         run_buyer_seller(
#             neg1,
#             neg2,
#             mechanism_type=GeneralizedTAUMechanism,
#             normalized=True,
#             seller_reserved=0.5,
#             buyer_reserved=0.6,
#             force_plot=FORCE_PLOT,
#             min_unique=INFINITE,
#         )
#
#
# # @pytest.mark.parametrize(
# #     ["neg1", "neg2", "parallel", "accept_any"],
# #     [
# #         _
# #         for _ in itertools.product(
# #             NEGOTIATORS, NEGOTIATORS, [True, False], [True, False]
# #         )
# #         if (_[0], _[1]) not in NORAISE
# #     ],
# # )
# # def test_buyer_seller_tau_bilateral_any_acceptance_extension_passes(
# #     neg1, neg2, parallel, accept_any
# # ):
# #     run_buyer_seller(
# #         neg1,
# #         neg2,
# #         mechanism_type=TAUMechanism,
# #         normalized=True,
# #         seller_reserved=0.5,
# #         buyer_reserved=0.6,
# #         force_plot=FORCE_PLOT,
# #         min_unique=0,
# #         mechanism_params=dict(parallel=parallel, accept_in_any_thread=accept_any),
# #     )
#
#
# def test_buyer_seller_tau_bilateral_any_acceptance_extension_ok():
#     for neg1, neg2 in itertools.product(NEGOTIATORS, NEGOTIATORS):
#         if (neg1, neg2) in NORAISE:
#             continue
#         results = []
#         for parallel, accept_any in itertools.product([True, False], [True, False]):
#             x = run_buyer_seller(
#                 neg1,
#                 neg2,
#                 mechanism_type=TAUMechanism,
#                 normalized=True,
#                 seller_reserved=0.5,
#                 buyer_reserved=0.6,
#                 force_plot=FORCE_PLOT,
#                 min_unique=0,
#                 mechanism_params=dict(
#                     parallel=parallel, accept_in_any_thread=accept_any
#                 ),
#             )
#             if x is not None:
#                 results.append(x.agreement)
#             else:
#                 results.append(x)
#         assert all(a == b for a, b in zip(results[1:], results[:-1]))
#
#
# @pytest.mark.parametrize(
#     ["neg1", "neg2", "mechanism"],
#     list(itertools.product(NEGOTIATORS, NEGOTIATORS, MECHS)),
# )
# def test_buyer_seller_beta0(neg1, neg2, mechanism):
#     run_buyer_seller(
#         neg1,
#         neg2,
#         mechanism_type=mechanism,
#         normalized=True,
#         seller_reserved=0.5,
#         buyer_reserved=0.6,
#         force_plot=FORCE_PLOT,
#         min_unique=0,
#     )
#
#
# def test_buyer_seller_beta0_example_gtau():
#     run_buyer_seller(
#         CABNegotiator,
#         CABNegotiator,
#         mechanism_type=GeneralizedTAUMechanism,
#         normalized=True,
#         seller_reserved=0.5,
#         buyer_reserved=0.6,
#         force_plot=FORCE_PLOT,
#         min_unique=0,
#     )
#
#
# def test_buyer_seller_beta0_example_tau():
#     run_buyer_seller(
#         CABNegotiator,
#         CABNegotiator,
#         mechanism_type=TAUMechanism,
#         normalized=True,
#         seller_reserved=0.5,
#         buyer_reserved=0.6,
#         force_plot=FORCE_PLOT,
#         min_unique=0,
#     )
#
#
# @pytest.mark.parametrize(
#     ["neg1", "neg2"], list(itertools.product(NEGOTIATORS, NEGOTIATORS))
# )
# def test_anac_scenario_example_single(neg1, neg2):
#     run_anac_example(
#         neg1,
#         neg2,
#         force_plot=FORCE_PLOT,
#         single_issue=True,
#         remove_under=False,
#     )
#
#
# def test_anac_scenario_example_gao_single():
#     run_anac_example(
#         AspirationNegotiator,
#         AspirationNegotiator,
#         mechanism_type=GAOMechanism,
#         force_plot=FORCE_PLOT,
#         single_issue=True,
#         remove_under=False,
#     )
#
#
# def test_anac_scenario_example_sao_single():
#     run_anac_example(
#         AspirationNegotiator,
#         AspirationNegotiator,
#         mechanism_type=SAOMechanism,
#         force_plot=FORCE_PLOT,
#         single_issue=True,
#         remove_under=False,
#     )
#
#
# @pytest.mark.skipif(not NEGMAS_RUN_GENIUS, reason="Skipping genius tests")
# def test_anac_scenario_example_genius_gao_single():
#     from negmas.genius.gnegotiators import Atlas3
#
#     run_anac_example(
#         Atlas3,
#         Atlas3,
#         mechanism_type=GAOMechanism,
#         force_plot=FORCE_PLOT,
#         single_issue=True,
#         remove_under=False,
#     )
#
#
# @pytest.mark.skipif(not NEGMAS_RUN_GENIUS, reason="Skipping genius tests")
# def test_anac_scenario_example_genius_sao_single():
#     from negmas.genius.gnegotiators import Atlas3
#
#     run_anac_example(
#         Atlas3,
#         Atlas3,
#         mechanism_type=SAOMechanism,
#         force_plot=FORCE_PLOT,
#         single_issue=True,
#         remove_under=False,
#     )
#
#
# @pytest.mark.parametrize(
#     ["neg1", "neg2"], list(itertools.product(NEGOTIATORS, NEGOTIATORS))
# )
# def test_anac_scenario_example(neg1, neg2):
#     run_anac_example(neg1, neg2, force_plot=FORCE_PLOT)
#
#
# def test_anac_scenario_example_gao():
#     run_anac_example(
#         AspirationNegotiator,
#         AspirationNegotiator,
#         mechanism_type=GAOMechanism,
#         force_plot=FORCE_PLOT,
#     )
#
#
# @pytest.mark.skipif(not NEGMAS_RUN_GENIUS, reason="Skipping genius tests")
# def test_anac_scenario_example_genius():
#     from negmas.genius.gnegotiators import Atlas3
#
#     run_anac_example(
#         Atlas3,
#         Atlas3,
#         mechanism_type=SAOMechanism,
#         force_plot=FORCE_PLOT,
#     )
#
#
# @pytest.mark.parametrize(
#     ["neg1", "neg2"], list(itertools.product(NEGOTIATORS, NEGOTIATORS))
# )
# def test_adversarial_case_easy(neg1, neg2):
#     run_adversarial_case(
#         neg1,
#         neg2,
#         force_plot=FORCE_PLOT,
#     )
#
#
# @pytest.mark.skipif(not NEGMAS_RUN_GENIUS, reason="Skipping genius tests")
# def test_adversarial_case_gao_easy_genius():
#     from negmas.genius.gnegotiators import Atlas3
#
#     run_adversarial_case(
#         Atlas3,
#         Atlas3,
#         force_plot=FORCE_PLOT,
#         mechanism_type=GAOMechanism,
#         do_asserts=False,
#     )
#
#
# def test_adversarial_case_gao_easy():
#     run_adversarial_case(
#         AspirationNegotiator,
#         AspirationNegotiator,
#         force_plot=FORCE_PLOT,
#         mechanism_type=GAOMechanism,
#         do_asserts=False,
#     )
#
#
# @given(
#     neg=st.sampled_from(NEGOTIATORS),
#     r1=st.floats(0, 1),
#     r2=st.floats(0, 1),
#     n1=st.integers(3, 5),
#     n2=st.integers(3, 10),
#     U1=st.sampled_from(
#         [
#             LU,
#         ]
#     ),
#     U2=st.sampled_from(
#         [
#             LU,
#         ]
#     ),
# )
# @settings(deadline=10_000, verbosity=Verbosity.verbose, max_examples=10)
# def test_a_tau_session(neg, r1, r2, n1, n2, U1, U2):
#     eps = 1e-3
#     time.perf_counter()
#     os: DiscreteCartesianOutcomeSpace = make_os([make_issue(n1), make_issue(n2)])  # type: ignore
#     p = TAUMechanism(outcome_space=os)
#     ufuns = [U1.random(os, reserved_value=r1), U2.random(os, reserved_value=r2)]
#     for i, u in enumerate(ufuns):
#         p.add(
#             neg(name=f"CAB{i}"),
#             preferences=u,
#         )
#     p.run()
#     front_utils, front_outcomes = p.pareto_frontier()
#     no_valid_outcomes = all(u1 <= r1 + eps or u2 <= r2 + eps for u1, u2 in front_utils)
#     assert len(p.history) > 0, f"{p.state}"
#     if (neg, neg) in COMPLETE:
#         assert (
#             p.agreement is not None or no_valid_outcomes
#         ), f"No agreement in a supposedly complete profile\n{_history(p)}{_plot(p, True, force=False)}"
#     if (neg, neg) in OPTIMAL:
#         assert (
#             p.agreement in front_outcomes or p.agreement is None
#         ), f"Suboptimal agreement in a supposedly optimal profile\n{_history(p)}{_plot(p, True, force=False)}"
#
#
# from negmas.gb.adapters.tau import TAUNegotiatorAdapter, UtilityAdapter
#
#
# @given(
#     neg=st.sampled_from(SAONEGOTIATORS),
#     U1=st.sampled_from(
#         [
#             LU,
#         ]
#     ),
#     U2=st.sampled_from(
#         [
#             LU,
#         ]
#     ),
# )
# @settings(deadline=10_000, verbosity=Verbosity.verbose, max_examples=10)
# def test_a_tau_session_with_adapter(neg, U1, U2):
#     n1, n2 = 5, 10
#     time.perf_counter()
#     os: DiscreteCartesianOutcomeSpace = make_os([make_issue(n1), make_issue(n2)])  # type: ignore
#     p = TAUMechanism(outcome_space=os)
#     ufuns = [U1.random(os, reserved_value=0.0), U2.random(os, reserved_value=0.0)]
#     negotiator = neg(name=f"{neg.__name__}{0}")
#     p.add(TAUNegotiatorAdapter(base=negotiator), preferences=ufuns[0])
#     for i, u in enumerate(ufuns[1:]):
#         p.add(CABNegotiator(name=f"CAB{i}"), preferences=u)
#     p.run()
#     assert len(p.full_trace) > 0, f"{p.state}{_plot(p)}"
#
#
# @pytest.mark.skipif(not NEGMAS_RUN_GENIUS, reason="Skipping genius tests")
# def test_buyer_seller_gao_easy_genius():
#     from negmas.genius.gnegotiators import Atlas3
#
#     run_buyer_seller(
#         Atlas3,
#         Atlas3,
#         normalized=True,
#         seller_reserved=0.1,
#         buyer_reserved=0.1,
#         force_plot=FORCE_PLOT,
#         mechanism_type=GAOMechanism,
#         do_asserts=False,
#     )
#
#
# @pytest.mark.skipif(not NEGMAS_RUN_GENIUS, reason="Skipping genius tests")
# def test_buyer_seller_sao_genius():
#     from negmas.genius.gnegotiators import Atlas3
#
#     run_buyer_seller(
#         Atlas3,
#         Atlas3,
#         normalized=True,
#         seller_reserved=0.5,
#         buyer_reserved=0.6,
#         force_plot=FORCE_PLOT,
#         mechanism_type=SAOMechanism,
#         do_asserts=False,
#     )
#
#
# @given(
#     neg=st.sampled_from(NEGOTIATORS),
#     r1=st.floats(0, 1),
#     r2=st.floats(0, 1),
#     n1=st.integers(3, 5),
#     n2=st.integers(3, 10),
#     U1=st.sampled_from(
#         [
#             LU,
#         ]
#     ),
#     U2=st.sampled_from(
#         [
#             LU,
#         ]
#     ),
# )
# @settings(deadline=10_000, verbosity=Verbosity.verbose, max_examples=10)
# def test_tau_matches_generalized(neg, r1, r2, n1, n2, U1, U2):
#     eps = 1e-3
#     time.perf_counter()
#     os: DiscreteCartesianOutcomeSpace = make_os([make_issue(n1), make_issue(n2)])  # type: ignore
#     results = []
#     ufuns = [U1.random(os, reserved_value=r1), U2.random(os, reserved_value=r2)]
#     for cls in (TAUMechanism, GeneralizedTAUMechanism):
#         p = cls(outcome_space=os)
#         for i, u in enumerate(ufuns):
#             p.add(
#                 neg(name=f"CAB{i}"),
#                 preferences=u,
#             )
#         p.run()
#         front_utils, front_outcomes = p.pareto_frontier()
#         no_valid_outcomes = all(
#             u1 <= r1 + eps or u2 <= r2 + eps for u1, u2 in front_utils
#         )
#         assert len(p.history) > 0, f"{p.state}"
#         if (neg, neg) in COMPLETE:
#             assert (
#                 p.agreement is not None or no_valid_outcomes
#             ), f"No agreement in a supposedly complete profile\n{_history(p)}{_plot(p, True, force=False)}"
#         if (neg, neg) in OPTIMAL:
#             assert (
#                 p.agreement in front_outcomes or p.agreement is None
#             ), f"Suboptimal agreement in a supposedly optimal profile\n{_history(p)}{_plot(p, True, force=False)}"
#         results.append(dict(steps=p.current_step, agreement=p.agreement, trace=p.trace))
#     assert (
#         results[0]["agreement"] == results[1]["agreement"]
#     ), f"{results[0]['trace']=}\n{results[1]['trace']=}"
#     assert (
#         results[0]["steps"] == results[1]["steps"]
#     ), f"{results[0]['trace']=}\n{results[1]['trace']=}"
#
#
# @given(vals=st.lists(st.floats(), min_size=2, max_size=100), r=st.floats())
# @example(
#     vals=[0.0, 0.0],
#     r=1.0,
# )
# def test_tau_adapter_adapts_correctly(vals, r):
#     r = (r - 0.5) * 2.5
#     # vals = [0.0, 0.0, 1.0, 2.0, 3.0, 6.999, 7.0, 7.0, 7.1, 7.22, 8.0, 10]
#     # r = random.choice(vals)
#     os = make_os([make_issue(len(vals))])
#     outcomes = list(os.enumerate_or_sample())
#     ufun = MappingUtilityFunction(
#         dict(zip(outcomes, vals)), outcome_space=os, reserved_value=r
#     )
#     utiladapter = UtilityAdapter(ufun)
#     for i in range(len(outcomes) - 1, -1, -1):
#         assert utiladapter(outcomes[i]) == outcomes[i]
#     rational_outcomes = [_ for _ in outcomes if ufun(_) >= r]
#     if len(rational_outcomes):
#         sample = random.choices(rational_outcomes, k=2 * len(vals))
#         outcomes = set(outcomes)
#         utiladapter = UtilityAdapter(ufun)
#         for outcome in sample:
#             shouldbesame = outcome not in utiladapter.offered
#             shouldrepeat = len(utiladapter.offered) == len(outcomes)
#             x = utiladapter(outcome)
#             if shouldrepeat:
#                 assert (
#                     x == utiladapter.last_offer
#                 ), f"{x=}, {outcome=}, {utiladapter.last_offer=}, {utiladapter.offered=}"
#             if shouldbesame:
#                 assert (
#                     x == outcome
#                 ), f"{x=}, {outcome=}, {utiladapter.last_offer=}, {utiladapter.offered=}"
#             if ufun(outcome) >= ufun.reserved_value:
#                 if ufun(x) < ufun.reserved_value:
#                     pass
#                 assert (
#                     ufun(x) >= ufun.reserved_value
#                 ), f"{x=}, {outcome=}, {utiladapter.last_offer=}, {utiladapter.offered=}"
#             assert (
#                 x in outcomes
#             ), f"{x=}, {outcome=}, {utiladapter.last_offer=}, {utiladapter.offered=}"
#             assert (
#                 x in utiladapter.offered
#             ), f"{x=}, {outcome=}, {utiladapter.last_offer=}, {utiladapter.offered=}"
#
#     # testing crazy startegy with some irrational outcomes
#     sample = random.choices(list(outcomes), k=100)
#     outcomes = set(outcomes)
#     utiladapter = UtilityAdapter(ufun)
#     for outcome in sample:
#         print(outcome)
#         shouldbesame = outcome not in utiladapter.offered
#         shouldrepeat = len(utiladapter.offered) == len(outcomes)
#         x = utiladapter(outcome)
#         if shouldrepeat:
#             assert (
#                 x == utiladapter.last_offer
#             ), f"{x=}, {outcome=}, {utiladapter.last_offer=}, {utiladapter.offered=}"
#         if shouldbesame:
#             assert (
#                 x == outcome
#             ), f"{x=}, {outcome=}, {utiladapter.last_offer=}, {utiladapter.offered=}"
#         if ufun(outcome) >= ufun.reserved_value:
#             if ufun(x) < ufun.reserved_value:
#                 pass
#             assert (
#                 ufun(x) >= ufun.reserved_value or x == utiladapter.last_offer
#             ), f"{x=}, {outcome=}, {utiladapter.last_offer=}, {utiladapter.offered=}"
#         assert (
#             x in outcomes
#         ), f"{x=}, {outcome=}, {utiladapter.last_offer=}, {utiladapter.offered=}"
#         assert (
#             x in utiladapter.offered
#         ), f"{x=}, {outcome=}, {utiladapter.last_offer=}, {utiladapter.offered=}"
#
#
# def test_a_tau_session_example_cab():
#     for _ in range(100):
#         r1, r2, n1, n2 = 0.2, 0.3, 2, 2
#         eps = 1e-3
#         time.perf_counter()
#         os: DiscreteCartesianOutcomeSpace = make_os([make_issue(n1), make_issue(n2)])  # type: ignore
#         p = TAUMechanism(outcome_space=os)
#         ufuns = [LU.random(os, reserved_value=r1), LU.random(os, reserved_value=r2)]
#         for i, u in enumerate(ufuns):
#             p.add(
#                 CABNegotiator(name=f"RCS{i}", id=f"RCS{i}"),
#                 preferences=u,
#             )
#         front_utils, front_outcomes = p.pareto_frontier()
#         no_valid_outcomes = all(
#             u1 <= r1 + eps or u2 <= r2 + eps for u1, u2 in front_utils
#         )
#         p.run()
#         assert len(p.history) > 0, f"{p.state}"
#         assert (
#             p.agreement is not None or no_valid_outcomes
#         ), f"No agreement in a supposedly complete profile {_history(p)}{_plot(p, True)}"
#         assert (
#             p.agreement in front_outcomes or p.agreement is None
#         ), f"Suboptimal agreement in a supposedly optimal profile {_history(p)}{_plot(p, True)}"
