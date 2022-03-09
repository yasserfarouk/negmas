from __future__ import annotations

import logging
import math
import random
from typing import Any, Optional

import pandas as pd

from negmas import warnings

from ..genius import GeniusNegotiator
from ..helpers import create_loggers, instantiate
from ..helpers.prob import ScipyDistribution
from ..inout import load_genius_domain_from_folder
from ..mechanisms import Mechanism
from ..models.acceptance import UncertainOpponentModel
from ..outcomes import Outcome
from ..preferences import IPUtilityFunction, MappingUtilityFunction, UtilityFunction
from ..sao import (
    AspirationNegotiator,
    LimitedOutcomesAcceptor,
    LimitedOutcomesNegotiator,
    RandomNegotiator,
    SAOMechanism,
    SAOState,
    TopFractionNegotiator,
    ToughNegotiator,
)
from .baseline import DummyElicitor, FullKnowledgeElicitor
from .expectors import BalancedExpector, MaxExpector, MinExpector
from .pandora import FullElicitor, RandomElicitor
from .user import User
from .voi import (
    Answer,
    BaseElicitor,
    EStrategy,
    MeanExpector,
    Query,
    RangeConstraint,
    VOIElicitor,
    VOIFastElicitor,
    VOINoUncertaintyElicitor,
    VOIOptimalElicitor,
    np,
    time,
)

__all__ = ["SAOElicitingMechanism"]


def uniform():
    loc = random.random()
    scale = random.random() * (1.0 - loc)
    return ScipyDistribution(type="uniform", loc=loc, scale=scale)


def current_aspiration(elicitor, outcome: Outcome, negotiation: Mechanism) -> float:
    return elicitor.utility_at(negotiation.relative_time)


def create_negotiator(
    negotiator_type, preferences, can_propose, outcomes, toughness, **kwargs
):
    if negotiator_type == "limited_outcomes":
        if can_propose:
            negotiator = LimitedOutcomesNegotiator(
                acceptable_outcomes=outcomes,
                acceptance_probabilities=list(preferences.mapping.values()),
                **kwargs,
            )
        else:
            negotiator = LimitedOutcomesAcceptor(
                acceptable_outcomes=outcomes,
                acceptance_probabilities=list(preferences.mapping.values()),
                **kwargs,
            )
    elif negotiator_type == "random":
        negotiator = RandomNegotiator(
            can_propose=can_propose,
        )
    elif negotiator_type == "tough":
        negotiator = ToughNegotiator(can_propose=can_propose)
    elif negotiator_type in ("only_best", "best_only", "best"):
        negotiator = TopFractionNegotiator(
            min_utility=None,
            top_fraction=1.0 - toughness,
            best_first=False,
            can_propose=can_propose,
        )
    elif negotiator_type.startswith("aspiration"):
        asp_kind = negotiator_type[len("aspiration") :]
        if asp_kind.startswith("_"):
            asp_kind = asp_kind[1:]
        try:
            asp_kind = float(asp_kind)
        except:
            pass
        if asp_kind == "":
            if toughness < 0.5:
                toughness *= 2
                toughness = 9.0 * toughness + 1.0
            elif toughness == 0.5:
                toughness = 1.0
            else:
                toughness = 2 * (toughness - 0.5)
                toughness = 1 - 0.9 * toughness
            asp_kind = toughness
        negotiator = AspirationNegotiator(
            aspiration_type=asp_kind,
            can_propose=can_propose,
            **kwargs,
        )
    elif negotiator_type.startswith("genius"):
        class_name = negotiator_type[len("genius") :]
        if class_name.startswith("_"):
            class_name = class_name[1:]
        if class_name == "auto" or len(class_name) < 1:
            negotiator = GeniusNegotiator.random_negotiator(can_propose=can_propose)
        else:
            negotiator = GeniusNegotiator(
                java_class_name=class_name,
                can_propose=can_propose,
            )
        negotiator.preferences = preferences
    else:
        raise ValueError(f"Unknown opponents type {negotiator_type}")
    return negotiator


def _beg(x):
    if isinstance(x, float):
        return x
    else:
        return x.loc


def _scale(x):
    if isinstance(x, float):
        return 0.0
    else:
        return x.scale


def _end(x):
    if isinstance(x, float):
        return x
    else:
        return x.loc + x.scale


class SAOElicitingMechanism(SAOMechanism):
    def __init__(
        self,
        priors,
        true_utilities,
        elicitor_reserved_value,
        cost,
        opp_utility,
        opponent,
        n_steps,
        time_limit,
        base_agent,
        opponent_model,
        elicitation_strategy="pingpong",
        toughness=0.95,
        elicitor_type="balanced",
        history_file_name: str = None,
        screen_log: bool = False,
        dynamic_queries=True,
        each_outcome_once=False,
        rational_answer_probs=True,
        update_related_queries=True,
        resolution=0.1,
        cost_assuming_titration=False,
        name: str | None = None,
    ):
        self.elicitation_state = {}
        initial_priors = priors
        self.xw_real = priors

        outcomes = list(initial_priors.distributions.keys())

        self.U = true_utilities

        super().__init__(
            issues=None,
            outcomes=outcomes,
            n_steps=n_steps,
            time_limit=time_limit,
            max_n_agents=2,
            dynamic_entry=False,
            name=name,
            extra_callbacks=True,
        )
        if elicitor_reserved_value is None:
            elicitor_reserved_value = 0.0
        self.logger = create_loggers(
            file_name=history_file_name,
            screen_level=logging.DEBUG if screen_log else logging.ERROR,
        )
        user = User(
            preferences=MappingUtilityFunction(
                dict(zip(self.outcomes, self.U)), reserved_value=elicitor_reserved_value
            ),
            cost=cost,
            nmi=self.nmi,
        )
        if resolution is None:
            resolution = max(elicitor_reserved_value / 4, 0.025)
        if "voi" in elicitor_type and "optimal" in elicitor_type:
            strategy = None
        else:
            strategy = EStrategy(strategy=elicitation_strategy, resolution=resolution)
            strategy.on_enter(nmi=self.nmi, preferences=initial_priors)

        def create_elicitor(type_, strategy=strategy, opponent_model=opponent_model):
            base_negotiator = create_negotiator(
                negotiator_type=base_agent,
                preferences=None,
                can_propose=True,
                outcomes=outcomes,
                toughness=toughness,
            )
            if type_ == "full":
                return FullElicitor(
                    strategy=strategy, user=user, base_negotiator=base_negotiator
                )

            if type_ == "dummy":
                return DummyElicitor(
                    strategy=strategy, user=user, base_negotiator=base_negotiator
                )

            if type_ == "full_knowledge":
                return FullKnowledgeElicitor(
                    strategy=strategy, user=user, base_negotiator=base_negotiator
                )

            if type_ == "random_deep":
                return RandomElicitor(
                    strategy=strategy,
                    deep_elicitation=True,
                    user=user,
                    base_negotiator=base_negotiator,
                )

            if type_ in ("random_shallow", "random"):
                return RandomElicitor(
                    strategy=strategy,
                    deep_elicitation=False,
                    user=user,
                    base_negotiator=base_negotiator,
                )
            if type_ in (
                "pessimistic",
                "optimistic",
                "balanced",
                "pandora",
                "fast",
                "mean",
            ):
                type_ = type_.title() + "Elicitor"
                return instantiate(
                    f"negmas.elicitation.{type_}",
                    strategy=strategy,
                    user=user,
                    base_negotiator=base_negotiator,
                    opponent_model_factory=lambda x: opponent_model,
                    single_elicitation_per_round=False,
                    assume_uniform=True,
                    user_model_in_index=True,
                    precalculated_index=False,
                )
            if "voi" in type_:
                expector_factory = MeanExpector
                if "balanced" in type_:
                    expector_factory = BalancedExpector
                elif "optimistic" in type_ or "max" in type_:
                    expector_factory = MaxExpector
                elif "pessimistic" in type_ or "min" in type_:
                    expector_factory = MinExpector

                if "fast" in type_:
                    factory = VOIFastElicitor
                elif "optimal" in type_:
                    prune = "prune" in type_ or "fast" in type_
                    if "no" in type_:
                        prune = not prune
                    return VOIOptimalElicitor(
                        user=user,
                        resolution=resolution,
                        opponent_model_factory=lambda x: opponent_model,
                        single_elicitation_per_round=False,
                        base_negotiator=base_negotiator,
                        each_outcome_once=each_outcome_once,
                        expector_factory=expector_factory,
                        update_related_queries=update_related_queries,
                        prune=prune,
                    )
                elif "no_uncertainty" in type_ or "full_knowledge" in type_:
                    factory = VOINoUncertaintyElicitor
                else:
                    factory = VOIElicitor

                if not dynamic_queries and "optimal" not in type_:
                    queries = []
                    for outcome in self.outcomes:
                        u = initial_priors(outcome)
                        scale = _scale(u)
                        if scale < resolution:
                            continue
                        bb, ee = _beg(u), _end(u)
                        n_q = int((ee - bb) / resolution)
                        limits = np.linspace(bb, ee, n_q, endpoint=False)[1:]
                        for i, limit in enumerate(limits):
                            if cost_assuming_titration:
                                qcost = cost * min(i, len(limits) - i - 1)
                            else:
                                qcost = cost
                            answers = [
                                Answer(
                                    outcomes=[outcome],
                                    constraint=RangeConstraint(rng=(0.0, limit)),
                                    name="yes",
                                ),
                                Answer(
                                    outcomes=[outcome],
                                    constraint=RangeConstraint(rng=(limit, 1.0)),
                                    name="no",
                                ),
                            ]
                            probs = (
                                [limit, 1.0 - limit]
                                if rational_answer_probs
                                else [0.5, 0.5]
                            )
                            query = Query(
                                answers=answers,
                                cost=qcost,
                                probs=probs,
                                name=f"{outcome}<{limit}",
                            )
                            queries.append((outcome, query, qcost))
                else:
                    queries = None
                return factory(
                    strategy=strategy if dynamic_queries else None,
                    user=user,
                    opponent_model_factory=lambda x: opponent_model,
                    single_elicitation_per_round=False,
                    dynamic_query_set=dynamic_queries,
                    queries=queries,
                    base_negotiator=base_negotiator,
                    each_outcome_once=each_outcome_once,
                    expector_factory=expector_factory,
                    update_related_queries=update_related_queries,
                )

        elicitor = create_elicitor(elicitor_type)

        if isinstance(opponent, GeniusNegotiator):
            if n_steps is not None and time_limit is not None:
                self.nmi.n_steps = None

        self.add(opponent, preferences=opp_utility)
        self.add(elicitor, preferences=initial_priors)
        if len(self.negotiators) != 2:
            raise ValueError(
                f"I could not add the two negotiators {elicitor.__class__.__name__}, {opponent.__class__.__name__}"
            )
        self.total_time = 0.0

    @classmethod
    def generate_config(
        cls,
        cost,
        n_outcomes: int = None,
        rand_preferencess=True,
        conflict: float = None,
        conflict_delta: float = None,
        winwin=None,  # only if rand_preferencess is false
        genius_folder: str = None,
        n_steps=None,
        time_limit=None,
        own_utility_uncertainty=0.5,
        own_uncertainty_variablility=0.0,
        own_reserved_value=0.0,
        own_base_agent="aspiration",
        opponent_model_uncertainty=0.5,
        opponent_model_adaptive=False,
        opponent_proposes=True,
        opponent_type="best_only",
        opponent_toughness=0.9,
        opponent_reserved_value=0.0,
    ) -> dict[str, Any]:
        config = {}
        if n_steps is None and time_limit is None and "aspiration" in opponent_type:
            raise ValueError(
                "Cannot use aspiration negotiators when no step limit or time limit is given"
            )
        if n_outcomes is None and genius_folder is None:
            raise ValueError(
                "Must specify a folder to run from or a number of outcomes"
            )
        if genius_folder is not None:
            d = load_genius_domain_from_folder(
                genius_folder,
                ignore_reserved=opponent_reserved_value is not None,
                ignore_discount=True,
            ).to_single_issue(numeric=True)
            domain = d.make_session(time_limit=120)

            n_outcomes = domain.nmi.n_outcomes  # type: ignore
            outcomes = domain.outcomes
            elicitor_indx = 0 + int(random.random() <= 0.5)
            opponent_indx = 1 - elicitor_indx
            preferences = d.ufuns[elicitor_indx]
            preferences.reserved_value = own_reserved_value
            opp_utility = d.ufuns[opponent_indx]
            opp_utility.reserved_value = opponent_reserved_value
        else:
            outcomes = [(_,) for _ in range(n_outcomes)]
            if rand_preferencess:
                preferences, opp_utility = UtilityFunction.generate_random_bilateral(
                    outcomes=outcomes
                )
            else:
                preferences, opp_utility = UtilityFunction.generate_bilateral(
                    outcomes=outcomes,
                    conflict_level=opponent_toughness,
                    conflict_delta=conflict_delta,
                    win_win=winwin,
                )
            preferences.reserved_value = own_reserved_value
            domain = SAOMechanism(
                outcomes=outcomes,
                n_steps=n_steps,
                time_limit=time_limit,
                max_n_agents=2,
                dynamic_entry=False,
                cache_outcomes=True,
            )

        true_utilities = list(preferences.mapping.values())
        priors = IPUtilityFunction.from_preferences(
            preferences,
            uncertainty=own_utility_uncertainty,
            variability=own_uncertainty_variablility,
        )

        outcomes = domain.nmi.outcomes

        opponent = create_negotiator(
            negotiator_type=opponent_type,
            can_propose=opponent_proposes,
            preferences=opp_utility,
            outcomes=outcomes,
            toughness=opponent_toughness,
        )
        opponent_model = UncertainOpponentModel(
            outcomes=outcomes,
            uncertainty=opponent_model_uncertainty,
            opponents=opponent,
            adaptive=opponent_model_adaptive,
        )
        config["n_steps"], config["time_limit"] = n_steps, time_limit
        config["priors"] = priors
        config["true_utilities"] = true_utilities
        config["elicitor_reserved_value"] = own_reserved_value
        config["cost"] = cost
        config["opp_utility"] = opp_utility
        config["opponent_model"] = opponent_model
        config["opponent"] = opponent
        config["base_agent"] = own_base_agent
        return config

    def loginfo(self, s: str) -> None:
        """logs nmi-level information

        Args:
            s (str): The string to log

        """
        self.logger.info(s.strip())

    def logdebug(self, s) -> None:
        """logs debug-level information

        Args:
            s (str): The string to log

        """
        self.logger.debug(s.strip())

    def logwarning(self, s) -> None:
        """logs warning-level information

        Args:
            s (str): The string to log

        """
        self.logger.warning(s.strip())

    def logerror(self, s) -> None:
        """logs error-level information

        Args:
            s (str): The string to log

        """
        self.logger.error(s.strip())

    def step(self) -> SAOState:
        start = time.perf_counter()
        _ = super().step()
        self.total_time += time.perf_counter() - start
        self.loginfo(
            f"[{self._step}] {self._current_proposer} offered {self._current_offer}"
        )
        return _

    def on_negotiation_start(self):
        if not super().on_negotiation_start():
            return False
        self.elicitation_state = {}
        self.elicitation_state["steps"] = None
        self.elicitation_state["relative_time"] = None
        self.elicitation_state["broken"] = False
        self.elicitation_state["timedout"] = False
        self.elicitation_state["agreement"] = None
        self.elicitation_state["agreed"] = False
        self.elicitation_state["utils"] = [
            0.0 for a in self.negotiators
        ]  # not even the reserved value
        self.elicitation_state["welfare"] = sum(self.elicitation_state["utils"])
        self.elicitation_state["elicitor"] = self.negotiators[
            1
        ].__class__.__name__.replace("Elicitor", "")
        self.elicitation_state["opponents"] = self.negotiators[
            0
        ].__class__.__name__.replace("Aget", "")
        self.elicitation_state["elicitor_utility"] = self.elicitation_state["utils"][1]
        self.elicitation_state["opponent_utility"] = self.elicitation_state["utils"][0]
        self.elicitation_state["opponent_params"] = str(self.negotiators[0])
        self.elicitation_state["elicitor_params"] = str(self.negotiators[1])
        self.elicitation_state["elicitation_cost"] = None
        self.elicitation_state["total_time"] = None
        self.elicitation_state["pareto"] = None
        self.elicitation_state["pareto_distance"] = None
        self.elicitation_state["_elicitation_time"] = None
        self.elicitation_state["real_asking_time"] = None
        self.elicitation_state["n_queries"] = 0
        return True

    def plot(
        self,
        visible_negotiators=(0, 1),
        consider_costs=False,
    ):
        try:
            import matplotlib.gridspec as gridspec
            import matplotlib.pyplot as plt

            if len(self.negotiators) > 2:
                warnings.warn(
                    "Cannot visualize negotiations with more than 2 negotiators"
                )
            else:
                # has_front = int(len(self.outcomes[0]) <2)
                has_front = 1
                n_agents = len(self.negotiators)
                history = pd.DataFrame(data=[_[1] for _ in self.history])
                history["time"] = [_[0].time for _ in self.history]
                history["relative_time"] = [_[0].relative_time for _ in self.history]
                history["step"] = [_[0].step for _ in self.history]
                history = history.loc[~history.offer.isnull(), :]
                # ufuns = self._get_preferencess(consider_costs=consider_costs)
                ufuns = self._get_preferencess()
                elicitor_dist = self.negotiators[1].ufun
                outcomes = self.outcomes

                utils = [tuple(f(o) for f in ufuns) for o in outcomes]
                agent_names = [
                    a.__class__.__name__ + ":" + a.name for a in self.negotiators
                ]
                history["offer_index"] = [outcomes.index(_) for _ in history.offer]
                frontier, frontier_outcome = self.pareto_frontier(sort_by_welfare=True)
                frontier_outcome_indices = [outcomes.index(_) for _ in frontier_outcome]
                fig_util, fig_outcome = plt.figure(), plt.figure()
                gs_util = gridspec.GridSpec(n_agents, has_front + 1)
                gs_outcome = gridspec.GridSpec(n_agents, has_front + 1)
                axs_util, axs_outcome = [], []

                agent_names_for_legends = [
                    agent_names[a]
                    .split(":")[0]
                    .replace("Negotiator", "")
                    .replace("Elicitor", "")
                    for a in range(n_agents)
                ]
                if agent_names_for_legends[0] == agent_names_for_legends[1]:
                    agent_names_for_legends = [
                        agent_names[a]
                        .split(":")[0]
                        .replace("Negotiator", "")
                        .replace("Elicitor", "")
                        + agent_names[a].split(":")[1]
                        for a in range(n_agents)
                    ]

                for a in range(n_agents):
                    if a == 0:
                        axs_util.append(fig_util.add_subplot(gs_util[a, has_front]))
                        axs_outcome.append(
                            fig_outcome.add_subplot(gs_outcome[a, has_front])
                        )
                    else:
                        axs_util.append(
                            fig_util.add_subplot(
                                gs_util[a, has_front], sharex=axs_util[0]
                            )
                        )
                        axs_outcome.append(
                            fig_outcome.add_subplot(
                                gs_outcome[a, has_front], sharex=axs_outcome[0]
                            )
                        )
                    axs_util[-1].set_ylabel(agent_names_for_legends[a])
                    axs_outcome[-1].set_ylabel(agent_names_for_legends[a])
                for a, (au, ao) in enumerate(zip(axs_util, axs_outcome)):
                    h = history.loc[
                        history.offerer == agent_names[a],
                        ["relative_time", "offer_index", "offer"],
                    ]
                    h["utility"] = h.offer.apply(ufuns[a])
                    ao.plot(h.relative_time, h.offer_index)
                    au.plot(h.relative_time, h.utility)
                    # if a == 1:
                    h["dist"] = h.offer.apply(elicitor_dist)
                    h["beg"] = h.dist.apply(_beg)
                    h["end"] = h.dist.apply(_end)
                    h["p_acceptance"] = h.offer.apply(
                        self.negotiators[1].opponent_model.probability_of_acceptance
                    )
                    au.plot(h.relative_time, h.end, color="r")
                    au.plot(h.relative_time, h.beg, color="r")
                    au.plot(h.relative_time, h.p_acceptance, color="g")
                    au.set_ylim(-0.1, 1.1)

                if has_front:
                    axu = fig_util.add_subplot(gs_util[:, 0])
                    axu.plot([0, 1], [0, 1], "g--")
                    axu.scatter(
                        [_[0] for _ in utils],
                        [_[1] for _ in utils],
                        label="outcomes",
                        color="yellow",
                        marker="s",
                        s=20,
                    )
                    axo = fig_outcome.add_subplot(gs_outcome[:, 0])
                    clrs = ("blue", "green")
                    for a in range(n_agents):
                        h = history.loc[
                            history.offerer == agent_names[a],
                            ["relative_time", "offer_index", "offer"],
                        ]
                        h["u0"] = h.offer.apply(ufuns[0])
                        h["u1"] = h.offer.apply(ufuns[1])

                        axu.scatter(
                            h.u0,
                            h.u1,
                            color=clrs[a],
                            label=f"{agent_names_for_legends[a]}",
                        )
                    steps = sorted(history.step.unique().tolist())
                    aoffers = [[], []]
                    for step in steps[::2]:
                        offrs = []
                        for a in range(n_agents):
                            a_offer = history.loc[
                                (history.offerer == agent_names[a])
                                & ((history.step == step) | (history.step == step + 1)),
                                "offer_index",
                            ]
                            if len(a_offer) > 0:
                                offrs.append(a_offer.values[-1])
                        if len(offrs) == 2:
                            aoffers[0].append(offrs[0])
                            aoffers[1].append(offrs[1])
                    axo.scatter(aoffers[0], aoffers[1], color=clrs[0], label=f"offers")

                    if self.state.agreement is not None:
                        axu.scatter(
                            [ufuns[0](self.state.agreement)],
                            [ufuns[1](self.state.agreement)],
                            color="black",
                            marker="*",
                            s=120,
                            label="SCMLAgreement",
                        )
                        axo.scatter(
                            [outcomes.index(self.state.agreement)],
                            [outcomes.index(self.state.agreement)],
                            color="black",
                            marker="*",
                            s=120,
                            label="SCMLAgreement",
                        )
                    f1, f2 = [_[0] for _ in frontier], [_[1] for _ in frontier]
                    axu.scatter(f1, f2, label="frontier", color="red", marker="x")
                    axo.scatter(
                        frontier_outcome_indices,
                        frontier_outcome_indices,
                        color="red",
                        marker="x",
                        label="frontier",
                    )
                    axu.legend()
                    axo.legend()
                    axo.set_xlabel(agent_names_for_legends[0])
                    axo.set_ylabel(agent_names_for_legends[1])

                    axu.set_xlabel(agent_names_for_legends[0] + " utility")
                    axu.set_ylabel(agent_names_for_legends[1] + " utility")
                    if self.agreement is not None:
                        pareto_distance = 1e9
                        cu = (ufuns[0](self.agreement), ufuns[1](self.agreement))
                        for pu in frontier:
                            dist = math.sqrt(
                                (pu[0] - cu[0]) ** 2 + (pu[1] - cu[1]) ** 2
                            )
                            if dist < pareto_distance:
                                pareto_distance = dist
                        axu.text(
                            0,
                            0.95,
                            f"Pareto-distance={pareto_distance:5.2}",
                            verticalalignment="top",
                            transform=axu.transAxes,
                        )

                fig_util.show()
                fig_outcome.show()
        except:
            pass

    def on_negotiation_end(self):
        super().on_negotiation_end()
        self.elicitation_state = {}
        self.elicitation_state["steps"] = self._step + 1
        self.elicitation_state["relative_time"] = self.relative_time
        self.elicitation_state["broken"] = self.state.broken
        self.elicitation_state["timedout"] = (
            not self.state.broken and self.state.agreement is None
        )
        self.elicitation_state["agreement"] = self.state.agreement
        self.elicitation_state["agreed"] = (
            self.state.agreement is not None and not self.state.broken
        )

        if self.elicitation_state["agreed"]:
            self.elicitation_state["utils"] = [
                a.user_preferences(self.state.agreement)
                if isinstance(a, BaseElicitor)
                else a.ufun(self.state.agreement)
                for a in self.negotiators
            ]
        else:
            self.elicitation_state["utils"] = [
                a.reserved_value if a.reserved_value is not None else 0.0
                for a in self.negotiators
            ]
        self.elicitation_state["welfare"] = sum(self.elicitation_state["utils"])
        self.elicitation_state["elicitor"] = self.negotiators[
            1
        ].__class__.__name__.replace("Elicitor", "")
        self.elicitation_state["opponents"] = self.negotiators[
            0
        ].__class__.__name__.replace("Aget", "")
        self.elicitation_state["elicitor_utility"] = self.elicitation_state["utils"][1]
        self.elicitation_state["opponent_utility"] = self.elicitation_state["utils"][0]
        self.elicitation_state["opponent_params"] = str(self.negotiators[0])
        self.elicitation_state["elicitor_params"] = str(self.negotiators[1])
        self.elicitation_state["elicitation_cost"] = self.negotiators[
            1
        ].elicitation_cost
        self.elicitation_state["total_time"] = self.total_time
        self.elicitation_state["_elicitation_time"] = self.negotiators[
            1
        ].elicitation_time
        self.elicitation_state["asking_time"] = self.negotiators[1].asking_time
        self.elicitation_state["pareto"], pareto_outcomes = self.pareto_frontier()
        if self.elicitation_state["agreed"]:
            if self.state.agreement in pareto_outcomes:
                min_dist = 0.0
            else:
                min_dist = 1e12
                for p in self.elicitation_state["pareto"]:
                    dist = 0.0
                    for par, real in zip(p, self.elicitation_state["utils"]):
                        dist += (par - real) ** 2
                    dist = math.sqrt(dist)
                    if dist < min_dist:
                        min_dist = dist
            self.elicitation_state["pareto_distance"] = (
                min_dist if min_dist < 1e12 else None
            )
        else:
            self.elicitation_state["pareto_distance"] = None
        try:
            self.elicitation_state["queries"] = [
                str(_) for _ in self.negotiators[1].user.elicited_queries()
            ]
        except:
            self.elicitation_state["queries"] = None
        try:
            self.elicitation_state["n_queries"] = len(
                self.negotiators[1].user.elicited_queries()
            )
        except:
            self.elicitation_state["n_queries"] = None
        if hasattr(self.negotiators[1], "total_voi"):
            self.elicitation_state["total_voi"] = self.negotiators[1].total_voi
        else:
            self.elicitation_state["total_voi"] = None
