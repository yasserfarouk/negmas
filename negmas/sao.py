"""
Implements Stacked Alternating Offers (SAO) mechanism and basic negotiators.
"""
import itertools
import math
import random
import time
import warnings
from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Any, Callable, Type
from typing import Sequence, Optional, List, Tuple, Iterable, Union

import numpy as np
import pandas as pd

from negmas.common import *
from negmas.common import _ShadowAgentMechanismInterface
from negmas.events import Notification, Event
from negmas.java import (
    JavaCallerMixin,
    to_java,
    from_java,
    java_link,
    to_dict,
    JNegmasGateway,
    python_identifier,
)
from negmas.mechanisms import MechanismRoundResult, Mechanism
from negmas.negotiators import Negotiator, AspirationMixin, Controller
from negmas.outcomes import (
    Outcome,
    outcome_is_valid,
    ResponseType,
    outcome_as_dict,
    outcome_is_complete,
    Issue,
)
from negmas.utilities import (
    MappingUtilityFunction,
    UtilityFunction,
    UtilityValue,
    JavaUtilityFunction,
    utility_range,
    outcome_with_utility,
)

__all__ = [
    "SAOState",
    "SAOMechanism",
    "SAOProtocol",
    "SAONegotiator",
    "RandomNegotiator",
    "LimitedOutcomesNegotiator",
    "LimitedOutcomesAcceptor",
    "AspirationNegotiator",
    "ToughNegotiator",
    "OnlyBestNegotiator",
    "NaiveTitForTatNegotiator",
    "SimpleTitForTatNegotiator",  # @todo remove this in future versions
    "NiceNegotiator",
    "SAOController",
    "JavaSAONegotiator",
    "PassThroughSAONegotiator",
    "SAOSyncController",
]


@dataclass
class SAOResponse:
    """A response to an offer given by an agent in the alternating offers protocol"""

    response: ResponseType = ResponseType.NO_RESPONSE
    outcome: Optional["Outcome"] = None


@dataclass
class SAOState(MechanismState):
    current_offer: Optional["Outcome"] = None
    current_proposer: Optional[str] = None
    n_acceptances: int = 0
    new_offers: List[Tuple[str, "Outcome"]] = field(default_factory=list)


@dataclass
class SAOAMI(AgentMechanismInterface):
    end_on_no_response: bool = True
    publish_proposer: bool = True
    publish_n_acceptances: bool = False


class SAOMechanism(Mechanism):
    """

    Remarks:

        Events:
            - negotiator_exception: Data=(negotiator, exception) raised whenever a negotiator raises an exception if
              ignore_negotiator_exceptions is set to True.
    """

    def __init__(
        self,
        issues=None,
        outcomes=None,
        n_steps=None,
        time_limit=None,
        step_time_limit=None,
        max_n_agents=None,
        dynamic_entry=True,
        keep_issue_names=False,
        cache_outcomes=True,
        max_n_outcomes: int = 1000000,
        annotation: Optional[Dict[str, Any]] = None,
        end_on_no_response=False,
        publish_proposer=True,
        publish_n_acceptances=False,
        enable_callbacks=False,
        avoid_ultimatum=False,
        check_offers=True,
        ignore_negotiator_exceptions=False,
        offering_is_accepting=True,
        name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            issues=issues,
            outcomes=outcomes,
            n_steps=n_steps,
            time_limit=time_limit if time_limit is not None else float("inf"),
            step_time_limit=step_time_limit
            if step_time_limit is not None
            else float("inf"),
            max_n_agents=max_n_agents,
            dynamic_entry=dynamic_entry,
            keep_issue_names=keep_issue_names,
            cache_outcomes=cache_outcomes,
            max_n_outcomes=max_n_outcomes,
            annotation=annotation,
            state_factory=SAOState,
            enable_callbacks=enable_callbacks,
            name=name,
            **kwargs,
        )
        self.ignore_negotiator_exceptions = ignore_negotiator_exceptions
        self._current_offer = None
        self._current_proposer = None
        self._last_checked_negotiator = -1
        self._n_accepting = 0
        self._avoid_ultimatum = n_steps is not None and avoid_ultimatum
        self.end_negotiation_on_refusal_to_propose = end_on_no_response
        self.publish_proposer = publish_proposer
        self.publish_n_acceptances = publish_n_acceptances
        self.check_offers = check_offers
        self._no_responses = 0
        self._new_offers = []
        self._offering_is_accepting = offering_is_accepting

    def join(
        self,
        ami: AgentMechanismInterface,
        state: MechanismState,
        *,
        ufun: Optional["UtilityFunction"] = None,
        role: str = "agent",
    ) -> bool:
        if not super().join(ami, state, ufun=ufun, role=role):
            return False
        if not self.ami.dynamic_entry and not any(
            [a.capabilities.get("propose", False) for a in self.negotiators]
        ):
            self._current_proposer = None
            self._last_checked_negotiator = -1
            self._current_offer = None
            self._n_accepting = 0
            return False
        return True

    def extra_state(self):
        return SAOState(
            current_offer=self._current_offer,
            new_offers=self._new_offers,
            current_proposer=self._current_proposer.id
            if self._current_proposer and self.publish_proposer
            else None,
            n_acceptances=self._n_accepting if self.publish_n_acceptances else 0,
        )

    def plot(
        self,
        visible_negotiators: Union[Tuple[int, int], Tuple[str, str]] = (0, 1),
        plot_utils=True,
        plot_outcomes=False,
        utility_range: Optional[Tuple[float, float]] = None,
    ):
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        if self.issues is not None and len(self.issues) > 1:
            plot_outcomes = False

        if len(self.negotiators) < 2:
            print("Cannot visualize negotiations with more less than 2 negotiators")
            return
        if len(visible_negotiators) > 2:
            print("Cannot visualize more than 2 agents")
            return
        if isinstance(visible_negotiators[0], str):
            tmp = []
            for _ in visible_negotiators:
                for n in self.negotiators:
                    if n.id == _:
                        tmp.append(n)
        else:
            visible_negotiators = [
                self.negotiators[visible_negotiators[0]],
                self.negotiators[visible_negotiators[1]],
            ]
        indx = dict(zip([_.id for _ in self.negotiators], range(len(self.negotiators))))
        history = []
        for state in self.history:
            for a, o in state.new_offers:
                history.append(
                    {
                        "current_proposer": a,
                        "current_offer": o,
                        "offer_index": self.outcomes.index(o),
                        "relative_time": state.relative_time,
                        "step": state.step,
                        "u0": visible_negotiators[0].utility_function(o),
                        "u1": visible_negotiators[1].utility_function(o),
                    }
                )
        history = pd.DataFrame(data=history)
        has_history = len(history) > 0
        has_front = 1
        n_negotiators = len(self.negotiators)
        n_agents = len(visible_negotiators)
        ufuns = self._get_ufuns()
        outcomes = self.outcomes
        utils = [tuple(f(o) for f in ufuns) for o in outcomes]
        agent_names = [a.name for a in visible_negotiators]
        if has_history:
            history["offer_index"] = [outcomes.index(_) for _ in history.current_offer]
        frontier, frontier_outcome = self.pareto_frontier(sort_by_welfare=True)
        frontier_indices = [
            i
            for i, _ in enumerate(frontier)
            if _[0] is not None
            and _[0] > float("-inf")
            and _[1] is not None
            and _[1] > float("-inf")
        ]
        frontier = [frontier[i] for i in frontier_indices]
        frontier_outcome = [frontier_outcome[i] for i in frontier_indices]
        frontier_outcome_indices = [outcomes.index(_) for _ in frontier_outcome]
        if plot_utils:
            fig_util = plt.figure()
        if plot_outcomes:
            fig_outcome = plt.figure()
        gs_util = gridspec.GridSpec(n_agents, has_front + 1) if plot_utils else None
        gs_outcome = (
            gridspec.GridSpec(n_agents, has_front + 1) if plot_outcomes else None
        )
        axs_util, axs_outcome = [], []

        for a in range(n_agents):
            if a == 0:
                if plot_utils:
                    axs_util.append(fig_util.add_subplot(gs_util[a, has_front]))
                if plot_outcomes:
                    axs_outcome.append(
                        fig_outcome.add_subplot(gs_outcome[a, has_front])
                    )
            else:
                if plot_utils:
                    axs_util.append(
                        fig_util.add_subplot(gs_util[a, has_front], sharex=axs_util[0])
                    )
                if plot_outcomes:
                    axs_outcome.append(
                        fig_outcome.add_subplot(
                            gs_outcome[a, has_front], sharex=axs_outcome[0]
                        )
                    )
            if plot_utils:
                axs_util[-1].set_ylabel(agent_names[a])
            if plot_outcomes:
                axs_outcome[-1].set_ylabel(agent_names[a])
        for a, (au, ao) in enumerate(
            zip(
                itertools.chain(axs_util, itertools.repeat(None)),
                itertools.chain(axs_outcome, itertools.repeat(None)),
            )
        ):
            if au is None and ao is None:
                break
            if has_history:
                h = history.loc[
                    history.current_proposer == visible_negotiators[a].id,
                    ["relative_time", "offer_index", "current_offer"],
                ]
                h["utility"] = h["current_offer"].apply(ufuns[a])
                if plot_outcomes:
                    ao.plot(h.relative_time, h["offer_index"])
                if plot_utils:
                    au.plot(h.relative_time, h.utility)
                    if utility_range is not None:
                        au.set_ylim(*utility_range)

        if has_front:
            if plot_utils:
                axu = fig_util.add_subplot(gs_util[:, 0])
                axu.scatter(
                    [_[0] for _ in utils],
                    [_[1] for _ in utils],
                    label="outcomes",
                    color="gray",
                    marker="s",
                    s=20,
                )
            if plot_outcomes:
                axo = fig_outcome.add_subplot(gs_outcome[:, 0])
            clrs = ("blue", "green")
            if plot_utils:
                f1, f2 = [_[0] for _ in frontier], [_[1] for _ in frontier]
                axu.scatter(f1, f2, label="frontier", color="red", marker="x")
                # axu.legend()
                axu.set_xlabel(agent_names[0] + " utility")
                axu.set_ylabel(agent_names[1] + " utility")
                if self.agreement is not None:
                    pareto_distance = 1e9
                    cu = (ufuns[0](self.agreement), ufuns[1](self.agreement))
                    for pu in frontier:
                        dist = math.sqrt((pu[0] - cu[0]) ** 2 + (pu[1] - cu[1]) ** 2)
                        if dist < pareto_distance:
                            pareto_distance = dist
                    axu.text(
                        0.05,
                        0.05,
                        f"Pareto-distance={pareto_distance:5.2}",
                        verticalalignment="top",
                        transform=axu.transAxes,
                    )

            if plot_outcomes:
                axo.scatter(
                    frontier_outcome_indices,
                    frontier_outcome_indices,
                    color="red",
                    marker="x",
                    label="frontier",
                )
                axo.legend()
                axo.set_xlabel(agent_names[0])
                axo.set_ylabel(agent_names[1])

            if plot_utils and has_history:
                for a in range(n_agents):
                    h = history.loc[
                        history.current_proposer == visible_negotiators[a].id,
                        ["relative_time", "offer_index", "current_offer"],
                    ]
                    h["u0"] = h["current_offer"].apply(ufuns[0])
                    h["u1"] = h["current_offer"].apply(ufuns[1])
                    axu.scatter(h.u0, h.u1, color=clrs[a], label=f"{agent_names[a]}")
                axu.scatter(
                    [frontier[0][0]],
                    [frontier[0][1]],
                    color="magenta",
                    label=f"Max. Welfare",
                )
                axu.annotate(
                    "Max. Welfare",
                    xy=frontier[0],  # theta, radius
                    xytext=(
                        frontier[0][0] + 0.02,
                        frontier[0][1] + 0.02,
                    ),  # fraction, fraction
                    horizontalalignment="left",
                    verticalalignment="bottom",
                )
            if plot_outcomes and has_history:
                steps = sorted(history.step.unique().tolist())
                aoffers = [[], []]
                for step in steps[::2]:
                    offrs = []
                    for a in range(n_agents):
                        a_offer = history.loc[
                            (history.current_proposer == agent_names[a])
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
                if plot_utils:
                    axu.scatter(
                        [ufuns[0](self.state.agreement)],
                        [ufuns[1](self.state.agreement)],
                        color="black",
                        marker="*",
                        s=120,
                        label="SCMLAgreement",
                    )
                if plot_outcomes:
                    axo.scatter(
                        [outcomes.index(self.state.agreement)],
                        [outcomes.index(self.state.agreement)],
                        color="black",
                        marker="*",
                        s=120,
                        label="Agreement",
                    )

        if plot_utils:
            fig_util.show()
        if plot_outcomes:
            fig_outcome.show()

    def round(self) -> MechanismRoundResult:
        """implements a round of the Stacked Alternating Offers Protocol.


        """
        self._new_offers = []
        negotiators: List[SAONegotiator] = self.negotiators
        n_negotiators = len(negotiators)

        def _safe_counter(negotiator, *args, **kwargs):
            try:
                if (
                    negotiator == self._current_proposer
                ) and self._offering_is_accepting:
                    self._n_accepting = 0
                    kwargs["offer"] = None
                    response = negotiator.counter(*args, **kwargs)
                else:
                    response = negotiator.counter(*args, **kwargs)
            except Exception as ex:
                if self.ignore_negotiator_exceptions:
                    self.announce(
                        Event(
                            "negotiator_exception",
                            {"negotiator": negotiator, "exception": ex},
                        )
                    )
                    return SAOResponse(ResponseType.END_NEGOTIATION, None)
                else:
                    raise ex
            if (
                self.check_offers
                and response.outcome is not None
                and (not outcome_is_complete(response.outcome, self.issues))
            ):
                return SAOResponse(response.response, None)
            return response

        proposers, proposer_indices = [], []
        for i, neg in enumerate(negotiators):
            if not neg.capabilities.get("propose", False):
                continue
            proposers.append(neg)
            proposer_indices.append(i)
        n_proposers = len(proposers)
        if n_proposers < 1:
            if not self.dynamic_entry:
                return MechanismRoundResult(
                    broken=True,
                    timedout=False,
                    agreement=None,
                    error=True,
                    error_details="No proposers and no dynamic entry",
                )
            else:
                return MechanismRoundResult(
                    broken=False, timedout=False, agreement=None
                )
        # if this is the first step (or no one has offered yet) which means that there is no _current_offer
        if self._current_offer is None and n_proposers > 1 and self._avoid_ultimatum:
            if not self.dynamic_entry and not self.state.step == 0:
                if self.end_negotiation_on_refusal_to_propose:
                    return MechanismRoundResult(broken=True)
            # assert self._current_proposer is None
            # assert self._last_checked_negotiator == -1

            # if we are trying to avoid an ultimatum, we take an offer from everyone and ignore them but one.
            # this way, the agent cannot know its order. For example, if we have two agents and 3 steps, this will
            # be the situation after each step:
            #
            # Case 1: Assume that it ignored the offer from agent 1
            # Step, Agent 0 calls received  , Agent 1 calls received    , relative time during last call
            # 0   , counter(None)->offer1*  , counter(None) -> offer0   , 0/3
            # 1   , counter(offer2)->offer3 , counter(offer1) -> offer2 , 1/3
            # 2   , counter(offer4)->offer5 , counter(offer3) -> offer4 , 2/3
            # 3   ,                         , counter(offer5)->offer6   , 3/3
            #
            # Case 2: Assume that it ignored the offer from agent 0
            # Step, Agent 0 calls received  , Agent 1 calls received    , relative time during last call
            # 0   , counter(None)->offer1   , counter(None) -> offer0*  , 0/3
            # 1   , counter(offer0)->offer2 , counter(offer2) -> offer3 , 1/3
            # 2   , counter(offer3)->offer4 , counter(offer4) -> offer5 , 2/3
            # 3   , counter(offer5)->offer6 ,                           , 3/3
            #
            # in both cases, the agent cannot know whether its last offer going to be passed to the other agent
            # (the ultimatum scenario) or not.
            responses = []
            for neg in proposers:
                if not neg.capabilities.get("propose", False):
                    continue
                strt = time.perf_counter()
                resp = _safe_counter(neg, state=self.state, offer=None)
                if time.perf_counter() - strt > self.ami.step_time_limit:
                    return MechanismRoundResult(
                        broken=False, timedout=True, agreement=None
                    )
                if self._enable_callbacks:
                    for other in self.negotiators:
                        if other is not neg:
                            other.on_partner_response(
                                state=self.state,
                                agent_id=neg.id,
                                outcome=None,
                                response=resp,
                            )
                if resp.response == ResponseType.END_NEGOTIATION:
                    return MechanismRoundResult(
                        broken=True, timedout=False, agreement=None
                    )
                if resp.response in (ResponseType.NO_RESPONSE, ResponseType.WAIT):
                    return MechanismRoundResult(
                        broken=True,
                        timedout=False,
                        agreement=None,
                        error=True,
                        error_details=f"When avoid_ultimatum is True, negotiators MUST return an offer when asked "
                        f"to propose the first time.\nNegotiator {neg.id} returned NO_RESPONSE/WAIT",
                    )
                responses.append(resp)
            if len(responses) < 1:
                if not self.dynamic_entry:
                    return MechanismRoundResult(
                        broken=True,
                        timedout=False,
                        agreement=None,
                        error=True,
                        error_details="No proposers and no dynamic entry",
                    )
                else:
                    return MechanismRoundResult(
                        broken=False, timedout=False, agreement=None
                    )
            # choose a random negotiator and set it as the current negotiator
            selected = random.randint(0, len(responses) - 1)
            resp = responses[selected]
            neg = proposers[selected]
            _first_proposer = proposer_indices[selected]
            self._n_accepting = 1 if self._offering_is_accepting else 0
            self._current_offer = resp.outcome
            self._current_proposer = neg
            self._last_checked_negotiator = _first_proposer
            self._new_offers.append((neg.id, resp.outcome))
            return MechanismRoundResult(broken=False, timedout=False, agreement=None)

        # this is not the first round. A round will get n_negotiators steps
        ordered_indices = [
            (_ + self._last_checked_negotiator + 1) % n_negotiators
            for _ in range(n_negotiators)
        ]
        for neg_indx in ordered_indices:
            self._last_checked_negotiator = neg_indx
            neg = self.negotiators[neg_indx]
            strt = time.perf_counter()
            resp = _safe_counter(neg, state=self.state, offer=self._current_offer)
            if time.perf_counter() - strt > self.ami.step_time_limit:
                return MechanismRoundResult(broken=False, timedout=True, agreement=None)
            if self._enable_callbacks:
                for other in self.negotiators:
                    if other is not neg:
                        other.on_partner_response(
                            state=self.state,
                            agent_id=neg.id,
                            outcome=self._current_offer,
                            response=resp,
                        )
            if resp.response == ResponseType.NO_RESPONSE:
                continue
            if resp.response == ResponseType.WAIT:
                self._last_checked_negotiator = neg_indx - 1
                if neg_indx < 0:
                    self._last_checked_negotiator = n_negotiators - 1
                return MechanismRoundResult(
                    broken=False, timedout=False, agreement=None, waiting=True
                )
            if resp.response == ResponseType.END_NEGOTIATION:
                return MechanismRoundResult(broken=True, timedout=False, agreement=None)
            if resp.response == ResponseType.ACCEPT_OFFER:
                self._n_accepting += 1
                if self._n_accepting == n_negotiators:
                    return MechanismRoundResult(
                        broken=False, timedout=False, agreement=self._current_offer
                    )
            if resp.response == ResponseType.REJECT_OFFER:
                proposal = resp.outcome
                if proposal is None:
                    if (
                        neg.capabilities.get("propose", False)
                        and self.end_negotiation_on_refusal_to_propose
                    ):
                        return MechanismRoundResult(
                            broken=True, timedout=False, agreement=None
                        )
                    continue
                else:
                    self._current_offer = proposal
                    self._current_proposer = neg
                    self._new_offers.append((neg.id, proposal))
                    self._n_accepting = 1 if self._offering_is_accepting else 0
                    if self._enable_callbacks:
                        for other in self.negotiators:
                            if other is neg:
                                continue
                            other.on_partner_proposal(
                                agent_id=neg.id, offer=proposal, state=self.state
                            )
        return MechanismRoundResult(broken=False, timedout=False, agreement=None)

    def negotiator_offers(self, negotiator_id: str) -> List[Outcome]:
        offers = []
        for state in self._history:
            offers += [o for n, o in state.new_offers if n == negotiator_id]
        return offers


class RandomResponseMixin(object):
    def init_ranodm_response(
        self,
        p_acceptance: float = 0.15,
        p_rejection: float = 0.25,
        p_ending: float = 0.1,
    ) -> None:
        """Constructor

        Args:
            p_acceptance (float): probability of accepting offers
            p_rejection (float): probability of rejecting offers
            p_ending (float): probability of ending negotiation

        Returns:
            None

        Remarks:
            - If the summation of acceptance, rejection and ending probabilities
                is less than 1.0 then with the remaining probability a
                NO_RESPONSE is returned from respond()
        """
        if not hasattr(self, "add_capabilities"):
            raise ValueError(
                f"self.__class__.__name__ is just a mixin for class Negotiator. You must inherit from Negotiator or"
                f" one of its descendents before inheriting from self.__class__.__name__"
            )
        self.add_capabilities({"respond": True})
        self.p_acceptance = p_acceptance
        self.p_rejection = p_rejection
        self.p_ending = p_ending
        self.wheel: List[Tuple[float, ResponseType]] = [(0.0, ResponseType.NO_RESPONSE)]
        if self.p_acceptance > 0.0:
            self.wheel = [
                (self.wheel[-1][0] + self.p_acceptance, ResponseType.ACCEPT_OFFER)
            ]
        if self.p_rejection > 0.0:
            self.wheel.append(
                (self.wheel[-1][0] + self.p_rejection, ResponseType.REJECT_OFFER)
            )
        if self.p_ending > 0.0:
            self.wheel.append(
                (self.wheel[-1][0] + self.p_ending, ResponseType.REJECT_OFFER)
            )
        if self.wheel[-1][0] > 1.0:
            raise ValueError("Probabilities of acceptance+rejection+ending>1")

        self.wheel = self.wheel[1:]

    # noinspection PyUnusedLocal,PyUnusedLocal
    def respond(self, state: MechanismState, offer: "Outcome") -> "ResponseType":
        r = random.random()
        for w in self.wheel:
            if w[0] >= r:
                return w[1]

        return ResponseType.NO_RESPONSE


# noinspection PyAttributeOutsideInit
class RandomProposalMixin(object):
    """The simplest possible agent.

    It just generates random offers and respond randomly to offers.
    """

    def init_random_proposal(self: Negotiator):
        self.add_capabilities(
            {
                "propose": True,
                "propose-with-value": False,
                "max-proposals": None,  # indicates infinity
            }
        )

    def propose(self, state: MechanismState) -> Optional["Outcome"]:
        if (
            hasattr(self, "_offerable_outcomes")
            and self._offerable_outcomes is not None
        ):
            return random.sample(self._offerable_outcomes, 1)[0]
        return self._ami.random_outcomes(1, astype=dict)[0]


class LimitedOutcomesAcceptorMixin(object):
    """An agent the accepts a limited set of outcomes.

    The agent accepts any of the given outcomes with the given probabilities.
    """

    def init_limited_outcomes_acceptor(
        self,
        outcomes: Optional[Union[int, Iterable["Outcome"]]] = None,
        acceptable_outcomes: Optional[Iterable["Outcome"]] = None,
        acceptance_probabilities: Optional[List[float]] = None,
        time_factor: Union[float, List[float]] = None,
        p_ending=0.05,
        p_no_response=0.0,
    ) -> None:
        """Constructor

        Args:
            acceptable_outcomes (Optional[Floats]): the set of acceptable
                outcomes. If None then it is assumed to be all the outcomes of
                the negotiation.
            acceptance_probabilities (Sequence[int]): probability of accepting
                each acceptable outcome. If None then it is assumed to be unity.
            p_no_response (float): probability of refusing to respond to offers
            p_ending (float): probability of ending negotiation

        Returns:
            None

        """
        self.add_capabilities({"respond": True})
        if acceptable_outcomes is not None and outcomes is None:
            raise ValueError(
                "If you are passing acceptable outcomes explicitly then outcomes must also be passed"
            )
        if isinstance(outcomes, int):
            outcomes = [(_,) for _ in range(outcomes)]
        self.outcomes = outcomes
        if acceptable_outcomes is not None:
            acceptable_outcomes = list(acceptable_outcomes)
        self.acceptable_outcomes = acceptable_outcomes
        if acceptable_outcomes is None:
            self.acceptable_outcomes = outcomes
            if acceptance_probabilities is None:
                acceptance_probabilities = [0.5] * len(outcomes)
        elif acceptance_probabilities is None:
            acceptance_probabilities = [1.0] * len(acceptable_outcomes)
            if outcomes is None:
                self.outcomes = [(_,) for _ in range(len(acceptable_outcomes))]
        if self.outcomes is None:
            raise ValueError(
                "Could not calculate all the outcomes. It is needed to assign a utility function"
            )
        self.acceptance_probabilities = acceptance_probabilities
        u = [0.0] * len(self.outcomes)
        for p, o in zip(self.acceptance_probabilities, self.acceptable_outcomes):
            u[self.outcomes.index(o)] = p
        self._utility_function = MappingUtilityFunction(dict(zip(self.outcomes, u)))
        self.p_no_response = p_no_response
        self.p_ending = p_ending + p_no_response
        self.time_factor = time_factor

    def respond(self, state: MechanismState, offer: "Outcome") -> "ResponseType":
        """Respond to an offer.

        Args:
            offer (Outcome): offer being tested

        Returns:
            ResponseType: The response to the offer

        """
        ami = self._ami
        r = random.random()
        if r < self.p_no_response:
            return ResponseType.NO_RESPONSE

        if r < self.p_ending:
            return ResponseType.END_NEGOTIATION

        # if self.acceptable_outcomes is None:
        #     if (
        #         outcome_is_valid(offer, ami.issues)
        #         and random.random() < self.acceptance_probabilities * pow(self.time_factor, state.step)
        #     ):
        #         return ResponseType.ACCEPT_OFFER
        #
        #     else:
        #         return ResponseType.REJECT_OFFER

        try:
            indx = self.acceptable_outcomes.index(offer)
        except ValueError:
            return ResponseType.REJECT_OFFER
        prob = self.acceptance_probabilities[indx]
        if indx < 0 or not outcome_is_valid(offer, ami.issues):
            return ResponseType.REJECT_OFFER

        if random.random() < prob:
            return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER


class LimitedOutcomesProposerMixin(object):
    """An agent the accepts a limited set of outcomes.

    The agent proposes randomly from the given set of outcomes.

    Args:
        proposable_outcomes (Optional[Outcomes]): the set of prooposable
            outcomes. If None then it is assumed to be all the outcomes of
            the negotiation


    """

    def init_limited_outcomes_proposer(
        self: Negotiator, proposable_outcomes: Optional[List["Outcome"]] = None
    ) -> None:
        self.add_capabilities(
            {
                "propose": True,
                "propose-with-value": False,
                "max-proposals": None,  # indicates infinity
            }
        )
        self._offerable_outcomes = proposable_outcomes
        if proposable_outcomes is not None:
            self._offerable_outcomes = list(proposable_outcomes)

    def propose(self, state: MechanismState) -> Optional["Outcome"]:
        if self._offerable_outcomes is None:
            return self._ami.random_outcomes(1)[0]
        else:
            return random.sample(self._offerable_outcomes, 1)[0]


class LimitedOutcomesMixin(LimitedOutcomesAcceptorMixin, LimitedOutcomesProposerMixin):
    """An agent the accepts a limited set of outcomes.

    The agent accepts any of the given outcomes with the given probabilities.
    """

    def init_limited_outcomes(
        self,
        outcomes: Optional[Union[int, Iterable["Outcome"]]] = None,
        acceptable_outcomes: Optional[Iterable["Outcome"]] = None,
        acceptance_probabilities: Optional[Union[float, List[float]]] = None,
        proposable_outcomes: Optional[Iterable["Outcome"]] = None,
        p_ending=0.0,
        p_no_response=0.0,
    ) -> None:
        """Constructor

        Args:
            acceptable_outcomes (Optional[Outcomes]): the set of acceptable
                outcomes. If None then it is assumed to be all the outcomes of
                the negotiation.
            acceptance_probabilities (Sequence[Float]): probability of accepting
                each acceptable outcome. If None then it is assumed to be unity.
            proposable_outcomes (Optional[Outcomes]): the set of outcomes from which the agent is allowed
                to propose. If None, then it is the same as acceptable outcomes with nonzero probability
            p_no_response (float): probability of refusing to respond to offers
            p_ending (float): probability of ending negotiation

        Returns:
            None

        """
        self.init_limited_outcomes_acceptor(
            outcomes=outcomes,
            acceptable_outcomes=acceptable_outcomes,
            acceptance_probabilities=acceptance_probabilities,
            p_ending=p_ending,
            p_no_response=p_no_response,
        )
        if proposable_outcomes is None and self.acceptable_outcomes is not None:
            if not isinstance(self.acceptance_probabilities, float):
                proposable_outcomes = [
                    _
                    for _, p in zip(
                        self.acceptable_outcomes, self.acceptance_probabilities
                    )
                    if p > 1e-9
                ]
        self.init_limited_outcomes_proposer(proposable_outcomes=proposable_outcomes)


class SAONegotiator(Negotiator):
    def __init__(
        self,
        assume_normalized=True,
        ufun: Optional[UtilityFunction] = None,
        name: Optional[str] = None,
        rational_proposal=True,
        parent: Controller = None,
        owner: "Agent" = None,
    ):
        super().__init__(name=name, ufun=ufun, parent=parent, owner=owner)
        self.assume_normalized = assume_normalized
        self.__end_negotiation = False
        self.my_last_proposal: Optional["Outcome"] = None
        self.my_last_proposal_utility: float = None
        self.rational_proposal = rational_proposal
        self.add_capabilities({"respond": True, "propose": True, "max-proposals": 1})

    def on_notification(self, notification: Notification, notifier: str):
        if notification.type == "end_negotiation":
            self.__end_negotiation = True

    def propose_(self, state: MechanismState) -> Optional["Outcome"]:
        if not self._capabilities["propose"] or self.__end_negotiation:
            return None
        if self._ufun_modified:
            self.on_ufun_changed()
        proposal = self.propose(state=state)

        # never return a proposal that is less than the reserved value
        if self.rational_proposal:
            utility = None
            if proposal is not None and self._utility_function is not None:
                utility = self._utility_function(proposal)
                if utility is not None and utility < self.reserved_value:
                    return None

            if utility is not None:
                self.my_last_proposal = proposal
                self.my_last_proposal_utility = utility

        return proposal

    def respond_(self, state: MechanismState, offer: "Outcome") -> "ResponseType":
        """Respond to an offer.

        Args:
            state: `MechanismState` giving current state of the negotiation.
            offer (Outcome): offer being tested

        Returns:
            ResponseType: The response to the offer

        Remarks:
            - The default implementation never ends the negotiation except if an earler end_negotiation notification is
              sent to the negotiator
            - The default implementation asks the negotiator to `propose`() and accepts the `offer` if its utility was
              at least as good as the offer that it would have proposed (and above the reserved value).

        """
        if self.__end_negotiation:
            return ResponseType.END_NEGOTIATION
        if self._ufun_modified:
            self.on_ufun_changed()
        return self.respond(state=state, offer=offer)

    def counter(
        self, state: MechanismState, offer: Optional["Outcome"]
    ) -> "SAOResponse":
        """
        Called to counter an offer

        Args:
            state: `MechanismState` giving current state of the negotiation.
            offer: The offer to be countered. None means no offer and the agent is requested to propose an offer

        Returns:
            Tuple[ResponseType, Outcome]: The response to the given offer with a counter offer if the response is REJECT

        """
        if self.__end_negotiation:
            return SAOResponse(ResponseType.END_NEGOTIATION, None)
        if offer is None:
            return SAOResponse(ResponseType.REJECT_OFFER, self.propose_(state=state))
        response = self.respond_(state=state, offer=offer)
        if response != ResponseType.REJECT_OFFER:
            return SAOResponse(response, None)
        return SAOResponse(response, self.propose_(state=state))

    def respond(self, state: MechanismState, offer: "Outcome") -> "ResponseType":
        """Respond to an offer.

        Args:
            state: `MechanismState` giving current state of the negotiation.
            offer (Outcome): offer being tested

        Returns:
            ResponseType: The response to the offer

        Remarks:
            - The default implementation never ends the negotiation
            - The default implementation asks the negotiator to `propose`() and accepts the `offer` if its utility was
              at least as good as the offer that it would have proposed (and above the reserved value).

        """
        if self._utility_function is None:
            return ResponseType.REJECT_OFFER
        if self._utility_function(offer) < self.reserved_value:
            return ResponseType.REJECT_OFFER
        utility = None
        if self.my_last_proposal_utility is not None:
            utility = self.my_last_proposal_utility
        if utility is None:
            myoffer = self.propose_(state=state)
            if myoffer is None:
                return ResponseType.NO_RESPONSE
            utility = self._utility_function(myoffer)
        if utility is not None and self._utility_function(offer) >= utility:
            return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER

    @property
    def eu(self) -> Callable[["Outcome"], Optional["UtilityValue"]]:
        """
        The utility function in the given negotiation taking opponent model into account.

        Remarks:
            - If no utility_function is internally stored, `eu` still returns a valid callable that returns None for
              everything.
        """
        if hasattr(self, "opponent_model"):
            return lambda x: self._utility_function(
                x
            ) * self.opponent_model.probability_of_acceptance(x)
        else:
            return self._utility_function

    # CALLBACK
    def on_partner_proposal(
        self, state: MechanismState, agent_id: str, offer: "Outcome"
    ) -> None:
        """
        A callback called by the mechanism when a partner proposes something

        Args:
            state: `MechanismState` giving the state of the negotiation when the offer was porposed.
            agent_id: The ID of the agent who proposed
            offer: The proposal.

        Returns:
            None

        """

    def on_partner_refused_to_propose(
        self, state: MechanismState, agent_id: str
    ) -> None:
        """
        A callback called by the mechanism when a partner refuses to propose

        Args:
            state: `MechanismState` giving the state of the negotiation when the partner refused to offer.
            agent_id: The ID of the agent who refused to propose

        Returns:
            None

        """

    def on_partner_response(
        self,
        state: MechanismState,
        agent_id: str,
        outcome: "Outcome",
        response: "SAOResponse",
    ) -> None:
        """
        A callback called by the mechanism when a partner responds to some offer

        Args:
            state: `MechanismState` giving the state of the negotiation when the partner responded.
            agent_id: The ID of the agent who responded
            outcome: The proposal being responded to.
            response: The response

        Returns:
            None

        """

    @abstractmethod
    def propose(self, state: MechanismState) -> Optional["Outcome"]:
        """Propose a set of offers

        Args:
            state: `MechanismState` giving current state of the negotiation.

        Returns:
            The outcome being proposed or None to refuse to propose

        Remarks:
            - This function guarantees that no agents can propose something with a utility value

        """

    class Java:
        implements = ["jnegmas.sao.SAONegotiator"]


class RandomNegotiator(Negotiator, RandomResponseMixin, RandomProposalMixin):
    """A negotiation agent that responds randomly in a single negotiation."""

    def __init__(
        self,
        outcomes: Union[int, List["Outcome"]],
        name: str = None,
        parent: Controller = None,
        reserved_value: float = float("-inf"),
        p_acceptance=0.15,
        p_rejection=0.25,
        p_ending=0.05,
        can_propose=True,
        ufun=None,
    ) -> None:
        super().__init__(name=name, parent=parent)
        # noinspection PyCallByClass
        self.init_ranodm_response(
            p_acceptance=p_acceptance, p_rejection=p_rejection, p_ending=p_ending
        )
        self.init_random_proposal()
        if isinstance(outcomes, int):
            outcomes = [(_,) for _ in range(outcomes)]
        self.capabilities["propose"] = can_propose
        self._utility_function = MappingUtilityFunction(
            dict(zip(outcomes, np.random.rand(len(outcomes)))),
            reserved_value=reserved_value,
        )


# noinspection PyCallByClass
class LimitedOutcomesNegotiator(LimitedOutcomesMixin, SAONegotiator):
    """A negotiation agent that uses a fixed set of outcomes in a single
    negotiation."""

    def __init__(
        self,
        name: str = None,
        parent: Controller = None,
        outcomes: Optional[Union[int, Iterable["Outcome"]]] = None,
        acceptable_outcomes: Optional[List["Outcome"]] = None,
        acceptance_probabilities: Optional[Union[float, List[float]]] = None,
        p_ending=0.0,
        p_no_response=0.0,
        ufun=None,
    ) -> None:
        super().__init__(name=name, parent=parent)
        self.init_limited_outcomes(
            p_ending=p_ending,
            p_no_response=p_no_response,
            acceptable_outcomes=acceptable_outcomes,
            acceptance_probabilities=acceptance_probabilities,
            outcomes=outcomes,
        )


# noinspection PyCallByClass
class LimitedOutcomesAcceptor(SAONegotiator, LimitedOutcomesAcceptorMixin):
    """A negotiation agent that uses a fixed set of outcomes in a single
    negotiation."""

    def respond(self, state: MechanismState, offer: "Outcome") -> "ResponseType":
        return LimitedOutcomesAcceptorMixin.respond(self, state=state, offer=offer)

    def __init__(
        self,
        name: str = None,
        parent: Controller = None,
        outcomes: Optional[Union[int, Iterable["Outcome"]]] = None,
        acceptable_outcomes: Optional[List["Outcome"]] = None,
        acceptance_probabilities: Optional[List[float]] = None,
        p_ending=0.0,
        p_no_response=0.0,
        ufun=None,
    ) -> None:
        SAONegotiator.__init__(self, name=name, parent=parent)
        self.init_limited_outcomes_acceptor(
            p_ending=p_ending,
            p_no_response=p_no_response,
            acceptable_outcomes=acceptable_outcomes,
            acceptance_probabilities=acceptance_probabilities,
            outcomes=outcomes,
        )
        self.add_capabilities({"propose": False})

    def propose(self, state: MechanismState) -> Optional["Outcome"]:
        return None


class AspirationNegotiator(SAONegotiator, AspirationMixin):
    """
    Represents a time-based negotiation strategy that is independent of the offers received during the negotiation.

    Args:
        name: The agent name
        ufun:  The utility function to attache with the agent
        parent: The parent which should be an ``SAOController``
        max_aspiration: The aspiration level to use for the first offer (or first acceptance decision).
        aspiration_type: The polynomial aspiration curve type. Here you can pass the exponent as a real value or
                         pass a string giving one of the predefined types: linear, conceder, boulware.
        dynamic_ufun: If True, the utility function will be assumed to be changing over time. This is depricated.
        randomize_offer: If True, the agent will propose outcomes with utility >= the current aspiration level not
                         outcomes just above it.
        can_propose: If True, the agent is allowed to propose
        assume_normalized: If True, the ufun will just be assumed to have the range [0, 1] inclusive
        ranking: If True, the aspiration level will not be based on the utility value but the ranking of the outcome
                 within the presorted list. It is only effective when presort is set to True
        ufun_max: The maximum utility value (used only when `presort` is True)
        ufun_min: The minimum utility value (used only when `presort` is True)
        presort: If True, the negotiator will catch a list of outcomes, presort them and only use them for offers
                 and responses. This is much faster then other option for general continuous utility functions
                 but with the obvious problem of only exploring a discrete subset of the issue space (Decided by
                 the `discrete_outcomes` property of the `AgentMechanismInterface` . If the number of outcomes is
                 very large (i.e. > 10000) and discrete, presort will be forced to be True. You can check if
                 presorting is active in realtime by checking the "presorted" attribute.
        tolerance: A tolerance used for sampling of outcomes when `presort` is set to False

    """

    def __init__(
        self,
        name=None,
        ufun=None,
        parent: Controller = None,
        max_aspiration=1.0,
        aspiration_type="boulware",
        dynamic_ufun=True,
        randomize_offer=False,
        can_propose=True,
        assume_normalized=False,
        ranking=False,
        ufun_max=None,
        ufun_min=None,
        presort: bool = True,
        tolerance: float = 0.01,
    ):
        self.ordered_outcomes = []
        self.ufun_max = ufun_max
        self.ufun_min = ufun_min
        self.ranking = ranking
        self.tolerance = tolerance
        if assume_normalized:
            self.ufun_max, self.ufun_min = 1.0, 0.0
        super().__init__(
            name=name, assume_normalized=assume_normalized, parent=parent, ufun=ufun
        )
        self.aspiration_init(
            max_aspiration=max_aspiration, aspiration_type=aspiration_type
        )
        if not dynamic_ufun:
            warnings.warn(
                "dynamic_ufun is deprecated. All Aspiration negotiators assume a dynamic ufun"
            )
        self.randomize_offer = randomize_offer
        self._max_aspiration = self.max_aspiration
        self.best_outcome, self.worst_outcome = None, None
        self.presort = presort
        self.presorted = False
        self.add_capabilities(
            {
                "respond": True,
                "propose": can_propose,
                "propose-with-value": False,
                "max-proposals": None,  # indicates infinity
            }
        )
        self.__last_offer_util, self.__last_offer = float("inf"), None
        self.n_outcomes_to_force_presort = 10000

    def on_ufun_changed(self):
        super().on_ufun_changed()
        presort = self.presort
        if (
            not presort
            and all(i.is_countable() for i in self._ami.issues)
            and Issue.num_outcomes(self._ami.issues) >= self.n_outcomes_to_force_presort
        ):
            presort = True
        if presort:
            outcomes = self._ami.discrete_outcomes()
            self.ordered_outcomes = sorted(
                [(self._utility_function(outcome), outcome) for outcome in outcomes],
                key=lambda x: float(x[0]) if x[0] is not None else float("-inf"),
                reverse=True,
            )
            if not self.assume_normalized:
                if self.ufun_max is None:
                    self.ufun_max = self.ordered_outcomes[0][0]

                if self.ufun_min is None:
                    # we set the minimum utility to the minimum finite value above both reserved_value
                    for j in range(len(outcomes) - 1, -1, -1):
                        self.ufun_min = self.ordered_outcomes[j][0]
                        if self.ufun_min is not None and self.ufun_min > float("-inf"):
                            break
                    if (
                        self.ufun_min is not None
                        and self.ufun_min < self.reserved_value
                    ):
                        self.ufun_min = self.reserved_value
        else:
            if self.ufun_min is None or self.ufun_max is None:
                mn, mx, self.worst_outcome, self.best_outcome = utility_range(
                    self.ufun, return_outcomes=True, issues=self._ami.issues
                )
                if self.ufun_min is None:
                    self.ufun_min = mn
                if self.ufun_max is None:
                    self.ufun_max = mx

        self.presorted = presort
        self.n_trials = 10

    def respond(self, state: MechanismState, offer: "Outcome") -> "ResponseType":
        if self.ufun_max is None or self.ufun_min is None:
            self.on_ufun_changed()
        if self._utility_function is None:
            return ResponseType.REJECT_OFFER
        u = self._utility_function(offer)
        if u is None:
            return ResponseType.REJECT_OFFER
        asp = (
            self.aspiration(state.relative_time) * (self.ufun_max - self.ufun_min)
            + self.ufun_min
        )
        if u >= asp and u > self.reserved_value:
            return ResponseType.ACCEPT_OFFER
        if asp < self.reserved_value:
            return ResponseType.END_NEGOTIATION
        return ResponseType.REJECT_OFFER

    def propose(self, state: MechanismState) -> Optional["Outcome"]:
        if self.ufun_max is None or self.ufun_min is None:
            self.on_ufun_changed()
        asp = (
            self.aspiration(state.relative_time) * (self.ufun_max - self.ufun_min)
            + self.ufun_min
        )
        if asp < self.reserved_value:
            return None
        if self.presorted:
            for i, (u, o) in enumerate(self.ordered_outcomes):
                if u is None:
                    continue
                if u < asp:
                    if u < self.reserved_value:
                        return None
                    if i == 0:
                        return self.ordered_outcomes[i][1]
                    if self.randomize_offer:
                        return random.sample(self.ordered_outcomes[:i], 1)[0][1]
                    return self.ordered_outcomes[i - 1][1]
            if self.randomize_offer:
                return random.sample(self.ordered_outcomes, 1)[0][1]
            return self.ordered_outcomes[-1][1]
        else:
            if asp >= 0.99999999999 and self.best_outcome is not None:
                return self.best_outcome
            if self.randomize_offer:
                return outcome_with_utility(
                    ufun=self._utility_function,
                    rng=(asp, float("inf")),
                    issues=self._ami.issues,
                )
            tol = self.tolerance
            for _ in range(self.n_trials):
                rng = self.ufun_max - self.ufun_min
                mx = min(asp + tol * rng, self.__last_offer_util)
                outcome = outcome_with_utility(
                    ufun=self._utility_function, rng=(asp, mx), issues=self._ami.issues
                )
                if outcome is not None:
                    break
                tol = math.sqrt(tol)
            else:
                outcome = (
                    self.best_outcome
                    if self.__last_offer is None
                    else self.__last_offer
                )
            self.__last_offer_util = self.utility_function(outcome)
            self.__last_offer = outcome
            return outcome


class NiceNegotiator(SAONegotiator, RandomProposalMixin):
    def __init__(self, *args, **kwargs):
        SAONegotiator.__init__(self, *args, **kwargs)
        self.init_random_proposal()

    def respond(self, state: MechanismState, offer: "Outcome") -> "ResponseType":
        return ResponseType.ACCEPT_OFFER

    def propose(self, state: MechanismState) -> Optional["Outcome"]:
        return RandomProposalMixin.propose(self=self, state=state)


class ToughNegotiator(SAONegotiator):
    def __init__(
        self,
        name=None,
        parent: Controller = None,
        dynamic_ufun=True,
        can_propose=True,
        ufun=None,
    ):
        super().__init__(name=name, parent=parent)
        self.best_outcome = None
        self._offerable_outcomes = None
        if not dynamic_ufun:
            warnings.warn(
                "dynamic_ufun is deprecated. All Aspiration negotiators assume a dynamic ufun"
            )
        self.add_capabilities(
            {
                "respond": True,
                "propose": can_propose,
                "propose-with-value": False,
                "max-proposals": None,  # indicates infinity
            }
        )

    def on_ufun_changed(self):
        super().on_ufun_changed()
        if self._utility_function is None:
            return
        outcomes = (
            self._ami.outcomes
            if self._offerable_outcomes is None
            else self._offerable_outcomes
        )
        self.best_outcome = max(
            [(self._utility_function(outcome), outcome) for outcome in outcomes]
        )[1]

    def respond(self, state: MechanismState, offer: "Outcome") -> "ResponseType":
        if offer == self.best_outcome:
            return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER

    def propose(self, state: MechanismState) -> Optional["Outcome"]:
        if not self._capabilities["propose"]:
            return None
        return self.best_outcome


class OnlyBestNegotiator(SAONegotiator):
    def __init__(
        self,
        name=None,
        parent: Controller = None,
        dynamic_ufun=True,
        min_utility=0.95,
        top_fraction=0.05,
        best_first=False,
        probabilisic_offering=True,
        can_propose=True,
        ufun=None,
    ):
        self._offerable_outcomes = None
        self.best_outcome = []
        self.ordered_outcomes = []
        self.acceptable_outcomes = []
        self.wheel = np.array([])
        self.offered = set([])
        super().__init__(name=name, parent=parent)
        if not dynamic_ufun:
            warnings.warn(
                "dynamic_ufun is deprecated. All Aspiration negotiators assume a dynamic ufun"
            )
        self.top_fraction = top_fraction
        self.min_utility = min_utility
        self.best_first = best_first
        self.probabilisic_offering = probabilisic_offering
        self.add_capabilities(
            {
                "respond": True,
                "propose": can_propose,
                "propose-with-value": False,
                "max-proposals": None,  # indicates infinity
            }
        )

    def on_ufun_changed(self):
        super().on_ufun_changed()
        outcomes = (
            self._ami.discrete_outcomes()
            if self._offerable_outcomes is None
            else self._offerable_outcomes
        )
        eu_outcome = [(self.eu(outcome), outcome) for outcome in outcomes]
        self.ordered_outcomes = sorted(eu_outcome, key=lambda x: x[0], reverse=True)
        if self.min_utility is None:
            selected, selected_utils = [], []
        else:
            util_limit = self.min_utility * self.ordered_outcomes[0][0]
            selected, selected_utils = [], []
            for u, o in self.ordered_outcomes:
                if u >= util_limit:
                    selected.append(o)
                    selected_utils.append(u)
                else:
                    break
        if self.top_fraction is not None:
            frac_limit = max(
                1, int(round(self.top_fraction * len(self.ordered_outcomes)))
            )
        else:
            frac_limit = len(outcomes)

        if frac_limit >= len(selected) > 0:
            sum = np.asarray(selected_utils).sum()
            if sum > 0.0:
                selected_utils /= sum
                selected_utils = np.cumsum(selected_utils)
            else:
                selected_utils = np.linspace(0.0, 1.0, len(selected_utils))
            self.acceptable_outcomes, self.wheel = selected, selected_utils
            return
        if frac_limit > 0:
            n_sel = len(selected)
            fsel = [_[1] for _ in self.ordered_outcomes[n_sel:frac_limit]]
            futil = [_[0] for _ in self.ordered_outcomes[n_sel:frac_limit]]
            selected_utils = selected_utils + futil
            sum = np.asarray(selected_utils).sum()
            if sum > 0.0:
                selected_utils /= sum
                selected_utils = np.cumsum(selected_utils)
            else:
                selected_utils = np.linspace(0.0, 1.0, len(selected_utils))
            self.acceptable_outcomes, self.wheel = selected + fsel, selected_utils
            return
        self.acceptable_outcomes, self.wheel = [], []
        return

    def respond(self, state: MechanismState, offer: "Outcome") -> "ResponseType":
        if offer in self.acceptable_outcomes:
            return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER

    def propose(self, state: MechanismState) -> Optional["Outcome"]:
        if not self._capabilities["propose"]:
            return None
        if self.best_first:
            for o in self.acceptable_outcomes:
                if o not in self.offered:
                    self.offered.add(o)
                    return o
        if len(self.acceptable_outcomes) > 0:
            if self.probabilisic_offering:
                r = random.random()
                for o, w in zip(self.acceptable_outcomes, self.wheel):
                    if w > r:
                        return o
                return random.sample(self.acceptable_outcomes, 1)[0]
            return random.sample(self.acceptable_outcomes, 1)[0]
        return None


class NaiveTitForTatNegotiator(SAONegotiator):
    """Implements a generalized tit-for-tat strategy"""

    def __init__(
        self,
        name: str = None,
        parent: Controller = None,
        ufun: Optional["UtilityFunction"] = None,
        kindness=0.0,
        randomize_offer=False,
        always_concede=True,
        initial_concession: Union[float, str] = "min",
    ):
        self.received_utilities = []
        self.proposed_utility = None
        self.ordered_outcomes = None
        self.sent_offer_index = None
        self.n_sent = 0
        super().__init__(name=name, ufun=ufun, parent=parent)
        self.kindness = kindness
        self.initial_concession = initial_concession
        self.randomize_offer = randomize_offer
        self.always_concede = always_concede

    def on_ufun_changed(self):
        super().on_ufun_changed()
        outcomes = self._ami.discrete_outcomes()
        self.ordered_outcomes = sorted(
            [(self._utility_function(outcome), outcome) for outcome in outcomes],
            key=lambda x: x[0],
            reverse=True,
        )

    def respond(self, state: MechanismState, offer: "Outcome") -> "ResponseType":
        if self._utility_function is None:
            return ResponseType.REJECT_OFFER
        offered_utility = self._utility_function(offer)
        if len(self.received_utilities) < 2:
            self.received_utilities.append(offered_utility)
        else:
            self.received_utilities[0] = self.received_utilities[1]
            self.received_utilities[-1] = offered_utility
        indx = self._propose(state=state)
        my_utility, my_offer = self.ordered_outcomes[indx]
        if offered_utility >= my_utility:
            return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER

    def _outcome_just_below(self, ulevel: float) -> int:
        for i, (u, o) in enumerate(self.ordered_outcomes):
            if u is None:
                continue
            if u < ulevel:
                if self.randomize_offer:
                    return random.randint(0, i)
                return i
        if self.randomize_offer:
            return random.randint(0, len(self.ordered_outcomes) - 1)
        return -1

    def _propose(self, state: MechanismState) -> int:
        if self.proposed_utility is None:
            return 0
        if len(self.received_utilities) < 2:
            if (
                isinstance(self.initial_concession, str)
                and self.initial_concession == "min"
            ):
                return self._outcome_just_below(ulevel=self.ordered_outcomes[0][0])
            else:
                asp = self.ordered_outcomes[0][0] * (1.0 - self.initial_concession)
            return self._outcome_just_below(ulevel=asp)

        if self.always_concede:
            opponent_concession = max(
                0.0, self.received_utilities[1] - self.received_utilities[0]
            )
        else:
            opponent_concession = (
                self.received_utilities[1] - self.received_utilities[0]
            )
        indx = self._outcome_just_below(
            ulevel=self.proposed_utility
            - opponent_concession
            - self.kindness * max(0.0, opponent_concession)
        )
        return indx

    def propose(self, state: MechanismState) -> Optional[Outcome]:
        indx = self._propose(state)
        self.proposed_utility = self.ordered_outcomes[indx][0]
        return self.ordered_outcomes[indx][1]


def _to_java_response(response: ResponseType) -> int:
    if response == ResponseType.ACCEPT_OFFER:
        return 0
    if response == ResponseType.REJECT_OFFER:
        return 1
    if response == ResponseType.END_NEGOTIATION:
        return 2
    if response == ResponseType.NO_RESPONSE:
        return 3
    raise ValueError(f"Unknown response{response}")


def _from_java_response(response: int) -> ResponseType:
    if response == 0:
        return ResponseType.ACCEPT_OFFER
    if response == 1:
        return ResponseType.REJECT_OFFER
    if response == 2:
        return ResponseType.END_NEGOTIATION
    if response == 3:
        return ResponseType.NO_RESPONSE
    raise ValueError(
        f"Unknown response type {response} returned from the Java underlying negotiator"
    )


class JavaSAONegotiator(SAONegotiator, JavaCallerMixin):
    def __init__(
        self,
        java_object,
        java_class_name: Optional[str],
        auto_load_java: bool = False,
        outcome_type: Type = dict,
    ):
        if java_class_name is not None or java_object is not None:
            self.init_java_bridge(
                java_object=java_object,
                java_class_name=java_class_name,
                auto_load_java=auto_load_java,
                python_shadow_object=None,
            )
            if java_object is None:
                self._java_object.fromMap(to_java(self))
        d = {
            python_identifier(k): v
            for k, v in JNegmasGateway.gateway.entry_point.toMap(
                self._java_object
            ).items()
        }
        ufun = d.get("utility_function", None)
        ufun = JavaUtilityFunction(ufun, None) if ufun is not None else None
        super().__init__(
            name=d.get("name", None),
            assume_normalized=d.get("assume_normalized", False),
            ufun=ufun,
            rational_proposal=d.get("rational_proposal", True),
            parent=d.get("parent", None),
        )
        self._outcome_type = outcome_type
        self.add_capabilities(
            {
                "respond": True,
                "propose": True,
                "propose-with-value": False,
                "max-proposals": None,  # indicates infinity
            }
        )

    def on_partner_proposal(
        self, state: MechanismState, agent_id: str, offer: "Outcome"
    ) -> None:
        self._java_object.onPartnerProposal(to_java(state), agent_id, to_java(offer))

    def on_partner_refused_to_propose(
        self, state: MechanismState, agent_id: str
    ) -> None:
        self._java_object.onPartnerRefusedToPropose(to_java(state), agent_id)

    def on_partner_response(
        self,
        state: MechanismState,
        agent_id: str,
        outcome: "Outcome",
        response: "SAOResponse",
    ) -> None:
        self._java_object.onPartnerResponse(
            to_java(state), agent_id, to_java(outcome), self._to_java_response(response)
        )

    def isin(self, negotiation_id: Optional[str]) -> bool:
        return self._java_object.isIn(negotiation_id)

    def join(
        self,
        ami: AgentMechanismInterface,
        state: MechanismState,
        *,
        ufun: Optional["UtilityFunction"] = None,
        role: str = "agent",
    ) -> bool:
        return self._java_object.join(
            java_link(_ShadowAgentMechanismInterface(ami)), to_java(state), ufun, role
        )

    def on_negotiation_start(self, state: MechanismState) -> None:
        self._java_object.onNegotiationStart(to_java(state))

    def on_round_start(self, state: MechanismState) -> None:
        self._java_object.onRoundStart(to_java(state))

    def on_mechanism_error(self, state: MechanismState) -> None:
        self._java_object.onMechanismError(to_java(state))

    def on_round_end(self, state: MechanismState) -> None:
        self._java_object.onRoundEnd(to_java(state))

    def on_leave(self, state: MechanismState) -> None:
        self._java_object.onLeave(to_java(state))

    def on_negotiation_end(self, state: MechanismState) -> None:
        self._java_object.onNegotiationEnd(to_java(state))

    def on_ufun_changed(self):
        self._java_object.onUfunChanged()

    @classmethod
    def from_dict(
        cls, java_object, *args, parent: Controller = None
    ) -> "JavaSAONegotiator":
        """Creates a Java negotiator from an object returned from the JVM implementing PySAONegotiator"""
        ufun = java_object.getUtilityFunction()
        if ufun is not None:
            ufun = JavaUtilityFunction.from_dict(java_object=ufun)
        return JavaCallerMixin.from_dict(
            java_object,
            name=java_object.getName(),
            assume_normalized=java_object.getAssumeNormalized(),
            rational_proposal=java_object.getRationalProposal(),
            parent=parent,
            ufun=ufun,
        )

    def on_notification(self, notification: Notification, notifier: str):
        super().on_notification(notification=notification, notifier=notifier)
        jnotification = {"type": notification.type, "data": to_java(notification.data)}
        self._java_object.on_notification(jnotification, notifier)

    def respond(self, state: MechanismState, offer: "Outcome"):
        return _from_java_response(
            self._java_object.respond(to_java(state), outcome_as_dict(offer))
        )

    def propose(self, state: MechanismState) -> Optional["Outcome"]:
        outcome = from_java(self._java_object.propose(to_java(state)))
        if outcome is None:
            return None
        if self._outcome_type == dict:
            return outcome
        if self._outcome_type == tuple:
            return tuple(outcome.values())
        return self._outcome_type(outcome)

    # class Java:
    #    implements = ['jnegmas.sao.SAONegotiator']


class _ShadowSAONegotiator:
    """A python shadow to a java negotiator"""

    class Java:
        implements = ["jnegmas.sao.SAONegotiator"]

    def to_java(self):
        return to_dict(self.shadow)

    def __init__(self, negotiator: SAONegotiator):
        self.shadow = negotiator

    def respond(self, state, outcome):
        return _to_java_response(
            self.shadow.respond(from_java(state), from_java(outcome))
        )

    def propose(self, state):
        return to_java(self.shadow.propose(from_java(state)))

    def isIn(self, negotiation_id):
        return to_java(self.shadow.isin(negotiation_id=negotiation_id))

    def join(self, ami, state, ufun, role):
        return to_java(
            self.shadow.join(
                ami=from_java(ami),
                state=from_java(state),
                ufun=JavaUtilityFunction(ufun, None) if ufun is not None else None,
                role=role,
            )
        )

    def onNegotiationStart(self, state):
        return to_java(self.shadow.on_negotiation_start(from_java(state)))

    def onNegotiationEnd(self, state):
        return to_java(self.shadow.on_negotiation_end(from_java(state)))

    def onMechanismError(self, state):
        return to_java(self.shadow.on_mechanism_error(from_java(state)))

    def onRoundStart(self, state):
        return to_java(self.shadow.on_round_start(from_java(state)))

    def onRoundEnd(self, state):
        return to_java(self.shadow.on_round_end(from_java(state)))

    def onLeave(self, state):
        return to_java(self.shadow.on_leave(from_java(state)))

    def onPartnerProposal(self, state, agent_id, offer):
        return to_java(
            self.shadow.on_partner_proposal(
                state=from_java(state), agent_id=agent_id, offer=from_java(offer)
            )
        )

    def onUfunChanged(self):
        return to_java(self.shadow.on_ufun_changed())

    def onPartnerRefusedToPropose(self, state, agent_id):
        return to_java(
            self.shadow.on_partner_refused_to_propose(
                state=from_java(state), agent_id=agent_id
            )
        )

    def onPartnerResponse(self, state, agent_id, offer, response: int, counter_offer):
        return to_java(
            self.shadow.on_partner_response(
                state=from_java(state),
                agent_id=agent_id,
                outcome=from_java(offer),
                response=SAOResponse(
                    response=ResponseType(response), outcome=from_java(counter_offer)
                ),
            )
        )

    def onNotification(self, notification, notifier):
        return to_java(self.shadow.on_notification(from_java(notification), notifier))

    def setUtilityFunction(self, ufun):
        self.shadow.utility_function = (
            ufun if ufun is None else JavaUtilityFunction(ufun, None)
        )

    def getUtilityFunction(self):
        return to_java(self.shadow._utility_function)

    def getID(self):
        return self.shadow.id

    def setID(self):
        return self.shadow.id


class PassThroughSAONegotiator(SAONegotiator):
    """A negotiator that acts as an end point to a parent Controller
    """

    def propose(self, state: MechanismState) -> Optional["Outcome"]:
        return self._Negotiator__parent.propose(self.id, state)

    def respond(self, state: MechanismState, offer: "Outcome") -> "ResponseType":
        return self._Negotiator__parent.respond(self.id, state, offer)

    def on_negotiation_end(self, state: MechanismState) -> None:
        return self._Negotiator__parent.on_negotiation_end(self.id, state)


class SAOController(Controller):
    """A controller that can manage multiple negotiators taking full or partial control from them."""

    def __init__(
        self,
        default_negotiator_type=PassThroughSAONegotiator,
        default_negotiator_params=None,
        name=None,
    ):
        super().__init__(
            default_negotiator_type=default_negotiator_type,
            default_negotiator_params=default_negotiator_params,
            name=name,
        )

    def propose(self, negotiator_id: str, state: MechanismState) -> Optional["Outcome"]:
        negotiator, cntxt = self._negotiators.get(negotiator_id, (None, None))
        if negotiator is None:
            raise ValueError(f"Unknown negotiator {negotiator_id}")
        return self.call(negotiator, "propose", state=state)

    def respond(
        self, negotiator_id: str, state: MechanismState, offer: "Outcome"
    ) -> "ResponseType":
        negotiator, cntxt = self._negotiators.get(negotiator_id, (None, None))
        if negotiator is None:
            raise ValueError(f"Unknown negotiator {negotiator_id}")
        return self.call(negotiator, "respond", state=state, offer=offer)

    def on_negotiation_end(self, negotiator_id: str, state: MechanismState) -> None:
        pass


class SAOSyncController(SAOController):
    """A controller that can manage multiple negotiators synchronously"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.offers: Dict[str, "Outcome"] = {}
        """Keeps the last offer received for each negotiation"""
        self.responses: Dict[str, ResponseType] = {}
        """Keeps the next response type for each negotiation"""
        self.proposals: Dict[str, "Outcome"] = {}
        """Keeps the next proposal for each negotiation"""
        self.offer_states: Dict[str, "SAOState"] = {}
        """Keeps the last state received for each negotiation"""
        self.n_waits: Dict[str, int] = defaultdict(int)

    def propose(self, negotiator_id: str, state: MechanismState) -> Optional["Outcome"]:
        # if there are no proposals yet, get first proposals
        if len(self.proposals) == 0:
            self.proposals = self.first_proposals()
        # get the saved proposal if it exists and return it
        proposal = self.proposals.get(negotiator_id, None)
        # if some proposal was there, delete it to force the controller to get a new one
        if proposal is not None:
            self.proposals[negotiator_id] = None
        return proposal

    def respond(
        self, negotiator_id: str, state: MechanismState, offer: "Outcome"
    ) -> "ResponseType":
        # get the saved response to this negotiator if any
        response = self.responses.get(negotiator_id, None)
        if response is not None:
            # remove the response and return it
            del self.responses[negotiator_id]
            self.n_waits[negotiator_id] = 0
            return response

        # set the saved offer for this negotiator
        self.offers[negotiator_id] = offer
        self.offer_states[negotiator_id] = state

        # if we got all the offers or waited long enough, counter all the offers so-far
        if len(self.offers) == len(self.negotiators) or self.n_waits[
            negotiator_id
        ] >= len(self.negotiators):
            responses = self.counter_all(offers=self.offers, states=self.offer_states)
            for nid in self.responses.keys():
                # register the responses for next time for all other negotiators
                if nid != negotiator_id:
                    self.responses[nid] = responses[nid].response
                self.proposals[nid] = responses[nid].outcome
            self.offers = dict()
            self.n_waits[negotiator_id] = 0
            return responses[negotiator_id].response
        self.n_waits[negotiator_id] += 1
        return ResponseType.WAIT

    @abstractmethod
    def counter_all(
        self, offers: Dict[str, "Outcome"], states: Dict[str, SAOState]
    ) -> Dict[str, SAOResponse]:
        """Calculate a response to all offers from all negotiators (negotiator ID is the key).

        Args:
            offers: Maps negotiator IDs to offers
            states: Maps negotiator IDs to offers AT the time the offers were made.

        Remarks:
            - The response type CANNOT be WAIT.
            - If the system determines that a loop is formed, the agent may receive this call for a subset of
              negotiations not all of them.

        """

    def first_proposals(self) -> Dict[str, "Outcome"]:
        """Gets a set of proposals to use for initializing the negotiation. To avoid offering anything, just return None
        for all of them. That is the default"""
        return dict(zip(self.negotiators.keys(), itertools.repeat(None)))


SAOProtocol = SAOMechanism
"""An alias for `SAOMechanism` object"""

SimpleTitForTatNegotiator = NaiveTitForTatNegotiator
"""A simple tit-for-tat negotiator"""
