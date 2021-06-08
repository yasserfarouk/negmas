"""
Implements Stacked Alternating Offers (SAO) mechanism.
"""
from collections import defaultdict
import functools
import itertools
import math
import random
import warnings
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid
import pathlib
import os

import pandas as pd

from .common import SAOResponse, SAOState
from ..common import AgentMechanismInterface, MechanismState
from ..events import Event
from ..mechanisms import Mechanism, MechanismRoundResult
from ..outcomes import Outcome, ResponseType, outcome_is_complete
from ..helpers import exception2str, TimeoutCaller, TimeoutError

__all__ = [
    "SAOMechanism",
    "SAOProtocol",
]


class SAOMechanism(Mechanism):
    """
    Implements Several variants of the Stacked Alternating Offers Protocol

    Args:
        issues: A list of issues defining the outcome-space of the negotiation
                (See `outcomes`).  Only one of `issues` and `outcomes` can be
                passed (the other must be `None`)
        outcomes: A list of outcomes defining the outcome-space of the negotiation
                  (See `issues`).  Only one of `issues` and `outcomes` can be
                  passed (the other must be `None`)
        n_steps: The maximum number of negotiaion rounds (see `time_limit` )
        time_limit: The maximum wall-time allowed for the negotiation (see `n_steps`)
        step_time_limit: The maximum wall-time allowed for a single negotiation round.
        max_n_agents: The maximum number of negotiators allowed to join the negotiation.
        dynamic_entry: Whether it is allowed for negotiators to join the negotiation after it starts
        keep_issue_names: DEPRICATED (use `outcome_type` instead). If true, `outcomes` and `discrete_outcomes` will be
                          dictionaries otherwise they will be tuples
        outcome_type: The outcome type to use for `outcomes` and `distrete_outcomes`. It can be `tuple`, `dict` or any
                      `OutcomeType`
        cache_outcomes: If true, the mechnism will catch `outcomes` and a discrete version (`discrete_outcomes`) that
                        can be accessed by any negotiator through their AMI.
        max_n_outcomes: Maximum number or outcomes to use when disctetizing the outcome-space
        annotation: A key-value mapping to keep around. Accessible through the AMI but not used by the mechanism.
        end_on_no_response: End the negotiation if any negotiator returns NO_RESPONSE from `respond`/`counter` or returns
                            REJECT_OFFER then refuses to give an offer (by returning `None` from `proposee/`counter`).
        publish_proposer: Put the last proposer in the state.
        publish_n_acceptances: Put the number of acceptances an offer got so far in the state.
        enable_callbacks: Enable callbacks like on_round_start, etc. Note that on_negotiation_end is always received
                          by the negotiators no matter what is the setting for this parameter.
        avoid_ultimatum: If true, a proposal is taken from every agent in the first round then all of them are discarded
                         except one (choose randomly) and the algorithm continues using this one. This prevents negotiators
                         from knowing their order in the round.
        check_offers: If true, offers are checked to see if they are valid for the negotiation
                      outcome-space and if not the offer is considered None which is the same as
                      refusing to offer (NO_RESPONSE).
        ignore_negotiator_exceptions: just silently ignore negotiator exceptions and consider them no-responses.
        offering_is_accepting: Offering an outcome implies accepting it. If not, the agent who proposed an offer will
                               be asked to respond to it after all other agents.
        name: Name of the mecnanism
        **kwargs: Extra paramters passed directly to the `Mechanism` constructor

    Remarks:

        - If both `n_steps` and `time_limit` are passed, the negotiation ends when either of the limits is reached.
        - Negotiations may take longer than `time_limit` because negotiators are not interrupted while they are
          executing their `respond` or `propose` methods.

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
        dynamic_entry=False,
        keep_issue_names=None,
        outcome_type=tuple,
        cache_outcomes=True,
        max_n_outcomes: int = 1_000_000,
        annotation: Optional[Dict[str, Any]] = None,
        end_on_no_response=False,
        publish_proposer=True,
        publish_n_acceptances=False,
        enable_callbacks=False,
        avoid_ultimatum=False,
        check_offers=True,
        ignore_negotiator_exceptions=False,
        offering_is_accepting=True,
        allow_offering_just_rejected_outcome=True,
        name: Optional[str] = None,
        max_wait: int = sys.maxsize,
        sync_calls: bool = False,
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
            outcome_type=outcome_type,
            cache_outcomes=cache_outcomes,
            max_n_outcomes=max_n_outcomes,
            annotation=annotation,
            state_factory=SAOState,
            enable_callbacks=enable_callbacks,
            name=name,
            **kwargs,
        )
        self._sync_calls = sync_calls
        self.params["end_on_no_response"] = end_on_no_response
        self.params["publish_proposer"] = publish_proposer
        self.params["publish_n_acceptances"] = publish_n_acceptances
        self.params["enable_callbacks"] = enable_callbacks
        self.params["avoid_ultimatum"] = avoid_ultimatum
        self.params["check_offers"] = check_offers
        self.params["offering_is_accepting"] = offering_is_accepting
        self.params[
            "allow_offering_just_rejected_outcome"
        ] = allow_offering_just_rejected_outcome
        self.ignore_negotiator_exceptions = ignore_negotiator_exceptions
        self.allow_offering_just_rejected_outcome = allow_offering_just_rejected_outcome
        self._current_offer = None
        self._current_proposer = None
        self._last_checked_negotiator = -1
        self._frozen_neg_list = None
        self._n_accepting = 0
        self._avoid_ultimatum = n_steps is not None and avoid_ultimatum
        self.end_negotiation_on_refusal_to_propose = end_on_no_response
        self.publish_proposer = publish_proposer
        self.publish_n_acceptances = publish_n_acceptances
        self.check_offers = check_offers
        self._no_responses = 0
        self._new_offers = []
        self._offering_is_accepting = offering_is_accepting
        self._n_waits = 0
        self._n_max_waits = max_wait if max_wait is not None else float("inf")
        self.params["max_wait"] = self._n_max_waits
        self._waiting_time: Dict[str, float] = defaultdict(float)
        self._waiting_start: Dict[str, float] = defaultdict(lambda: float("inf"))
        self._selected_first = 0

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

    def set_sync_call(self, v: bool):
        self._sync_call = v

    def extra_state(self):
        current_proposer_agent = (
            self._current_proposer.owner if self._current_proposer else None
        )
        if current_proposer_agent and self.publish_proposer:
            current_proposer_agent = current_proposer_agent.id
        new_offerer_agents = []
        for neg_id, _ in self._new_offers:
            neg = self._negotiator_map.get(neg_id, None)
            agent = neg.owner if neg else None
            if agent is not None and self.publish_proposer:
                new_offerer_agents.append(agent.id)
            else:
                new_offerer_agents.append(None)
        return dict(
            current_offer=self._current_offer,
            new_offers=self._new_offers,
            current_proposer=self._current_proposer.id
            if self._current_proposer and self.publish_proposer
            else None,
            current_proposer_agent=current_proposer_agent,
            n_acceptances=self._n_accepting if self.publish_n_acceptances else 0,
            new_offerer_agents=new_offerer_agents,
        )

    def plot(
        self,
        visible_negotiators: Union[Tuple[int, int], Tuple[str, str]] = (0, 1),
        plot_utils=True,
        plot_outcomes=False,
        utility_range: Optional[Tuple[float, float]] = None,
        save_fig: bool = False,
        path: str = None,
        fig_name: str = None,
    ):
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        if self.issues is not None and len(self.issues) > 1:
            plot_outcomes = False

        if len(self.negotiators) < 2:
            warnings.warn(
                "Cannot visualize negotiations with more less than 2 negotiators"
            )
            return
        if len(visible_negotiators) > 2:
            warnings.warn("Cannot visualize more than 2 agents")
            return
        if isinstance(visible_negotiators[0], str):
            tmp = []
            for _ in visible_negotiators:
                for n in self.negotiators:
                    if n.id == _:
                        tmp.append(n)
        else:
            vnegotiators = [
                self.negotiators[visible_negotiators[0]],
                self.negotiators[visible_negotiators[1]],
            ]
        # indx = dict(zip([_.id for _ in self.negotiators], range(len(self.negotiators))))
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
                        "u0": vnegotiators[0].utility_function(o),
                        "u1": vnegotiators[1].utility_function(o),
                    }
                )
        history = pd.DataFrame(data=history)
        has_history = len(history) > 0
        has_front = 1
        # n_negotiators = len(self.negotiators)
        n_agents = len(vnegotiators)
        ufuns = self._get_ufuns()
        outcomes = self.outcomes
        utils = [tuple(f(o) for f in ufuns) for o in outcomes]
        agent_names = [a.name for a in vnegotiators]
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
                    history.current_proposer == vnegotiators[a].id,
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
                        history.current_proposer == vnegotiators[a].id,
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

        if save_fig:
            if fig_name is None:
                fig_name = str(uuid.uuid4()) + ".png"
            if path is None:
                path = pathlib.Path().absolute()
            else:
                pathlib.Path(path).mkdir(parents=True, exist_ok=True)
            if plot_utils:
                fig_util.savefig(os.path.join(path, fig_name))
            if plot_outcomes:
                fig_outcome.savefig(os.path.join(path, fig_name))
        else:
            if plot_utils:
                fig_util.show()
            if plot_outcomes:
                fig_outcome.show()

    def round(self) -> MechanismRoundResult:
        """implements a round of the Stacked Alternating Offers Protocol."""
        if self._frozen_neg_list is None:
            self._new_offers = []
        negotiators: List["SAONegotiator"] = self.negotiators
        n_negotiators = len(negotiators)
        # times = dict(zip([_.id for _ in negotiators], itertools.repeat(0.0)))
        times = defaultdict(float, self._waiting_time)
        exceptions = dict(
            zip([_.id for _ in negotiators], [list() for _ in negotiators])
        )

        def _safe_counter(negotiator, *args, **kwargs) -> Tuple[SAOResponse, bool]:
            rem = self.remaining_time
            if rem is None:
                rem = float("inf")
            timeout = min(
                self.ami.negotiator_time_limit - times[negotiator.id],
                self.ami.step_time_limit,
                rem,
            )
            if timeout is None or timeout == float("inf") or self._sync_calls:
                __strt = time.perf_counter()
                try:
                    if (
                        negotiator == self._current_proposer
                    ) and self._offering_is_accepting:
                        self._n_accepting = 0
                        response = negotiator.counter(*args, **kwargs)
                    else:
                        response = negotiator.counter(*args, **kwargs)
                except TimeoutError:
                    response = None
                    try:
                        negotiator.cancel()
                    except:
                        pass
                except Exception as ex:
                    exceptions[negotiator.id].append(exception2str())
                    if self.ignore_negotiator_exceptions:
                        self.announce(
                            Event(
                                "negotiator_exception",
                                {"negotiator": negotiator, "exception": ex},
                            )
                        )
                        times[negotiator.id] += time.perf_counter() - __strt
                        return SAOResponse(ResponseType.END_NEGOTIATION, None), True
                    else:
                        raise ex
                times[negotiator.id] += time.perf_counter() - __strt
            else:
                fun = functools.partial(negotiator.counter, *args, **kwargs)
                __strt = time.perf_counter()
                try:
                    if (
                        negotiator == self._current_proposer
                    ) and self._offering_is_accepting:
                        self._n_accepting = 0
                        response = TimeoutCaller.run(fun, timeout=timeout)
                    else:
                        response = TimeoutCaller.run(fun, timeout=timeout)
                except TimeoutError:
                    response = None
                except Exception as ex:
                    exceptions[negotiator.id].append(exception2str())
                    if self.ignore_negotiator_exceptions:
                        self.announce(
                            Event(
                                "negotiator_exception",
                                {"negotiator": negotiator, "exception": ex},
                            )
                        )
                        times[negotiator.id] += time.perf_counter() - __strt
                        return SAOResponse(ResponseType.END_NEGOTIATION, None), True
                    else:
                        raise ex
                times[negotiator.id] += time.perf_counter() - __strt
            if (
                self.check_offers
                and response is not None
                and response.outcome is not None
                and (not outcome_is_complete(response.outcome, self.issues))
            ):
                return SAOResponse(response.response, None), False
            return response, False

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
                    times=times,
                    exceptions=exceptions,
                )
            else:
                return MechanismRoundResult(
                    broken=False,
                    timedout=False,
                    agreement=None,
                    times=times,
                    exceptions=exceptions,
                )
        # if this is the first step (or no one has offered yet) which means that there is no _current_offer
        if self._current_offer is None and n_proposers > 1 and self._avoid_ultimatum:
            if not self.dynamic_entry and not self.state.step == 0:
                if self.end_negotiation_on_refusal_to_propose:
                    return MechanismRoundResult(
                        broken=True,
                        times=times,
                        exceptions=exceptions,
                    )
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
                resp, has_exceptions = _safe_counter(neg, state=self.state, offer=None)
                if has_exceptions:
                    return MechanismRoundResult(
                        broken=True,
                        timedout=False,
                        agreement=None,
                        times=times,
                        exceptions=exceptions,
                        error=True,
                        error_details=str(exceptions[neg.id]),
                    )
                if resp is None:
                    return MechanismRoundResult(
                        broken=False,
                        timedout=True,
                        agreement=None,
                        times=times,
                        exceptions=exceptions,
                        error=False,
                        error_details="",
                    )
                if resp.response != ResponseType.WAIT:
                    self._waiting_time[neg.id] = 0.0
                    self._waiting_start[neg.id] = float("inf")
                else:
                    self._waiting_start[neg.id] = min(self._waiting_start[neg.id], strt)
                    self._waiting_time[neg.id] += (
                        time.perf_counter() - self._waiting_start[neg.id]
                    )
                if resp is None:
                    return MechanismRoundResult(
                        broken=False,
                        timedout=True,
                        agreement=None,
                        times=times,
                        exceptions=exceptions,
                    )
                if time.perf_counter() - strt > self.ami.step_time_limit:
                    return MechanismRoundResult(
                        broken=False,
                        timedout=True,
                        agreement=None,
                        times=times,
                        exceptions=exceptions,
                    )
                if resp.response == ResponseType.END_NEGOTIATION:
                    return MechanismRoundResult(
                        broken=True,
                        timedout=False,
                        agreement=None,
                        times=times,
                        exceptions=exceptions,
                    )
                if resp.response in (ResponseType.NO_RESPONSE, ResponseType.WAIT):
                    continue
                responses.append(resp)
            if len(responses) < 1:
                if not self.dynamic_entry:
                    return MechanismRoundResult(
                        broken=True,
                        timedout=False,
                        agreement=None,
                        error=True,
                        error_details="No proposers and no dynamic entry. This may happen if no negotiators responded to their first proposal request with an offer",
                        times=times,
                        exceptions=exceptions,
                    )
                else:
                    return MechanismRoundResult(
                        broken=False,
                        timedout=False,
                        agreement=None,
                        times=times,
                        exceptions=exceptions,
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
            self._selected_first = _first_proposer
            return MechanismRoundResult(
                broken=False,
                timedout=False,
                agreement=None,
                times=times,
                exceptions=exceptions,
            )

        # this is not the first round. A round will get n_negotiators steps
        if self._frozen_neg_list is not None:
            ordered_indices = self._frozen_neg_list
        else:
            ordered_indices = [
                (_ + self._last_checked_negotiator + 1) % n_negotiators
                for _ in range(n_negotiators)
            ]

        for neg_indx in ordered_indices:
            self._last_checked_negotiator = neg_indx
            neg = self.negotiators[neg_indx]
            strt = time.perf_counter()
            resp, has_exceptions = _safe_counter(
                neg, state=self.state, offer=self._current_offer
            )
            if has_exceptions:
                return MechanismRoundResult(
                    broken=True,
                    timedout=False,
                    agreement=None,
                    times=times,
                    exceptions=exceptions,
                    error=True,
                    error_details=str(exceptions[neg.id]),
                )
            if resp is None:
                return MechanismRoundResult(
                    broken=False,
                    timedout=True,
                    agreement=None,
                    times=times,
                    exceptions=exceptions,
                    error=False,
                    error_details="",
                )
            if resp.response == ResponseType.WAIT:
                self._waiting_start[neg.id] = min(self._waiting_start[neg.id], strt)
                self._waiting_time[neg.id] += time.perf_counter() - strt
                if self._frozen_neg_list is None:
                    self._frozen_neg_list = ordered_indices[
                        self._last_checked_negotiator :
                    ]
                self._n_waits += 1
            else:
                self._waiting_time[neg.id] = 0.0
                self._waiting_start[neg.id] = float("inf")
                self._n_waits = 0
                self._frozen_neg_list = None

            if resp is None:
                return MechanismRoundResult(
                    broken=False,
                    timedout=True,
                    agreement=None,
                    times=times,
                    exceptions=exceptions,
                )
            if time.perf_counter() - strt > self.ami.step_time_limit:
                return MechanismRoundResult(
                    broken=False,
                    timedout=True,
                    agreement=None,
                    times=times,
                    exceptions=exceptions,
                )
            if self._enable_callbacks:
                if self._current_offer is not None:
                    for other in self.negotiators:
                        if other is not neg:
                            other.on_partner_response(
                                state=self.state,
                                partner_id=neg.id,
                                outcome=self._current_offer,
                                response=resp.response,
                            )
            if resp.response == ResponseType.NO_RESPONSE:
                continue
            if resp.response == ResponseType.WAIT:
                if self._n_waits > self._n_max_waits:
                    self._n_waits = 0
                    self._waiting_time[neg.id] = 0.0
                    self._waiting_start[neg.id] = float("inf")
                    self._frozen_neg_list = None
                    return MechanismRoundResult(
                        broken=False,
                        timedout=True,
                        agreement=None,
                        waiting=True,
                        times=times,
                        exceptions=exceptions,
                    )
                self._last_checked_negotiator = neg_indx - 1
                if neg_indx < 0:
                    self._last_checked_negotiator = n_negotiators - 1
                return MechanismRoundResult(
                    broken=False,
                    timedout=False,
                    agreement=None,
                    waiting=True,
                    times=times,
                    exceptions=exceptions,
                )
            if resp.response == ResponseType.END_NEGOTIATION:
                return MechanismRoundResult(
                    broken=True,
                    timedout=False,
                    agreement=None,
                    times=times,
                    exceptions=exceptions,
                )
            if resp.response == ResponseType.ACCEPT_OFFER:
                self._n_accepting += 1
                if self._n_accepting == n_negotiators:
                    return MechanismRoundResult(
                        broken=False,
                        timedout=False,
                        agreement=self._current_offer,
                        times=times,
                        exceptions=exceptions,
                    )
            if resp.response == ResponseType.REJECT_OFFER:
                proposal = resp.outcome
                if (
                    not self.allow_offering_just_rejected_outcome
                    and proposal == self._current_offer
                ):
                    proposal = None
                if proposal is None:
                    if (
                        neg.capabilities.get("propose", True)
                        and self.end_negotiation_on_refusal_to_propose
                    ):
                        return MechanismRoundResult(
                            broken=True,
                            timedout=False,
                            agreement=None,
                            times=times,
                            exceptions=exceptions,
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
                                partner_id=neg.id, offer=proposal, state=self.state
                            )
        return MechanismRoundResult(
            broken=False,
            timedout=False,
            agreement=None,
            times=times,
            exceptions=exceptions,
        )

    @property
    def trace(self) -> List[Tuple[str, Outcome]]:
        """Returns the negotiation history as a list of negotiator/offer tuples"""
        offers = []
        for state in self._history:
            offers += [(n, o) for n, o in state.new_offers]

        def not_equal(a, b):
            if isinstance(a, dict):
                a = a.values()
            if isinstance(b, dict):
                b = b.values()
            return any(x != y for x, y in zip(a, b))

        if (
            self.agreement is not None
            and offers
            and not_equal(offers[-1][1], self.agreement)
        ):
            offers.append(
                (
                    self._history[-1].current_proposer,
                    self.agreement,
                )
            )
        return offers

    def negotiator_offers(self, negotiator_id: str) -> List[Outcome]:
        """Returns the offers given by a negotiator (in order)"""
        return [o for n, o in self.trace if n == negotiator_id]

    @property
    def offers(self) -> List[Outcome]:
        """Returns the negotiation history as a list of offers"""
        return [o for _, o in self.trace]


SAOProtocol = SAOMechanism
"""An alias for `SAOMechanism` object"""
