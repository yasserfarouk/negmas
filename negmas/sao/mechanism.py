"""
Implements Stacked Alternating Offers (SAO) mechanism.
"""
from __future__ import annotations

import functools
import sys
import time
from collections import defaultdict
from typing import TYPE_CHECKING

from attr import asdict
from rich import print

from negmas import warnings
from negmas.gb.negotiators import GBNegotiator
from negmas.helpers.strings import humanize_time

from ..common import TraceElement
from ..events import Event
from ..helpers import TimeoutCaller, TimeoutError, exception2str
from ..mechanisms import Mechanism, MechanismStepResult
from ..outcomes.common import Outcome
from ..outcomes.outcome_ops import cast_value_types, outcome_types_are_ok
from .common import SAONMI, ResponseType, SAOResponse, SAOState

if TYPE_CHECKING:
    from negmas.preferences import Preferences

    from .negotiators import SAONegotiator

__all__ = [
    "SAOMechanism",
    "SAOProtocol",
    "TraceElement",
]

DEFAULT_COLORMAP = "jet"


class SAOMechanism(Mechanism):
    """
    Implements Several variants of the Stacked Alternating Offers Protocol

    Args:
        outcome_space: The negotiation agenda
        issues: A list of issues defining the outcome-space of the negotiation
        outcomes: A list of outcomes defining the outcome-space of the negotiation
        n_steps: The maximum number of negotiaion rounds (see `time_limit` )
        time_limit: The maximum wall-time allowed for the negotiation (see `n_steps`)
        step_time_limit: The maximum wall-time allowed for a single negotiation round.
        max_n_agents: The maximum number of negotiators allowed to join the negotiation.
        dynamic_entry: Whether it is allowed for negotiators to join the negotiation after it starts
        cache_outcomes: If true, the mechnism will catch `outcomes` and a discrete version (`discrete_outcomes`) that
                        can be accessed by any negotiator through their AMI.
        max_cardinality: Maximum number or outcomes to use when disctetizing the outcome-space
        annotation: A key-value mapping to keep around. Accessible through the AMI but not used by the mechanism.
        end_on_no_response: End the negotiation if any negotiator returns NO_RESPONSE from `respond`/`counter` or returns
                            REJECT_OFFER then refuses to give an offer (by returning `None` from `proposee/`counter`).
        enable_callbacks: Enable callbacks like on_round_start, etc. Note that on_negotiation_end is always received
                          by the negotiators no matter what is the setting for this parameter.
        check_offers: If true, offers are checked to see if they are valid for the negotiation
                      outcome-space and if not the offer is considered None which is the same as
                      refusing to offer (NO_RESPONSE).
        enforce_issue_types: If True, the type of each issue is enforced depending on the value of `cast_offers`
        cast_offers: If true, each issue value is cast using the issue's type otherwise an incorrect type will be considered an invalid offer. See `check_offers`. Only
                     used if `enforce_issue_types`
        ignore_negotiator_exceptions: just silently ignore negotiator exceptions and consider them no-responses.
        offering_is_accepting: Offering an outcome implies accepting it. If not, the agent who proposed an offer will
                               be asked to respond to it after all other agents.
        name: Name of the mecnanism
        **kwargs: Extra paramters passed directly to the `Mechanism` constructor

    Remarks:

        - One and only one of `outcome_space`, `issues`, `outcomes` can be given
        - If both `n_steps` and `time_limit` are passed, the negotiation ends when either of the limits is reached.
        - Negotiations may take longer than `time_limit` because negotiators are not interrupted while they are
          executing their `respond` or `propose` methods.

    Events:

        - negotiator_exception: Data=(negotiator, exception) raised whenever a negotiator raises an exception if
          ignore_negotiator_exceptions is set to True.
    """

    def __init__(
        self,
        dynamic_entry=False,
        extra_callbacks=True,
        end_on_no_response=True,
        avoid_ultimatum=False,
        check_offers=False,
        enforce_issue_types=False,
        cast_offers=False,
        ignore_negotiator_exceptions=False,
        offering_is_accepting=True,
        allow_offering_just_rejected_outcome=True,
        name: str | None = None,
        max_wait: int = sys.maxsize,
        sync_calls: bool = False,
        initial_state: SAOState | None = None,
        one_offer_per_step: bool = False,
        **kwargs,
    ):
        if avoid_ultimatum:
            warnings.warn(
                "Support for Avoid-Ultimatum will be removed soon. We will force avoid_ultimatum to False",
                warnings.NegmasWarning,
            )
            avoid_ultimatum = False
        super().__init__(
            dynamic_entry=dynamic_entry,
            extra_callbacks=extra_callbacks,
            initial_state=SAOState() if not initial_state else initial_state,
            name=name,
            **kwargs,
        )
        self.nmi = SAONMI(
            **{
                **asdict(self.nmi, recurse=False),
                **dict(
                    end_on_no_response=end_on_no_response,
                    one_offer_per_step=one_offer_per_step,
                ),
            }
        )
        assert self.nmi.end_on_no_response == end_on_no_response
        assert self.nmi.one_offer_per_step == one_offer_per_step
        self._one_offer_per_step = one_offer_per_step
        self._current_state: SAOState

        # self._history: list[SAOState] = []
        n_steps, time_limit = self.n_steps, self.time_limit
        if (n_steps is None or n_steps == float("inf")) and (
            time_limit is None or time_limit == float("inf")
        ):
            warnings.warn(
                "You are passing no time_limit and no n_steps to an SAOMechanism. The mechanism may never finish!!",
                warnings.NegmasInfiniteNegotiationWarning,
            )
        self._sync_calls = sync_calls
        self.params["one_offer_per_step"] = one_offer_per_step
        self.params["end_on_no_response"] = end_on_no_response
        self.params["enable_callbacks"] = extra_callbacks
        self.params["sync_calls"] = sync_calls
        self.params["check_offers"] = check_offers
        self.params["offering_is_accepting"] = offering_is_accepting
        self.params["enforce_issue_types"] = enforce_issue_types
        self.params["cast_offers"] = cast_offers
        self.params[
            "allow_offering_just_rejected_outcome"
        ] = allow_offering_just_rejected_outcome
        self._n_max_waits = max_wait if max_wait is not None else float("inf")
        self.params["max_wait"] = self._n_max_waits
        self.ignore_negotiator_exceptions = ignore_negotiator_exceptions
        self.allow_offering_just_rejected_outcome = allow_offering_just_rejected_outcome
        self.end_negotiation_on_refusal_to_propose = end_on_no_response
        self.check_offers = check_offers
        self._enforce_issue_types = enforce_issue_types
        self._cast_offers = cast_offers

        self._last_checked_negotiator = -1
        self._current_proposer = None
        self._frozen_neg_list = None
        self._no_responses = 0
        self._offering_is_accepting = offering_is_accepting
        self._n_waits = 0
        self._waiting_time: dict[str, float] = defaultdict(float)
        self._waiting_start: dict[str, float] = defaultdict(lambda: float("inf"))
        self._selected_first = 0

    @property
    def state(self) -> SAOState:
        """Returns the current state.

        Override `extra_state` if you want to keep extra state
        """
        return self._current_state

    def add(  # type: ignore
        self,
        negotiator: SAONegotiator | GBNegotiator,
        *,
        preferences: Preferences | None = None,
        role: str | None = None,
        **kwargs,
    ) -> bool | None:
        from ..genius.negotiator import GeniusNegotiator

        added = super().add(negotiator, preferences=preferences, role=role, **kwargs)
        if (
            added
            and isinstance(negotiator, GeniusNegotiator)
            and self.nmi.time_limit is not None
            and self.nmi.time_limit != float("inf")
            and self.nmi.n_steps is not None
            and self.nmi.n_steps != float("inf")
        ):
            warnings.warn(
                f"{negotiator.id} of type {negotiator.__class__.__name__} is joining "
                f"SAOMechanism which has a time_limit of {self.nmi.time_limit} seconds "
                f"and a n_steps of {self.nmi.n_steps}. This agnet will only know about the "
                f"time_limit and will not know about the n_steps!!!",
                warnings.NegmasStepAndTimeLimitWarning,
            )
        return added

    def set_sync_call(self, v: bool):
        self._sync_call = v

    def _agent_info(self):
        state = self._current_state
        current_proposer_agent = (
            self._current_proposer.owner if self._current_proposer else None
        )
        if current_proposer_agent:
            current_proposer_agent = current_proposer_agent.id
        new_offerer_agents = []
        for neg_id, _ in state.new_offers:
            neg = self._negotiator_map.get(neg_id, None)
            agent = neg.owner if neg else None
            if agent is not None:
                new_offerer_agents.append(agent.id)
            else:
                new_offerer_agents.append(None)
        return current_proposer_agent, new_offerer_agents

    def next_negotitor_ids(self) -> list[str]:
        """Returns a list of negotiator IDs in the order they are to be run in the next call to step()"""
        n_negotiators = len(self.negotiators)
        if self._frozen_neg_list is not None:
            ordered_indices = self._frozen_neg_list
        else:
            ordered_indices = [
                (_ + self._last_checked_negotiator + 1) % n_negotiators
                for _ in range(n_negotiators)
            ]

        if ordered_indices and self._one_offer_per_step:
            ordered_indices = ordered_indices[:1]
        ids = self.negotiator_ids
        return [ids[_] for _ in ordered_indices]

    def _stop_waiting(self, negotiator_id):
        self._waiting_time[negotiator_id] = 0.0
        self._waiting_start[negotiator_id] = float("inf")
        self._n_waits = 0
        self._frozen_neg_list = None

    def __call__(  # type: ignore
        self, state: SAOState, action: dict[str, SAOResponse] | None = None
    ) -> MechanismStepResult:
        """
        implements a round or a single step of the Stacked Alternating Offers Protocol.

        Args:
            state: Current state of the mechanism
            action: The action to use as a mapping from negotiator ID (key) to its response (value).
                    If not given, the negotiator(s) is called to generate its response.
        """
        state = self._current_state
        if self._frozen_neg_list is None:
            state.new_offers = []
        negotiators: list[SAONegotiator] = self.negotiators
        n_negotiators = len(negotiators)
        # times = dict(zip([_.id for _ in negotiators], itertools.repeat(0.0)))
        times = defaultdict(float, self._waiting_time)
        exceptions = dict(
            zip([_.id for _ in negotiators], [list() for _ in negotiators])
        )

        def _safe_counter(
            negotiator, *args, **kwargs
        ) -> tuple[SAOResponse | None, bool]:
            assert (
                not state.waiting or negotiator.id == state.current_proposer
            ), f"We are waiting with {state.current_proposer} as the last offerer but we are asking {negotiator.id} to offer\n{state}"
            if self.verbosity > 2:
                print(
                    f"{self.name}: {negotiator.name} called after {humanize_time(time.perf_counter() - self._start_time, show_ms=True) if self._start_time else 0}",
                    flush=True,
                )
            rem = self.remaining_time
            if rem is None:
                rem = float("inf")
            timeout = min(
                self.nmi.negotiator_time_limit - times[negotiator.id],
                self.nmi.step_time_limit,
                rem,
                self._hidden_time_limit - self.time,
            )
            given_response = action.pop(negotiator.id, None) if action else None
            if timeout is None or timeout == float("inf") or self._sync_calls:
                __strt = time.perf_counter()
                try:
                    if (
                        negotiator == self._current_proposer
                    ) and self._offering_is_accepting:
                        self._current_state.n_acceptances = 0
                        response = (
                            given_response
                            if given_response
                            else negotiator(*args, **kwargs)
                        )
                    else:
                        response = (
                            given_response
                            if given_response
                            else negotiator(*args, **kwargs)
                        )
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
                fun = functools.partial(negotiator, *args, **kwargs)
                __strt = time.perf_counter()
                try:
                    if (
                        negotiator == self._current_proposer
                    ) and self._offering_is_accepting:
                        state.n_acceptances = 0
                        response = (
                            given_response
                            if given_response
                            else TimeoutCaller.run(fun, timeout=timeout)
                        )
                    else:
                        response = (
                            given_response
                            if given_response
                            else TimeoutCaller.run(fun, timeout=timeout)
                        )
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
            ):
                if not self.outcome_space.is_valid(response.outcome):
                    return SAOResponse(response.response, None), False
                # todo: do not use .issues here as they are not guaranteed to exist (if it is not a cartesial outcome space)
                if self._enforce_issue_types and hasattr(self.outcome_space, "issues"):
                    if outcome_types_are_ok(
                        response.outcome, self.outcome_space.issues  # type: ignore
                    ):
                        return response, False
                    elif self._cast_offers:
                        return (
                            SAOResponse(
                                response.response,
                                cast_value_types(
                                    response.outcome, self.outcome_space.issues  # type: ignore
                                ),
                            ),
                            False,
                        )
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
                state.broken = True
                state.has_error = True
                state.error_details = "No proposers and no dynamic entry"
                return MechanismStepResult(state, times=times, exceptions=exceptions)
            else:
                return MechanismStepResult(state, times=times, exceptions=exceptions)
        if self._frozen_neg_list is not None:
            ordered_indices = self._frozen_neg_list
        else:
            ordered_indices = [
                (_ + self._last_checked_negotiator + 1) % n_negotiators
                for _ in range(n_negotiators)
            ]

        if ordered_indices and self._one_offer_per_step:
            ordered_indices = ordered_indices[:1]

        for _, neg_indx in enumerate(ordered_indices):
            self._last_checked_negotiator = neg_indx
            neg = self.negotiators[neg_indx]
            strt = time.perf_counter()
            resp, has_exceptions = _safe_counter(neg, state=self.state)
            self._negotiator_times[neg.id] += time.perf_counter() - strt
            if has_exceptions:
                state.broken = True
                state.has_error = True
                state.error_details = str(exceptions[neg.id])
                return MechanismStepResult(
                    state,
                    times=times,
                    exceptions=exceptions,
                )
            if resp is None:
                state.timedout = True
                return MechanismStepResult(
                    state,
                    times=times,
                    exceptions=exceptions,
                )
            if resp.response == ResponseType.WAIT:
                self._waiting_start[neg.id] = min(self._waiting_start[neg.id], strt)
                self._waiting_time[neg.id] += time.perf_counter() - strt
                self._last_checked_negotiator = (neg_indx - 1) % n_negotiators
                offered = {self._negotiator_index[_[0]] for _ in state.new_offers}
                did_not_offer = sorted(
                    list(set(range(n_negotiators)).difference(offered))
                )
                assert neg_indx in did_not_offer
                indx = did_not_offer.index(neg_indx)
                assert (
                    self._frozen_neg_list is None
                    or self._frozen_neg_list[0] == neg_indx
                )
                self._frozen_neg_list = did_not_offer[indx:] + did_not_offer[:indx]
                self._n_waits += 1
            else:
                self._stop_waiting(neg.id)

            if resp is None or time.perf_counter() - strt > self.nmi.step_time_limit:
                state.timedout = True
                return MechanismStepResult(
                    state,
                    times=times,
                    exceptions=exceptions,
                )
            if self._extra_callbacks:
                if state.current_offer is not None:
                    for other in self.negotiators:
                        if other is not neg:
                            other.on_partner_response(
                                state=self.state,
                                partner_id=neg.id,
                                outcome=state.current_offer,
                                response=resp.response,
                            )
            if resp.response == ResponseType.NO_RESPONSE:
                continue
            if resp.response == ResponseType.WAIT:
                if self._n_waits > self._n_max_waits:
                    self._stop_waiting(neg.id)
                    state.timedout = True
                    state.waiting = False
                    return MechanismStepResult(
                        state,
                        times=times,
                        exceptions=exceptions,
                    )
                state.waiting = True
                return MechanismStepResult(
                    state,
                    times=times,
                    exceptions=exceptions,
                )
            if resp.response == ResponseType.END_NEGOTIATION:
                state.broken = True
                return MechanismStepResult(
                    state,
                    times=times,
                    exceptions=exceptions,
                )
            if resp.response == ResponseType.ACCEPT_OFFER:
                state.n_acceptances += 1
                if state.n_acceptances == n_negotiators:
                    state.agreement = self._current_state.current_offer
                    return MechanismStepResult(
                        state,
                        timedout=False,
                        agreement=state.current_offer,
                        times=times,
                        exceptions=exceptions,
                        broken=False,
                    )
            if resp.response == ResponseType.REJECT_OFFER:
                proposal = resp.outcome
                if (
                    not self.allow_offering_just_rejected_outcome
                    and proposal == state.current_offer
                ):
                    proposal = None
                if proposal is None:
                    if (
                        neg.capabilities.get("propose", True)
                        and self.end_negotiation_on_refusal_to_propose
                    ):
                        state.broken = True
                        return MechanismStepResult(
                            state,
                            times=times,
                            exceptions=exceptions,
                        )
                    state.n_acceptances = 0
                else:
                    state.n_acceptances = 1 if self._offering_is_accepting else 0
                    if self._extra_callbacks:
                        for other in self.negotiators:
                            if other is neg:
                                continue
                            other.on_partner_proposal(
                                partner_id=neg.id, offer=proposal, state=self.state
                            )
                state.current_offer = proposal
                self._current_proposer = neg
                state.current_proposer = neg.id
                state.new_offers.append((neg.id, proposal))
                if self._last_checked_negotiator >= 0:
                    state.last_negotiator = self.negotiators[
                        self._last_checked_negotiator
                    ].name
                else:
                    state.last_negotiator = ""
                (
                    self._current_proposer_agent,
                    state.new_offerer_agents,
                ) = self._agent_info()

        # if action is not None:
        #     assert (
        #         not action
        #     ), f"Not all negotiator actions were used in this step: {action}"
        return MechanismStepResult(
            state,
            times=times,
            exceptions=exceptions,
        )

    @property
    def full_trace(self) -> list[TraceElement]:
        """Returns the negotiation history as a list of relative_time/step/negotiator/offer tuples"""

        def response(state: SAOState):
            if state.agreement:
                return "agreement"
            if state.timedout:
                return "timedout"
            if state.ended:
                return "ended"
            if state.has_error:
                return "error"
            return "continuing"

        def asint(state: SAOState):
            if state.ended:
                return int(ResponseType.END_NEGOTIATION)
            if state.timedout or state.has_error:
                return int(ResponseType.NO_RESPONSE)
            if state.agreement:
                return int(ResponseType.ACCEPT_OFFER)
            return int(ResponseType.REJECT_OFFER)

        offers = []
        for state in self._history:
            state: SAOState
            offers += [
                TraceElement(
                    state.time,
                    state.relative_time,
                    state.step,
                    n,
                    o,
                    (asint(state),),
                    response(state),
                )
                for n, o in state.new_offers
            ]

        def not_equal(a, b):
            return any(x != y for x, y in zip(a, b))

        self._history: list[SAOState]
        # if the agreement does not appear as the last offer in the trace, add it.
        # this should not happen though!!
        if (
            self.agreement is not None
            and offers
            and not_equal(offers[-1][-1], self.agreement)
        ):
            offers.append(
                TraceElement(
                    self._history[-1].time,
                    self._history[-1].relative_time,
                    self._history[-1].step,
                    self._history[-1].current_proposer,
                    self.agreement,
                    (asint(self._history[-1]),),
                    response(self._history[-1]),
                )
            )

        return offers

    @property
    def extended_trace(self) -> list[tuple[int, str, Outcome]]:
        """Returns the negotiation history as a list of step/negotiator/offer tuples"""
        offers = []
        for state in self._history:
            state: SAOState
            offers += [(state.step, n, o) for n, o in state.new_offers]

        def not_equal(a, b):
            return any(x != y for x, y in zip(a, b))

        self._history: list[SAOState]  # type: ignore
        # if the agreement does not appear as the last offer in the trace, add it.
        # this should not happen though!!
        if (
            self.agreement is not None
            and offers
            and not_equal(offers[-1][-1], self.agreement)
        ):
            offers.append(
                (
                    self._history[-1].step,
                    self._history[-1].current_proposer,
                    self.agreement,
                )
            )

        return offers

    @property
    def trace(self) -> list[tuple[str, Outcome]]:
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
            and not_equal(offers[-1][-1], self.agreement)
        ):
            offers.append(
                (
                    self._history[-1].current_proposer,
                    self.agreement,
                )
            )

        return offers

    def negotiator_offers(self, negotiator_id: str) -> list[Outcome]:
        """Returns the offers given by a negotiator (in order)"""
        return [o for n, o in self.trace if n == negotiator_id]

    def negotiator_full_trace(
        self, negotiator_id: str
    ) -> list[tuple[float, float, int, Outcome, str]]:
        """Returns the (time/relative-time/step/outcome/response) given by a negotiator (in order)"""
        return [
            (t, rt, s, o, a)
            for t, rt, s, n, o, _, a in self.full_trace
            if n == negotiator_id
        ]

    @property
    def offers(self) -> list[Outcome]:
        """Returns the negotiation history as a list of offers"""
        return [o for _, o in self.trace]

    @property
    def _step(self):
        """
        A private property used by the checkpoint system
        """
        return self._current_state.step

    def plot(
        self,
        plotting_negotiators: tuple[int, int] | tuple[str, str] = (0, 1),
        save_fig: bool = False,
        path: str | None = None,
        fig_name: str | None = None,
        ignore_none_offers: bool = True,
        with_lines: bool = True,
        show_agreement: bool = False,
        show_pareto_distance: bool = True,
        show_nash_distance: bool = True,
        show_kalai_distance: bool = True,
        show_max_welfare_distance: bool = True,
        show_max_relative_welfare_distance: bool = False,
        show_end_reason: bool = True,
        show_last_negotiator: bool = True,
        show_annotations: bool = False,
        show_reserved: bool = True,
        show_total_time=True,
        show_relative_time=True,
        show_n_steps=True,
        colors: list | None = None,
        markers: list[str] | None = None,
        colormap: str = DEFAULT_COLORMAP,
        ylimits: tuple[float, float] | None = None,
        common_legend: bool = True,
        xdim: str = "relative_time",
        only2d: bool = False,
        no2d: bool = False,
        fast: bool = False,
        simple_offers_view: bool = False,
        mark_offers_view: bool = True,
        mark_pareto_points: bool = True,
        mark_all_outcomes: bool = True,
        mark_nash_points: bool = True,
        mark_kalai_points: bool = True,
        mark_max_welfare_points: bool = True,
        **kwargs,
    ):
        from negmas.plots.util import plot_mechanism_run

        extra_annotation = (
            f"Last: {self._current_state.last_negotiator}"
            if show_last_negotiator
            else ""
        )
        return plot_mechanism_run(
            mechanism=self,
            negotiators=plotting_negotiators,
            save_fig=save_fig,
            path=path,
            fig_name=fig_name,
            ignore_none_offers=ignore_none_offers,
            with_lines=with_lines,
            only2d=only2d,
            show_agreement=show_agreement,
            show_pareto_distance=show_pareto_distance,
            show_nash_distance=show_nash_distance,
            show_kalai_distance=show_kalai_distance,
            show_max_welfare_distance=show_max_welfare_distance,
            show_max_relative_welfare_distance=show_max_relative_welfare_distance,
            show_end_reason=show_end_reason,
            show_annotations=show_annotations,
            show_reserved=show_reserved,
            colors=colors,
            markers=markers,
            colormap=colormap,
            ylimits=ylimits,
            common_legend=common_legend,
            extra_annotation=extra_annotation,
            xdim=xdim,
            colorizer=lambda _: 1.0,
            show_total_time=show_total_time,
            show_relative_time=show_relative_time,
            show_n_steps=show_n_steps,
            fast=fast,
            no2d=no2d,
            simple_offers_view=simple_offers_view,
            mark_offers_view=mark_offers_view,
            mark_pareto_points=mark_pareto_points,
            mark_all_outcomes=mark_all_outcomes,
            mark_nash_points=mark_nash_points,
            mark_kalai_points=mark_kalai_points,
            mark_max_welfare_points=mark_max_welfare_points,
            **kwargs,
        )


SAOProtocol = SAOMechanism
"""An alias for `SAOMechanism` object"""
