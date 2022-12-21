from __future__ import annotations

from random import shuffle
from typing import TYPE_CHECKING, Any, Callable

from attr import define

from negmas.common import TraceElement
from negmas.helpers.types import get_full_type_name, instantiate
from negmas.mechanisms import Mechanism, MechanismRoundResult
from negmas.outcomes import Outcome
from negmas.plots.util import default_colorizer
from negmas.preferences import BaseUtilityFunction, Preferences

from ..common import GBResponse, GBState, ResponseType, ThreadState
from ..constraints.base import LocalOfferingConstraint, OfferingConstraint
from ..evaluators.base import EvaluationStrategy, LocalEvaluationStrategy, all_accept

if TYPE_CHECKING:
    from negmas.gb.negotiators.base import GBNegotiator
    from negmas.plots.util import Colorizer

__all__ = ["GBMechanism", "ParallelGBMechanism", "SerialGBMechanism"]


@define
class GBThread:
    mechanism: GBMechanism
    negotiator: GBNegotiator
    responders: list[GBNegotiator]
    indx: int
    state: ThreadState
    evaluator: LocalEvaluationStrategy | None = None
    constraint: LocalOfferingConstraint | None = None

    @property
    def current(self):
        return self.state.current_offer

    def run(self) -> tuple[ThreadState, GBResponse | None]:
        mechanism_state: GBState = self.mechanism.state  # type: ignore
        history: list[GBState] = self.mechanism.history  # type: ignore
        source = self.negotiator.id
        offer = self.negotiator.propose(mechanism_state)
        self.state.new_offer = offer
        if self.constraint and not self.constraint(
            mechanism_state.threads[source],
            [_.threads[source] for _ in history],
        ):
            self.state.new_offer = offer = None

        if offer is None:
            if self.mechanism._extra_callbacks:
                for n in self.responders:
                    n.on_partner_ended(source)
            self.state.new_responses = dict()
            return (
                self.state,
                self.evaluator.eval(
                    source,
                    self.state,
                    mechanism_state.thread_history(history, source),
                    mechanism_state.base_state,
                )
                if self.evaluator
                else None,
            )
        if self.mechanism._extra_callbacks:
            for n in self.responders:
                n.on_partner_proposal(mechanism_state, source, offer)
        responses = [_.respond(mechanism_state, offer, source) for _ in self.responders]
        if self.mechanism._extra_callbacks:
            for n, r in zip(self.responders, responses):
                n.on_partner_response(mechanism_state, n.id, offer, r)
        self.state.current_offer = (
            offer
            if all(_ == ResponseType.ACCEPT_OFFER for _ in responses)
            else self.current
        )
        self.state.new_responses = dict(
            zip(tuple(_.id for _ in self.responders), responses)
        )

        return (
            self.state,
            self.evaluator.eval(
                source,
                self.state,
                mechanism_state.thread_history(history, source),
                mechanism_state.base_state,
            )
            if self.evaluator
            else None,
        )


class GBMechanism(Mechanism):
    def __init__(
        self,
        *args,
        evaluator_type: type[EvaluationStrategy] | None = None,
        evaluator_params: dict[str, Any] | None = None,
        local_evaluator_type: type[LocalEvaluationStrategy] | None = None,
        local_evaluator_params: dict[str, Any] | None = None,
        constraint_type: type[OfferingConstraint] | None = None,
        constraint_params: dict[str, Any] | None = None,
        local_constraint_type: type[LocalOfferingConstraint] | None = None,
        local_constraint_params: dict[str, Any] | None = None,
        response_combiner: Callable[[list[GBResponse]], GBResponse] = all_accept,
        dynamic_entry=False,
        extra_callbacks=False,
        check_offers=False,
        enforce_issue_types=False,
        cast_offers=False,
        end_on_no_response=True,
        ignore_negotiator_exceptions=False,
        parallel: bool = True,
        sync_calls: bool = False,
        initial_state: GBState | None = None,
        **kwargs,
    ):
        if not evaluator_type and not local_evaluator_type:
            raise ValueError(
                "You must pass either `evaluator_type` or `local_evaluator_type` to GBMechanism"
            )
        super().__init__(
            *args,
            dynamic_entry=dynamic_entry,
            extra_callbacks=extra_callbacks,
            initial_state=GBState() if not initial_state else initial_state,
            **kwargs,
        )
        self._current_state: GBState
        if not constraint_params:
            constraint_params = dict()
        self._global_constraint: OfferingConstraint | None = (
            instantiate(constraint_type, **constraint_params)
            if constraint_type
            else None
        )
        self._local_constraint_type: type[
            LocalOfferingConstraint
        ] | None = local_constraint_type
        self._local_constraint_params = (
            local_constraint_params if local_constraint_params else dict()
        )
        if not evaluator_params:
            evaluator_params = dict()
        self._global_evaluator = (
            instantiate(evaluator_type, **evaluator_params) if evaluator_type else None
        )
        self._local_evaluator_type = local_evaluator_type
        self._local_evaluator_params = (
            local_evaluator_params if local_evaluator_params else dict()
        )
        self._combiner = response_combiner
        self._parallel = parallel
        self._threads: list[GBThread] = []
        self._ignore_negotiator_exceptions = ignore_negotiator_exceptions
        self._dynamic_entry = dynamic_entry
        self._extra_callbacks = extra_callbacks
        self._check_offers = check_offers
        self._enforce_issue_types = enforce_issue_types
        self._cast_offers = cast_offers
        self._end_on_no_response = enforce_issue_types
        self._ignore_negotiator_exceptions = enforce_issue_types
        self._parallel = parallel
        self._sync_calls = sync_calls
        self.params["check_offers"] = check_offers
        self.params["enforce_issue_types"] = enforce_issue_types
        self.params["ignore_negotiator_exceptions"] = ignore_negotiator_exceptions
        self.params["cast_offers"] = cast_offers
        self.params["end_on_no_response"] = end_on_no_response
        self.params["enable_callbacks"] = extra_callbacks
        self.params["parallel"] = parallel
        self.params["sync_calls"] = sync_calls
        self.params["evaluator_type"] = get_full_type_name(evaluator_type)  # type: ignore
        self.params["evaluator_params"] = evaluator_params
        self.params["constraint_type"] = get_full_type_name(constraint_type)  # type: ignore
        self.params["constraint_params"] = constraint_params
        self.params["local_evaluator_type"] = get_full_type_name(local_evaluator_type)  # type: ignore
        self.params["local_evaluator_params"] = local_evaluator_params
        self.params["local_constraint_type"] = get_full_type_name(local_constraint_type)  # type: ignore
        self.params["local_constraint_params"] = local_constraint_params

    def add(
        self,
        negotiator: GBNegotiator,
        *,
        preferences: Preferences | None = None,
        role: str | None = None,
        ufun: BaseUtilityFunction | None = None,
    ) -> bool | None:
        added = super().add(negotiator, preferences=preferences, role=role, ufun=ufun)
        if added:
            for thread in self._threads:
                thread.responders.append(negotiator)
            evaluator, constraint = None, None
            if self._local_evaluator_type:
                evaluator = instantiate(
                    self._local_evaluator_type, **self._local_evaluator_params
                )
            if self._local_constraint_type:
                constraint = instantiate(
                    self._local_constraint_type, **self._local_constraint_params
                )

            thread = GBThread(
                self,
                negotiator,
                responders=(
                    responders := [
                        _ for _ in self.negotiators if id(_) != id(negotiator)
                    ]
                ),
                indx=len(self._threads),
                evaluator=evaluator,
                constraint=constraint,
                state=ThreadState(
                    None,
                    None,
                    {_.id: ResponseType.REJECT_OFFER for _ in responders},
                ),
            )
            self._threads.append(thread)
            self._current_state.threads[negotiator.id] = thread.state

        return added

    def set_sync_call(self, v: bool):
        self._sync_call = v

    def run_threads(self) -> dict[str, tuple[ThreadState, GBResponse | None]]:
        if not self._parallel:
            threads = self._threads
            return {thread.negotiator.id: thread.run() for thread in threads}
        # todo: make this really parallel
        indices = [_ for _ in range(len(self._threads))]
        shuffle(indices)
        threads = [self._threads[_] for _ in indices]
        results = dict()
        for t in threads:
            results[t.negotiator.id] = t.run()
        return results

    def __call__(self, state: GBState) -> MechanismRoundResult:
        # print(f"Round {self._current_state.step}")
        results = self.run_threads()
        # if state.step > self.outcome_space.cardinality:
        #     self.plot()
        #     breakpoint()
        responses = []
        for source, (tstate, response) in results.items():
            state.threads[source] = tstate
            if self._local_evaluator_type:
                responses.append(response)
        if self._global_evaluator:
            responses.append(
                self._global_evaluator(self.negotiator_ids, state, self._history)
            )
        if self._global_constraint:
            responses.append(self._global_constraint(state, self._history))

        e = self._combiner(responses)

        if e == "continue":
            return MechanismRoundResult(state)
        if e is None:
            state.broken = True
        state.agreement = e
        return MechanismRoundResult(state)

    @property
    def full_trace(self) -> list[TraceElement]:
        def response(state: GBState):
            if state.timedout:
                return "timedout"
            if state.ended:
                return "ended"
            if state.has_error:
                return "error"
            if state.agreement:
                return "accepted"
            return "rejected"

        offers = []
        self._history: list[GBState]  # type: ignore
        for state in self._history:
            state: GBState
            offers += [
                TraceElement(
                    state.time,
                    state.relative_time,
                    state.step,
                    source,
                    t.new_offer,
                    {neg: int(resp) for neg, resp in t.new_responses.items()},
                    response(state),
                )
                for source, t in state.threads.items()
            ]

        return offers

    @property
    def extended_trace(self) -> list[tuple[int, str, Outcome]]:
        """Returns the negotiation history as a list of step/negotiator/offer tuples"""
        offers = []
        self._history: list[GBState]  # type: ignore
        for state in self._history:
            offers += [
                (state.step, source, t.new_offer) for source, t in state.threads.items()
            ]
        return offers

    @property
    def trace(self) -> list[tuple[str, Outcome]]:
        """Returns the negotiation history as a list of negotiator/offer tuples"""
        offers = []
        for state in self._history:
            offers += [(source, t.new_offer) for source, t in state.threads.items()]

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
        show_end_reason: bool = True,
        show_annotations: bool = False,
        show_reserved: bool = True,
        show_total_time=True,
        show_relative_time=True,
        show_n_steps=True,
        colors: list | None = None,
        markers: list[str] | None = None,
        colormap: str = "jet",
        ylimits: tuple[float, float] | None = None,
        common_legend: bool = True,
        xdim: str = "step",
        colorizer: Colorizer | None = default_colorizer,
    ):
        from negmas.plots.util import plot_mechanism_run

        return plot_mechanism_run(
            mechanism=self,  # type: ignore
            negotiators=plotting_negotiators,
            save_fig=save_fig,
            path=path,
            fig_name=fig_name,
            ignore_none_offers=ignore_none_offers,
            with_lines=with_lines,
            show_agreement=show_agreement,
            show_pareto_distance=show_pareto_distance,
            show_nash_distance=show_nash_distance,
            show_end_reason=show_end_reason,
            show_annotations=show_annotations,
            show_reserved=show_reserved,
            colors=colors,
            markers=markers,
            colormap=colormap,
            ylimits=ylimits,
            common_legend=common_legend,
            extra_annotation="",
            xdim=xdim,
            colorizer=colorizer,
            show_total_time=show_total_time,
            show_relative_time=show_relative_time,
            show_n_steps=show_n_steps,
        )


class ParallelGBMechanism(GBMechanism):
    def __init__(self, *args, **kwargs):
        kwargs["parallel"] = True
        super().__init__(*args, **kwargs)


class SerialGBMechanism(GBMechanism):
    def __init__(self, *args, **kwargs):
        kwargs["parallel"] = False
        super().__init__(*args, **kwargs)
