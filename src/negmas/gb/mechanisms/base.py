"""Mechanisms base classes."""

from __future__ import annotations

from random import shuffle
from time import perf_counter
from typing import TYPE_CHECKING, Any, Callable

from attrs import define

from negmas.common import TraceElement
from negmas.helpers import humanize_time
from negmas.helpers.types import get_full_type_name, instantiate
from negmas.mechanisms import Mechanism, MechanismStepResult
from negmas.outcomes import Outcome
from negmas.plots.util import default_colorizer
from negmas.preferences import BaseUtilityFunction, Preferences

from ..common import GBAction, GBResponse, GBState, ResponseType, ThreadState, GBNMI
from ..constraints.base import LocalOfferingConstraint, OfferingConstraint
from ..evaluators.base import EvaluationStrategy, LocalEvaluationStrategy, all_accept
from negmas.gb.negotiators.base import GBNegotiator

if TYPE_CHECKING:
    from negmas.plots.util import Colorizer

__all__ = ["GBMechanism", "ParallelGBMechanism", "SerialGBMechanism"]

DEFAULT_COLORMAP = "jet"


@define
class GBThread:
    """GBThread implementation."""

    mechanism: BaseGBMechanism
    negotiator: GBNegotiator
    responders: list[GBNegotiator]
    indx: int
    state: ThreadState
    evaluator: LocalEvaluationStrategy | None = None
    constraint: LocalOfferingConstraint | None = None

    @property
    def accepted_offers(self) -> list[Outcome]:
        """Accepted offers.

        Returns:
            list[Outcome]: The result.
        """
        return self.state.accepted_offers

    def run(
        self, action: dict[str, GBAction] | None = None
    ) -> tuple[ThreadState, GBResponse | None]:
        """Run.

        Args:
            action: Action.

        Returns:
            tuple[ThreadState, GBResponse | None]: The result.
        """
        mechanism_state: GBState = self.mechanism.state  #
        history: list[GBState] = self.mechanism.history  #
        source = self.negotiator.id
        if action is None:
            strt = perf_counter()
            dest = (
                self.mechanism.dest_separator.join([_.id for _ in self.responders])
                if self.mechanism.dest_separator
                else None
            )
            offer = self.negotiator.propose(mechanism_state, dest=dest)
            self.mechanism._negotiator_times[self.negotiator.id] += (
                perf_counter() - strt
            )
        else:
            offer = action.get(source, None)
            if not offer:
                offer = None
        # assert offer is None or isinstance(offer, Outcome)
        self.state.new_offer = offer  # type: ignore
        if self.constraint and not self.constraint(
            mechanism_state.threads[source], [_.threads[source] for _ in history]
        ):
            self.state.new_offer = offer = None

        if offer is None:
            if self.mechanism._extra_callbacks:
                for n in self.responders:
                    strt = perf_counter()
                    n.on_partner_ended(source)
                    self.mechanism._negotiator_times[n.id] += perf_counter() - strt
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
                strt = perf_counter()
                n.on_partner_proposal(mechanism_state, source, offer)  # type: ignore
                self.mechanism._negotiator_times[n.id] += perf_counter() - strt
        responses = []
        for responder in self.responders:
            strt = perf_counter()
            responses.append(responder.respond(mechanism_state, source=source))
            self.mechanism._negotiator_times[responder.id] += perf_counter() - strt
        if self.mechanism._extra_callbacks:
            for n, r in zip(self.responders, responses):
                strt = perf_counter()
                n.on_partner_response(mechanism_state, n.id, offer, r)  # type: ignore
                self.mechanism._negotiator_times[n.id] += perf_counter() - strt
        if all(_ == ResponseType.ACCEPT_OFFER for _ in responses):
            self.state.accepted_offers.append(offer)  # type: ignore
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


class BaseGBMechanism(Mechanism[GBNMI, GBState, GBAction, GBNegotiator]):
    """
    Base for all Generalized Bargaining Mechanisms
    """

    def __init__(
        self,
        *args,
        dynamic_entry=False,
        extra_callbacks=True,
        check_offers=False,
        enforce_issue_types=False,
        cast_offers=False,
        end_on_no_response=True,
        ignore_negotiator_exceptions=False,
        parallel: bool = True,
        sync_calls: bool = False,
        initial_state: GBState | None = None,
        dest_separator: str | None = ";",
        **kwargs,
    ):
        """Initialize the instance.

        Args:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        self.dest_separator = dest_separator
        super().__init__(
            *args,
            dynamic_entry=dynamic_entry,
            extra_callbacks=extra_callbacks,
            initial_state=GBState() if not initial_state else initial_state,
            **kwargs,
        )
        self._current_state: GBState
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

    @property
    def state(self) -> GBState:
        """Returns the current state.

        Override `extra_state` if you want to keep extra state
        """
        return self._current_state

    def set_sync_call(self, v: bool):
        """Set sync call.

        Args:
            v: V.
        """
        self._sync_call = v

    def run_threads(
        self, action: dict[str, GBAction] | None = None
    ) -> dict[str, tuple[ThreadState, GBResponse | None]]:
        """Run threads.

        Args:
            action: Action.

        Returns:
            dict[str, tuple[ThreadState, GBResponse | None]]: The result.
        """

        def _do_run(idd, thread: GBThread):
            if self.verbosity > 2:
                print(
                    f"{self.name}: Thread {thread.negotiator.name} starts after {humanize_time(perf_counter() - self._start_time, show_ms=True) if self._start_time else 0}",
                    flush=True,
                )
            r = thread.run(action)
            state: GBState = self.state
            state.last_thread = idd
            return r

        if not self._parallel:
            threads = self._threads
            return {
                thread.negotiator.id: _do_run(thread.negotiator.id, thread)
                for thread in threads
            }
        # todo: make this really parallel
        indices = [_ for _ in range(len(self._threads))]
        shuffle(indices)
        threads = [self._threads[_] for _ in indices]
        results = dict()
        for t in threads:
            results[t.negotiator.id] = _do_run(t.negotiator.id, t)
        return results

    @property
    def full_trace(self) -> list[TraceElement]:
        """Full trace.

        Returns:
            list[TraceElement]: The result.
        """

        def response(state: GBState):
            """Response.

            Args:
                state: Current state.
            """
            if state.timedout:
                return "timedout"
            if state.ended:
                return "ended"
            if state.has_error:
                return "error"
            if state.agreement:
                return "agreement"
            return "continuing"

        offers = []
        self._history: list[GBState]  #
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
        """Returns the negotiation history as a list of step/negotiator/offer
        tuples."""
        offers = []
        self._history: list[GBState]  #
        for state in self._history:
            offers += [
                (state.step, source, t.new_offer) for source, t in state.threads.items()
            ]
        return offers

    @property
    def trace(self) -> list[tuple[str, Outcome]]:
        """Returns the negotiation history as a list of negotiator/offer
        tuples."""
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
        """Returns the (time/relative-time/step/outcome/response) given by a
        negotiator (in order)"""
        return [
            (t, rt, s, o, a)
            for t, rt, s, n, o, _, a in self.full_trace
            if n == negotiator_id
        ]

    @property
    def offers(self) -> list[Outcome]:
        """Returns the negotiation history as a list of offers."""
        return [o for _, o in self.trace]

    @property
    def _step(self):
        """A private property used by the checkpoint system."""
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
        xdim: str = "step",
        colorizer: Colorizer | None = default_colorizer,
        only2d: bool = False,
        fast=False,
        simple_offers_view=False,
        **kwargs,
    ):
        """Plot.

        Args:
            plotting_negotiators: Plotting negotiators.
            save_fig: Save fig.
            path: Path.
            fig_name: Fig name.
            ignore_none_offers: Ignore none offers.
            with_lines: With lines.
            show_agreement: Show agreement.
            show_pareto_distance: Show pareto distance.
            show_nash_distance: Show nash distance.
            show_kalai_distance: Show kalai distance.
            show_max_welfare_distance: Show max welfare distance.
            show_max_relative_welfare_distance: Show max relative welfare distance.
            show_end_reason: Show end reason.
            show_annotations: Show annotations.
            show_reserved: Show reserved.
            show_total_time: Show total time.
            show_relative_time: Show relative time.
            show_n_steps: Show n steps.
            colors: Colors.
            markers: Markers.
            colormap: Colormap.
            ylimits: Ylimits.
            common_legend: Common legend.
            xdim: Xdim.
            colorizer: Colorizer.
            only2d: Only2d.
            fast: Fast.
            simple_offers_view: Simple offers view.
            **kwargs: Additional keyword arguments.
        """
        from negmas.plots.util import plot_mechanism_run

        return plot_mechanism_run(
            mechanism=self,
            negotiators=plotting_negotiators,
            save_fig=save_fig,
            path=path,
            fig_name=fig_name,
            ignore_none_offers=ignore_none_offers,
            with_lines=with_lines,
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
            extra_annotation="",
            xdim=xdim,
            colorizer=colorizer,
            show_total_time=show_total_time,
            show_relative_time=show_relative_time,
            show_n_steps=show_n_steps,
            only2d=only2d,
            fast=fast,
            simple_offers_view=simple_offers_view,
            **kwargs,
        )

    def add(  #
        self,
        negotiator: GBNegotiator,
        *,
        preferences: Preferences | None = None,
        role: str | None = None,
        ufun: BaseUtilityFunction | None = None,
    ) -> bool | None:
        """Add.

        Args:
            negotiator: Negotiator.

        Returns:
            bool | None: The result.
        """
        added = super().add(negotiator, preferences=preferences, role=role, ufun=ufun)
        if added:
            for thread in self._threads:
                thread.responders.append(negotiator)

            thread = GBThread(
                self,  #
                negotiator,
                responders=(
                    responders := [
                        _ for _ in self.negotiators if id(_) != id(negotiator)
                    ]
                ),
                indx=len(self._threads),
                evaluator=None,
                constraint=None,
                state=ThreadState(
                    new_offer=None,
                    new_responses={_.id: ResponseType.REJECT_OFFER for _ in responders},
                    accepted_offers=[],
                ),
            )
            self._threads.append(thread)
            self._current_state.threads[negotiator.id] = thread.state

        return added


class GBMechanism(BaseGBMechanism):
    """GB mechanism."""

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
        """Initialize the instance.

        Args:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
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
        self.params["evaluator_type"] = get_full_type_name(evaluator_type)  #
        self.params["evaluator_params"] = evaluator_params
        self.params["constraint_type"] = get_full_type_name(constraint_type)  #
        self.params["constraint_params"] = constraint_params
        self.params["local_evaluator_type"] = get_full_type_name(
            local_evaluator_type
        )  #
        self.params["local_evaluator_params"] = local_evaluator_params
        self.params["local_constraint_type"] = get_full_type_name(
            local_constraint_type
        )  #
        self.params["local_constraint_params"] = local_constraint_params

    def add(
        self,
        negotiator: GBNegotiator,
        *,
        preferences: Preferences | None = None,
        role: str | None = None,
        ufun: BaseUtilityFunction | None = None,
    ) -> bool | None:
        """Add.

        Args:
            negotiator: Negotiator.

        Returns:
            bool | None: The result.
        """
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
                    new_offer=None,
                    new_responses={_.id: ResponseType.REJECT_OFFER for _ in responders},
                    accepted_offers=[],
                ),
            )
            self._threads.append(thread)
            self._current_state.threads[negotiator.id] = thread.state

        return added

    def set_sync_call(self, v: bool):
        """Set sync call.

        Args:
            v: V.
        """
        self._sync_call = v

    def __call__(  #
        self, state: GBState, action: dict[str, GBAction] | None = None
    ) -> MechanismStepResult:
        # print(f"Round {self._current_state.step}")
        """Make instance callable.

        Args:
            state: Current state.
            action: Action.

        Returns:
            MechanismStepResult: The result.
        """
        results = self.run_threads(action)
        # if state.step > self.outcome_space.cardinality:
        #     self.plot()
        #     breakpoint()
        if self.verbosity > 2:
            print(
                f"{self.name}: Central Processing after {humanize_time(perf_counter() - self._start_time, show_ms=True) if self._start_time else 0}",
                flush=True,
            )
        responses = []
        for source, (tstate, response) in results.items():
            state.threads[source] = tstate
            if self._local_evaluator_type:
                responses.append(response)
        if self.verbosity > 2:
            print(
                f"{self.name}: Global Evaluator after {humanize_time(perf_counter() - self._start_time, show_ms=True) if self._start_time else 0}",
                flush=True,
            )
        if self._global_evaluator:
            responses.append(
                self._global_evaluator(self.negotiator_ids, state, self._history)
            )
        if self.verbosity > 2:
            print(
                f"{self.name}: Global Constraint after {humanize_time(perf_counter() - self._start_time, show_ms=True) if self._start_time else 0}",
                flush=True,
            )
        if self._global_constraint:
            responses.append(self._global_constraint(state, self._history))  # type: ignore

        e = self._combiner(responses)

        if e == "continue":
            return MechanismStepResult(state)
        if e is None:
            state.broken = True
        state.agreement = e
        return MechanismStepResult(state)

    @property
    def full_trace(self) -> list[TraceElement]:
        """Full trace.

        Returns:
            list[TraceElement]: The result.
        """

        def response(state: GBState):
            """Response.

            Args:
                state: Current state.
            """
            if state.timedout:
                return "timedout"
            if state.ended:
                return "ended"
            if state.has_error:
                return "error"
            if state.agreement:
                return "agreement"
            return "continuing"

        offers = []
        self._history: list[GBState]  #
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
        """Returns the negotiation history as a list of step/negotiator/offer
        tuples."""
        offers = []
        self._history: list[GBState]  #
        for state in self._history:
            offers += [
                (state.step, source, t.new_offer) for source, t in state.threads.items()
            ]
        return offers

    @property
    def trace(self) -> list[tuple[str, Outcome]]:
        """Returns the negotiation history as a list of negotiator/offer
        tuples."""
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
        """Returns the (time/relative-time/step/outcome/response) given by a
        negotiator (in order)"""
        return [
            (t, rt, s, o, a)
            for t, rt, s, n, o, _, a in self.full_trace
            if n == negotiator_id
        ]

    @property
    def offers(self) -> list[Outcome]:
        """Returns the negotiation history as a list of offers."""
        return [o for _, o in self.trace]

    @property
    def _step(self):
        """A private property used by the checkpoint system."""
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
        xdim: str = "step",
        colorizer: Colorizer | None = default_colorizer,
        only2d: bool = False,
        fast=False,
        simple_offers_view=False,
        **kwargs,
    ):
        """Plot.

        Args:
            plotting_negotiators: Plotting negotiators.
            save_fig: Save fig.
            path: Path.
            fig_name: Fig name.
            ignore_none_offers: Ignore none offers.
            with_lines: With lines.
            show_agreement: Show agreement.
            show_pareto_distance: Show pareto distance.
            show_nash_distance: Show nash distance.
            show_kalai_distance: Show kalai distance.
            show_max_welfare_distance: Show max welfare distance.
            show_max_relative_welfare_distance: Show max relative welfare distance.
            show_end_reason: Show end reason.
            show_annotations: Show annotations.
            show_reserved: Show reserved.
            show_total_time: Show total time.
            show_relative_time: Show relative time.
            show_n_steps: Show n steps.
            colors: Colors.
            markers: Markers.
            colormap: Colormap.
            ylimits: Ylimits.
            common_legend: Common legend.
            xdim: Xdim.
            colorizer: Colorizer.
            only2d: Only2d.
            fast: Fast.
            simple_offers_view: Simple offers view.
            **kwargs: Additional keyword arguments.
        """
        from negmas.plots.util import plot_mechanism_run

        return plot_mechanism_run(
            mechanism=self,  #
            negotiators=plotting_negotiators,
            save_fig=save_fig,
            path=path,
            fig_name=fig_name,
            ignore_none_offers=ignore_none_offers,
            with_lines=with_lines,
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
            extra_annotation="",
            xdim=xdim,
            colorizer=colorizer,
            show_total_time=show_total_time,
            show_relative_time=show_relative_time,
            show_n_steps=show_n_steps,
            only2d=only2d,
            fast=fast,
            simple_offers_view=simple_offers_view,
            **kwargs,
        )


class ParallelGBMechanism(GBMechanism):
    """ParallelGB mechanism."""

    def __init__(self, *args, **kwargs):
        """Initialize the instance.

        Args:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        kwargs["parallel"] = True
        super().__init__(*args, **kwargs)


class SerialGBMechanism(GBMechanism):
    """SerialGB mechanism."""

    def __init__(self, *args, **kwargs):
        """Initialize the instance.

        Args:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        kwargs["parallel"] = False
        super().__init__(*args, **kwargs)
