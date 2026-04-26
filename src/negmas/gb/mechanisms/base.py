"""Mechanisms base classes."""

from __future__ import annotations

from random import shuffle
from time import perf_counter, perf_counter_ns
from typing import TYPE_CHECKING, Any, Callable

from attrs import define

from negmas.common import TraceElement
from negmas.helpers import humanize_time
from negmas.helpers.types import get_full_type_name, instantiate
from negmas.mechanisms import Mechanism, MechanismStepResult
from negmas.outcomes import Outcome
from negmas.outcomes.common import ExtendedOutcome, extract_data, extract_outcome
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
        """Returns the list of offers that have been accepted in this thread."""
        return self.state.accepted_offers

    def run(
        self, action: dict[str, GBAction] | None = None
    ) -> tuple[ThreadState, GBResponse | None]:
        """Execute one round of the negotiation thread.

        Args:
            action: Optional mapping of negotiator IDs to their pre-specified actions.
                    If None, the thread's negotiator proposes and responders respond.

        Returns:
            A tuple of the updated thread state and the optional evaluation response.
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
        self.state.new_offer = extract_outcome(offer)
        self.state.new_data = extract_data(offer)
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
        """Create a new GB mechanism with the specified configuration.

        Args:
            dynamic_entry: Whether negotiators can join after the mechanism starts.
            extra_callbacks: Whether to call additional callbacks on partner actions.
            check_offers: Whether to validate offers against the outcome space.
            enforce_issue_types: Whether to enforce correct types for issue values.
            cast_offers: Whether to cast offer values to expected types.
            end_on_no_response: Whether to end negotiation when a negotiator stops responding.
            ignore_negotiator_exceptions: Whether to ignore exceptions from negotiators.
            parallel: Whether to run threads in parallel (randomized order) or serial.
            sync_calls: Whether to use synchronous callback execution.
            initial_state: Optional initial state for the mechanism.
            dest_separator: Separator for combining destination negotiator IDs, or None to disable.
            *args: Additional positional arguments passed to the parent class.
            **kwargs: Additional keyword arguments passed to the parent class.
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
        """Enable or disable synchronous callback execution."""
        self._sync_call = v

    def run_threads(
        self, action: dict[str, GBAction] | None = None
    ) -> dict[str, tuple[ThreadState, GBResponse | None]]:
        """Execute all negotiation threads for one round.

        Args:
            action: Optional mapping of negotiator IDs to their pre-specified actions.
                    If None, each thread's negotiator proposes independently.

        Returns:
            A mapping from negotiator IDs to their thread state and evaluation response.
        """

        def _do_run(idd, thread: GBThread):
            if self.verbosity > 2:
                print(
                    f"{self.name}: Thread {thread.negotiator.name} starts after {humanize_time((perf_counter_ns() - self._start_time) / 1_000_000_000, show_ms=True) if self._start_time else 0}",
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
        """Returns the complete negotiation history with timing, offers, responses, and metadata."""

        def response(state: GBState):
            """Determine the response status string from the given state."""
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
                    t.new_data.get("text", None) if t.new_data else None,
                    t.new_data,
                )
                for source, t in state.threads.items()
            ]

        return offers

    @property
    def extended_trace(self) -> list[tuple[int, str, Outcome | None]]:
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
    def trace(self) -> list[tuple[str, Outcome | None]]:
        """Returns the negotiation history as a list of negotiator/offer
        tuples."""
        offers = []
        for state in self._history:
            offers += [(source, t.new_offer) for source, t in state.threads.items()]

        return offers

    def negotiator_offers(
        self, negotiator_id: str
    ) -> list[Outcome | ExtendedOutcome | None]:
        """Returns the offers given by a negotiator (in order)"""
        return [o for n, o in self.trace if n == negotiator_id]

    def negotiator_full_trace(
        self, negotiator_id: str
    ) -> list[
        tuple[float, float, int, Outcome, str, str | None, dict[str, Any] | None]
    ]:
        """Returns the (time/relative-time/step/outcome/response) given by a
        negotiator (in order)"""
        return [
            (t, rt, s, o, a, text, data)
            for t, rt, s, n, o, _, a, text, data in self.full_trace
            if n == negotiator_id
        ]

    @property
    def offers(self) -> list[Outcome | None]:
        """Returns the negotiation history as a list of offers."""
        return [extract_outcome(o) for _, o in self.trace]

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
        colorizer: Colorizer | None = None,
        only2d: bool = False,
        fast=False,
        simple_offers_view=False,
        **kwargs,
    ):
        """Visualize the negotiation session showing offers, utilities, and outcome metrics.

        Args:
            plotting_negotiators: Indices or IDs of the two negotiators to plot.
            save_fig: Whether to save the figure to disk.
            path: Directory path for saving the figure.
            fig_name: Filename for the saved figure.
            ignore_none_offers: Whether to skip None offers in the plot.
            with_lines: Whether to connect offer points with lines.
            show_agreement: Whether to highlight the final agreement point.
            show_pareto_distance: Whether to display distance to Pareto frontier.
            show_nash_distance: Whether to display distance to Nash solution.
            show_kalai_distance: Whether to display distance to Kalai-Smorodinsky solution.
            show_max_welfare_distance: Whether to display distance to max welfare point.
            show_max_relative_welfare_distance: Whether to display distance to max relative welfare.
            show_end_reason: Whether to annotate the reason for negotiation end.
            show_annotations: Whether to show offer annotations on the plot.
            show_reserved: Whether to show reserved value lines.
            show_total_time: Whether to display total elapsed time.
            show_relative_time: Whether to display relative time progress.
            show_n_steps: Whether to display the number of negotiation steps.
            colors: Custom color sequence for plotting negotiators.
            markers: Custom marker styles for each negotiator.
            colormap: Matplotlib colormap name for gradient coloring.
            ylimits: Y-axis limits as (min, max) tuple.
            common_legend: Whether to use a shared legend for all subplots.
            xdim: X-axis dimension, either "step" or "time".
            colorizer: Custom function for determining point colors.
            only2d: Whether to show only the 2D utility plot.
            fast: Whether to use fast rendering (less detail).
            simple_offers_view: Whether to use simplified offer visualization.
            **kwargs: Additional arguments passed to the plotting function.
        """
        from negmas.plots.util import default_colorizer, plot_mechanism_run

        if colorizer is None:
            colorizer = default_colorizer

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
        """Add a negotiator to the mechanism and create its corresponding thread.

        Args:
            negotiator: The negotiator instance to add to this mechanism.
            preferences: Optional preferences to assign to the negotiator.
            role: Optional role identifier for the negotiator.
            ufun: Optional utility function to assign to the negotiator.

        Returns:
            True if successfully added, False if rejected, None if already present.
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
    """Generalized Bargaining (GB) mechanism.

    Implements the Generalized Bargaining Protocol framework for automated negotiation.
    This mechanism supports configurable evaluation strategies and offering constraints
    that can be applied globally or per-thread.

    References:
        Mohammad, Y. (2023). Generalized Bargaining Protocols.
        In: Australasian Joint Conference on Artificial Intelligence (AI 2023).
        Springer. https://doi.org/10.1007/978-981-99-8391-9_37
    """

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
        allow_negotiators_to_leave: bool = True,
        **kwargs,
    ):
        """Create a GB mechanism with evaluation strategies and offering constraints.

        Args:
            evaluator_type: Global evaluation strategy class for determining negotiation outcomes.
            evaluator_params: Parameters to pass when instantiating the global evaluator.
            local_evaluator_type: Per-thread evaluation strategy class for local decisions.
            local_evaluator_params: Parameters to pass when instantiating local evaluators.
            constraint_type: Global constraint class limiting valid offers.
            constraint_params: Parameters to pass when instantiating the global constraint.
            local_constraint_type: Per-thread constraint class limiting valid offers.
            local_constraint_params: Parameters to pass when instantiating local constraints.
            response_combiner: Function to combine multiple thread responses into a final response.
            dynamic_entry: Whether negotiators can join after the mechanism starts.
            extra_callbacks: Whether to call additional callbacks on partner actions.
            check_offers: Whether to validate offers against the outcome space.
            enforce_issue_types: Whether to enforce correct types for issue values.
            cast_offers: Whether to cast offer values to expected types.
            end_on_no_response: Whether to end negotiation when a negotiator stops responding.
            ignore_negotiator_exceptions: Whether to ignore exceptions from negotiators.
            parallel: Whether to run threads in parallel (randomized order) or serial.
            sync_calls: Whether to use synchronous callback execution.
            initial_state: Optional initial state for the mechanism.
            allow_negotiators_to_leave: If True (default), LEAVE marks the negotiator as left
                and allows remaining negotiators to continue. The negotiation ends when fewer
                than 2 negotiators remain (unless dynamic_entry is True).
                If False, LEAVE is treated exactly like END_NEGOTIATION.
            *args: Additional positional arguments passed to the parent class.
            **kwargs: Additional keyword arguments passed to the parent class.
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
        self._local_constraint_type: type[LocalOfferingConstraint] | None = (
            local_constraint_type
        )
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
        self.params["allow_negotiators_to_leave"] = allow_negotiators_to_leave
        self.allow_negotiators_to_leave = allow_negotiators_to_leave

    def add(
        self,
        negotiator: GBNegotiator,
        *,
        preferences: Preferences | None = None,
        role: str | None = None,
        ufun: BaseUtilityFunction | None = None,
    ) -> bool | None:
        """Add a negotiator to the mechanism with its evaluator and constraint.

        Args:
            negotiator: The negotiator instance to add to this mechanism.
            preferences: Optional preferences to assign to the negotiator.
            role: Optional role identifier for the negotiator.
            ufun: Optional utility function to assign to the negotiator.

        Returns:
            True if successfully added, False if rejected, None if already present.
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

    @property
    def participating_negotiators(self) -> list[GBNegotiator]:
        """Returns negotiators still participating (those who haven't left).

        Unlike `negotiators`, this excludes any negotiator that has returned
        a LEAVE response. The original negotiator list and indices remain
        unchanged to avoid breaking any saved references.

        Returns:
            List of negotiator objects still active in the negotiation.
        """
        left = self._current_state.left_negotiators
        return [n for n in self._negotiators if n.id not in left]

    @property
    def agreement_partners(self) -> list[GBNegotiator]:
        """Returns negotiators who are part of the final agreement.

        This is the same as `participating_negotiators` - it includes all
        negotiators who haven't left the negotiation.

        Returns:
            List of negotiator objects who participated in the final outcome.
        """
        return self.participating_negotiators

    @property
    def n_participating(self) -> int:
        """Number of negotiators still participating (not left).

        Returns:
            Count of active negotiators.
        """
        return len(self._negotiators) - len(self._current_state.left_negotiators)

    def _remove_negotiator_on_leave(self, negotiator: GBNegotiator) -> bool:
        """Mark a negotiator as having left and notify others.

        Instead of removing the negotiator from the list (which would break
        indices and references), we mark them as left in the state. The
        negotiator remains in `negotiators` but is excluded from
        `participating_negotiators` and `agreement_partners`.

        Args:
            negotiator: The negotiator leaving the mechanism.

        Returns:
            True if successfully marked as left, False if already left or not found.
        """
        neg_id = negotiator.id
        # Check if negotiator exists and hasn't already left
        if neg_id not in self._negotiator_map:
            return False
        if neg_id in self._current_state.left_negotiators:
            return False  # Already left

        # Mark as left in the state (don't remove from lists)
        self._current_state.left_negotiators.add(neg_id)

        # Call on_leave for the leaving negotiator
        if self._extra_callbacks:
            negotiator.on_leave(self.state)

        # Notify remaining negotiators that this one left
        if self._extra_callbacks:
            for other in self.participating_negotiators:
                if hasattr(other, "on_negotiator_left"):
                    other.on_negotiator_left(neg_id, self.state)

        return True

    def set_sync_call(self, v: bool):
        """Enable or disable synchronous callback execution."""
        self._sync_call = v

    def __call__(  #
        self, state: GBState, action: dict[str, GBAction] | None = None
    ) -> MechanismStepResult:
        # print(f"Round {self._current_state.step}")
        """Execute one step of the mechanism, processing all threads and evaluating results.

        Args:
            state: The current mechanism state containing thread states and history.
            action: Optional mapping of negotiator IDs to pre-specified actions.

        Returns:
            The step result containing updated state and any agreement reached.
        """
        results = self.run_threads(action)
        # if state.step > self.outcome_space.cardinality:
        #     self.plot()
        #     breakpoint()
        if self.verbosity > 2:
            print(
                f"{self.name}: Central Processing after {humanize_time((perf_counter_ns() - self._start_time) / 1_000_000_000, show_ms=True) if self._start_time else 0}",
                flush=True,
            )

        # Handle LEAVE responses from threads
        negotiators_to_mark_left = []
        for source, (tstate, response) in results.items():
            state.threads[source] = tstate
            # Check for LEAVE responses in thread responses
            for responder_id, resp in tstate.new_responses.items():
                if resp == ResponseType.LEAVE:
                    if not self.allow_negotiators_to_leave:
                        # Treat LEAVE exactly like END_NEGOTIATION
                        state.broken = True
                        return MechanismStepResult(state)
                    else:
                        # Mark the responder for leaving
                        negotiators_to_mark_left.append(responder_id)

        # Mark negotiators who chose to leave
        for neg_id in set(negotiators_to_mark_left):  # Use set to avoid duplicates
            neg = self._negotiator_map.get(neg_id)
            if neg:
                self._remove_negotiator_on_leave(neg)

        # Check if we have enough participating negotiators to continue
        n_participating = self.n_participating
        if n_participating < 2 and not self.dynamic_entry:
            # End negotiation - too few negotiators remain and no new ones can join
            state.broken = True
            return MechanismStepResult(state)

        responses = []
        for source, (tstate, response) in results.items():
            if source in state.threads:  # Only process if negotiator is still present
                if self._local_evaluator_type:
                    responses.append(response)
        if self.verbosity > 2:
            print(
                f"{self.name}: Global Evaluator after {humanize_time((perf_counter_ns() - self._start_time) / 1_000_000_000, show_ms=True) if self._start_time else 0}",
                flush=True,
            )
        if self._global_evaluator:
            responses.append(
                self._global_evaluator(self.negotiator_ids, state, self._history)
            )
        if self.verbosity > 2:
            print(
                f"{self.name}: Global Constraint after {humanize_time((perf_counter_ns() - self._start_time) / 1_000_000_000, show_ms=True) if self._start_time else 0}",
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
        """Returns the complete negotiation history with timing, offers, responses, and metadata."""

        def response(state: GBState):
            """Determine the response status string from the given state."""
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
                    t.new_data.get("text", None) if t.new_data else None,
                    t.new_data,
                )
                for source, t in state.threads.items()
            ]

        return offers

    @property
    def extended_trace(self) -> list[tuple[int, str, Outcome | None]]:
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
    def trace(self) -> list[tuple[str, Outcome | None]]:
        """Returns the negotiation history as a list of negotiator/offer
        tuples."""
        offers = []
        for state in self._history:
            offers += [(source, t.new_offer) for source, t in state.threads.items()]

        return offers

    def negotiator_offers(self, negotiator_id: str) -> list[Outcome | None]:
        """Returns the offers given by a negotiator (in order)"""
        return [o for n, o in self.trace if n == negotiator_id]

    def negotiator_full_trace(
        self, negotiator_id: str
    ) -> list[
        tuple[float, float, int, Outcome, str, str | None, dict[str, Any] | None]
    ]:
        """Returns the (time/relative-time/step/outcome/response) given by a
        negotiator (in order)"""
        return [
            (t, rt, s, o, a, text, data)
            for t, rt, s, n, o, _, a, text, data in self.full_trace
            if n == negotiator_id
        ]

    @property
    def offers(self) -> list[Outcome | None]:
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
        colorizer: Colorizer | None = None,
        only2d: bool = False,
        fast=False,
        simple_offers_view=False,
        **kwargs,
    ):
        """Visualize the negotiation session showing offers, utilities, and outcome metrics.

        Args:
            plotting_negotiators: Indices or IDs of the two negotiators to plot.
            save_fig: Whether to save the figure to disk.
            path: Directory path for saving the figure.
            fig_name: Filename for the saved figure.
            ignore_none_offers: Whether to skip None offers in the plot.
            with_lines: Whether to connect offer points with lines.
            show_agreement: Whether to highlight the final agreement point.
            show_pareto_distance: Whether to display distance to Pareto frontier.
            show_nash_distance: Whether to display distance to Nash solution.
            show_kalai_distance: Whether to display distance to Kalai-Smorodinsky solution.
            show_max_welfare_distance: Whether to display distance to max welfare point.
            show_max_relative_welfare_distance: Whether to display distance to max relative welfare.
            show_end_reason: Whether to annotate the reason for negotiation end.
            show_annotations: Whether to show offer annotations on the plot.
            show_reserved: Whether to show reserved value lines.
            show_total_time: Whether to display total elapsed time.
            show_relative_time: Whether to display relative time progress.
            show_n_steps: Whether to display the number of negotiation steps.
            colors: Custom color sequence for plotting negotiators.
            markers: Custom marker styles for each negotiator.
            colormap: Matplotlib colormap name for gradient coloring.
            ylimits: Y-axis limits as (min, max) tuple.
            common_legend: Whether to use a shared legend for all subplots.
            xdim: X-axis dimension, either "step" or "time".
            colorizer: Custom function for determining point colors.
            only2d: Whether to show only the 2D utility plot.
            fast: Whether to use fast rendering (less detail).
            simple_offers_view: Whether to use simplified offer visualization.
            **kwargs: Additional arguments passed to the plotting function.
        """
        from negmas.plots.util import default_colorizer, plot_mechanism_run

        if colorizer is None:
            colorizer = default_colorizer

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
        """Initializes the instance."""
        kwargs["parallel"] = True
        super().__init__(*args, **kwargs)


class SerialGBMechanism(GBMechanism):
    """SerialGB mechanism."""

    def __init__(self, *args, **kwargs):
        """Initializes the instance."""
        kwargs["parallel"] = False
        super().__init__(*args, **kwargs)
