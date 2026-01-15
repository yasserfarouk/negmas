"""
Implements single text negotiation mechanisms.
"""

from __future__ import annotations
import math
import random
import time
from copy import deepcopy

from attrs import define

from negmas.common import NegotiatorMechanismInterface, MechanismAction
from negmas.negotiators.simple import BinaryComparatorNegotiator

from .mechanisms import Mechanism, MechanismState, MechanismStepResult
from .outcomes import Outcome

__all__ = ["VetoSTMechanism", "HillClimbingSTMechanism"]


@define
class STState(MechanismState):
    """Defines extra values to keep in the mechanism state. This is accessible to all negotiators"""

    current_offer: Outcome | None = None
    new_offer: Outcome | None = None


class VetoSTMechanism(
    Mechanism[
        NegotiatorMechanismInterface,
        STState,
        MechanismAction,
        BinaryComparatorNegotiator,
    ]
):
    """Base class for all single text mechanisms

    Args:
        *args: positional arguments to be passed to the base Mechanism
        **kwargs: keyword arguments to be passed to the base Mechanism
        initial_outcome: initial outcome. If None, it will be selected by `next_outcome` which by default will choose it
                         randomly.
        initial_responses: Initial set of responses.

    Remarks:

        - initial_responses is only of value when the number of negotiators that will join the negotiation is less then
          or equal to its length. By default it is not used for anything. Nevertheless, it is here because
          `next_outcome` may decide to use it with the `initial_outcome`


    """

    def __init__(
        self,
        *args,
        epsilon: float = 1e-6,
        initial_outcome=None,
        initial_responses: tuple[bool] = tuple(),
        initial_state: STState | None = None,
        **kwargs,
    ):
        """Initialize the instance.

        Args:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self._current_state = initial_state if initial_state else STState()
        state = self._current_state

        self.add_requirements(
            {"compare-binary": True}
        )  # assert that all agents must have compare-binary capability
        state.current_offer = initial_outcome
        """The current offer"""
        self.initial_outcome = deepcopy(initial_outcome)
        """The initial offer"""
        self.last_responses = list(initial_responses)
        """The responses of all negotiators for the last offer"""
        self.initial_responses = deepcopy(self.last_responses)
        """The initial set of responses. See the remarks of this class to understand its role."""
        self.epsilon = epsilon
        state.new_offer = initial_outcome
        """The new offer generated in this step"""

    def next_outcome(self, outcome: Outcome | None) -> Outcome | None:
        """Generate the next outcome given some outcome.

        Args:
             outcome: The current outcome

        Returns:
            a new outcome or None to end the mechanism run

        """
        return self.random_outcomes(1)[0]

    def __call__(self, state: STState, action=None) -> MechanismStepResult:
        """Single round of the protocol"""

        new_offer = self.next_outcome(state.current_offer)
        responses = []

        for neg in self.negotiators:
            strt = time.perf_counter()
            responses.append(neg.is_better(new_offer, state.current_offer))
            if time.perf_counter() - strt > self.nmi.step_time_limit:
                state.timedout = True
                return MechanismStepResult(state)

        self.last_responses = responses
        state.new_offer = new_offer
        if all(responses):
            state.current_offer = new_offer

        return MechanismStepResult(state)

    def on_negotiation_end(self) -> None:
        """Used to pass the final offer for agreement between all negotiators"""

        state: STState = self._current_state  # type: ignore
        if state.current_offer is not None and all(
            neg.is_acceptable_as_agreement(state.current_offer)
            for neg in self.negotiators
        ):
            state.agreement = state.current_offer

        super().on_negotiation_end()

    def plot(
        self,
        visible_negotiators: tuple[int, int] | tuple[str, str] = (0, 1),
        show_all_offers=False,
        show: bool = True,
        **kwargs,
    ):
        """Plot.

        Args:
            visible_negotiators: Visible negotiators.
            show_all_offers: Show all offers.
            show: Whether to display the figure immediately.
            **kwargs: Additional keyword arguments.

        Returns:
            The plotly Figure object.
        """
        import pandas as pd
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        if len(self.negotiators) < 2:
            print("Cannot visualize negotiations with more less than 2 negotiators")
            return None
        if len(visible_negotiators) > 2:
            print("Cannot visualize more than 2 agents")
            return None
        vnegs = [
            self.negotiators[_] if isinstance(_, int) else self.get_negotiator_raise(_)
            for _ in visible_negotiators
        ]
        assert all(_ is not None and _.ufun is not None for _ in vnegs)
        # indx = dict(zip([_.id for _ in self.negotiators], range(len(self.negotiators))))
        history = []
        for state in self.history:  # type: ignore
            state: STState
            offer = state.new_offer if show_all_offers else state.current_offer
            history.append(
                {
                    "current_offer": offer,
                    "relative_time": state.relative_time,
                    "step": state.step,
                    "u0": vnegs[0].ufun(offer) if vnegs[0].ufun is not None else 1 / 0,
                    "u1": vnegs[1].ufun(offer) if vnegs[1].ufun is not None else 1 / 0,
                }
            )
        history = pd.DataFrame(data=history)
        has_history = len(history) > 0
        has_front = True
        # n_negotiators = len(self.negotiators)
        n_agents = len(vnegs)
        ufuns = self._get_preferences()
        outcomes = self.outcomes
        assert outcomes is not None
        utils = [tuple(f(o) for f in ufuns) for o in outcomes]
        agent_names = [a.name for a in vnegs]
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
        # frontier_outcome_indices = [outcomes.index(_) for _ in frontier_outcome]

        # Create subplot layout: n_agents rows, 2 columns (front + utility plots)
        # Column 1: Pareto frontier (spans all rows)
        # Column 2: Individual utility plots per agent
        fig = make_subplots(
            rows=n_agents,
            cols=2,
            column_widths=[0.5, 0.5],
            specs=[
                [{"rowspan": n_agents}, {}] if i == 0 else [None, {}]
                for i in range(n_agents)
            ],
            subplot_titles=["Utility Space"]
            + [f"{agent_names[a]} Utility" for a in range(n_agents)],
        )

        # Plot utility over time for each agent (right column)
        for a in range(n_agents):
            if has_history:
                h = history.loc[:, ["step", "current_offer", "u0", "u1"]]
                h["utility"] = h[f"u{a}"]
                fig.add_trace(
                    go.Scatter(
                        x=h["step"],
                        y=h["utility"],
                        mode="lines",
                        name=f"{agent_names[a]} utility over time",
                        showlegend=False,
                    ),
                    row=a + 1,
                    col=2,
                )
            fig.update_yaxes(range=[0.0, 1.0], title_text="Utility", row=a + 1, col=2)
            fig.update_xaxes(title_text="Step", row=a + 1, col=2)

        # Plot Pareto frontier and utility space (left column, spanning all rows)
        if has_front:
            # All outcomes
            fig.add_trace(
                go.Scatter(
                    x=[_[0] for _ in utils],
                    y=[_[1] for _ in utils],
                    mode="markers",
                    name="Outcomes",
                    marker=dict(color="gray", symbol="square", size=8),
                ),
                row=1,
                col=1,
            )

            # Pareto frontier
            f1, f2 = [_[0] for _ in frontier], [_[1] for _ in frontier]
            fig.add_trace(
                go.Scatter(
                    x=f1,
                    y=f2,
                    mode="markers",
                    name="Frontier",
                    marker=dict(color="red", symbol="x", size=10),
                ),
                row=1,
                col=1,
            )

            fig.update_xaxes(title_text=f"{agent_names[0]} utility", row=1, col=1)
            fig.update_yaxes(title_text=f"{agent_names[1]} utility", row=1, col=1)

            if self.agreement is not None:
                pareto_distance = 1e9
                cu = (ufuns[0](self.agreement), ufuns[1](self.agreement))
                for pu in frontier:
                    dist = math.sqrt((pu[0] - cu[0]) ** 2 + (pu[1] - cu[1]) ** 2)
                    if dist < pareto_distance:
                        pareto_distance = dist
                fig.add_annotation(
                    x=0.05,
                    y=0.05,
                    xref="x domain",
                    yref="y domain",
                    text=f"Pareto-distance={pareto_distance:5.2f}",
                    showarrow=False,
                    xanchor="left",
                    yanchor="bottom",
                    row=1,
                    col=1,
                )

            if has_history:
                h = history.loc[:, ["step", "current_offer", "u0", "u1"]]
                fig.add_trace(
                    go.Scatter(
                        x=h["u0"],
                        y=h["u1"],
                        mode="markers",
                        name="Mediator's Offer",
                        marker=dict(color="green", size=8),
                    ),
                    row=1,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=[frontier[0][0]],
                        y=[frontier[0][1]],
                        mode="markers",
                        name="Max Welfare",
                        marker=dict(color="blue", size=10),
                    ),
                    row=1,
                    col=1,
                )

            if self.state.agreement is not None:
                fig.add_trace(
                    go.Scatter(
                        x=[ufuns[0](self.state.agreement)],
                        y=[ufuns[1](self.state.agreement)],
                        mode="markers",
                        name="Agreement",
                        marker=dict(color="black", symbol="star", size=14),
                    ),
                    row=1,
                    col=1,
                )

        fig.update_layout(
            title="Single Text Negotiation", showlegend=True, height=300 * n_agents
        )

        if show:
            fig.show()
            return None

        return fig

    @property
    def current_offer(self):
        """Returns the current offer being considered in the negotiation."""
        return self._current_state.current_offer


class HillClimbingSTMechanism(VetoSTMechanism):
    """A single text mechanism that use hill climbing

    Args:
        *args: positional arguments to be passed to the base Mechanism
        **kwargs: keyword arguments to be passed to the base Mechanism
    """

    def neighbors(self, outcome: Outcome) -> list[Outcome]:
        """Returns all neighbors

        Neighbor is an outcome that differs any one of the issues from the original outcome.
        """

        neighbors = []
        for i, issue in enumerate(self.issues):
            values = []
            if isinstance(issue.values, list):
                values = issue.values
            if isinstance(issue.values, int):
                values = [max(0, outcome[i] - 1), min(outcome[i] + 1, issue.values)]
            if isinstance(issue.values, tuple):
                delta = random.random() * (issue.values[0] - issue.values[0])
                values.append(max(issue.values[0], outcome[i] - delta))
                values.append(min(outcome[i] + delta, issue.values[0]))

            for value in values:
                neighbor = list(deepcopy(outcome))
                if neighbor[i] == value:
                    continue
                neighbor[i] = value
                neighbors.append(tuple(neighbor))

        return neighbors

    def __init__(self, *args, **kwargs):
        """Initialize the instance.

        Args:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(*args, **kwargs)

        for issue in self.issues:
            if issue.is_discrete() is False:
                raise ValueError("This mechanism assume discrete issues")

        if self.initial_outcome is None:
            self.initial_outcome = self.random_outcomes(1)[0]

        self._current_state.current_offer = self.initial_outcome
        self.possible_offers = self.neighbors(self._current_state.current_offer)

    def next_outcome(self, outcome: Outcome | None) -> Outcome | None:
        """Generate the next outcome given some outcome.

        Args:
             outcome: The current outcome

        Returns:
            a new outcome or None to end the mechanism run

        """

        if len(self.possible_offers) == 0:
            return None
        return self.possible_offers.pop(
            random.randint(0, len(self.possible_offers)) - 1
        )

    def __call__(self, state: STState, action=None) -> MechanismStepResult:
        """Single round of the protocol"""

        new_offer = self.next_outcome(state.current_offer)
        if new_offer is None:
            state.agreement = (state.current_offer,)
            return MechanismStepResult(state)

        responses = []
        for neg in self.negotiators:
            strt = time.perf_counter()
            responses.append(neg.is_better(new_offer, state.current_offer))
            if time.perf_counter() - strt > self.nmi.step_time_limit:
                return MechanismStepResult(state)

        self.last_responses = responses

        if all(responses):
            state.current_offer = new_offer
            self.possible_offers = self.neighbors(state.current_offer)

        return MechanismStepResult(state)

    @property
    def current_offer(self):
        """Returns the current offer being considered in the negotiation."""
        return self._current_state.current_offer
