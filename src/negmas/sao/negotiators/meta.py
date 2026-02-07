"""SAO Meta-negotiator base class and aggregation-based implementation."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Iterable

from negmas.common import MechanismState, NegotiatorMechanismInterface
from negmas.negotiators.meta import MetaNegotiator
from negmas.outcomes import Outcome
from negmas.outcomes.common import ExtendedOutcome

from ..common import ResponseType, SAOState
from .base import SAONegotiator

if TYPE_CHECKING:
    from negmas.preferences import BaseUtilityFunction, Preferences

__all__ = [
    "SAOMetaNegotiator",
    "SAOAggMetaNegotiator",
    "RangeMetaNegotiator",
    "MeanMetaNegotiator",
    "OSMeanMetaNegotiator",
]


class SAOMetaNegotiator(MetaNegotiator, SAONegotiator):
    """
    Abstract base class for SAO meta-negotiators that manage multiple SAONegotiator instances.

    This class provides the infrastructure for managing sub-negotiators in SAO protocols,
    including lifecycle callback forwarding, NMI/ufun sharing, and proper joining.

    Subclasses must implement `propose` and `respond` to define how the meta-negotiator
    behaves. This is straightforward - just decide how to use your sub-negotiators.

    Args:
        negotiators: An iterable of `SAONegotiator` instances to manage.
        negotiator_names: Optional names for the negotiators.
        share_ufun: If True (default), sub-negotiators will share the parent's ufun.
        share_nmi: If True (default), sub-negotiators will receive the parent's NMI on join.
        *args: Additional positional arguments passed to the base class.
        **kwargs: Additional keyword arguments passed to the base class.

    Remarks:
        - All lifecycle callbacks are delegated to all sub-negotiators.
        - Subclasses can implement any strategy: delegation to a single negotiator,
          aggregation of multiple negotiators, or custom logic that uses sub-negotiators
          for advice/context only.

    Example:
        A simple passthrough meta-negotiator that delegates to its first sub-negotiator::

            class PassthroughMeta(SAOMetaNegotiator):
                def propose(self, state, dest=None):
                    return self._negotiators[0].propose(state, dest=dest)

                def respond(self, state, source=None):
                    return self._negotiators[0].respond(state, source=source)

        A filtering meta-negotiator that modifies proposals::

            class FilteringMeta(SAOMetaNegotiator):
                def propose(self, state, dest=None):
                    proposal = self._negotiators[0].propose(state, dest=dest)
                    if proposal and sum(proposal) < 10:
                        return self.ufun.extreme_outcomes()[1]  # best outcome
                    return proposal

                def respond(self, state, source=None):
                    return self._negotiators[0].respond(state, source=source)

    See Also:
        SAOAggMetaNegotiator: Implementation that aggregates proposals/responses
            from multiple sub-negotiators.
    """

    def __init__(
        self,
        *args,
        negotiators: Iterable[SAONegotiator],
        negotiator_names: Iterable[str] | None = None,
        share_ufun: bool = True,
        share_nmi: bool = True,
        **kwargs,
    ):
        """Initialize the SAOMetaNegotiator.

        Args:
            *args: Positional arguments for the base MetaNegotiator.
            negotiators: The SAO sub-negotiators to manage.
            negotiator_names: Optional names for the sub-negotiators.
            share_ufun: Whether sub-negotiators should share the parent's ufun.
            share_nmi: Whether sub-negotiators should receive the parent's NMI.
            **kwargs: Keyword arguments for the base MetaNegotiator.
        """
        # Initialize with empty negotiators first, then add them
        super().__init__(
            *args,
            negotiators=[],  # Start with empty, add later
            negotiator_names=None,
            share_ufun=share_ufun,
            share_nmi=share_nmi,
            **kwargs,
        )
        self._negotiators: list[SAONegotiator]  # type: ignore

        # Now add the negotiators
        import itertools

        for neg, name in zip(
            negotiators,
            negotiator_names if negotiator_names else itertools.repeat(None),
        ):
            self.add_negotiator(neg, name=name)

    @property
    def sao_negotiators(self) -> tuple[SAONegotiator, ...]:
        """Return the tuple of SAO sub-negotiators.

        Returns:
            A tuple of all SAO sub-negotiators.
        """
        return tuple(self._negotiators)  # type: ignore

    # Abstract methods that subclasses must implement

    @abstractmethod
    def propose(
        self, state: SAOState, dest: str | None = None
    ) -> Outcome | ExtendedOutcome | None:
        """Generate a proposal for the negotiation.

        Subclasses must implement this method to define how proposals are generated.
        This could involve delegating to a single sub-negotiator, aggregating proposals
        from multiple sub-negotiators, or using custom logic.

        Args:
            state: The current SAO state.
            dest: The destination partner ID (if applicable).

        Returns:
            The proposal, or None to refuse to propose.
        """
        ...

    @abstractmethod
    def respond(self, state: SAOState, source: str | None = None) -> ResponseType:
        """Respond to an offer in the negotiation.

        Subclasses must implement this method to define how responses are generated.
        This could involve delegating to a single sub-negotiator, aggregating responses
        from multiple sub-negotiators, or using custom logic.

        Args:
            state: The current SAO state.
            source: The source partner ID.

        Returns:
            The response to the current offer.
        """
        ...

    # Override join to handle sub-negotiators properly
    def join(
        self,
        nmi: NegotiatorMechanismInterface,
        state: MechanismState,
        *,
        preferences: Preferences | None = None,
        ufun: BaseUtilityFunction | None = None,
        role: str = "negotiator",
    ) -> bool:
        """Join a negotiation and have sub-negotiators join too.

        Args:
            nmi: The negotiator-mechanism interface.
            state: The current mechanism state.
            preferences: Optional preferences for this negotiator.
            ufun: Optional utility function (overrides preferences).
            role: The role in the negotiation.

        Returns:
            True if successfully joined, False otherwise.
        """
        # Use SAONegotiator's join (which handles capabilities etc.)
        joined = SAONegotiator.join(
            self, nmi, state, preferences=preferences, ufun=ufun, role=role
        )
        if not joined:
            return False

        if self._share_nmi:
            # Have sub-negotiators join with shared NMI and optionally shared ufun
            sub_ufun = self.ufun if self._share_ufun else None
            sub_prefs = self.preferences if self._share_ufun and not sub_ufun else None
            for neg in self._negotiators:
                neg.join(nmi, state, preferences=sub_prefs, ufun=sub_ufun, role=role)

        return True

    # Lifecycle callbacks - delegate to both parent and sub-negotiators

    def on_negotiation_start(self, state: MechanismState) -> None:
        """Notify all sub-negotiators that negotiation has started."""
        for neg in self._negotiators:
            neg.on_negotiation_start(state)

    def on_round_start(self, state: MechanismState) -> None:
        """Notify all sub-negotiators that a round has started."""
        for neg in self._negotiators:
            neg.on_round_start(state)

    def on_round_end(self, state: MechanismState) -> None:
        """Notify all sub-negotiators that a round has ended."""
        for neg in self._negotiators:
            neg.on_round_end(state)

    def on_leave(self, state: MechanismState) -> None:
        """Notify all sub-negotiators that we're leaving the negotiation."""
        for neg in self._negotiators:
            neg.on_leave(state)
        SAONegotiator.on_leave(self, state)

    def on_negotiation_end(self, state: MechanismState) -> None:
        """Notify all sub-negotiators that negotiation has ended."""
        for neg in self._negotiators:
            neg.on_negotiation_end(state)

    def on_mechanism_error(self, state: MechanismState) -> None:
        """Notify all sub-negotiators of a mechanism error."""
        for neg in self._negotiators:
            neg.on_mechanism_error(state)


class SAOAggMetaNegotiator(SAOMetaNegotiator):
    """
    An SAO meta-negotiator that aggregates proposals and responses from sub-negotiators.

    This class collects proposals and responses from all sub-negotiators and uses
    abstract aggregation methods to combine them into a single proposal or response.

    Subclasses must implement `aggregate_proposals` and `aggregate_responses`
    to define how proposals and responses from sub-negotiators are combined.

    Args:
        negotiators: An iterable of `SAONegotiator` instances to manage.
        negotiator_names: Optional names for the negotiators.
        share_ufun: If True (default), sub-negotiators will share the parent's ufun.
        share_nmi: If True (default), sub-negotiators will receive the parent's NMI on join.
        *args: Additional positional arguments passed to the base class.
        **kwargs: Additional keyword arguments passed to the base class.

    Remarks:
        - `propose` collects proposals from all sub-negotiators and aggregates them.
        - `respond` collects responses from all sub-negotiators and aggregates them.

    Example:
        >>> from negmas.sao.negotiators import (
        ...     SAOAggMetaNegotiator,
        ...     BoulwareTBNegotiator,
        ... )
        >>> from negmas.sao import SAOMechanism
        >>> from negmas.preferences import LinearAdditiveUtilityFunction as U
        >>> from negmas.outcomes import make_issue
        >>>
        >>> class MajorityVoteNegotiator(SAOAggMetaNegotiator):
        ...     def aggregate_proposals(self, state, proposals, dest=None):
        ...         for neg, proposal in proposals:
        ...             if proposal is not None:
        ...                 return proposal
        ...         return None
        ...
        ...     def aggregate_responses(self, state, responses, offer, source=None):
        ...         accept_count = sum(
        ...             1 for _, r in responses if r == ResponseType.ACCEPT_OFFER
        ...         )
        ...         if accept_count > len(responses) / 2:
        ...             return ResponseType.ACCEPT_OFFER
        ...         return ResponseType.REJECT_OFFER
    """

    # Abstract aggregation methods that subclasses must implement

    @abstractmethod
    def aggregate_proposals(
        self,
        state: SAOState,
        proposals: list[tuple[SAONegotiator, Outcome | ExtendedOutcome | None]],
        dest: str | None = None,
    ) -> Outcome | ExtendedOutcome | None:
        """Aggregate proposals from all sub-negotiators into a single proposal.

        Args:
            state: The current SAO state.
            proposals: List of (negotiator, proposal) tuples from sub-negotiators.
            dest: The destination partner ID (if applicable).

        Returns:
            The aggregated proposal, or None to refuse to propose.
        """
        ...

    @abstractmethod
    def aggregate_responses(
        self,
        state: SAOState,
        responses: list[tuple[SAONegotiator, ResponseType]],
        offer: Outcome | None,
        source: str | None = None,
    ) -> ResponseType:
        """Aggregate responses from all sub-negotiators into a single response.

        Args:
            state: The current SAO state.
            responses: List of (negotiator, response) tuples from sub-negotiators.
            offer: The offer being responded to.
            source: The source partner ID (if applicable).

        Returns:
            The aggregated response.
        """
        ...

    # SAO protocol methods - collect from sub-negotiators and aggregate

    def propose(
        self, state: SAOState, dest: str | None = None
    ) -> Outcome | ExtendedOutcome | None:
        """Collect proposals from all sub-negotiators and aggregate them.

        Args:
            state: The current SAO state.
            dest: The destination partner ID (if applicable).

        Returns:
            The aggregated proposal.
        """
        proposals: list[tuple[SAONegotiator, Outcome | ExtendedOutcome | None]] = []
        for neg in self._negotiators:
            proposal = neg.propose(state, dest=dest)
            proposals.append((neg, proposal))  # type: ignore
        return self.aggregate_proposals(state, proposals, dest=dest)

    def respond(self, state: SAOState, source: str | None = None) -> ResponseType:
        """Collect responses from all sub-negotiators and aggregate them.

        Args:
            state: The current SAO state.
            source: The source partner ID.

        Returns:
            The aggregated response.
        """
        offer = state.current_offer
        responses: list[tuple[SAONegotiator, ResponseType]] = []
        for neg in self._negotiators:
            response = neg.respond(state, source=source)
            responses.append((neg, response))  # type: ignore
        return self.aggregate_responses(state, responses, offer, source=source)


class RangeMetaNegotiator(SAOAggMetaNegotiator):
    """
    An SAO meta-negotiator that samples outcomes within the utility range of sub-negotiator proposals.

    This negotiator collects proposals from all sub-negotiators, computes their utilities,
    and samples an outcome from the outcome space that falls within the min/max utility range
    defined by those proposals.

    For response aggregation, it uses majority voting among sub-negotiators.

    Args:
        negotiators: An iterable of `SAONegotiator` instances to manage.
        negotiator_names: Optional names for the negotiators.
        max_cardinality: Maximum number of outcomes to sample when searching for outcomes
            in the utility range. Defaults to 10000.
        *args: Additional positional arguments passed to the base class.
        **kwargs: Additional keyword arguments passed to the base class.

    Example:
        >>> from negmas.sao.negotiators import (
        ...     SAORangeMetaNegotiator,
        ...     BoulwareTBNegotiator,
        ...     ConcederTBNegotiator,
        ... )
        >>> from negmas.sao import SAOMechanism
        >>> from negmas.preferences import LinearAdditiveUtilityFunction as U
        >>> from negmas.outcomes import make_issue, make_os
        >>>
        >>> issues = [make_issue(10, "price")]
        >>> os = make_os(issues)
        >>> ufun1 = U.random(os, reserved_value=0.0)
        >>> ufun2 = U.random(os, reserved_value=0.0)
        >>>
        >>> # Create a range meta-negotiator with diverse strategies
        >>> meta = SAORangeMetaNegotiator(
        ...     negotiators=[BoulwareTBNegotiator(), ConcederTBNegotiator()], ufun=ufun1
        ... )
        >>> opponent = ConcederTBNegotiator(ufun=ufun2)
        >>>
        >>> mechanism = SAOMechanism(issues=issues, n_steps=50)
        >>> _ = mechanism.add(meta)
        >>> _ = mechanism.add(opponent)
        >>> result = mechanism.run()
        >>> result.running
        False
    """

    def __init__(
        self,
        *args,
        negotiators: Iterable[SAONegotiator],
        negotiator_names: Iterable[str] | None = None,
        max_cardinality: int = 10000,
        **kwargs,
    ):
        super().__init__(
            *args, negotiators=negotiators, negotiator_names=negotiator_names, **kwargs
        )
        self._max_cardinality = max_cardinality

    def aggregate_proposals(
        self,
        state: SAOState,
        proposals: list[tuple[SAONegotiator, Outcome | ExtendedOutcome | None]],
        dest: str | None = None,
    ) -> Outcome | ExtendedOutcome | None:
        """Aggregate proposals by sampling an outcome within the utility range.

        Computes the utility of each sub-negotiator's proposal, finds the min/max
        utility range, and samples an outcome from the outcome space that falls
        within this range.

        Args:
            state: The current SAO state.
            proposals: List of (negotiator, proposal) tuples from sub-negotiators.
            dest: The destination partner ID (if applicable).

        Returns:
            An outcome within the utility range of sub-negotiator proposals,
            or None if no valid outcome can be found.
        """
        import random

        if self.ufun is None or self.nmi is None:
            return None

        # Collect utilities of non-None proposals
        utilities: list[float] = []
        valid_proposals: list[Outcome | ExtendedOutcome] = []
        for _, proposal in proposals:
            if proposal is not None:
                u = self.ufun(proposal)
                if u is not None:
                    utilities.append(float(u))
                    valid_proposals.append(proposal)

        if not utilities:
            return None

        # If only one valid proposal, return it
        if len(utilities) == 1:
            return valid_proposals[0]

        min_util = min(utilities)
        max_util = max(utilities)

        # Sample outcomes in the utility range
        outcomes = [
            o
            for o in self.nmi.outcome_space.enumerate_or_sample(
                max_cardinality=self._max_cardinality
            )
            if (u := self.ufun(o)) is not None and min_util <= float(u) <= max_util
        ]

        if outcomes:
            return random.choice(outcomes)

        # Fallback: return any valid proposal
        return valid_proposals[0] if valid_proposals else None

    def aggregate_responses(
        self,
        state: SAOState,
        responses: list[tuple[SAONegotiator, ResponseType]],
        offer: Outcome | None,
        source: str | None = None,
    ) -> ResponseType:
        """Aggregate responses using majority voting.

        Args:
            state: The current SAO state.
            responses: List of (negotiator, response) tuples from sub-negotiators.
            offer: The offer being responded to.
            source: The source partner ID (if applicable).

        Returns:
            The most common response among sub-negotiators.
        """
        from collections import Counter

        if not responses:
            return ResponseType.REJECT_OFFER

        response_counts = Counter(r for _, r in responses)
        most_common = response_counts.most_common(1)
        if most_common:
            return most_common[0][0]
        return ResponseType.REJECT_OFFER


class MeanMetaNegotiator(SAOAggMetaNegotiator):
    """
    An SAO meta-negotiator that samples outcomes around the mean utility of sub-negotiator proposals.

    This negotiator collects proposals from all sub-negotiators, computes the mean utility
    of those proposals, and tries to sample an outcome with utility close to the mean.
    If no outcome is found within a small epsilon around the mean, it progressively expands
    the search range until it covers the full min/max range of sub-negotiator proposals.

    For response aggregation, it uses majority voting among sub-negotiators.

    Args:
        negotiators: An iterable of `SAONegotiator` instances to manage.
        negotiator_names: Optional names for the negotiators.
        initial_epsilon: Initial utility range around the mean to search. Defaults to 0.05.
        epsilon_step: How much to expand the range on each iteration. Defaults to 0.1.
        max_cardinality: Maximum number of outcomes to sample when searching. Defaults to 10000.
        *args: Additional positional arguments passed to the base class.
        **kwargs: Additional keyword arguments passed to the base class.

    Example:
        >>> from negmas.sao.negotiators import (
        ...     MeanMetaNegotiator,
        ...     BoulwareTBNegotiator,
        ...     ConcederTBNegotiator,
        ... )
        >>> from negmas.sao import SAOMechanism
        >>> from negmas.preferences import LinearAdditiveUtilityFunction as U
        >>> from negmas.outcomes import make_issue, make_os
        >>>
        >>> issues = [make_issue(10, "price")]
        >>> os = make_os(issues)
        >>> ufun1 = U.random(os, reserved_value=0.0)
        >>> ufun2 = U.random(os, reserved_value=0.0)
        >>>
        >>> # Create a mean meta-negotiator with diverse strategies
        >>> meta = MeanMetaNegotiator(
        ...     negotiators=[BoulwareTBNegotiator(), ConcederTBNegotiator()], ufun=ufun1
        ... )
        >>> opponent = ConcederTBNegotiator(ufun=ufun2)
        >>>
        >>> mechanism = SAOMechanism(issues=issues, n_steps=50)
        >>> _ = mechanism.add(meta)
        >>> _ = mechanism.add(opponent)
        >>> result = mechanism.run()
        >>> result.running
        False
    """

    def __init__(
        self,
        *args,
        negotiators: Iterable[SAONegotiator],
        negotiator_names: Iterable[str] | None = None,
        initial_epsilon: float = 0.05,
        epsilon_step: float = 0.1,
        max_cardinality: int = 10000,
        **kwargs,
    ):
        super().__init__(
            *args, negotiators=negotiators, negotiator_names=negotiator_names, **kwargs
        )
        self._initial_epsilon = initial_epsilon
        self._epsilon_step = epsilon_step
        self._max_cardinality = max_cardinality

    def aggregate_proposals(
        self,
        state: SAOState,
        proposals: list[tuple[SAONegotiator, Outcome | ExtendedOutcome | None]],
        dest: str | None = None,
    ) -> Outcome | ExtendedOutcome | None:
        """Aggregate proposals by sampling an outcome near the mean utility.

        Computes the mean utility of sub-negotiator proposals and tries to find
        an outcome within a small epsilon around the mean. If not found, progressively
        expands the search range up to the full min/max range.

        Args:
            state: The current SAO state.
            proposals: List of (negotiator, proposal) tuples from sub-negotiators.
            dest: The destination partner ID (if applicable).

        Returns:
            An outcome near the mean utility of sub-negotiator proposals,
            or None if no valid outcome can be found.
        """
        import random

        if self.ufun is None or self.nmi is None:
            return None

        # Collect utilities of non-None proposals
        utilities: list[float] = []
        valid_proposals: list[Outcome | ExtendedOutcome] = []
        for _, proposal in proposals:
            if proposal is not None:
                u = self.ufun(proposal)
                if u is not None:
                    utilities.append(float(u))
                    valid_proposals.append(proposal)

        if not utilities:
            return None

        # If only one valid proposal, return it
        if len(utilities) == 1:
            return valid_proposals[0]

        mean_util = sum(utilities) / len(utilities)
        min_util = min(utilities)
        max_util = max(utilities)

        # Pre-compute outcomes with their utilities for efficiency
        outcomes_with_utils: list[tuple[Outcome, float]] = []
        for o in self.nmi.outcome_space.enumerate_or_sample(
            max_cardinality=self._max_cardinality
        ):
            u = self.ufun(o)
            if u is not None:
                outcomes_with_utils.append((o, float(u)))

        # Progressively expand the search range
        epsilon = self._initial_epsilon
        max_range = max_util - min_util

        while epsilon <= max_range + self._epsilon_step:
            low = max(min_util, mean_util - epsilon)
            high = min(max_util, mean_util + epsilon)

            matching = [o for o, u in outcomes_with_utils if low <= u <= high]
            if matching:
                return random.choice(matching)

            epsilon += self._epsilon_step

        # Final fallback: search the entire min/max range
        matching = [o for o, u in outcomes_with_utils if min_util <= u <= max_util]
        if matching:
            return random.choice(matching)

        # Ultimate fallback: return any valid proposal
        return valid_proposals[0] if valid_proposals else None

    def aggregate_responses(
        self,
        state: SAOState,
        responses: list[tuple[SAONegotiator, ResponseType]],
        offer: Outcome | None,
        source: str | None = None,
    ) -> ResponseType:
        """Aggregate responses using majority voting.

        Args:
            state: The current SAO state.
            responses: List of (negotiator, response) tuples from sub-negotiators.
            offer: The offer being responded to.
            source: The source partner ID (if applicable).

        Returns:
            The most common response among sub-negotiators.
        """
        from collections import Counter

        if not responses:
            return ResponseType.REJECT_OFFER

        response_counts = Counter(r for _, r in responses)
        most_common = response_counts.most_common(1)
        if most_common:
            return most_common[0][0]
        return ResponseType.REJECT_OFFER


class OSMeanMetaNegotiator(SAOAggMetaNegotiator):
    """
    An SAO meta-negotiator that aggregates proposals by averaging in outcome space.

    Unlike `MeanMetaNegotiator` which works in utility space, this negotiator works
    directly in the outcome space by averaging issue values from sub-negotiator proposals.

    For numeric issues (integers, floats), it computes the mean value across all
    sub-negotiator proposals and finds/generates an outcome with values near that mean.
    For non-numeric issues (categorical), it samples from the values proposed by
    sub-negotiators.

    For response aggregation, it uses majority voting among sub-negotiators.

    Args:
        negotiators: An iterable of `SAONegotiator` instances to manage.
        negotiator_names: Optional names for the negotiators.
        *args: Additional positional arguments passed to the base class.
        **kwargs: Additional keyword arguments passed to the base class.

    Remarks:
        - For continuous issues, the mean value is used directly.
        - For discrete numeric issues (integers), the mean is rounded to the nearest
          valid value within the issue's range.
        - For categorical issues, a value is randomly sampled from those proposed
          by the sub-negotiators.

    Example:
        >>> from negmas.sao.negotiators import (
        ...     OSMeanMetaNegotiator,
        ...     BoulwareTBNegotiator,
        ...     ConcederTBNegotiator,
        ... )
        >>> from negmas.sao import SAOMechanism
        >>> from negmas.preferences import LinearAdditiveUtilityFunction as U
        >>> from negmas.outcomes import make_issue, make_os
        >>>
        >>> issues = [make_issue(10, "price"), make_issue(5, "quantity")]
        >>> os = make_os(issues)
        >>> ufun1 = U.random(os, reserved_value=0.0)
        >>> ufun2 = U.random(os, reserved_value=0.0)
        >>>
        >>> # Create an outcome-space mean meta-negotiator
        >>> meta = OSMeanMetaNegotiator(
        ...     negotiators=[BoulwareTBNegotiator(), ConcederTBNegotiator()], ufun=ufun1
        ... )
        >>> opponent = ConcederTBNegotiator(ufun=ufun2)
        >>>
        >>> mechanism = SAOMechanism(issues=issues, n_steps=50)
        >>> _ = mechanism.add(meta)
        >>> _ = mechanism.add(opponent)
        >>> result = mechanism.run()
        >>> result.running
        False
    """

    def aggregate_proposals(
        self,
        state: SAOState,
        proposals: list[tuple[SAONegotiator, Outcome | ExtendedOutcome | None]],
        dest: str | None = None,
    ) -> Outcome | ExtendedOutcome | None:
        """Aggregate proposals by averaging issue values in outcome space.

        For numeric issues, computes the mean and finds/generates a valid value.
        For non-numeric issues, samples from the values proposed by sub-negotiators.

        Args:
            state: The current SAO state.
            proposals: List of (negotiator, proposal) tuples from sub-negotiators.
            dest: The destination partner ID (if applicable).

        Returns:
            An outcome with averaged issue values, or None if no valid proposals.
        """
        import random

        if self.nmi is None:
            return None

        # Collect valid proposals
        valid_proposals: list[Outcome | ExtendedOutcome] = [
            p for _, p in proposals if p is not None
        ]

        if not valid_proposals:
            return None

        # If only one valid proposal, return it
        if len(valid_proposals) == 1:
            return valid_proposals[0]

        issues = self.nmi.outcome_space.issues

        # Build the aggregated outcome issue by issue
        aggregated_values: list = []

        for i, issue in enumerate(issues):
            # Collect values for this issue from all valid proposals
            issue_values = [p[i] for p in valid_proposals]

            if issue.is_numeric():
                # Numeric issue: compute mean and find valid value
                mean_val = sum(issue_values) / len(issue_values)
                aggregated_values.append(
                    self._find_valid_numeric_value(issue, mean_val)
                )
            else:
                # Non-numeric (categorical) issue: sample from proposed values
                aggregated_values.append(random.choice(issue_values))

        return tuple(aggregated_values)

    def _find_valid_numeric_value(self, issue, target_value: float):
        """Find a valid value for a numeric issue closest to the target.

        Args:
            issue: The issue to find a value for.
            target_value: The target value (e.g., mean of proposals).

        Returns:
            A valid value for the issue closest to the target.
        """
        from negmas.outcomes.continuous_issue import ContinuousIssue

        # Clamp to issue bounds
        min_val = issue.min_value
        max_val = issue.max_value
        clamped = max(min_val, min(max_val, target_value))

        if isinstance(issue, ContinuousIssue):
            # Continuous issue: return the clamped value directly
            return float(clamped)

        if issue.is_integer():
            # Integer issue: round to nearest integer within bounds
            rounded = int(round(clamped))
            # Ensure it's within bounds
            return max(min_val, min(max_val, rounded))

        # Discrete numeric issue (e.g., DiscreteCardinalIssue, DiscreteOrdinalIssue)
        # Find the closest valid value from the issue's values
        best_value = None
        best_distance = float("inf")

        for v in issue.all:
            distance = abs(v - target_value)
            if distance < best_distance:
                best_distance = distance
                best_value = v

        return best_value if best_value is not None else issue.rand()

    def aggregate_responses(
        self,
        state: SAOState,
        responses: list[tuple[SAONegotiator, ResponseType]],
        offer: Outcome | None,
        source: str | None = None,
    ) -> ResponseType:
        """Aggregate responses using majority voting.

        Args:
            state: The current SAO state.
            responses: List of (negotiator, response) tuples from sub-negotiators.
            offer: The offer being responded to.
            source: The source partner ID (if applicable).

        Returns:
            The most common response among sub-negotiators.
        """
        from collections import Counter

        if not responses:
            return ResponseType.REJECT_OFFER

        response_counts = Counter(r for _, r in responses)
        most_common = response_counts.most_common(1)
        if most_common:
            return most_common[0][0]
        return ResponseType.REJECT_OFFER
