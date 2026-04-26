"""
Acceptance modeling.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Collection, Iterable

import numpy as np

from negmas.common import NegotiatorMechanismInterface
from negmas.outcomes import Outcome
from negmas.sao import ResponseType

if TYPE_CHECKING:
    from negmas.sao import SAONegotiator

__all__ = [
    "AcceptanceModelType",
    "DiscreteAcceptanceModel",
    "AdaptiveDiscreteAcceptanceModel",
    "RandomDiscreteAcceptanceModel",
    "PeekingDiscreteAcceptanceModel",
    "AggregatingDiscreteAcceptanceModel",
    "UncertainOpponentModel",
    "PeekingProbabilisticDiscreteAcceptanceModel",
]


class AcceptanceModelType(Enum):
    """Represents types of acceptance models."""

    ACCEPTANCE_MODEL_RANDOM = -100
    ACCEPTANCE_MODEL_AUTO = -1
    ACCEPTANCE_MODEL_STATIC = 0
    ACCEPTANCE_MODEL_HOMOGENEOUS_CONST = 1
    ACCEPTANCE_MODEL_HOMOGENEOUS = 2
    ACCEPTANCE_MODEL_MONOTONIC = 3
    ACCEPTANCE_MODEL_GENERAL = 4
    ACCEPTANCE_MODEL_BINARY_STATIC = 7
    ACCEPTANCE_MODEL_BINARY_HOMOGENEOUS_CONST = 8
    ACCEPTANCE_MODEL_BINARY_HOMOGENEOUS = 9
    ACCEPTANCE_MODEL_BINARY_MONOTONIC = 10
    ACCEPTANCE_MODEL_BINARY = 11


class DiscreteAcceptanceModel(ABC):
    """Abstract base class for modeling opponent acceptance behavior over discrete outcomes.

    This model estimates the probability that opponents will accept each possible
    outcome in a negotiation, and updates these estimates based on observed behavior.
    """

    def __init__(self, outcomes: Iterable[Outcome]):
        """Initializes the acceptance model with a set of possible outcomes.

        Args:
            outcomes: The possible outcomes in the negotiation space.
        """
        outcomes = list(outcomes)
        self.outcomes = outcomes
        self.indx = dict(zip(outcomes, range(len(outcomes))))

    @abstractmethod
    def probability_of_acceptance_indx(self, outcome_index: int) -> float:
        """Returns the estimated probability that opponents will accept the outcome at the given index.

        Args:
            outcome_index: Index into the outcomes list.

        Returns:
            float: Probability in [0, 1] that the outcome will be accepted.
        """
        raise NotImplementedError()

    def probability_of_acceptance(self, outcome: Outcome):
        """Returns the estimated probability that opponents will accept the given outcome.

        Args:
            outcome: The outcome to evaluate.
        """
        indx = self.indx.get(outcome, None)
        if indx is None:
            return 0.0
        return self.probability_of_acceptance_indx(indx)

    @abstractmethod
    def update_rejected_indx(self, outcome_index: int):
        """Updates the model after observing a rejection of the outcome at the given index.

        Args:
            outcome_index: Index into the outcomes list for the rejected outcome.
        """
        raise NotImplementedError()

    def update_rejected(self, outcome: Outcome):
        """Updates the model after observing a rejection of the given outcome.

        Args:
            outcome: The outcome that was rejected.
        """
        if outcome is None:
            return
        return self.update_rejected_indx(self.indx[outcome])

    @abstractmethod
    def update_offered_indx(self, outcome_index: int):
        """Updates the model after observing an offer of the outcome at the given index.

        Args:
            outcome_index: Index into the outcomes list for the offered outcome.
        """
        raise NotImplementedError()

    def update_offered(self, outcome):
        """Updates the model after observing an offer of the given outcome.

        Args:
            outcome: The outcome that was offered.
        """
        if outcome is None:
            return
        return self.update_offered_indx(self.indx[outcome])

    def update_accepted(self, outcome):
        """Updates the model after observing acceptance of the given outcome.

        Args:
            outcome: The outcome that was accepted.
        """
        return self.update_offered(outcome=outcome)

    def acceptance_probabilities(self) -> np.ndarray:
        """Returns the acceptance probabilities for all outcomes as an array.

        Returns:
            np.ndarray: Array of acceptance probabilities, one per outcome.
        """
        return np.array(
            [self.probability_of_acceptance_indx(_) for _ in range(len(self.outcomes))]
        )


class AdaptiveDiscreteAcceptanceModel(DiscreteAcceptanceModel):
    """An acceptance model that adapts its probability estimates based on observed rejections and offers.

    The model starts with initial probabilities and updates them when outcomes are
    rejected (decreasing probability) or offered by opponents (increasing probability).
    """

    def __init__(
        self,
        outcomes: Iterable[Outcome],
        n_negotiators: int = 2,
        prob: float | list[float] = 0.5,
        end_prob=0.0,
        p_accept_after_reject=0.0,
        p_reject_after_accept=0.0,
        rejection_discount=0.98,
        rejection_delta=0.0,
        not_offering_rejection_ratio=0.75,
    ):
        """Initializes the adaptive acceptance model.

        Args:
            outcomes: The possible outcomes in the negotiation space.
            n_negotiators: Number of negotiators in the negotiation.
            prob: Initial acceptance probability, either a single value for all outcomes or a list per outcome.
            end_prob: Probability used at negotiation end.
            p_accept_after_reject: Minimum probability after a rejection is observed.
            p_reject_after_accept: Probability of rejection after an acceptance (used for decay).
            rejection_discount: Multiplicative factor applied to probability after rejection (0-1).
            rejection_delta: Additive reduction to probability after rejection.
            not_offering_rejection_ratio: Ratio controlling implicit rejection when outcome is not offered.
        """
        super().__init__(outcomes=outcomes)
        outcomes = self.outcomes
        if isinstance(prob, list) and len(outcomes) != len(prob):
            raise ValueError(
                f"{len(outcomes)} outcomes but {len(prob)} probabilities. Cannot initialize simple "
                f"opponents model"
            )
        self.n_agents = n_negotiators
        if not isinstance(prob, Collection):
            self.p = np.array([prob for _ in range(len(outcomes))])
        else:
            self.p = np.array(list(prob))
        self.end_prob = end_prob
        self.p_accept_after_reject = p_accept_after_reject
        self.p_accept_after_accept = 1 - p_reject_after_accept
        self.delta = rejection_delta
        self.discount = rejection_discount
        self.first = True
        self.not_offered = set(list(range(len(self.outcomes))))
        self.not_offering_rejection_ratio = not_offering_rejection_ratio
        self.not_offering_discount = self.discount + (
            1.0 - self.not_offering_rejection_ratio
        ) * (1.0 - self.discount)

    @classmethod
    def from_negotiation(
        cls,
        nmi: NegotiatorMechanismInterface,
        prob: float | list = 0.5,
        end_prob=0.0,
        p_accept_after_reject=0.0,
        p_reject_after_accept=0.0,
    ) -> AdaptiveDiscreteAcceptanceModel:
        """Creates an acceptance model from a negotiation mechanism interface.

        Args:
            nmi: The negotiator-mechanism interface providing negotiation context.
            prob: Initial acceptance probability, either a single value or list per outcome.
            end_prob: Probability used at negotiation end.
            p_accept_after_reject: Minimum probability after a rejection is observed.
            p_reject_after_accept: Probability of rejection after an acceptance.

        Returns:
            AdaptiveDiscreteAcceptanceModel: A new model initialized from the negotiation.
        """
        if not nmi.n_outcomes or nmi.outcomes is None:
            raise ValueError(
                "Cannot initialize this simple opponents model for a negotiation with uncountable outcomes"
            )
        return cls(
            outcomes=nmi.outcomes,
            n_negotiators=nmi.n_negotiators,
            prob=prob,
            end_prob=end_prob,
            p_accept_after_reject=p_accept_after_reject,
            p_reject_after_accept=p_reject_after_accept,
        )

    def probability_of_acceptance_indx(self, outcome_index: int) -> float:
        """Returns the estimated acceptance probability for the outcome at the given index.

        Args:
            outcome_index: Index into the outcomes list.

        Returns:
            float: The current acceptance probability estimate.
        """
        return self.p[outcome_index]

    def acceptance_probabilities(self):
        """Probability of acceptance for all outcomes"""
        return self.p

    def _update(self, p: float, real_rejection: bool) -> float:
        if real_rejection:
            return min(
                self.p_accept_after_reject, min(1.0, (p - self.delta) * self.discount)
            )
        else:
            return max(
                self.p_accept_after_reject,
                min(
                    1.0,
                    (p - self.delta * self.not_offering_rejection_ratio)
                    * self.not_offering_discount,
                ),
            )

    def update_rejected_indx(self, outcome_index: int):
        """Reduces acceptance probability for the outcome at the given index after observing rejection."""
        self.p[outcome_index] = self._update(self.p[outcome_index], real_rejection=True)

    def update_offered_indx(self, outcome_index: int):
        """Updates acceptance probability when opponent offers the outcome at the given index."""
        try:
            self.not_offered.remove(outcome_index)
            self.p[outcome_index] = self.p_accept_after_accept
            # for i in self.not_offered:
            #    self.p[i] = self._update(self.p[i], real_rejection=False)
        except KeyError:
            pass


class RandomDiscreteAcceptanceModel(DiscreteAcceptanceModel):
    """An acceptance model that returns random probabilities, ignoring observed behavior.

    Useful as a baseline or for experimentation with uncertainty effects.
    """

    def __init__(self, outcomes: Collection[Outcome], **kwargs):
        """Initializes the random acceptance model."""
        super().__init__(outcomes=outcomes)

    def probability_of_acceptance_indx(self, outcome_index: int) -> float:
        """Returns a random acceptance probability between 0 and 1.

        Args:
            outcome_index: Index into the outcomes list (ignored).

        Returns:
            float: A random probability value.
        """
        return random.random()

    def update_rejected_indx(self, outcome_index: int):
        """No-op: random model does not learn from rejections."""
        pass

    def update_offered_indx(self, outcome_index: int):
        """No-op: random model does not learn from offers."""
        pass


class ConstantDiscreteAcceptanceModel(DiscreteAcceptanceModel):
    """An acceptance model that always returns a constant probability of 0.5.

    Useful as a simple baseline that assumes equal likelihood of acceptance/rejection.
    """

    def __init__(self, outcomes: Collection[Outcome], **kwargs):
        """Initializes the constant acceptance model."""
        super().__init__(outcomes=outcomes)

    def probability_of_acceptance_indx(self, outcome_index: int) -> float:
        """Returns a constant acceptance probability of 0.5.

        Args:
            outcome_index: Index into the outcomes list (ignored).

        Returns:
            float: Always returns 0.5.
        """
        return 0.5

    def update_rejected_indx(self, outcome_index: int):
        """No-op: constant model does not learn from rejections."""
        pass

    def update_offered_indx(self, outcome_index: int):
        """No-op: constant model does not learn from offers."""
        pass


class PeekingDiscreteAcceptanceModel(DiscreteAcceptanceModel):
    """An acceptance model that queries opponents directly to determine acceptance.

    This model "peeks" at opponent negotiators by calling their respond_ method
    to get their actual response to each outcome. Returns 1.0 if all opponents
    accept, 0.0 otherwise.
    """

    def __init__(
        self,
        outcomes: Collection[Outcome],
        opponents: SAONegotiator | Collection[SAONegotiator],
    ):
        """Initializes the peeking acceptance model.

        Args:
            outcomes: The possible outcomes in the negotiation space.
            opponents: The opponent negotiator(s) to query for acceptance.
        """
        super().__init__(outcomes=outcomes)
        if not isinstance(opponents, Collection):
            opponents = [opponents]
        self.opponents = opponents

    def probability_of_acceptance_indx(self, outcome_index: int) -> float:
        """Queries all opponents and returns 1.0 if all accept, 0.0 otherwise.

        Args:
            outcome_index: Index into the outcomes list.

        Returns:
            float: 1.0 if all opponents accept, 0.0 if any rejects.
        """
        outcome = self.outcomes[outcome_index]
        for opponent in self.opponents:
            if opponent is self:
                continue
            if opponent._nmi is None:
                response = ResponseType.REJECT_OFFER
            else:
                response = opponent.respond_(state=opponent._nmi.state, offer=outcome)  # type: ignore
            if response != ResponseType.ACCEPT_OFFER:
                return 0.0
        return 1.0

    def update_rejected_indx(self, outcome_index: int):
        """No-op: peeking model queries opponents directly."""
        pass

    def update_offered_indx(self, outcome_index: int):
        """No-op: peeking model queries opponents directly."""
        pass


class PeekingProbabilisticDiscreteAcceptanceModel(DiscreteAcceptanceModel):
    """An acceptance model that estimates probability using opponents' utility functions.

    This model "peeks" at opponent utility functions and returns the product
    of their utilities for each outcome as the acceptance probability.
    """

    def __init__(
        self,
        outcomes: Collection[Outcome],
        opponents: SAONegotiator | Collection[SAONegotiator],
    ):
        """Initializes the probabilistic peeking acceptance model.

        Args:
            outcomes: The possible outcomes in the negotiation space.
            opponents: The opponent negotiator(s) whose utility functions are used.
        """
        super().__init__(outcomes=outcomes)
        if not isinstance(opponents, Collection):
            opponents = [opponents]
        self.opponents = opponents

    def probability_of_acceptance_indx(self, outcome_index: int) -> float:
        """Returns the product of opponent utilities as the acceptance probability.

        Args:
            outcome_index: Index into the outcomes list.

        Returns:
            float: Product of all opponent utilities for this outcome.
        """
        outcome = self.outcomes[outcome_index]
        if outcome is None:
            return 0.0
        prod = 1.0
        for o in self.opponents:
            prod *= o.ufun(outcome)  # type: ignore
        return prod

    def update_rejected_indx(self, outcome_index: int):
        """No-op: probabilistic peeking model uses utility functions directly."""
        pass

    def update_offered_indx(self, outcome_index: int):
        """No-op: probabilistic peeking model uses utility functions directly."""
        pass


class AggregatingDiscreteAcceptanceModel(DiscreteAcceptanceModel):
    """An acceptance model that combines multiple models using weighted averaging.

    The final acceptance probability is computed as a weighted sum of the
    probabilities from each constituent model.
    """

    def __init__(
        self,
        outcomes: Collection[Outcome],
        models: list[DiscreteAcceptanceModel],
        weights: list[float] | None = None,
    ):
        """Initializes the aggregating acceptance model.

        Args:
            outcomes: The possible outcomes in the negotiation space.
            models: List of acceptance models to aggregate.
            weights: Optional weights for each model (normalized internally). Defaults to equal weights.
        """
        super().__init__(outcomes=outcomes)
        if weights is None:
            weights = [1.0] * len(self.outcomes)
        s = sum(weights)
        weights = [_ / s for _ in weights]
        self.models = models
        self.weights = weights

    def probability_of_acceptance_indx(self, outcome_index: int) -> float:
        """Returns weighted average of acceptance probabilities from all models.

        Args:
            outcome_index: Index into the outcomes list.

        Returns:
            float: Weighted average probability clamped to [0, 1].
        """
        p = 0.0
        for model, w in zip(self.models, self.weights):
            p += w * model.probability_of_acceptance_indx(outcome_index=outcome_index)
        return min(1.0, max(0.0, p))

    def update_rejected_indx(self, outcome_index: int):
        """Propagates rejection update to all constituent models."""
        for model in self.models:
            model.update_rejected_indx(outcome_index=outcome_index)

    def update_offered_indx(self, outcome_index: int):
        """Propagates offer update to all constituent models."""
        for model in self.models:
            model.update_offered_indx(outcome_index=outcome_index)


class UncertainOpponentModel(AggregatingDiscreteAcceptanceModel):
    """A model for which the uncertainty about the acceptance probability of different negotiators is controllable.

    This is not a realistic model but it can be used to experiment with effects of this uncertainty on different
    negotiation related algorithms (e.g. elicitation algorithms)

    Args:
        outcomes: The list of possible outcomes
        uncertainty (float): The uncertainty level. Zero means no uncertainty and 1.0 means maximum uncertainty
        adaptive (bool): If true then the random part will learn from experience with the opponents otherwise it will not.
        rejection_discount: Only effective if adaptive is True. See `AdaptiveDiscreteAcceptanceModel`
        rejection_delta: Only effective if adaptive is True. See `AdaptiveDiscreteAcceptanceModel`

    """

    def __init__(
        self,
        outcomes: Collection[Outcome],
        opponents: SAONegotiator | Collection[SAONegotiator],
        uncertainty: float = 0.5,
        adaptive: bool = False,
        rejection_discount: float = 0.95,
        rejection_delta: float = 0.0,
        constant_base=True,
        accesses_real_acceptance=False,
    ):
        """Initializes the uncertain opponent model.

        Args:
            outcomes: The possible outcomes in the negotiation space.
            opponents: The opponent negotiator(s) to model.
            uncertainty: Level of uncertainty from 0 (perfect knowledge) to 1 (no knowledge).
            adaptive: If True, the random component learns from observed behavior.
            rejection_discount: Multiplicative discount on probability after rejection (if adaptive).
            rejection_delta: Additive reduction to probability after rejection (if adaptive).
            constant_base: If True, uses constant 0.5 probability; otherwise uses random.
            accesses_real_acceptance: If True, queries opponents directly; otherwise uses utilities.
        """
        randomizing_model: DiscreteAcceptanceModel
        peaking_model: DiscreteAcceptanceModel
        if adaptive:
            randomizing_model = AdaptiveDiscreteAcceptanceModel(
                outcomes=outcomes,
                rejection_discount=rejection_discount,
                rejection_delta=rejection_delta,
            )
        elif constant_base:
            randomizing_model = ConstantDiscreteAcceptanceModel(outcomes=outcomes)
        else:
            randomizing_model = RandomDiscreteAcceptanceModel(outcomes=outcomes)
        if accesses_real_acceptance:
            peaking_model = PeekingDiscreteAcceptanceModel(
                opponents=opponents, outcomes=outcomes
            )
        else:
            peaking_model = PeekingProbabilisticDiscreteAcceptanceModel(
                opponents=opponents, outcomes=outcomes
            )
        if uncertainty < 1e-7:
            super().__init__(outcomes=outcomes, models=[peaking_model], weights=[1.0])
        elif uncertainty > 1.0 - 1e-7:
            super().__init__(
                outcomes=outcomes, models=[randomizing_model], weights=[1.0]
            )
        else:
            super().__init__(
                outcomes=outcomes,
                models=[peaking_model, randomizing_model],
                weights=[1.0 - uncertainty, uncertainty],
            )
