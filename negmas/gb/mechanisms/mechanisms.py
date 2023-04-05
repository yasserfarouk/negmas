from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Set

from negmas import check_one_and_only, ensure_os
from negmas.gb.common import GBState, ResponseType
from negmas.gb.constraints import RepeatFinalOfferOnly
from negmas.gb.evaluators import (
    INFINITE,
    GAOEvaluationStrategy,
    TAUEvaluationStrategy,
    any_accept,
)
from negmas.gb.mechanisms.base import BaseGBMechanism, SerialGBMechanism
from negmas.mechanisms import MechanismStepResult
from negmas.outcomes import Issue, Outcome, OutcomeSpace
from negmas.plots.util import opacity_colorizer

if TYPE_CHECKING:
    from negmas.plots.util import Colorizer

__all__ = ["GAOMechanism", "TAUMechanism", "GeneralizedTAUMechanism"]


class GAOMechanism(SerialGBMechanism):
    def __init__(self, *args, **kwargs):
        kwargs["local_evaluator_type"] = GAOEvaluationStrategy
        kwargs["response_combiner"] = any_accept
        super().__init__(*args, **kwargs)

    def plot(self, *args, colorizer: Colorizer = opacity_colorizer, **kwargs):
        return super().plot(*args, colorizer=colorizer, **kwargs)


class GeneralizedTAUMechanism(SerialGBMechanism):
    def __init__(
        self,
        *args,
        cardinality=INFINITE,
        min_unique=0,
        outcome_space: OutcomeSpace | None = None,
        issues: list[Issue] | None = None,
        outcomes: list[Outcome] | int | None = None,
        **kwargs,
    ):
        check_one_and_only(outcome_space, issues, outcomes)
        outcome_space = ensure_os(outcome_space, issues, outcomes)
        kwargs["evaluator_type"] = TAUEvaluationStrategy
        kwargs["evaluator_params"] = dict(
            cardinality=cardinality, n_outcomes=outcome_space.cardinality
        )
        kwargs["local_constraint_type"] = RepeatFinalOfferOnly
        kwargs["local_constraint_params"] = dict(n=min_unique)
        super().__init__(
            *args,
            outcome_space=outcome_space,
            **kwargs,
        )


class TAUMechanism(BaseGBMechanism):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._offers: dict[str, set[Outcome]] = defaultdict(set)
        self._acceptances: dict[Outcome, set[str]] = defaultdict(set)
        self._proposals: dict[Outcome, set[str]] = defaultdict(set)
        self._last_offer: dict[str, Outcome] = dict()

    def __call__(self, state: GBState) -> MechanismStepResult:
        # print(f"Round {self._current_state.step}")
        results = self.run_threads()

        # check that the offer is not repeated
        # Find any offers that have been accepted by anyone.
        # If someone ended the negotiation, return
        new_offers = []

        n = len(self.negotiators)
        # todo: support dynamic entry by:
        # keeping track of who left in this round. Once we hit 0 negotiators, optionally end the negotiation
        # when a negotiator leaves, remove it from all sets we keep and from the threads
        for thread_id, (tstate, _) in results.items():
            offer = tstate.new_offer
            last_offer = self._last_offer.get(thread_id, None)
            if offer is None or (
                offer in self._offers[thread_id]
                and (last_offer is None or last_offer != offer)
            ):
                state.broken = True
                return MechanismStepResult(state)

            self._last_offer[thread_id] = offer
            self._proposals[offer].add(thread_id)
            self._offers[thread_id].add(offer)
            new_offers.append(offer)
            offer = tstate.new_offer
            if any(
                _ == ResponseType.END_NEGOTIATION for _ in tstate.new_responses.keys()
            ):
                state.broken = True
                return MechanismStepResult(state)
            # finf out who accepted this offer
            acceptedby = [
                neg
                for neg, response in tstate.new_responses.items()
                if response == ResponseType.ACCEPT_OFFER
            ]
            if not acceptedby:
                continue
            # we know that at least someone accpeted this offer.
            for neg in acceptedby:
                assert offer is not None
                self._acceptances[offer].add(neg)
                if len(self._acceptances[offer]) == n and (
                    n == 2 or len(self._proposals[offer]) == n
                ):
                    state.agreement = offer
                    return MechanismStepResult(state)
        # final check for any acceptable offers we may have missed
        for offer in new_offers:
            if len(self._acceptances[offer]) == len(self.negotiators) and (
                n == 2 or len(self._proposals[offer]) == n
            ):
                state.agreement = offer
                return MechanismStepResult(state)

        return MechanismStepResult(state)
