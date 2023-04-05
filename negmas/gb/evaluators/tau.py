from __future__ import annotations

from collections import defaultdict
from sys import maxsize
from typing import Literal

from attrs import define, field

from negmas.outcomes import Outcome

from ..common import GBState, ResponseType
from ..evaluators.base import EvaluationStrategy

INFINITE = maxsize
"""Stands for an infinite int value"""

__all__ = ["TAUEvaluationStrategy", "INFINITE"]


@define
class TAUEvaluationStrategy(EvaluationStrategy):
    """
    Implements the Tentative-Accept Unique-Offers Generalized Bargaining Protocol.
    """

    n_outcomes: int = INFINITE
    cardinality: int = INFINITE
    _accepted: dict[Outcome | None, set[str]] = field(factory=lambda: defaultdict(set))
    _offered: dict[Outcome | None, set[str]] = field(factory=lambda: defaultdict(set))
    _repeating: dict[str, bool] = field(factory=lambda: defaultdict(bool))
    _last: dict[str, Outcome | None] = field(factory=lambda: defaultdict(Outcome))

    def __call__(
        self, negotiator_ids: list[str], state: GBState, history: list[GBState]
    ) -> Outcome | None | Literal["continue"]:
        for source, t in state.threads.items():
            offer = t.new_offer
            self._repeating[source] = offer == self._last[source]
            self._last[source] = offer
        # end the negotiation once all negotiators are repeating
        if (len(self._repeating) == state.n_negotiators) and all(
            list(self._repeating.values())
        ):
            return None

        # it is impossible to have more rounds than the number of outcomes. We should never hit this condition.
        if state.step > self.n_outcomes:
            return None

        # now we can start checking for agreement
        accepted, offered = self._accepted, self._offered

        def register(negotiator, offer, responses):
            """Register the offer and response in offered/accepted dicts"""
            if offer is None:
                return False
            offered[offer].add(negotiator)
            for responder, response in responses.items():
                if response == ResponseType.END_NEGOTIATION:
                    return False
                if response == ResponseType.ACCEPT_OFFER:
                    accepted[offer].add(responder)
            return True

        def registerall(s: GBState):
            """Updates offered/accepted dicts given a state"""
            for source, t in s.threads.items():
                offer, responses = t.new_offer, t.new_responses
                if not register(source, offer, responses):
                    return False
            return True

        # re-calcuate accepted, offered if we need to use only a part of the history
        nh, c = len(history), self.cardinality
        if 0 < c <= nh:
            accepted, offered = defaultdict(set), defaultdict(set)
            for s in history[nh - c + 1 :]:
                if not registerall(s):
                    return None

        if not registerall(state):
            return None

        outcomes = set(accepted.keys()).union(set(offered.keys()))
        n_negotiators = len(negotiator_ids)

        for outcome in outcomes:
            if len(accepted[outcome]) == len(offered[outcome]) == n_negotiators:
                return outcome
        return "continue"
