from __future__ import annotations

from collections import defaultdict

from negmas.gb.common import GBState, ResponseType
from negmas.gb.mechanisms.base import BaseGBMechanism
from negmas.mechanisms import MechanismStepResult
from negmas.outcomes import Outcome

__all__ = ["TAUMechanism"]


class TAUMechanism(BaseGBMechanism):
    def __init__(
        self, *args, accept_in_any_thread: bool = True, parallel: bool = True, **kwargs
    ):
        super().__init__(*args, parallel=parallel, **kwargs)
        self._offers: dict[str, set[Outcome]] = defaultdict(set)
        self._acceptances: dict[Outcome, set[str]] = defaultdict(set)
        self._acceptances_per_negotiator: dict[
            Outcome, dict[str, set[str]]
        ] = defaultdict(lambda: defaultdict(set))
        self._proposals: dict[Outcome, set[str]] = defaultdict(set)
        self._last_offer: dict[str, Outcome] = dict()
        self._accept_in_any_thread = accept_in_any_thread

    def __call__(self, state: GBState, action=None) -> MechanismStepResult:
        assert (
            action is None
        ), "passing action != None to TAUMechanism is not yet supported"
        n = len(self.negotiators)

        def _check_agreement(offer, acceptor, n=n):
            if not self._accept_in_any_thread:
                if acceptor:
                    self._acceptances_per_negotiator[offer][acceptor].add(thread_id)
                return (
                    len(self._acceptances_per_negotiator[offer]) == n
                    and (len(self._proposals[offer]) == n)
                    and all(
                        len(_) == n - 1
                        for _ in self._acceptances_per_negotiator[offer].values()
                    )
                )
            if acceptor:
                self._acceptances[offer].add(acceptor)
            return len(self._acceptances[offer]) == n and (
                len(self._proposals[offer]) == n
            )

        # print(f"Round {self._current_state.step}")
        results = self.run_threads()

        # check that the offer is not repeated
        # Find any offers that have been accepted by anyone.
        # If someone ended the negotiation, return
        new_offers = []

        # TODO: support dynamic entry by:
        # keeping track of who left in this round. Once we hit 0 negotiators, optionally end the negotiation
        # when a negotiator leaves, remove it from all sets we keep and from the threads
        if state.step > self.outcome_space.cardinality + 1:
            raise RuntimeError("TAU should never run for more than N. Outcomes rounds")
        repeating = []
        for thread_id, (tstate, _) in results.items():
            offer = tstate.new_offer
            last_offer = self._last_offer.get(thread_id, None)
            if offer == last_offer:
                repeating.append(thread_id)
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
            # find out who accepted this offer
            acceptedby = [
                neg  # if not self._accept_in_every_thread else (neg, thread_id)
                for neg, response in tstate.new_responses.items()
                if response == ResponseType.ACCEPT_OFFER
            ]
            # In a bilateral negotiation, the offer cannot become an agreement except if it was just accepted in this round
            # That is because it is not possible that the offer was already accepted by both agents but was not offered by
            # one of them. In a multilateral negotiation, it may happen that an offer was accepted by all negotiators but
            # not yet offered by one of them and can become an agreement once this last one offers it. For example we may
            # have three agents A, B, C (three threads). A, B offered the outcome and it was accepted by B, C in thread A
            # and by A in thread C. This will not make it an agreement yet. Nevertheless, once C offers it, it now becomes
            # an agreement immediately even if no one accepts it now.
            # This whole discussion assumes that in a agreement means being offered at least once and accepted at least once
            # (in any thread) by every negotiator. If, on the other hand, we insist that an agreement means being offered at least
            # once by every negotiator and accepted by every negotiator IN EVERY thread not owned by it; then an agreement will
            # only happen at a round in which it is accepted by someone like in the bilateral case.
            # TODO: I think the second approach is better but I need to think about it more. I added it but need to test is
            if self._accept_in_any_thread and n == 2 and not acceptedby:
                continue
            # we know that at least someone accepted this offer.
            for neg in acceptedby:
                assert offer is not None
                if _check_agreement(offer, neg, n):
                    state.agreement = offer
                    return MechanismStepResult(state)
        # If everyone is repeating, end the negotaition
        if len(repeating) == n:
            state.broken = True
            return MechanismStepResult(state)

        # final check for any acceptable offers we may have missed
        for offer in new_offers:
            if _check_agreement(offer, None, n):
                state.agreement = offer
                return MechanismStepResult(state)

        return MechanismStepResult(state)
