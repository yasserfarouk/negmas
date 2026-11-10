from __future__ import annotations

import pytest

from negmas import SAOMechanism, make_issue
from negmas.models.acceptance import (
    PeekingDiscreteAcceptanceModel,
    UncertainOpponentModel,
)
from negmas.preferences import LinearAdditiveUtilityFunction
from negmas.sao import AspirationNegotiator, ResponseType


class _Threshold(AspirationNegotiator):
    """Rejects everything in the real negotiation; when peeked, accepts x >= 5."""

    peek = False

    def respond(self, state, source=None):
        if not self.peek:
            return ResponseType.REJECT_OFFER
        return (
            ResponseType.ACCEPT_OFFER
            if state.current_offer is not None and state.current_offer[0] >= 5
            else ResponseType.REJECT_OFFER
        )


def _running_negotiation():
    issues = [make_issue(10, "x")]
    m = SAOMechanism(issues=issues, n_steps=20)
    a, b = AspirationNegotiator(), _Threshold()
    for n in (a, b):
        m.add(
            n,
            ufun=LinearAdditiveUtilityFunction.random(
                issues=issues, reserved_value=0.0
            ),
        )
    m.step()
    assert m.state.running
    b.peek = True
    return m, b, list(m.outcome_space.enumerate_or_sample())


@pytest.mark.parametrize("uncertain", [False, True])
def test_peeking_acceptance_model_queries_opponent(uncertain):
    m, b, outcomes = _running_negotiation()
    current_before = m.state.current_offer
    if uncertain:
        model = UncertainOpponentModel(
            outcomes=outcomes,
            opponents=b,
            uncertainty=0.0,
            accesses_real_acceptance=True,
        )
    else:
        model = PeekingDiscreteAcceptanceModel(outcomes=outcomes, opponents=b)
    probs = [model.probability_of_acceptance(o) for o in outcomes]
    assert probs == [1.0 if o[0] >= 5 else 0.0 for o in outcomes]
    # peeking must not alter the real negotiation state
    assert m.state.current_offer == current_before
