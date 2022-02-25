from __future__ import annotations

import time

import pytest

from negmas.outcomes.base_issue import make_issue
from negmas.preferences import LinearAdditiveUtilityFunction as LUFun
from negmas.preferences.crisp.nonlinear import HyperRectangleUtilityFunction
from negmas.preferences.value_fun import AffineFun, IdentityFun, LinearFun
from negmas.sao.mechanism import SAOMechanism
from negmas.sao.negotiators import AspirationNegotiator, NaiveTitForTatNegotiator
from negmas.sao.negotiators.limited import LimitedOutcomesNegotiator


def test_a_session():
    time.perf_counter()
    n = 50
    p = SAOMechanism(outcomes=n, n_steps=50)
    for _ in range(4):
        p.add(
            LimitedOutcomesNegotiator(p_ending=0.01, name=f"agent {_}"),
            preferences=HyperRectangleUtilityFunction(
                [None], [lambda x: x[0]], outcomes=((_,) for _ in range(n))
            ),
        )
    p.run()
    # print(f'{len(p.negotiators)} negotiators')
    assert len(p.history) > 0
    # print(f'Took {time.perf_counter()-start}')


@pytest.mark.skip(
    "Will check this later. Someone breaks the negotiation and there is a difference based on utility normalizerion!!"
)
def test_buy_sell_session():
    # create negotiation agenda (issues)
    issues = [
        make_issue(name="price", values=10),
        make_issue(name="quantity", values=(1, 11)),
        make_issue(name="delivery_time", values=10),
    ]

    # create the mechanism
    session = SAOMechanism(issues=issues, n_steps=20)

    # define buyer and seller utilities
    seller_utility = LUFun(
        values=[IdentityFun(), LinearFun(0.2), AffineFun(-1, bias=9.0)],
        outcome_space=session.outcome_space,
    ).scale_max(1.0)

    buyer_utility = LUFun(
        values={
            "price": AffineFun(-1, bias=9.0),
            "quantity": LinearFun(0.2),
            "delivery_time": IdentityFun(),
        },
        outcome_space=session.outcome_space,
    ).scale_max(1.0)

    # create and add buyer and seller negotiators
    session.add(NaiveTitForTatNegotiator(name="buyer"), preferences=buyer_utility)
    session.add(AspirationNegotiator(name="seller"), preferences=seller_utility)

    # run the negotiation and show the results
    state = session.run()
    import matplotlib.pyplot as plt

    assert state.agreement is not None, f"{neg.plot()}{plt.show()}{neg.extended_trace}"


if __name__ == "__main__":
    pytest.main(args=[__file__])
