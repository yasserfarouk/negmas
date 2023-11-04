from __future__ import annotations

import matplotlib.pyplot as plt
import pytest

from negmas import NaiveTitForTatNegotiator, SAOMechanism, make_issue
from negmas.gb.negotiators.micro import MiCRONegotiator
from negmas.preferences import LinearAdditiveUtilityFunction as LUFun
from negmas.preferences.value_fun import AffineFun, IdentityFun, LinearFun, TableFun
from negmas.sao.negotiators.timebased import BoulwareTBNegotiator
from tests.switches import NEGMAS_RUN_TEMP_FAILING

SHOW_PLOTS = False


def run_buyer_seller(buyer, seller, normalized=False, callbacks=False, n_steps=100):
    # create negotiation agenda (issues)
    issues = [
        make_issue(name="price", values=10),
        make_issue(name="quantity", values=(1, 11)),
        make_issue(name="delivery_time", values=["today", "tomorrow", "nextweek"]),
    ]

    # create the mechanism
    session = SAOMechanism(
        issues=issues, n_steps=n_steps, time_limit=None, extra_callbacks=callbacks
    )

    # define buyer and seller utilities
    seller_utility = LUFun(
        values=[  # type: ignore
            IdentityFun(),
            LinearFun(0.2),
            TableFun(dict(today=1.0, tomorrow=0.2, nextweek=0.0)),
        ],
        outcome_space=session.outcome_space,
    )

    buyer_utility = LUFun(
        values={  # type: ignore
            "price": AffineFun(-1, bias=9.0),
            "quantity": LinearFun(0.2),
            "delivery_time": TableFun(dict(today=0, tomorrow=0.7, nextweek=1.0)),
        },
        outcome_space=session.outcome_space,
    )
    if normalized:
        seller_utility = seller_utility.scale_max(1.0)
        buyer_utility = buyer_utility.scale_max(1.0)

    # create and add buyer and seller negotiators
    b = buyer(name="buyer")
    s = seller(name="seller")
    s.name += f"{s.short_type_name}"
    b.name += f"{b.short_type_name}"
    session.add(b, ufun=buyer_utility)
    session.add(s, ufun=seller_utility)

    session.run()
    if SHOW_PLOTS:
        session.plot()
        from matplotlib import pyplot as plt

        plt.show()
    return session


def kind_tft(*args, **kwargs):
    kwargs["kindness"] = 0.01
    kwargs["punish"] = False
    return NaiveTitForTatNegotiator(*args, **kwargs)


def test_buy_sell_asp_asp():
    session = run_buyer_seller(BoulwareTBNegotiator, BoulwareTBNegotiator)
    assert session.agreement


# @pytest.mark.skipif(
#     not NEGMAS_RUN_TEMP_FAILING,
#     reason="Not always getting to an greement. This is not a bug necesarily but should be investigated",
# )
# def test_buy_sell_tft_tft():
#     # todo: find out why this is not always getting and agreeming
#     session = run_buyer_seller(kind_tft, NaiveTitForTatNegotiator)
#     # assert session.agreement, f"{session.plot()}{plt.show()}{session.extended_trace}"
#     assert session.agreement, f"{session.extended_trace}"


def test_buy_sell_asp_tft():
    session = run_buyer_seller(BoulwareTBNegotiator, NaiveTitForTatNegotiator)
    assert session.state.agreement


def test_buy_sell_micro():
    session = run_buyer_seller(
        MiCRONegotiator, MiCRONegotiator, callbacks=True, n_steps=10000
    )
    assert session.agreement


if __name__ == "__main__":
    SHOW_PLOTS = True
