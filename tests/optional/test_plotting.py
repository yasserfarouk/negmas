import pytest

from negmas.gb.negotiators.timebased import AspirationNegotiator
from negmas.outcomes import make_issue, make_os
from negmas.plots.util import plot_offline_run
from negmas.preferences.crisp.linear import LinearAdditiveUtilityFunction
from negmas.sao.mechanism import SAOMechanism


@pytest.mark.skip("A test added for manually testing plotting")
def test_mechanism_plot():
    os = make_os([make_issue(10) for _ in range(3)])
    m = SAOMechanism(n_steps=100, outcome_space=os)
    ufuns = LinearAdditiveUtilityFunction.generate_random_bilateral(
        list(os.enumerate_or_sample())
    )
    for i, u in enumerate(ufuns):
        u.outcome_space = os
        m.add(
            AspirationNegotiator(
                id=f"a{i}",
                name=f"n{i}",
                ufun=u,
            )
        )
    m.run()
    m.plot()
    plt.show()


@pytest.mark.skip("A test added for manually testing plotting")
def test_offline_plot():
    os = make_os([make_issue(10) for _ in range(3)])
    m = SAOMechanism(n_steps=100, outcome_space=os)
    ufuns = LinearAdditiveUtilityFunction.generate_random_bilateral(
        list(os.enumerate_or_sample())
    )
    for i, u in enumerate(ufuns):
        u.outcome_space = os
        m.add(
            AspirationNegotiator(
                id=f"a{i}",
                name=f"n{i}",
                ufun=u,
            )
        )
    m.run()
    trace = m.full_trace
    ids = m.negotiator_ids
    names = m.negotiator_names
    state = m.state
    plot_offline_run(trace, ids, ufuns, state, names)
    plt.show()
