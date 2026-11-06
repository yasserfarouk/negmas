"""Tests for per-object seeding (`RandomStream`, `seeded`) and seeded ids."""

from __future__ import annotations

import copy
import pickle
import random
import uuid

import numpy as np
import pytest

from negmas.helpers.rand import (
    RandomStream,
    apply_seed,
    seeded,
    uses_random_stream,
    uuid4,
)
from negmas.outcomes import IntIssue, StrIssue, make_issue, make_os
from negmas.preferences import LinearAdditiveUtilityFunction
from negmas.preferences.inv_ufun import SamplingInverseUtilityFunction
from negmas.preferences.pareto_sampler import (
    AdaptiveParetoSampler,
    BruteForceParetoSampler,
    IPSParetoSampler,
    MOBANOSParetoSampler,
    NB3ParetoSampler,
)
from negmas.preferences.value_fun import LambdaFun, TableFun


@pytest.fixture(autouse=True)
def _restore_global_seed_state():
    """Keep these tests from leaking seed state into the rest of the suite."""
    import negmas.helpers.rand as m

    saved = (m._seed, m._id_rng, random.getstate(), np.random.get_state())
    yield
    m._seed, m._id_rng = saved[0], saved[1]
    random.setstate(saved[2])
    np.random.set_state(saved[3])


def _draws(n: int = 5) -> tuple:
    return tuple(random.random() for _ in range(n)) + tuple(np.random.rand(n))


# --- RandomStream / seeded --------------------------------------------------


def test_random_stream_same_seed_same_draws():
    with RandomStream(3):
        a = _draws()
    with RandomStream(3):
        b = _draws()
    with RandomStream(4):
        c = _draws()
    assert a == b
    assert a != c


def test_random_stream_restores_global_state():
    random.seed(11)
    np.random.seed(11)
    expected = _draws()
    random.seed(11)
    np.random.seed(11)
    with RandomStream(99):
        _draws(50)
    assert _draws() == expected


def test_random_stream_continues_and_ignores_outside_draws():
    whole = RandomStream(5)
    with whole:
        reference = _draws(6)
    stream = RandomStream(5)
    with stream:
        first = tuple(random.random() for _ in range(3))
    _draws(17)  # the rest of the program draws in between
    with stream:
        second = tuple(random.random() for _ in range(3))
    assert first + second == reference[:6]


def test_random_stream_none_is_a_no_op():
    random.seed(1)
    np.random.seed(1)
    expected = _draws()
    random.seed(1)
    np.random.seed(1)
    with RandomStream(None) as s:
        got = _draws()
    assert got == expected
    assert s.seed is None


def test_random_stream_is_reentrant():
    flat = RandomStream(8)
    with flat:
        reference = [random.random() for _ in range(4)]
    nested = RandomStream(8)
    with nested:
        a = [random.random() for _ in range(2)]
        with nested:
            b = [random.random() for _ in range(2)]
    assert a + b == reference


def test_random_stream_restores_after_exception():
    random.seed(2)
    expected = random.random()
    random.seed(2)
    with pytest.raises(ZeroDivisionError):
        with RandomStream(1):
            random.random()
            _ = 1 / 0
    assert random.random() == expected


def test_random_stream_reseed():
    s = RandomStream(1)
    with s:
        a = random.random()
    s.reseed(1)
    with s:
        assert random.random() == a
    with s:
        with pytest.raises(RuntimeError):
            s.reseed(2)


def test_random_stream_survives_pickle_and_copy():
    s = RandomStream(21)
    with s:
        random.random()
    for clone in (pickle.loads(pickle.dumps(s)), copy.deepcopy(s)):
        with clone:
            x = _draws()
        fresh = pickle.loads(pickle.dumps(s))
        with fresh:
            assert _draws() == x


def test_seeded_block_is_reproducible():
    os = make_os([make_issue(100, "a"), make_issue(100, "b")])
    with seeded(7):
        a = [os.random_outcome() for _ in range(10)]
    with seeded(7):
        b = [os.random_outcome() for _ in range(10)]
    assert a == b


def test_uses_random_stream_decorator():
    class Thing:
        def __init__(self, seed):
            self._random_stream = RandomStream(seed)

        @uses_random_stream
        def draw(self):
            return random.random()

    assert Thing(4).draw() == Thing(4).draw()


# --- uuid4 ------------------------------------------------------------------


def test_uuid4_unseeded_is_random(monkeypatch):
    import negmas.helpers.rand as m

    monkeypatch.setattr(m, "_id_rng", None)
    assert uuid4() != uuid4()
    assert uuid4().version == 4


def test_uuid4_reproducible_once_seeded():
    apply_seed(10)
    a = [uuid4() for _ in range(3)]
    apply_seed(10)
    b = [uuid4() for _ in range(3)]
    apply_seed(11)
    c = [uuid4() for _ in range(3)]
    assert a == b
    assert a != c
    assert len(set(a)) == 3
    assert all(isinstance(x, uuid.UUID) and x.version == 4 for x in a)


def test_uuid4_does_not_consume_the_global_stream():
    apply_seed(10)
    expected = _draws()
    apply_seed(10)
    for _ in range(5):
        uuid4()
    assert _draws() == expected


def test_seeded_runs_repeat_object_ids():
    from negmas.sao import SAOMechanism
    from negmas.sao.negotiators import AspirationNegotiator

    def ids():
        apply_seed(3)
        m = SAOMechanism(outcomes=10, n_steps=5)
        n = AspirationNegotiator()
        return m.id, n.id, n.name

    assert ids() == ids()


# --- Mechanism --------------------------------------------------------------


def test_mechanism_does_not_draw_per_wall_second_without_pend_per_second(monkeypatch):
    """The global stream must not depend on how long the steps took."""
    from negmas.sao import SAOMechanism
    from negmas.sao.negotiators import RandomNegotiator

    def run(slow: bool):
        apply_seed(17)
        os = make_os([make_issue(20, "a"), make_issue(20, "b")])
        m = SAOMechanism(outcome_space=os, n_steps=30)
        if slow:
            # every step looks like it started in a new wall-clock second
            ticks = iter(range(1, 10_000))
            monkeypatch.setattr(
                SAOMechanism, "time", property(lambda self: float(next(ticks)))
            )
        for _ in range(2):
            m.add(
                RandomNegotiator(p_acceptance=0.05, p_rejection=0.0),
                ufun=LinearAdditiveUtilityFunction.random(outcome_space=os),
            )
        m.run()
        monkeypatch.undo()
        return [tuple(_.current_offer or ()) for _ in m.history], random.random()

    assert run(False) == run(True)


# --- samplers ---------------------------------------------------------------


def _infinite_scenario():
    """An outcome space the samplers can only sample, never enumerate."""
    os = make_os(
        [
            IntIssue("q", min=1, typical=(1, 50)),
            StrIssue("n", max_length=10),
            make_issue(["a", "b"], "k"),
        ]
    )

    def mk(target, sign):
        return LinearAdditiveUtilityFunction(
            [
                LambdaFun(lambda x: -abs(x - target), min_value=-1e9, max_value=0.0),
                LambdaFun(lambda s: sign * len(s) / 10, min_value=-1.0, max_value=1.0),
                TableFun({"a": float(sign > 0), "b": float(sign < 0)}),
            ],
            outcome_space=os,
        )

    return os, mk(10, 1), mk(40, -1)


SAMPLERS = [
    AdaptiveParetoSampler,
    BruteForceParetoSampler,
    IPSParetoSampler,
    MOBANOSParetoSampler,
    NB3ParetoSampler,
]


def _front(cls, seed, scenario=None):
    _, a, b = scenario or _infinite_scenario()
    s = cls(seed=seed)
    s.init(a, b)
    return list(s.pareto_outcomes())


@pytest.mark.parametrize("cls", SAMPLERS)
def test_pareto_sampler_seed_is_reproducible(cls):
    first = _front(cls, 5)
    random.seed()  # disturb the global generators between the two runs
    np.random.seed()
    _draws(13)
    assert first
    assert _front(cls, 5) == first


@pytest.mark.parametrize("cls", SAMPLERS)
def test_pareto_sampler_seed_leaves_global_stream_alone(cls):
    # built first: creating ufuns draws their names from the global stream
    scenario = _infinite_scenario()
    random.seed(0)
    np.random.seed(0)
    expected = _draws()
    random.seed(0)
    np.random.seed(0)
    _front(cls, 5, scenario)
    assert _draws() == expected


@pytest.mark.parametrize("cls", SAMPLERS)
def test_pareto_sampler_default_is_unseeded(cls):
    assert cls().seed is None
    assert cls(seed=3).seed == 3


def test_pareto_sampler_seed_actually_matters():
    """Different seeds explore different candidates on an infinite space."""
    fronts = {tuple(_front(BruteForceParetoSampler, s)) for s in range(6)}
    assert len(fronts) > 1


def test_make_pareto_sampler_forwards_seed():
    _, a, b = _infinite_scenario()
    s = a.make_pareto_sampler(b, pareto_sampler=BruteForceParetoSampler, seed=9)
    assert s.seed == 9  # type: ignore[attr-defined]


def test_nice_tit_for_tat_forwards_sampler_seed():
    from negmas.gb.negotiators.titfortat import NiceTitForTatNegotiator

    n = NiceTitForTatNegotiator(sampler_seed=12)
    assert n._offering.sampler_seed == 12  # type: ignore[attr-defined]


def test_sampling_inverter_seed_is_reproducible():
    os = make_os([make_issue(30, "a"), make_issue(30, "b"), make_issue(30, "c")])

    def run():
        u = LinearAdditiveUtilityFunction.random(outcome_space=os, normalized=True)
        return u

    with seeded(1):
        u = run()

    def answers(seed):
        inv = SamplingInverseUtilityFunction(u, seed=seed)
        inv.init()
        return [inv.one_in((0.3, 0.7), normalized=True) for _ in range(5)] + [
            inv.some((0.2, 0.8), normalized=True, n=4)
        ]

    a = answers(4)
    _draws(9)
    assert answers(4) == a
    assert SamplingInverseUtilityFunction(u).seed is None


def test_random_stream_is_decorrelated_from_the_global_seed():
    apply_seed(42)
    global_draws = _draws()
    with RandomStream(42):
        assert _draws() != global_draws


# --- ufun generators and outcome-space sampling ------------------------------


def _small_os():
    return make_os([make_issue(4, "a"), make_issue(3, "b"), make_issue(5, "c")])


def _numeric_issues():
    return [make_issue((0, 9), "x"), make_issue((0, 4), "y")]


def _signature(ufun, os):
    return ([float(ufun(o)) for o in os.enumerate()], float(ufun.reserved_value))


def _ufun_makers():
    """Each maker builds its inputs, then returns ``seed -> (ufun, space)``.

    Inputs are built up front because creating issues and outcome spaces draws
    their names from the global generators.
    """
    import negmas.preferences as P

    def on_os(cls, **kw):
        def build():
            os = _small_os()
            return lambda seed: (cls.random(outcome_space=os, seed=seed, **kw), os)

        return build

    def on_issues(cls, **kw):
        def build():
            issues = _numeric_issues()
            os = make_os(issues)
            return lambda seed: (cls.random(issues, seed=seed, **kw), os)

        return build

    def positional(cls):
        def build():
            os = _small_os()
            return lambda seed: (cls.random(os, (0.0, 1.0), seed=seed), os)

        return build

    return {
        "linear": on_os(P.LinearAdditiveUtilityFunction),
        "gpa": on_os(P.GPAUtilityFunction),
        "gla": on_os(P.GLAUtilityFunction),
        "pa": on_os(P.PAUtilityFunction),
        "mapping": on_os(P.MappingUtilityFunction),
        "affine": on_issues(P.AffineUtilityFunction),
        "const": on_issues(P.ConstUtilityFunction, normalized=False),
        "exp_discounted": on_issues(P.ExpDiscountedUFun),
        "lin_discounted": on_issues(P.LinDiscountedUFun),
        "weighted": positional(P.WeightedUtilityFunction),
        "complex": positional(P.ComplexNonlinearUtilityFunction),
    }


@pytest.mark.parametrize("kind", sorted(_ufun_makers()))
def test_random_ufun_seed_is_reproducible(kind):
    make = _ufun_makers()[kind]()
    first = _signature(*make(5))
    _draws(11)
    assert _signature(*make(5)) == first


@pytest.mark.parametrize("kind", sorted(_ufun_makers()))
def test_random_ufun_seed_leaves_global_stream_alone(kind):
    make = _ufun_makers()[kind]()
    random.seed(0)
    np.random.seed(0)
    expected = _draws()
    random.seed(0)
    np.random.seed(0)
    make(5)
    assert _draws() == expected


def test_random_ufun_different_seeds_differ():
    make = _ufun_makers()["linear"]()
    assert len({str(_signature(*make(s))) for s in range(5)}) > 1


def test_random_ufun_without_seed_is_unchanged():
    """No seed: the result follows the global generators, as before."""
    from negmas.preferences import LinearAdditiveUtilityFunction as U

    os = _small_os()
    random.seed(3)
    np.random.seed(3)
    a = _signature(U.random(outcome_space=os), os)
    random.seed(3)
    np.random.seed(3)
    b = _signature(U.random(outcome_space=os), os)
    assert a == b


def test_prob_mapping_ufun_seed_is_reproducible():
    from negmas.preferences import ProbMappingUtilityFunction as U

    os = _small_os()

    def sig(seed):
        u = U.random(os, seed=seed)
        return [str(u(o)) for o in os.enumerate()]

    first = sig(2)
    _draws(5)
    assert sig(2) == first


def test_ufun_class_generators_seed_is_reproducible():
    from negmas.preferences.crisp_ufun import UtilityFunction

    os = _small_os()
    outcomes = list(os.enumerate())

    def sig(seed):
        many = UtilityFunction.generate_random(3, outcomes, seed=seed)
        a, b = UtilityFunction.generate_bilateral(outcomes, seed=seed)
        c, d = UtilityFunction.generate_random_bilateral(outcomes, seed=seed)
        return [[float(u(o)) for o in outcomes] for u in [*many, a, b, c, d]]

    first = sig(4)
    _draws(5)
    assert sig(4) == first


def test_generators_module_seed_is_reproducible():
    from negmas.preferences.generators import (
        generate_multi_issue_ufuns,
        generate_single_issue_ufuns,
        generate_ufuns_for,
        make_pareto,
        sample_between,
    )

    os = make_os([make_issue(5, "a"), make_issue(4, "b")])

    def sig(seed):
        single = generate_single_issue_ufuns(10, 20, seed=seed)
        multi = generate_multi_issue_ufuns(2, sizes=(4, 5), seed=seed)
        fors = generate_ufuns_for(os, seed=seed)
        out = [
            [float(u(o)) for o in u.outcome_space.enumerate()]  # type: ignore
            for u in (*single, *multi, *fors)
        ]
        out.append(list(sample_between(0.0, 1.0, 6, seed=seed)))
        out.append(make_pareto([(0.0, 1.0), (1.0, 0.0)], 8, seed=seed))
        return out

    first = sig(6)
    _draws(5)
    assert sig(6) == first


@pytest.mark.parametrize(
    "os",
    [
        _small_os(),
        make_os([IntIssue("q", min=1, typical=(1, 50)), StrIssue("n", max_length=6)]),
    ],
    ids=["discrete", "infinite"],
)
def test_outcome_space_sampling_seed_is_reproducible(os):
    def sig(seed):
        return (
            os.random_outcome(seed=seed),
            list(os.sample(5, with_replacement=True, seed=seed)),
            list(os.enumerate_or_sample(max_cardinality=10, seed=seed)),
        )

    first = sig(8)
    _draws(5)
    assert sig(8) == first
    assert len({str(sig(s)) for s in range(5)}) > 1


def test_outcome_space_sampling_without_seed_follows_global_generators():
    os = _small_os()
    random.seed(1)
    np.random.seed(1)
    a = list(os.sample(5, with_replacement=True))
    random.seed(1)
    np.random.seed(1)
    assert list(os.sample(5, with_replacement=True)) == a


def test_distribution_and_prob_ufun_sample_seed():
    from negmas.helpers.prob import ScipyDistribution
    from negmas.preferences import IPUtilityFunction

    d = ScipyDistribution("uniform", loc=0.0, scale=1.0)
    first = list(d.sample(4, seed=3))
    _draws(5)
    assert list(d.sample(4, seed=3)) == first

    os = _small_os()
    outcomes = list(os.enumerate())
    u = IPUtilityFunction(
        outcomes=outcomes,
        distributions=[
            ScipyDistribution("uniform", loc=0.0, scale=1.0) for _ in outcomes
        ],
    )

    def sig(seed):
        crisp = u.sample(seed=seed)
        return [float(crisp(o)) for o in outcomes]

    first2 = sig(1)
    _draws(5)
    assert sig(1) == first2
