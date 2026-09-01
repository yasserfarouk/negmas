"""Tests for saving/loading cached inverse utility functions with a scenario.

The safety property under test throughout is: a cached inverse is used **only**
when it is provably equivalent to one built from scratch, and any doubt at all
falls back to rebuilding.
"""

from __future__ import annotations

import json
import random

import numpy as np
import pytest

from negmas import Scenario, make_issue
from negmas.outcomes import make_os
from negmas.preferences import LinearAdditiveUtilityFunction as LU
from negmas.preferences.inv_ufun import (
    DefaultInverseUtilityFunction,
    PresortingInverseUtilityFunction,
)
from negmas.preferences.inv_ufun.persistence import (
    DEFAULT_INVERSE_CONFIGS,
    INVERSE_FOLDER_NAME,
    INVERSE_MANIFEST_NAME,
    can_cache_inverse,
    effective_config,
    load_state,
    outcome_space_fingerprint,
)
from negmas.sao import AspirationNegotiator
from negmas.sao.mechanism import SAOMechanism


def _os(n_issues: int = 3, n_values: int = 8, name: str = "D"):
    """A small discrete outcome space."""
    return make_os(
        [make_issue(list(range(n_values)), f"i{k}") for k in range(n_issues)], name=name
    )


def _scenario(n_ufuns: int = 2, reserved: float = 0.0, **kwargs) -> Scenario:
    """A scenario with random linear-additive ufuns over a small discrete space."""
    space = _os(**kwargs)
    return Scenario(
        outcome_space=space,
        ufuns=tuple(
            LU.random(outcome_space=space, reserved_value=reserved)
            for _ in range(n_ufuns)
        ),
    )


def _cache_dir(folder):
    """The inverse-cache folder inside a scenario folder."""
    return folder / INVERSE_FOLDER_NAME


def _served_from_cache(ufun, **kwargs) -> bool:
    """Whether a fresh inverter for ``ufun`` would be restored from disk.

    Exercises exactly the production path: build the inverter the way any caller
    would, then ask the ufun to restore it (which is what its ``init()`` does).
    """
    inverter = DefaultInverseUtilityFunction(ufun, **kwargs)
    return ufun.restore_inverse_from_cache(inverter)


def _count_evals(ufun, fn):
    """Runs ``fn`` and returns how many times ``ufun``'s class evaluated an outcome."""
    cls = type(ufun)
    original = cls.eval
    calls = [0]

    def counting(self, offer):
        calls[0] += 1
        return original(self, offer)

    cls.eval = counting
    try:
        result = fn()
    finally:
        cls.eval = original
    return result, calls[0]


# ---------------------------------------------------------------------------
# Saving is opt-in
# ---------------------------------------------------------------------------


def test_no_inverse_saved_by_default(tmp_path):
    """`dumpas` must never write an inverse cache unless explicitly asked."""
    _scenario().dumpas(tmp_path, type="yml")
    assert not _cache_dir(tmp_path).exists()


def test_inverse_saved_when_requested(tmp_path):
    """`save_inverse=True` writes a manifest plus arrays for every cached config."""
    s = _scenario()
    s.dumpas(tmp_path, type="yml", save_inverse=True)
    folder = _cache_dir(tmp_path)
    assert folder.is_dir()
    assert (folder / INVERSE_MANIFEST_NAME).is_file()
    expected = len(s.ufuns) * len(DEFAULT_INVERSE_CONFIGS)
    assert len(list(folder.glob("*.npz"))) == expected


def test_saving_without_inverse_deletes_existing_cache(tmp_path):
    """A re-save without ``save_inverse`` must delete a stale cache, not keep it."""
    s = _scenario()
    s.dumpas(tmp_path, type="yml", save_inverse=True)
    assert _cache_dir(tmp_path).is_dir()
    s.dumpas(tmp_path, type="yml")
    assert not _cache_dir(tmp_path).exists()


def test_update_preserves_existing_cache(tmp_path):
    """`update()` defaults to preserving whatever caching state the folder has."""
    _scenario().dumpas(tmp_path, type="yml", save_inverse=True)
    loaded = Scenario.load(tmp_path, load_cached_inverse=True)
    assert loaded is not None
    loaded.source = tmp_path
    assert loaded.update()
    assert _cache_dir(tmp_path).is_dir()


def test_update_does_not_create_cache_when_absent(tmp_path):
    """`update()` must not start caching for a scenario that never asked for it."""
    _scenario().dumpas(tmp_path, type="yml")
    loaded = Scenario.load(tmp_path, load_cached_inverse=True)
    assert loaded is not None
    loaded.source = tmp_path
    assert loaded.update()
    assert not _cache_dir(tmp_path).exists()


def test_update_can_force_off(tmp_path):
    """An explicit ``save_inverse=False`` removes the cache even via `update()`."""
    _scenario().dumpas(tmp_path, type="yml", save_inverse=True)
    loaded = Scenario.load(tmp_path, load_cached_inverse=True)
    assert loaded is not None
    loaded.source = tmp_path
    assert loaded.update(save_inverse=False)
    assert not _cache_dir(tmp_path).exists()


def test_no_pickle_in_saved_arrays(tmp_path):
    """Arrays must load with ``allow_pickle=False`` so loading can never execute code."""
    _scenario().dumpas(tmp_path, type="yml", save_inverse=True)
    for path in _cache_dir(tmp_path).glob("*.npz"):
        with np.load(path, allow_pickle=False) as data:
            assert "utils" in data


# ---------------------------------------------------------------------------
# Loading and equivalence
# ---------------------------------------------------------------------------


def test_cache_is_actually_used(tmp_path):
    """A loaded inverse must cost far fewer ufun evaluations than a rebuild."""
    space_size = 8**3
    _scenario().dumpas(tmp_path, type="yml", save_inverse=True)

    cached = Scenario.load(tmp_path, load_cached_inverse=True)
    fresh = Scenario.load(tmp_path)  # default: no cache
    assert cached is not None and fresh is not None

    _, cached_evals = _count_evals(cached.ufuns[0], cached.ufuns[0].invert)
    _, fresh_evals = _count_evals(fresh.ufuns[0], fresh.ufuns[0].invert)

    assert fresh_evals >= space_size
    assert cached_evals < fresh_evals


def test_loading_a_cache_requires_opting_in(tmp_path):
    """Loading a saved inverse must never happen by default, on any loader."""
    s = _scenario(n_ufuns=1)
    s.dumpas(tmp_path, type="yml", save_inverse=True)
    for loader in (
        lambda: Scenario.load(tmp_path),
        lambda: Scenario.from_yaml_folder(tmp_path),
    ):
        loaded = loader()
        assert loaded is not None
        ufun = loaded.ufuns[0]
        assert ufun._inverse_cache_folder is None
        assert not _served_from_cache(ufun)


def test_every_loader_honours_the_opt_in(tmp_path):
    """`load_cached_inverse=True` must work through each loader entry point."""
    _scenario(n_ufuns=1).dumpas(tmp_path, type="yml", save_inverse=True)
    for loader in (
        lambda: Scenario.load(tmp_path, load_cached_inverse=True),
        lambda: Scenario.from_yaml_folder(tmp_path, load_cached_inverse=True),
    ):
        loaded = loader()
        assert loaded is not None
        ufun = loaded.ufuns[0]
        assert ufun._inverse_cache_folder is not None
        assert _served_from_cache(ufun)


def test_lazily_built_inverter_uses_the_cache(tmp_path):
    """An inverter constructed directly and `init()`-ed later must still hit the cache.

    This is the path `make_inverter` takes (it deliberately returns an
    uninitialized inverter), so the cache has to be consulted inside ``init()``
    rather than only in `invert`.
    """
    _scenario(n_ufuns=1).dumpas(tmp_path, type="yml", save_inverse=True)
    loaded = Scenario.load(tmp_path, load_cached_inverse=True)
    assert loaded is not None
    ufun = loaded.ufuns[0]

    inverter = DefaultInverseUtilityFunction(ufun)
    assert not inverter.initialized
    _, evals = _count_evals(ufun, inverter.init)
    assert inverter.initialized
    assert evals < 8**3  # far fewer than a full rebuild


def test_make_inverter_uses_the_cache(tmp_path):
    """The component helper every time-based negotiator goes through must benefit."""
    from negmas.gb.components.inverter import make_inverter

    _scenario(n_ufuns=1).dumpas(tmp_path, type="yml", save_inverse=True)
    loaded = Scenario.load(tmp_path, load_cached_inverse=True)
    assert loaded is not None
    ufun = loaded.ufuns[0]
    inverter = make_inverter(ufun)
    _, evals = _count_evals(ufun, inverter.init)
    assert inverter.initialized
    assert evals < 8**3


def test_genius_xml_lossy_round_trip_rejects_the_cache(tmp_path):
    """Genius XML loses issue types, so its cache must be rejected, not reused.

    Saving to XML turns integer-valued cardinal issues into *string*-valued
    categorical ones, so the reloaded ufun is evaluated over genuinely different
    outcome tuples. The outcome-space gate must catch that.
    """
    _scenario(n_ufuns=2).dumpas(tmp_path, type="xml", save_inverse=True)
    assert _cache_dir(tmp_path).is_dir()
    loaded = Scenario.load(tmp_path, load_cached_inverse=True)
    assert loaded is not None
    for u in loaded.ufuns:
        assert not _served_from_cache(u)
        # ... and the ufun still gets a correct, freshly built inverter.
        assert u.invert().initialized


def test_genius_xml_serves_the_cache_when_the_round_trip_is_lossless(tmp_path):
    """With string-valued categorical issues the XML round-trip is faithful."""
    space = make_os(
        [make_issue([f"v{v}" for v in range(6)], f"i{k}") for k in range(3)], name="Cat"
    )
    scenario = Scenario(
        outcome_space=space,
        ufuns=tuple(
            LU.random(outcome_space=space, reserved_value=0.0) for _ in range(2)
        ),
    )
    scenario.dumpas(tmp_path, type="xml", save_inverse=True)
    loaded = Scenario.load(tmp_path, load_cached_inverse=True)
    assert loaded is not None
    for u in loaded.ufuns:
        assert _served_from_cache(u)


@pytest.mark.parametrize(
    "negotiator_name",
    ["AspirationNegotiator", "BoulwareTBNegotiator", "NaiveTitForTatNegotiator"],
)
def test_negotiation_does_not_invalidate_the_cache(tmp_path, negotiator_name):
    """No negotiator may write to a ufun in a way that discards its cache."""
    import negmas.sao as sao_mod

    cls = getattr(sao_mod, negotiator_name)
    _scenario().dumpas(tmp_path, type="yml", save_inverse=True)
    loaded = Scenario.load(tmp_path, load_cached_inverse=True)
    assert loaded is not None
    mechanism = SAOMechanism(outcome_space=loaded.outcome_space, n_steps=25)
    for u in loaded.ufuns:
        mechanism.add(cls(), ufun=u)
    mechanism.run()
    assert all(not u.modified for u in loaded.ufuns), negotiator_name


def _count_inversions(fn):
    """Runs ``fn``, returning ``(inversions, served_from_disk)``.

    Counts real presorting builds against restores, which is the only honest
    measure of whether a negotiator benefits from a cache.
    """
    import negmas.preferences.base_ufun as base_ufun_mod
    import negmas.preferences.inv_ufun.presorting as presorting_mod

    counts = {"inits": 0, "restores": 0}
    cls = presorting_mod.PresortingInverseUtilityFunction
    original_init, original_restore = (
        cls.init,
        base_ufun_mod.BaseUtilityFunction.restore_inverse_from_cache,
    )

    def counting_init(self):
        counts["inits"] += 1
        return original_init(self)

    def counting_restore(self, inverter):
        served = original_restore(self, inverter)
        counts["restores"] += bool(served)
        return served

    cls.init = counting_init
    base_ufun_mod.BaseUtilityFunction.restore_inverse_from_cache = counting_restore
    try:
        fn()
    finally:
        cls.init = original_init
        base_ufun_mod.BaseUtilityFunction.restore_inverse_from_cache = original_restore
    return counts["inits"], counts["restores"]


def test_tit_for_tat_concession_uses_the_cache(tmp_path):
    """The tit-for-tat family inverts via `KindConcessionRecommender`.

    That path calls ``ufun.invert()``, so it must be served from disk like any
    other. Paired against a tough opponent so the concession branch is reached.
    """
    from negmas.sao import NaiveTitForTatNegotiator, ToughNegotiator

    _scenario().dumpas(tmp_path, type="yml", save_inverse=True)
    loaded = Scenario.load(tmp_path, load_cached_inverse=True)
    assert loaded is not None

    def run():
        mechanism = SAOMechanism(outcome_space=loaded.outcome_space, n_steps=300)
        mechanism.add(NaiveTitForTatNegotiator(must_concede=True), ufun=loaded.ufuns[0])
        mechanism.add(ToughNegotiator(), ufun=loaded.ufuns[1])
        mechanism.run()

    inversions, served = _count_inversions(run)
    assert inversions > 0, "the concession branch never inverted"
    assert served == inversions
    assert all(not u.modified for u in loaded.ufuns)


@pytest.mark.parametrize(
    "negotiator_name",
    ["AspirationNegotiator", "BoulwareTBNegotiator", "TopFractionNegotiator"],
)
def test_every_inversion_is_served_from_disk(tmp_path, negotiator_name):
    """No negotiator may rebuild an inverse the cache could have provided."""
    import negmas.sao as sao_mod

    cls = getattr(sao_mod, negotiator_name)
    _scenario().dumpas(tmp_path, type="yml", save_inverse=True)
    loaded = Scenario.load(tmp_path, load_cached_inverse=True)
    assert loaded is not None

    def run():
        mechanism = SAOMechanism(outcome_space=loaded.outcome_space, n_steps=30)
        for u in loaded.ufuns:
            mechanism.add(cls(), ufun=u)
        mechanism.run()

    inversions, served = _count_inversions(run)
    assert inversions > 0, f"{negotiator_name} never inverted"
    assert served == inversions, f"{negotiator_name}: {served}/{inversions} served"


def test_loaded_inverter_is_initialized(tmp_path):
    """A restored inverter must be ready to answer queries without `init()`."""
    _scenario().dumpas(tmp_path, type="yml", save_inverse=True)
    loaded = Scenario.load(tmp_path, load_cached_inverse=True)
    assert loaded is not None
    assert loaded.ufuns[0].invert().initialized


@pytest.mark.parametrize("normalized", [True, False])
def test_cached_and_fresh_answer_identically(tmp_path, normalized):
    """Deterministic queries must give identical answers cached vs. rebuilt.

    ``one_in`` and ``some`` are excluded because they deliberately cycle among
    equal-utility outcomes and so differ even between two freshly built inverters.
    """
    _scenario(n_ufuns=1).dumpas(tmp_path, type="yml", save_inverse=True)
    cached = Scenario.load(tmp_path, load_cached_inverse=True)
    fresh = Scenario.load(tmp_path)  # default: no cache
    assert cached is not None and fresh is not None
    a, b = cached.ufuns[0].invert(), fresh.ufuns[0].invert()

    lo_hi = np.linspace(0, 1, 11) if normalized else np.linspace(*a.minmax(), 11)
    for lo in lo_hi:
        for hi in lo_hi[lo_hi >= lo]:
            rng = (float(lo), float(hi))
            assert a.worst_in(rng, normalized=normalized) == b.worst_in(
                rng, normalized=normalized
            )
            assert a.best_in(rng, normalized=normalized) == b.best_in(
                rng, normalized=normalized
            )
        assert a.closest(float(lo), normalized=normalized) == b.closest(
            float(lo), normalized=normalized
        )


def test_cached_inverse_preserves_full_negotiation_trace(tmp_path):
    """The strongest check: a whole negotiation must replay identically.

    This is what catches subtle ordering differences that a round-trip test of
    the arrays alone would miss.
    """

    def run(folder, load_cached_inverse):
        random.seed(1234)
        np.random.seed(1234)
        scenario = Scenario.load(folder, load_cached_inverse=load_cached_inverse)
        assert scenario is not None
        mechanism = SAOMechanism(outcome_space=scenario.outcome_space, n_steps=50)
        for u in scenario.ufuns:
            mechanism.add(AspirationNegotiator(), ufun=u)
        mechanism.run()
        return mechanism.agreement, [
            (s.step, s.current_offer) for s in mechanism.history
        ]

    _scenario().dumpas(tmp_path, type="yml", save_inverse=True)
    assert run(tmp_path, True) == run(tmp_path, False)


def test_all_outcomes_and_utilities_round_trip(tmp_path):
    """Every stored outcome/utility must come back byte-for-byte equivalent."""
    s = _scenario(n_ufuns=1)
    s.dumpas(tmp_path, type="yml", save_inverse=True)
    loaded = Scenario.load(tmp_path, load_cached_inverse=True)
    assert loaded is not None
    original = s.ufuns[0].invert()
    restored = loaded.ufuns[0].invert()
    assert restored.delegate.outcomes == original.delegate.outcomes
    assert np.allclose(restored.delegate.utils, original.delegate.utils)


@pytest.mark.parametrize("reserved", [0.0, 0.1, 0.3, 0.45])
def test_rational_only_restore_matches_a_fresh_build(reserved):
    """`rational_only` partitions the arrays, so a restore must reproduce it exactly.

    With a reserved value above the minimum, ``init()`` puts the sorted rational
    outcomes first and appends the irrational ones. `restore_state` recomputes
    that split rather than trusting the file, so it must agree with a fresh build
    on the outcome order, the rational boundary and the tie groups.
    """
    space = _os(n_issues=3, n_values=6)
    ufun = LU.random(outcome_space=space, reserved_value=reserved)
    fresh = PresortingInverseUtilityFunction(ufun, rational_only=True)
    fresh.init()

    restored = PresortingInverseUtilityFunction(ufun, rational_only=True)
    assert restored.restore_state(fresh.persist_state())

    assert restored.outcomes == fresh.outcomes
    assert np.allclose(restored.utils, fresh.utils)
    assert restored._last_rational == fresh._last_rational
    assert restored._near_range == fresh._near_range
    assert restored.minmax() == fresh.minmax()


def test_continuous_additive_scenario_saves_and_loads_gracefully(tmp_path):
    """A continuous additive space is inverted by BIDS, which is not persistable.

    Nothing should be cached, and loading must still work — the point is that an
    un-persistable inverter degrades silently rather than failing the save.
    """
    space = make_os(
        [make_issue((0.0, 1.0), "c0"), make_issue((0.0, 1.0), "c1")], name="C"
    )
    s = Scenario(
        outcome_space=space, ufuns=(LU.random(outcome_space=space, reserved_value=0.0),)
    )
    s.dumpas(tmp_path, type="yml", save_inverse=True)
    assert not list(_cache_dir(tmp_path).glob("*.npz"))
    loaded = Scenario.load(tmp_path, load_cached_inverse=True)
    assert loaded is not None
    assert loaded.ufuns[0].invert().initialized


def test_continuous_presorting_inverse_round_trips(tmp_path):
    """A presorting inverse over a continuous space must restore exactly.

    Continuous issues are discretized onto a deterministic grid, so the restored
    inverter must reproduce that grid and its utilities verbatim.
    """
    space = make_os(
        [make_issue((0.0, 1.0), "c0"), make_issue((0.0, 1.0), "c1")], name="C"
    )
    u = LU.random(outcome_space=space, reserved_value=0.0)
    original = u.invert(inverter=PresortingInverseUtilityFunction)
    state = original.persist_state()

    restored = PresortingInverseUtilityFunction(u)
    assert restored.restore_state(state)
    assert restored.outcomes == original.outcomes
    assert np.allclose(restored.utils, original.utils)
    assert restored.minmax() == original.minmax()


def test_ufun_order_permutation_is_handled(tmp_path):
    """Ufun files load in sorted-name order, so caches must be keyed by name.

    Built as a regression test: positional keys silently attached ufun 0's cache
    to ufun 1 whenever the names happened to sort the other way.
    """
    for _ in range(12):
        s = _scenario(n_ufuns=2, n_values=5)
        s.dumpas(tmp_path, type="yml", save_inverse=True)
        loaded = Scenario.load(tmp_path, load_cached_inverse=True)
        assert loaded is not None
        for u in loaded.ufuns:
            inverter = u.invert()
            # Each restored inverse must agree with its OWN ufun.
            for outcome in list(u.outcome_space.enumerate_or_sample())[:20]:
                idx = inverter.delegate.outcomes.index(outcome)
                assert float(u.eval(outcome)) == pytest.approx(
                    float(inverter.delegate.utils[idx])
                )


# ---------------------------------------------------------------------------
# Validity gates: every one of these must force a rebuild
# ---------------------------------------------------------------------------


def _load_after(tmp_path, mutate):
    """Saves a scenario, mutates the reloaded ufun, and tries to load its cache."""
    _scenario(n_ufuns=1).dumpas(tmp_path, type="yml", save_inverse=True)
    loaded = Scenario.load(tmp_path, load_cached_inverse=True)
    assert loaded is not None
    ufun = loaded.ufuns[0]
    mutate(ufun)
    return _served_from_cache(ufun)


def test_changed_reserved_value_rejects_cache(tmp_path):
    """The reserved value decides which outcomes are rational, so it must gate the cache."""

    def mutate(u):
        u.reserved_value = 0.7

    assert not _load_after(tmp_path, mutate)


def test_changed_weights_reject_cache(tmp_path):
    """A changed ufun must never be served a stale inverse.

    The weights are mutated through the private slot deliberately: it is the
    route that bypasses every explicit hook, so it exercises the behavioral
    verification gate rather than the `__setattr__` backstop.
    """

    def mutate(u):
        object.__setattr__(u, "_weights", tuple(reversed(list(u.weights))))

    assert not _load_after(tmp_path, mutate)


def test_marking_modified_rejects_cache(tmp_path):
    """`mark_modified()` is the escape hatch for in-place mutation."""
    assert not _load_after(tmp_path, lambda u: u.mark_modified())


def test_any_attribute_assignment_marks_modified(tmp_path):
    """The `__setattr__` backstop must catch assignments it does not know about."""
    _scenario(n_ufuns=1).dumpas(tmp_path, type="yml", save_inverse=True)
    loaded = Scenario.load(tmp_path, load_cached_inverse=True)
    assert loaded is not None
    ufun = loaded.ufuns[0]
    assert not ufun.modified
    ufun.name = "renamed"
    assert ufun.modified


def test_using_the_inverter_does_not_mark_modified(tmp_path):
    """Bookkeeping writes (cache slots) must not count as modification."""
    _scenario(n_ufuns=1).dumpas(tmp_path, type="yml", save_inverse=True)
    loaded = Scenario.load(tmp_path, load_cached_inverse=True)
    assert loaded is not None
    ufun = loaded.ufuns[0]
    ufun.invert()
    ufun.minmax()
    ufun.extreme_outcomes()
    assert not ufun.modified


def test_owner_assignment_does_not_mark_modified(tmp_path):
    """A negotiator taking ownership of a ufun must not discard its cache."""
    _scenario(n_ufuns=1).dumpas(tmp_path, type="yml", save_inverse=True)
    loaded = Scenario.load(tmp_path, load_cached_inverse=True)
    assert loaded is not None
    ufun = loaded.ufuns[0]
    ufun.owner = object()
    assert not ufun.modified


def test_reassigning_an_equal_value_does_not_mark_modified(tmp_path):
    """negmas re-points `outcome_space` at negotiation start; that changes nothing."""
    _scenario(n_ufuns=1).dumpas(tmp_path, type="yml", save_inverse=True)
    loaded = Scenario.load(tmp_path, load_cached_inverse=True)
    assert loaded is not None
    ufun = loaded.ufuns[0]
    ufun.outcome_space = ufun.outcome_space
    ufun.reserved_value = ufun.reserved_value
    assert not ufun.modified


def test_cache_survives_a_whole_negotiation(tmp_path):
    """The end-to-end guard: running a negotiation must not invalidate the cache."""
    _scenario().dumpas(tmp_path, type="yml", save_inverse=True)
    loaded = Scenario.load(tmp_path, load_cached_inverse=True)
    assert loaded is not None
    mechanism = SAOMechanism(outcome_space=loaded.outcome_space, n_steps=20)
    for u in loaded.ufuns:
        mechanism.add(AspirationNegotiator(), ufun=u)
    mechanism.run()
    assert all(not u.modified for u in loaded.ufuns)


def test_unattached_ufun_never_marks_modified():
    """Ufuns with no disk cache must not pay for modification tracking."""
    space = _os()
    u = LU.random(outcome_space=space, reserved_value=0.0)
    u.name = "whatever"
    assert not u.modified


def test_reordered_issues_reject_cache(tmp_path):
    """Outcomes are positional tuples, so issue order must gate the cache."""
    _scenario(n_ufuns=1).dumpas(tmp_path, type="yml", save_inverse=True)
    loaded = Scenario.load(tmp_path, load_cached_inverse=True)
    assert loaded is not None
    ufun = loaded.ufuns[0]
    reordered = make_os(list(reversed(list(ufun.outcome_space.issues))), name="D")
    assert outcome_space_fingerprint(reordered) != outcome_space_fingerprint(
        ufun.outcome_space
    )


def test_mismatched_config_rejects_cache(tmp_path):
    """A config that changes the stored arrays must not be served from cache."""
    _scenario(n_ufuns=1).dumpas(tmp_path, type="yml", save_inverse=True)
    loaded = Scenario.load(tmp_path, load_cached_inverse=True)
    assert loaded is not None
    ufun = loaded.ufuns[0]
    # `levels` changes the discretization, so it changes the stored arrays and
    # is not among the cached configurations.
    assert not _served_from_cache(ufun, levels=3)


def test_default_configs_are_all_cached(tmp_path):
    """Every configuration negmas' own negotiators ask for must hit the cache."""
    _scenario(n_ufuns=1).dumpas(tmp_path, type="yml", save_inverse=True)
    loaded = Scenario.load(tmp_path, load_cached_inverse=True)
    assert loaded is not None
    ufun = loaded.ufuns[0]
    for config in DEFAULT_INVERSE_CONFIGS:
        assert _served_from_cache(ufun, **config), config


def test_corrupt_npz_falls_back_silently(tmp_path):
    """A truncated cache must never break loading — just rebuild."""
    _scenario(n_ufuns=1).dumpas(tmp_path, type="yml", save_inverse=True)
    for path in _cache_dir(tmp_path).glob("*.npz"):
        path.write_bytes(b"not really an npz")
    loaded = Scenario.load(tmp_path, load_cached_inverse=True)
    assert loaded is not None
    inverter = loaded.ufuns[0].invert()
    assert inverter.initialized


def test_corrupt_manifest_falls_back_silently(tmp_path):
    """An unparseable manifest must be ignored rather than raise."""
    _scenario(n_ufuns=1).dumpas(tmp_path, type="yml", save_inverse=True)
    (_cache_dir(tmp_path) / INVERSE_MANIFEST_NAME).write_text(
        "{not json", encoding="utf-8"
    )
    loaded = Scenario.load(tmp_path, load_cached_inverse=True)
    assert loaded is not None
    assert loaded.ufuns[0].invert().initialized


def test_tampered_utilities_are_detected(tmp_path):
    """The behavioral gate must reject a cache whose utilities no longer match."""
    _scenario(n_ufuns=1).dumpas(tmp_path, type="yml", save_inverse=True)
    for path in _cache_dir(tmp_path).glob("*.npz"):
        with np.load(path, allow_pickle=False) as data:
            arrays = {k: data[k] for k in data.files}
        arrays["utils"] = arrays["utils"] + 0.25
        np.savez_compressed(path, **arrays)

    loaded = Scenario.load(tmp_path, load_cached_inverse=True)
    assert loaded is not None
    ufun = loaded.ufuns[0]
    for config in DEFAULT_INVERSE_CONFIGS:
        effective = effective_config(ufun, DefaultInverseUtilityFunction, dict(config))
        assert effective is not None
        assert (
            load_state(tmp_path, ufun, ufun.name, effective, verification_samples=0)
            is None
        ), config
    # ... and the ufun still ends up with a working, rebuilt inverter.
    assert ufun.invert().initialized


def test_missing_cache_folder_is_a_no_op(tmp_path):
    """Attaching when nothing was saved must simply do nothing."""
    _scenario().dumpas(tmp_path, type="yml")
    loaded = Scenario.load(tmp_path, load_cached_inverse=True)
    assert loaded is not None
    assert not loaded.attach_inverses(tmp_path)
    assert loaded.ufuns[0].invert().initialized


# ---------------------------------------------------------------------------
# Ufuns that must never be cached
# ---------------------------------------------------------------------------


def test_constrained_ufun_is_not_cacheable():
    """Constraints are arbitrary callables and cannot be fingerprinted."""
    space = _os()
    u = LU.random(outcome_space=space, reserved_value=0.0)
    assert can_cache_inverse(u)[0]
    u.add_constraint(lambda o: o[0] > 1)
    ok, reason = can_cache_inverse(u)
    assert not ok and "constraint" in reason


def test_constrained_ufun_is_skipped_on_save(tmp_path):
    """A scenario with a constrained ufun still saves the ufuns it *can* cache."""
    s = _scenario(n_ufuns=2)
    s.ufuns[0].add_constraint(lambda o: o[0] > 1)
    s.dumpas(tmp_path, type="yml", save_inverse=True)
    assert len(list(_cache_dir(tmp_path).glob("*.npz"))) == len(DEFAULT_INVERSE_CONFIGS)


def test_discounted_ufun_is_not_cacheable():
    """A time-dependent ufun's inverse is only valid at one instant."""
    from negmas.preferences.discounted import ExpDiscountedUFun

    space = _os()
    u = ExpDiscountedUFun(
        LU.random(outcome_space=space, reserved_value=0.0), discount=0.9
    )
    ok, reason = can_cache_inverse(u)
    assert not ok and "stationary" in reason


# ---------------------------------------------------------------------------
# `invert()` caching semantics
# ---------------------------------------------------------------------------


def test_invert_reuses_for_same_kwargs():
    """Repeated identical calls must return the very same object."""
    u = LU.random(outcome_space=_os(), reserved_value=0.0)
    assert u.invert() is u.invert()
    assert u.invert(rational_only=True) is u.invert(rational_only=True)


def test_invert_rebuilds_for_different_kwargs():
    """A different configuration must not be silently served the cached inverter."""
    u = LU.random(outcome_space=_os(), reserved_value=0.0)
    default = u.invert()
    assert u.invert(rational_only=True) is not default


def test_invert_respects_explicit_inverter_type():
    """Asking for a specific inverter class must yield that class."""
    u = LU.random(outcome_space=_os(), reserved_value=0.0)
    inverter = u.invert(inverter=PresortingInverseUtilityFunction)
    assert isinstance(inverter, PresortingInverseUtilityFunction)


def test_changing_reserved_value_leaves_an_uncached_ufun_alone():
    """With no saved inverse attached, behaviour is exactly as it was before.

    Inverse caching is entirely opt-in: a ufun that was never pointed at a cache
    must behave identically to how it did before the feature existed, including
    keeping its in-memory inverter when the reserved value changes.
    """
    u = LU.random(outcome_space=_os(), reserved_value=0.0)
    inverter = u.invert()
    u.reserved_value = 0.6
    assert u._cached_inverse is inverter
    assert not u.modified


def test_changing_reserved_value_invalidates_an_attached_cache(tmp_path):
    """With a cache attached, the reserved value must invalidate it."""
    _scenario(n_ufuns=1).dumpas(tmp_path, type="yml", save_inverse=True)
    loaded = Scenario.load(tmp_path, load_cached_inverse=True)
    assert loaded is not None
    ufun = loaded.ufuns[0]
    ufun.invert()
    ufun.reserved_value = 0.6
    assert ufun.modified
    assert ufun._cached_inverse is None
    assert not _served_from_cache(ufun)


# ---------------------------------------------------------------------------
# Manifest shape
# ---------------------------------------------------------------------------


def test_manifest_records_expected_fields(tmp_path):
    """The manifest must carry everything the validity gates need."""
    _scenario(n_ufuns=1).dumpas(tmp_path, type="yml", save_inverse=True)
    manifest = json.loads(
        (_cache_dir(tmp_path) / INVERSE_MANIFEST_NAME).read_text(encoding="utf-8")
    )
    assert manifest["version"] == 1
    assert len(manifest["entries"]) == len(DEFAULT_INVERSE_CONFIGS)
    for entry in manifest["entries"].values():
        for key in (
            "file",
            "ufun",
            "inverter",
            "n_outcomes",
            "kinds",
            "outcome_space",
            "reserved_value",
            "config",
            "extra",
        ):
            assert key in entry


def test_string_and_int_issues_round_trip(tmp_path):
    """Mixed value types must survive the pickle-free column encoding."""
    space = make_os(
        [
            make_issue([0, 1, 2, 3], "ints"),
            make_issue(["a", "b", "c"], "strs"),
            make_issue([0.5, 1.5], "floats"),
        ],
        name="M",
    )
    s = Scenario(
        outcome_space=space, ufuns=(LU.random(outcome_space=space, reserved_value=0.0),)
    )
    s.dumpas(tmp_path, type="yml", save_inverse=True)
    loaded = Scenario.load(tmp_path, load_cached_inverse=True)
    assert loaded is not None
    restored = loaded.ufuns[0].invert()
    assert set(restored.delegate.outcomes) == set(space.enumerate_or_sample())
    for outcome in restored.delegate.outcomes:
        assert isinstance(outcome[0], int)
        assert isinstance(outcome[1], str)
        assert isinstance(outcome[2], float)
