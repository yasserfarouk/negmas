"""Tests for the global seeding facility (`negmas.helpers.rand`)."""

from __future__ import annotations

import os
import random
import subprocess
import sys

import numpy as np
import pytest

from negmas.helpers.rand import get_seed, register_seeder, seed_all, seed_from_env


@pytest.fixture(autouse=True)
def _restore_global_seed_state():
    """Keep these tests from leaking seed state into the rest of the suite."""
    import negmas.helpers.rand as m

    saved = (m._seed, list(m._seeders), random.getstate(), np.random.get_state())
    yield
    m._seed, m._seeders[:] = saved[0], saved[1]
    random.setstate(saved[2])
    np.random.set_state(saved[3])


def test_seed_all_makes_stdlib_and_numpy_reproducible():
    seed_all(1234)
    first = (random.random(), np.random.random())
    seed_all(1234)
    assert (random.random(), np.random.random()) == first


def test_seed_all_with_different_seeds_differs():
    seed_all(1)
    first = (random.random(), np.random.random())
    seed_all(2)
    assert (random.random(), np.random.random()) != first


def test_seed_all_returns_the_applied_seed_and_updates_get_seed():
    assert seed_all(99) == 99
    assert get_seed() == 99


def test_seed_all_none_is_a_no_op():
    seed_all(5)
    assert seed_all(None) is None
    # get_seed is left alone so seed_all(get_seed()) can be called safely
    assert get_seed() == 5


def test_registered_seeder_is_called_by_seed_all():
    seen = []
    register_seeder(seen.append)
    seed_all(3)
    assert seen == [3]


def test_registering_after_seeding_applies_the_current_seed():
    seed_all(11)
    seen = []
    register_seeder(seen.append)
    assert seen == [11], "a library imported after seeding must still be seeded"


def test_seed_from_env_does_nothing_when_unset(monkeypatch):
    monkeypatch.delenv("NEGMAS_RAND_SEED", raising=False)
    assert seed_from_env() is None
    assert get_seed() is None


@pytest.mark.parametrize("value", ["", "none", "None", "random"])
def test_seed_from_env_treats_these_as_a_request_for_fresh_entropy(monkeypatch, value):
    monkeypatch.setenv("NEGMAS_RAND_SEED", value)
    assert seed_from_env() is None
    assert get_seed() is None


def test_seed_from_env_warns_and_ignores_a_non_integer_seed(monkeypatch):
    monkeypatch.setenv("NEGMAS_RAND_SEED", "not-a-number")
    with pytest.warns(Warning):
        assert seed_from_env() is None
    assert get_seed() is None


def test_seed_from_env_applies_the_environment_seed(monkeypatch):
    monkeypatch.setenv("NEGMAS_RAND_SEED", "123")
    assert seed_from_env() == 123
    assert get_seed() == 123


def _run_repro_script(env_seed: str | None) -> str:
    """Run a tiny negotiation in a fresh interpreter and return its outcome."""
    script = (
        "from negmas.outcomes import make_issue\n"
        "from negmas.preferences import LinearAdditiveUtilityFunction as U\n"
        "from negmas.sao import SAOMechanism, AspirationNegotiator\n"
        "issues = [make_issue([f'v{i}' for i in range(10)], 'a'), make_issue(10, 'b')]\n"
        "m = SAOMechanism(issues=issues, n_steps=30)\n"
        "for _ in range(2):\n"
        "    m.add(AspirationNegotiator("
        "preferences=U.random(issues=issues, reserved_value=0.0)))\n"
        "s = m.run()\n"
        "print(s.agreement, s.step, len(m.trace))\n"
    )
    env = dict(os.environ)
    env.pop("NEGMAS_RAND_SEED", None)
    if env_seed is not None:
        env["NEGMAS_RAND_SEED"] = env_seed
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    ).stdout


def test_env_seed_makes_a_whole_negotiation_reproducible_across_processes():
    assert _run_repro_script("7") == _run_repro_script("7")
