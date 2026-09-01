"""Tests for the global seeding facility (`negmas.helpers.rand`)."""

from __future__ import annotations

import os
import random
import subprocess
import sys

import numpy as np
import pytest

from negmas.helpers.rand import (
    get_seed,
    register_seeder,
    seed_all,
    seed_environment,
    seed_from_env,
    task_seed,
)


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


_REPRO_SCRIPT = (
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


def _run_repro_script(env_seed: str | None) -> str:
    """Run a tiny negotiation in a fresh interpreter and return its outcome."""
    env = dict(os.environ)
    env.pop("NEGMAS_RAND_SEED", None)
    if env_seed is not None:
        env["NEGMAS_RAND_SEED"] = env_seed
    return subprocess.run(
        [sys.executable, "-c", _REPRO_SCRIPT],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    ).stdout


def test_env_seed_makes_a_whole_negotiation_reproducible_across_processes():
    assert _run_repro_script("7") == _run_repro_script("7")


def test_seed_environment_covers_negmas_and_hash_randomization():
    env = seed_environment(44)
    assert env["NEGMAS_RAND_SEED"] == "44"
    assert env["PYTHONHASHSEED"] == "44"


def test_negmas_seed_command_prints_shell_exports():
    out = subprocess.run(
        [sys.executable, "-m", "negmas.scripts.app", "seed", "44"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    lines = out.strip().splitlines()
    assert all(line.startswith("export ") for line in lines), out
    assert "export NEGMAS_RAND_SEED=44" in lines
    assert "export PYTHONHASHSEED=44" in lines


def test_negmas_seed_command_can_print_bare_assignments():
    out = subprocess.run(
        [sys.executable, "-m", "negmas.scripts.app", "seed", "7", "--no-export"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "NEGMAS_RAND_SEED=7" in out.splitlines()
    assert "export " not in out


def test_evaluating_the_seed_command_makes_a_run_reproducible():
    """The documented `eval "$(negmas seed N)"` workflow, end to end."""
    env = dict(os.environ)
    env.pop("NEGMAS_RAND_SEED", None)
    for name, value in seed_environment(44).items():
        env[name] = value
    first = subprocess.run(
        [sys.executable, "-c", _REPRO_SCRIPT],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    ).stdout
    second = subprocess.run(
        [sys.executable, "-c", _REPRO_SCRIPT],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    ).stdout
    assert first == second


def _draw(_task):
    """A picklable task whose result depends only on the RNG state it starts from."""
    import negmas  # noqa: F401  (a spawned worker re-imports and re-seeds)

    return round(random.random(), 9)


def _run_tasks(runner, **kwargs):
    """Drive one of negmas's dispatchers over four identical tasks."""
    from negmas.helpers.parallel import run_parallel_tasks  # noqa: F401

    results = {}
    tasks = [(i, _draw, (i,), {}) for i in range(4)]
    runner(
        tasks,
        on_result=lambda info, res, i, n: results.__setitem__(info, res),
        **kwargs,
    )
    return [results[i] for i in sorted(results)]


def test_task_seed_is_none_when_nothing_is_seeded():
    assert task_seed(0) is None


def test_task_seeds_are_distinct_and_reproducible():
    seed_all(42)
    first = [task_seed(i) for i in range(8)]
    assert len(set(first)) == 8, "every task must get its own stream"
    seed_all(42)
    assert [task_seed(i) for i in range(8)] == first


def test_task_seed_does_not_move_the_base_seed():
    """Deriving a task's seed must not change what later tasks derive from."""
    seed_all(42)
    task_seed(0)
    assert get_seed() == 42


def test_identical_tasks_do_not_collapse_onto_one_result_in_parallel(monkeypatch):
    """Regression: every spawned worker used to start from the same stream.

    Workers re-import negmas and re-apply NEGMAS_RAND_SEED for themselves, so
    without a per-task seed all four "repetitions" below returned the same
    number -- reproducible, but with the variance they exist to measure gone.
    """
    monkeypatch.setenv("NEGMAS_RAND_SEED", "42")
    seed_from_env()
    from negmas.helpers.parallel import make_process_executor, run_parallel_tasks

    with make_process_executor(max_workers=2) as executor:
        results = _run_tasks(run_parallel_tasks, executor=executor)
    assert len(set(results)) == len(results), f"repetitions collapsed: {results}"


def test_a_serial_run_reproduces_a_parallel_one(monkeypatch):
    """njobs must not change the results of a seeded run."""
    monkeypatch.setenv("NEGMAS_RAND_SEED", "42")
    seed_from_env()
    from negmas.helpers.parallel import (
        make_process_executor,
        run_parallel_tasks,
        run_serial_tasks,
    )

    serial = _run_tasks(run_serial_tasks)
    seed_from_env()
    with make_process_executor(max_workers=2) as executor:
        parallel = _run_tasks(run_parallel_tasks, executor=executor)
    assert serial == parallel


def test_a_seeded_serial_run_repeats_in_the_same_process(monkeypatch):
    monkeypatch.setenv("NEGMAS_RAND_SEED", "42")
    from negmas.helpers.parallel import run_serial_tasks

    seed_from_env()
    first = _run_tasks(run_serial_tasks)
    seed_from_env()
    assert _run_tasks(run_serial_tasks) == first


def test_unseeded_dispatch_is_left_exactly_as_it_was(monkeypatch):
    """With no seed in effect the dispatchers must not wrap the task at all."""
    monkeypatch.delenv("NEGMAS_RAND_SEED", raising=False)
    import negmas.helpers.rand as m
    from negmas.helpers.parallel import _seeded

    m._seed = None
    assert _seeded(_draw, 0) is _draw


def _negotiate(_task):
    """A picklable task doing real negotiation work, not just one RNG draw.

    The serial/parallel equality below is only meaningful if the task does the
    kind of work a tournament actually dispatches -- building random ufuns and
    running a mechanism -- rather than drawing a single number immediately
    after being seeded.
    """
    from negmas.outcomes import make_issue
    from negmas.preferences import LinearAdditiveUtilityFunction as U
    from negmas.sao import AspirationNegotiator, SAOMechanism

    issues = [make_issue([f"v{i}" for i in range(10)], "a"), make_issue(10, "b")]
    mechanism = SAOMechanism(issues=issues, n_steps=30)
    for _ in range(2):
        mechanism.add(
            AspirationNegotiator(
                preferences=U.random(issues=issues, reserved_value=0.0)
            )
        )
    state = mechanism.run()
    return (state.agreement, state.step)


def _run_negotiations(runner, **kwargs):
    results = {}
    tasks = [(i, _negotiate, (i,), {}) for i in range(4)]
    runner(
        tasks,
        on_result=lambda info, res, i, n: results.__setitem__(info, res),
        **kwargs,
    )
    return [results[i] for i in sorted(results)]


def test_a_serial_negotiation_run_reproduces_a_parallel_one(monkeypatch):
    """njobs must not change the results of a seeded run of real negotiations."""
    monkeypatch.setenv("NEGMAS_RAND_SEED", "42")
    from negmas.helpers.parallel import (
        make_process_executor,
        run_parallel_tasks,
        run_serial_tasks,
    )

    seed_from_env()
    serial = _run_negotiations(run_serial_tasks)
    seed_from_env()
    with make_process_executor(max_workers=2) as executor:
        parallel = _run_negotiations(run_parallel_tasks, executor=executor)
    assert serial == parallel
    assert len(set(serial)) > 1, f"repetitions collapsed: {serial}"


def test_isolated_dispatch_still_serializes_a_closure_when_seeded(monkeypatch):
    """The isolated path cloudpickles tasks; wrapping must not break that.

    `run_isolated_tasks` exists to run tasks that stdlib pickle cannot handle --
    closures and locally defined negotiators. Wrapping them for seeding must
    keep them cloudpickle-able, or a seeded run would start failing tasks that
    used to serialize fine.
    """
    monkeypatch.setenv("NEGMAS_RAND_SEED", "42")
    from negmas.helpers.parallel import run_isolated_tasks

    seed_from_env()
    scale = 1000  # closed over, so `job` is a genuine closure

    def job(_rep):
        import random

        import negmas  # noqa: F401

        return int(random.random() * scale)

    results = {}
    errors = {}
    run_isolated_tasks(
        [(i, job, (i,), {}) for i in range(4)],
        max_workers=2,
        timeout=60,
        on_result=lambda info, res, i, n: results.__setitem__(info, res),
        on_error=lambda e, info, i, n: errors.__setitem__(info, e),
    )
    assert not errors, f"seeded isolated dispatch failed tasks: {errors}"
    assert len(results) == 4
    assert len(set(results.values())) == 4, "isolated repetitions collapsed"
