"""Global, best-effort seeding of every random number generator NegMAS uses.

NegMAS draws its randomness from the global generators of the standard library
(:mod:`random`) and of numpy (:mod:`numpy.random`), so seeding those two makes
a run reproducible. `register_seeder` lets libraries built on NegMAS attach
their own generators to the same switch.

The seed is read from the ``NEGMAS_RAND_SEED`` environment variable (or the
``rand_seed`` config key) when :mod:`negmas` is imported. Leaving it unset --
the default -- seeds nothing, so every run draws fresh entropy exactly as it
did before this module existed.

Hash randomization is not covered: ``PYTHONHASHSEED`` can only be set before
the interpreter starts.

Besides that global switch, a single object or block of code can get a private,
reproducible stream of its own with `RandomStream` (or the `seeded` context
manager). Components that draw random numbers -- the Pareto samplers, for
example -- take a ``seed`` argument built on it. Their default (``seed=None``)
draws from the global generators, exactly as before.

Unique identifiers (`uuid4`) are reproducible too once a seed is in effect.
"""

from __future__ import annotations

import functools
import random
import types
import uuid
from typing import Any, Callable, TypeVar

import numpy as np
from numpy.random import SeedSequence

from negmas.warnings import warn

__all__ = [
    "RandomStream",
    "seeded",
    "seedable",
    "uses_random_stream",
    "uuid4",
    "seed_all",
    "apply_seed",
    "get_seed",
    "register_seeder",
    "seed_environment",
    "task_seed",
]

_NOT_A_SEED = ("", "none", "random")
"""Values of the setting that explicitly ask for fresh entropy"""

_seed: int | None = None
"""The seed passed to the last `seed_all` call (`None` if never seeded)"""

_seeders: list[Callable[[int], None]] = []
"""Callbacks registered by libraries that seed their own generators"""

_id_rng: random.Random | None = None
"""Generator behind `uuid4` while a seed is in effect (`None` = use the OS)"""

_ID_STREAM = 0x1D5
"""Spawn key separating the identifier stream from every task seed"""

_COMPONENT_STREAM = 0x5EED
"""Spawn key separating `RandomStream` seeds from the global and task seeds"""


def get_seed() -> int | None:
    """The seed currently in effect, or `None` if nothing was ever seeded."""
    return _seed


def seed_all(seed: int | None) -> int | None:
    """Seed every random number generator NegMAS is known to use.

    Seeds the global generators of :mod:`random` and :mod:`numpy.random` then
    calls everything passed to `register_seeder`.

    Args:
        seed: The seed to use. `None` seeds nothing, so `seed_all(get_seed())`
              is a safe no-op.

    Returns:
        The seed that was applied, or `None` if nothing was seeded.
    """
    global _seed
    if seed is None:
        return None
    seed = int(seed)
    apply_seed(seed)
    _seed = seed
    return seed


def apply_seed(seed: int) -> None:
    """Seed every generator without recording `seed` as the run's base seed.

    This is what a per-task seed needs. `task_seed` derives each task's seed
    from the base, so applying one must not make it the base itself --
    otherwise task 2 would be derived from task 1's seed rather than from the
    run's, and repeating a run in the same process would not repeat.

    Use `seed_all` to start a run; use this to seed one task within it. Code
    outside negmas that dispatches its own parallel work needs both, together
    with `task_seed`:

    >>> from negmas.helpers.rand import apply_seed, task_seed
    >>> seed = task_seed(index)  # doctest: +SKIP
    >>> if seed is not None:  # doctest: +SKIP
    ...     apply_seed(seed)
    """
    global _id_rng
    random.seed(seed)
    np.random.seed(seed % (2**32))
    # identifiers get a stream of their own so that creating an object never
    # shifts the draws everything else sees
    _id_rng = random.Random(_derive(seed, _ID_STREAM))
    for seeder in _seeders:
        seeder(seed)


def _derive(seed: int, key: int) -> int:
    """A decorrelated seed derived from ``seed`` for the stream named ``key``."""
    return int(SeedSequence(seed, spawn_key=(key,)).generate_state(1)[0])


def uuid4() -> uuid.UUID:
    """A random UUID that is reproducible once a seed is in effect.

    Without a seed this is exactly `uuid.uuid4` (fresh OS entropy). After
    `seed_all` (or `apply_seed`) the UUIDs come from a dedicated generator
    derived from the seed, so the ids of the objects a run creates repeat with
    the run. That generator is separate from :mod:`random`, so creating objects
    never changes the numbers any other code draws.

    Remarks:
        Two runs seeded with the same seed in the *same process* produce the
        same ids. That is the point, but objects of the first run that are
        still alive share their ids with those of the second.
    """
    if _id_rng is None:
        return uuid.uuid4()
    return uuid.UUID(int=_id_rng.getrandbits(128), version=4)


def register_seeder(seeder: Callable[[int], None]) -> None:
    """Register a callback seeding generators NegMAS itself knows nothing about.

    The callback is called with the seed by every later `seed_all`, and
    immediately if a seed is already in effect -- so a library imported after
    the seed was applied is still seeded.
    """
    _seeders.append(seeder)
    if _seed is not None:
        seeder(_seed)


def task_seed(index: int) -> int | None:
    """A distinct but reproducible seed for the ``index``-th task of a run.

    Returns `None` when nothing is seeded, so the caller leaves the task alone
    and it draws fresh entropy exactly as it does today.

    Remarks:
        - Parallel workers are separate processes that each re-apply
          ``NEGMAS_RAND_SEED`` as they import negmas, so without this they all
          start from the *same* stream: two tasks that draw the same sequence
          produce the same result, and repetitions of one configuration
          collapse onto a single outcome instead of sampling the variation
          they exist to measure.
        - The seed is derived from the task's index rather than from the worker
          that happens to run it, so a run reproduces whatever the scheduling
          order or the number of workers -- and a serial run reproduces a
          parallel one.
        - `SeedSequence` is used rather than ``seed + index`` because it is
          built to turn one seed into many decorrelated ones.
    """
    if _seed is None:
        return None
    return _derive(_seed, index)


def seed_from_env() -> int | None:
    """Apply the seed configured through ``NEGMAS_RAND_SEED``, if any.

    Called once when :mod:`negmas` is imported. Returns the applied seed, or
    `None` when the setting is absent or asks for fresh entropy.
    """
    from negmas.config import negmas_config

    value = negmas_config("rand_seed", None)
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in _NOT_A_SEED:
        return None
    try:
        return seed_all(int(text))
    except ValueError:
        warn(
            f"NEGMAS_RAND_SEED is set to {text!r} which is not an integer. "
            "Ignoring it and leaving all generators unseeded."
        )
        return None


SEED_ENVIRONMENT_VARIABLES = (
    # negmas itself (see `seed_from_env`)
    ("NEGMAS_RAND_SEED", "{seed}"),
    # python's hash randomization; only read at interpreter start-up
    ("PYTHONHASHSEED", "{seed}"),
    # read by pytorch-lightning's seed_everything() when called without a seed
    ("PL_GLOBAL_SEED", "{seed}"),
    # cuBLAS needs a fixed workspace for torch's deterministic algorithms
    ("CUBLAS_WORKSPACE_CONFIG", ":4096:8"),
    # tensorflow's deterministic kernels
    ("TF_DETERMINISTIC_OPS", "1"),
    ("TF_CUDNN_DETERMINISTIC", "1"),
)
"""Environment variables that make a run reproducible, and their values"""


def seed_environment(seed: int) -> dict[str, str]:
    """The environment variables that make a run with this seed reproducible.

    Covers NegMAS and the seeding knobs of the common libraries used alongside
    it that can only be set from the environment. Applying these is the job of
    the shell (``negmas seed``), because ``PYTHONHASHSEED`` is read before the
    interpreter starts and so cannot be set from inside a running process.

    Only the NegMAS entries are guaranteed to do something: the rest are read
    by torch, tensorflow and pytorch-lightning if those are used at all, and
    are simply ignored otherwise.
    """
    return {
        name: value.format(seed=int(seed)) for name, value in SEED_ENVIRONMENT_VARIABLES
    }


class RandomStream:
    """A private, reproducible stream of random numbers for one component.

    Code run inside ``with stream:`` draws from this stream instead of the
    global generators of :mod:`random` and :mod:`numpy.random`. Nothing has to
    be passed down: functions deep inside negmas that call ``random.random()``
    or ``np.random.rand()`` are covered as they are. When the block ends the
    global generators are put back exactly as they were, and the stream keeps
    its own position, so the next ``with stream:`` continues where the last one
    stopped -- the sequence of draws depends on the seed alone, never on what
    the rest of the program drew in between.

    With ``seed=None`` (the default) the stream does nothing at all: the code
    inside the block draws from the global generators as it would without it.

    Args:
        seed: The seed of the stream, or `None` to use the global generators.

    Examples:
        >>> import random
        >>> stream = RandomStream(42)
        >>> with stream:
        ...     first = [random.random() for _ in range(3)]
        >>> with RandomStream(42):
        ...     again = [random.random() for _ in range(3)]
        >>> first == again
        True

    Remarks:
        - Covers :mod:`random` and the global (legacy) :mod:`numpy.random`
          generator; a `numpy.random.Generator` created elsewhere is untouched.
        - The stream is derived from ``seed`` rather than started from it, so
          ``RandomStream(42)`` and ``seed_all(42)`` give unrelated sequences.
        - Swapping the state of the global generators is not thread-safe. A
          stream left active in a worker thread (e.g. one abandoned on a
          timeout by a threaded runner) rewinds the global generators when it
          finally exits. Do not use seeded components under threaded runners.
        - Re-entering a stream that is already active (a seeded method calling
          another seeded method of the same object) simply continues it.
    """

    __slots__ = ("_seed", "_py_state", "_np_state", "_saved", "_depth")

    def __init__(self, seed: int | None = None) -> None:
        self._seed = None if seed is None else int(seed)
        self._py_state: Any = None
        self._np_state: Any = None
        self._saved: tuple[Any, Any] | None = None
        self._depth = 0

    @property
    def seed(self) -> int | None:
        """The seed of the stream (`None` when it uses the global generators)."""
        return self._seed

    def reseed(self, seed: int | None) -> None:
        """Restart the stream from ``seed`` (`None` switches it off)."""
        if self._depth:
            raise RuntimeError("Cannot reseed a RandomStream while it is active")
        self._seed = None if seed is None else int(seed)
        self._py_state = self._np_state = None

    def __enter__(self) -> RandomStream:
        if self._seed is None:
            return self
        self._depth += 1
        if self._depth > 1:
            return self
        self._saved = (random.getstate(), np.random.get_state())
        if self._py_state is None:
            # derived rather than used as is, so that `RandomStream(42)` does
            # not replay the very numbers `seed_all(42)` gives everything else
            derived = _derive(self._seed, _COMPONENT_STREAM)
            random.seed(derived)
            np.random.seed(derived % (2**32))
        else:
            random.setstate(self._py_state)
            np.random.set_state(self._np_state)
        return self

    def __exit__(self, *_: Any) -> None:
        if self._seed is None:
            return
        self._depth -= 1
        if self._depth:
            return
        self._py_state, self._np_state = random.getstate(), np.random.get_state()
        assert self._saved is not None
        random.setstate(self._saved[0])
        np.random.set_state(self._saved[1])
        self._saved = None

    def __getstate__(self) -> dict[str, Any]:
        return {
            "seed": self._seed,
            "py_state": self._py_state,
            "np_state": self._np_state,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self._seed = state["seed"]
        self._py_state = state["py_state"]
        self._np_state = state["np_state"]
        self._saved = None
        self._depth = 0

    def __repr__(self) -> str:
        return f"RandomStream(seed={self._seed})"


def seeded(seed: int | None) -> RandomStream:
    """Run a block of code with its own reproducible random numbers.

    A fresh `RandomStream` started from ``seed``. ``with seeded(s): ...``
    repeats its random draws whenever it is run with the same ``s``, whatever
    the program did before, and leaves the global generators as it found them.
    ``seeded(None)`` changes nothing.

    Examples:
        >>> from negmas import make_issue, make_os
        >>> os = make_os([make_issue(100, "price"), make_issue(100, "quantity")])
        >>> with seeded(7):
        ...     a = [os.random_outcome() for _ in range(5)]
        >>> with seeded(7):
        ...     b = [os.random_outcome() for _ in range(5)]
        >>> a == b
        True
    """
    return RandomStream(seed)


_F = TypeVar("_F", bound=Callable[..., Any])


def uses_random_stream(method: _F) -> _F:
    """Run a method inside its object's ``_random_stream`` (a `RandomStream`).

    Decorate the public methods of a component that draws random numbers and
    give it a ``_random_stream`` attribute (typically
    ``RandomStream(seed)`` built from a ``seed`` constructor argument). An
    object without that attribute runs unchanged.
    """

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        stream = getattr(self, "_random_stream", None)
        if stream is None:
            return method(self, *args, **kwargs)
        with stream:
            return method(self, *args, **kwargs)

    return wrapper  # type: ignore[return-value]


def seedable(fn: _F) -> _F:
    """Let ``fn`` be made reproducible through its ``seed`` keyword argument.

    The decorated function declares a keyword-only ``seed: int | None = None``
    parameter (and otherwise ignores it). Called with an integer ``seed`` it
    runs inside ``seeded(seed)``, so the same seed gives the same result
    whatever the program drew before; a generator it returns is consumed
    inside the block and returned as a list, so its draws are covered too.
    Called without a seed it runs exactly as it would undecorated.

    Remarks:
        Ids (`uuid4`) are not affected by ``seed``: they follow the global
        seed only, so two objects generated with the same ``seed`` are equal
        in content but keep distinct ids.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        seed = kwargs.get("seed")
        if seed is None:
            return fn(*args, **kwargs)
        with RandomStream(seed):
            result = fn(*args, **kwargs)
            if isinstance(result, types.GeneratorType):
                result = list(result)
        return result

    return wrapper  # type: ignore[return-value]
