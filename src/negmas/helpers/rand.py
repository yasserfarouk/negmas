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
"""

from __future__ import annotations

import random
from typing import Callable

import numpy as np

from negmas.warnings import warn

__all__ = ["seed_all", "get_seed", "register_seeder", "seed_environment"]

_NOT_A_SEED = ("", "none", "random")
"""Values of the setting that explicitly ask for fresh entropy"""

_seed: int | None = None
"""The seed passed to the last `seed_all` call (`None` if never seeded)"""

_seeders: list[Callable[[int], None]] = []
"""Callbacks registered by libraries that seed their own generators"""


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
    random.seed(seed)
    np.random.seed(seed % (2**32))
    _seed = seed
    for seeder in _seeders:
        seeder(seed)
    return seed


def register_seeder(seeder: Callable[[int], None]) -> None:
    """Register a callback seeding generators NegMAS itself knows nothing about.

    The callback is called with the seed by every later `seed_all`, and
    immediately if a seed is already in effect -- so a library imported after
    the seed was applied is still seeded.
    """
    _seeders.append(seeder)
    if _seed is not None:
        seeder(_seed)


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
