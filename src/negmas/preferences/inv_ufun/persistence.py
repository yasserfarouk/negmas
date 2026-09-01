"""Saving and loading initialized inverse utility functions to/from disk.

Inverting a utility function is expensive: the outcome space is enumerated (or
sampled) and every outcome is evaluated and sorted. When the *same* scenario is
negotiated repeatedly, that cost is paid again on every run. This module lets an
initialized inverter be persisted next to the scenario it belongs to, and
restored on load so that :meth:`BaseUtilityFunction.invert` returns immediately.

.. warning::

    Reusing a stale inverse silently corrupts every negotiation that uses it, so
    caching here is **opt-in and fail-closed**:

    * Nothing is ever written unless the caller explicitly asks for it
      (``Scenario.dumpas(..., save_inverse=True)``).
    * A cache is only reused when every validity gate passes (see
      :func:`load_state`); any doubt at all — a changed outcome space, a
      changed reserved value, a mutated ufun, a corrupt file — results in the
      cache being ignored and the inverter rebuilt from scratch.
    * Nothing is ever unpickled. Arrays are stored in ``.npz`` with
      ``allow_pickle=False`` so loading a scenario can never execute code.

Layout
------

Caches live in an ``_inverses`` subfolder of the scenario folder (kept out of the
scenario folder proper so the ufun/domain file finders never see them)::

    <scenario>/
        MyDomain.yml
        ufun0.yml
        ufun1.yml
        _inverses/
            manifest.json      # format version + one entry per cached ufun
            ufun0.npz          # outcome columns + utilities
            ufun1.npz
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from negmas.outcomes import Outcome

if TYPE_CHECKING:
    from ..base_ufun import BaseUtilityFunction
    from ..protocols import InverseUFun

__all__ = [
    "INVERSE_FOLDER_NAME",
    "INVERSE_MANIFEST_NAME",
    "DEFAULT_INVERSE_CONFIGS",
    "can_cache_inverse",
    "save_inverter",
    "remove_saved_inverses",
    "outcome_space_fingerprint",
    "entry_key",
    "effective_config",
    "load_state",
]

INVERSE_FOLDER_NAME = "_inverses"
"""Name of the sub-folder (inside a scenario folder) holding cached inverses."""

INVERSE_MANIFEST_NAME = "manifest.json"
"""Name of the manifest file inside `INVERSE_FOLDER_NAME`."""

INVERSE_FORMAT_VERSION = 1
"""Bumped whenever the on-disk layout changes incompatibly."""

DEFAULT_INVERSE_CONFIGS: tuple[dict[str, Any], ...] = (
    {},
    {"rational_only": True, "eps": -1, "rel_eps": -1},
    # `negmas.gb.components.selectors.make_inverter` caps the cache size, which
    # changes the stored arrays and so needs its own entry.
    {"max_cache_size": 10_000},
)
"""Inverter configurations cached by default.

Different components ask for different configurations, and each produces
*different* sorted arrays, so a cache is only useful if it covers the
configuration actually requested. These are the two used across negmas' own
negotiators.
"""

DEFAULT_VERIFICATION_SAMPLES = 256
"""How many stored outcomes are re-evaluated to verify a cache by default."""

VERIFICATION_TOLERANCE = 1e-9
"""Absolute tolerance when comparing re-evaluated utilities against stored ones."""


# --------------------------------------------------------------------------
# Fingerprinting
# --------------------------------------------------------------------------


def _hash(payload: Any) -> str:
    """Stable sha256 of a JSON-serializable payload."""
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def outcome_space_fingerprint(outcome_space) -> str:
    """A fingerprint of an outcome space's *structure*.

    Outcomes are stored as positional tuples, so issue **order** matters just as
    much as issue contents. Reordering, renaming, adding, or removing an issue —
    or changing any issue's values — all change the fingerprint.
    """
    if outcome_space is None:
        return ""
    issues = []
    for issue in outcome_space.issues:
        try:
            values = list(issue.all)
        except Exception:
            # Continuous / uncountable issue: describe it by its bounds instead.
            values = ["__continuous__", *(str(_) for _ in issue.values)]
        issues.append(
            {
                "name": issue.name,
                "type": type(issue).__name__,
                "values": [str(_) for _ in values],
            }
        )
    return _hash({"issues": issues})


def _reserved_fingerprint(ufun: BaseUtilityFunction) -> str:
    """Fingerprint of the reserved value (which changes ``rational_only`` results)."""
    r = getattr(ufun, "_reserved_value", None)
    try:
        return repr(float(r))  # type: ignore[arg-type]
    except Exception:
        return repr(r)


def can_cache_inverse(ufun: BaseUtilityFunction) -> tuple[bool, str]:
    """Whether an inverse of this ufun may be cached to (or loaded from) disk.

    Returns:
        ``(True, "")`` when caching is allowed, otherwise ``(False, reason)``.

    Remarks:
        Caching is refused for:

        - Ufuns with no outcome space (nothing to enumerate).
        - Non-stationary ufuns: a time-dependent ufun's inverse is only valid at
          the instant it was built, so persisting it is meaningless.
        - Ufuns carrying constraints: constraints are arbitrary callables that
          force utilities to ``-inf`` and cannot be fingerprinted.
    """
    if getattr(ufun, "outcome_space", None) is None:
        return False, "the ufun has no outcome space"
    stability = getattr(ufun, "_stability", None)
    if stability is None or not stability.is_stationary:
        return False, "the ufun is not stationary"
    if getattr(ufun, "constraints", None):
        return False, "the ufun has constraints (which cannot be fingerprinted)"
    return True, ""


# --------------------------------------------------------------------------
# Outcome (de)serialization without pickle
# --------------------------------------------------------------------------


def _outcomes_to_columns(outcomes: list[Outcome]) -> tuple[dict[str, Any], list[str]]:
    """Splits a list of outcome tuples into one array per issue.

    Storing per-issue *columns* (rather than an object array of tuples) is what
    lets the arrays be written with ``allow_pickle=False``.

    Returns:
        ``(arrays, kinds)`` where ``arrays`` maps ``"col{i}"`` to a numpy array
        and ``kinds`` records the original Python type of each column so values
        round-trip as ``int``/``float``/``str`` rather than numpy scalars.
    """
    if not outcomes:
        return {}, []
    width = len(outcomes[0])
    arrays: dict[str, Any] = {}
    kinds: list[str] = []
    for i in range(width):
        column = [o[i] for o in outcomes]
        first = column[0]
        if isinstance(first, bool):
            kind = "bool"
            arr = np.asarray(column, dtype=bool)
        elif isinstance(first, (int, np.integer)):
            kind = "int"
            arr = np.asarray(column, dtype=np.int64)
        elif isinstance(first, (float, np.floating)):
            kind = "float"
            arr = np.asarray(column, dtype=np.float64)
        else:
            kind = "str"
            arr = np.asarray([str(_) for _ in column], dtype=np.str_)
        arrays[f"col{i}"] = arr
        kinds.append(kind)
    return arrays, kinds


def _columns_to_outcomes(data, kinds: list[str], n: int) -> list[Outcome]:
    """Rebuilds outcome tuples from per-issue columns written by `_outcomes_to_columns`."""
    if not kinds or n == 0:
        return []
    columns = []
    for i, kind in enumerate(kinds):
        arr = data[f"col{i}"]
        if kind == "bool":
            columns.append([bool(_) for _ in arr])
        elif kind == "int":
            columns.append([int(_) for _ in arr])
        elif kind == "float":
            columns.append([float(_) for _ in arr])
        else:
            columns.append([str(_) for _ in arr])
    return [tuple(col[j] for col in columns) for j in range(n)]


# --------------------------------------------------------------------------
# Saving
# --------------------------------------------------------------------------


def _persistable(inverter: InverseUFun):
    """Returns the concrete inverter carrying persistable arrays, or ``None``.

    `AdaptiveInverseUtilityFunction` (the default) is a thin forwarder, so this
    unwraps to its delegate.
    """
    target = inverter
    delegate = getattr(target, "delegate", None)
    if delegate is not None:
        target = delegate
    if hasattr(target, "persist_state") and hasattr(target, "restore_state"):
        return target
    return None


def save_inverter(
    folder: Path | str, ufun: BaseUtilityFunction, inverter: InverseUFun, name: str
) -> bool:
    """Saves one initialized inverter into ``<folder>/_inverses``.

    Args:
        folder: The *scenario* folder (the ``_inverses`` subfolder is created inside it).
        ufun: The ufun the inverter belongs to.
        inverter: An **initialized** inverter.
        name: The key used to match this cache back to its ufun on load (the
            ufun's serialized file stem).

    Returns:
        True if a cache was written, False if this inverter/ufun cannot be cached.
    """
    ok, _ = can_cache_inverse(ufun)
    if not ok:
        return False
    target = _persistable(inverter)
    if target is None or not getattr(target, "initialized", False):
        return False
    state = target.persist_state()
    if state is None:
        return False
    key = entry_key(name, state.get("config", {}))

    folder = Path(folder) / INVERSE_FOLDER_NAME
    folder.mkdir(parents=True, exist_ok=True)

    outcomes: list[Outcome] = state["outcomes"]
    utils = np.asarray(state["utils"], dtype=np.float64)
    arrays, kinds = _outcomes_to_columns(outcomes)
    np.savez_compressed(folder / f"{key}.npz", utils=utils, **arrays)

    entry = {
        "file": f"{key}.npz",
        "ufun": name,
        "inverter": type(target).__name__,
        "n_outcomes": len(outcomes),
        "kinds": kinds,
        "outcome_space": outcome_space_fingerprint(ufun.outcome_space),
        "reserved_value": _reserved_fingerprint(ufun),
        "config": state.get("config", {}),
        "extra": state.get("extra", {}),
    }
    manifest = _read_manifest(folder.parent)
    manifest["entries"][key] = entry
    _write_manifest(folder.parent, manifest)
    return True


def _read_manifest(scenario_folder: Path) -> dict[str, Any]:
    """Reads the inverse manifest, returning an empty one when absent or unreadable."""
    path = Path(scenario_folder) / INVERSE_FOLDER_NAME / INVERSE_MANIFEST_NAME
    empty: dict[str, Any] = {"version": INVERSE_FORMAT_VERSION, "entries": {}}
    if not path.is_file():
        return empty
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return empty
    if not isinstance(d, dict) or d.get("version") != INVERSE_FORMAT_VERSION:
        return empty
    if not isinstance(d.get("entries"), dict):
        return empty
    return d


def _write_manifest(scenario_folder: Path, manifest: dict[str, Any]) -> None:
    """Writes the inverse manifest for a scenario folder."""
    folder = Path(scenario_folder) / INVERSE_FOLDER_NAME
    folder.mkdir(parents=True, exist_ok=True)
    (folder / INVERSE_MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )


def remove_saved_inverses(folder: Path | str) -> bool:
    """Deletes any cached inverses under ``folder``.

    Called whenever a scenario is saved *without* ``save_inverse``, so a stale
    cache can never outlive the scenario it was built for.

    Returns:
        True if a cache folder was found and removed.
    """
    path = Path(folder) / INVERSE_FOLDER_NAME
    if not path.is_dir():
        return False
    import shutil

    shutil.rmtree(path, ignore_errors=True)
    return not path.exists()


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def _verify_against_ufun(
    ufun: BaseUtilityFunction, outcomes: list[Outcome], utils, n_samples: int
) -> bool:
    """Re-evaluates the ufun on a sample of stored outcomes and compares utilities.

    This is the check that actually protects against a *changed ufun*: rather than
    fingerprinting a serialization (which is fragile — value functions may contain
    lambdas that serialize differently across runs), it directly asks whether this
    ufun still agrees with the cached utilities.

    Remarks:
        With ``n_samples < len(outcomes)`` this is a probabilistic check: a change
        affecting only a handful of outcomes in a very large space may be missed.
        Pass ``n_samples <= 0`` to verify **every** stored outcome (as expensive as
        rebuilding, so only useful for testing).
    """
    n = len(outcomes)
    if n == 0:
        return True
    if n_samples <= 0 or n_samples >= n:
        indices = range(n)
    else:
        rng = np.random.default_rng(0)
        # Always include the extremes: they set the normalization range, so an
        # error there corrupts every normalized query.
        picked = set(rng.integers(0, n, size=n_samples).tolist())
        picked.update((0, n - 1))
        indices = sorted(picked)
    try:
        for i in indices:
            expected = float(utils[i])
            actual = float(ufun.eval(outcomes[i]))
            if math.isinf(expected) or math.isinf(actual):
                if expected != actual:
                    return False
                continue
            if not math.isclose(
                actual, expected, rel_tol=1e-9, abs_tol=VERIFICATION_TOLERANCE
            ):
                return False
    except Exception:
        return False
    return True


def load_state(
    folder: Path | str,
    ufun: BaseUtilityFunction,
    name: str,
    config: dict[str, Any],
    verification_samples: int = DEFAULT_VERIFICATION_SAMPLES,
) -> dict[str, Any] | None:
    """Reads and validates one cached inverse, returning restorable state.

    This is where every validity gate lives. All of the following must hold,
    otherwise ``None`` is returned and the caller rebuilds normally:

    1. The ufun is cacheable at all (stationary, has an outcome space, no
       constraints) - see :func:`can_cache_inverse`.
    2. A manifest entry exists for this ufun *and* configuration, with a readable
       ``.npz`` beside it.
    3. The outcome-space fingerprint matches exactly (issue order included).
    4. The reserved value matches.
    5. Re-evaluating the ufun on a sample of the stored outcomes reproduces the
       stored utilities.

    Args:
        folder: The scenario folder containing the ``_inverses`` subfolder.
        ufun: The ufun the cache must agree with.
        name: The cache key (the ufun's serialized file stem).
        config: The array-affecting configuration to look up.
        verification_samples: How many stored outcomes to re-evaluate in gate 5.
            ``<= 0`` verifies all of them.

    Returns:
        A dict with ``outcomes``, ``utils`` and ``extra`` suitable for an
        inverter's ``restore_state``, or ``None``.
    """
    ok, _ = can_cache_inverse(ufun)
    if not ok:
        return None
    folder = Path(folder)
    manifest = _read_manifest(folder)
    entry = manifest["entries"].get(entry_key(name, config))
    if not entry:
        return None
    if entry.get("outcome_space") != outcome_space_fingerprint(ufun.outcome_space):
        return None
    if entry.get("reserved_value") != _reserved_fingerprint(ufun):
        return None
    if _canonical_config(config) != _canonical_config(entry.get("config", {})):
        return None

    path = folder / INVERSE_FOLDER_NAME / entry.get("file", "")
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            utils = np.asarray(data["utils"], dtype=np.float64)
            outcomes = _columns_to_outcomes(
                data, list(entry.get("kinds", [])), int(entry.get("n_outcomes", 0))
            )
    except Exception:
        # A truncated/corrupt cache must never break scenario loading.
        return None
    if len(outcomes) != len(utils):
        return None
    if not _verify_against_ufun(ufun, outcomes, utils, verification_samples):
        return None
    return {"outcomes": outcomes, "utils": utils, "extra": entry.get("extra", {})}


def _canonical_config(config: dict[str, Any]) -> str:
    """Config comparison that ignores key order and int/float spelling."""
    return json.dumps({k: str(v) for k, v in sorted(config.items())})


def _config_digest(config: dict[str, Any]) -> str:
    """A short, stable digest of an effective inverter configuration."""
    return hashlib.sha256(_canonical_config(config).encode("utf-8")).hexdigest()[:16]


def entry_key(name: str, config: dict[str, Any]) -> str:
    """The manifest key for one ufun cached under one configuration.

    A single ufun may be inverted under several configurations (agents differ in
    e.g. ``rational_only``), and those produce *different* arrays, so each gets
    its own entry rather than one overwriting the other.
    """
    return f"{name}@{_config_digest(config)}"


def effective_config(
    ufun: BaseUtilityFunction, inverter_type: type | None, kwargs: dict[str, Any]
) -> dict[str, Any] | None:
    """The array-affecting configuration an inverter would actually be built with.

    Constructs the inverter (cheap — all the work happens in ``init()``) and asks
    it, so caller-omitted parameters are filled in with real defaults instead of
    being compared as absent.
    """
    if inverter_type is None:
        from .adaptive import AdaptiveInverseUtilityFunction

        inverter_type = AdaptiveInverseUtilityFunction
    try:
        getter = getattr(inverter_type(ufun, **kwargs), "persistable_config", None)
        return getter() if getter is not None else None
    except Exception:
        return None
