"""`_nearest_around` must answer exactly as the loop it replaced, on any array.

The loop scanned the whole array on every call -- its first statement,
``n = len(a) - 1``, overwrote the caller's search radius -- which on a
188,160-outcome domain cost 13ms per call and 64% of the runtime of a whole
negotiation. It is now vectorized.

The reach cannot be narrowed to a window around ``i``: `a` is not sorted end to end,
because the presorting inverters append the irrational outcomes to the sorted rational
ones without sorting them. So these tests cover unsorted arrays too, and the tie cases
that decide which of several equally-near entries wins.
"""

import random

import numpy as np
import pytest

from negmas.preferences.inv_ufun._common import EPS, _nearest_around


def scan(x, a, i, mn, mx, eps=EPS):
    """The original implementation, kept verbatim as the oracle."""
    n = len(a) - 1
    best, best_diff = i, abs(a[i] - x)
    for j in range(i - n, i + n + 1):
        if j < 0 or j > n:
            continue
        if not (mn <= a[j] <= mx):
            continue
        d = abs(a[j] - x)
        if d < best_diff:
            best, best_diff = j, d
    if abs(a[best] - x) > eps and best != i:
        return None
    return best


def _array(rng, style, n):
    if style == "random":
        return [round(rng.uniform(0, 1), rng.choice([2, 8])) for _ in range(n)]
    if style == "sorted":
        return sorted(round(rng.uniform(0, 1), rng.choice([2, 8])) for _ in range(n))
    if style == "plateau":  # many equal utilities: the tie-breaking case
        return sorted(round(rng.uniform(0, 1), 1) for _ in range(n))
    if style == "identical":
        return [0.5] * n
    # A sorted rational prefix followed by an unsorted irrational tail, which is
    # exactly the shape `PresortingInverseUtilityFunction.utils` has.
    k = rng.randint(1, n)
    return sorted(round(rng.uniform(0.5, 1), 3) for _ in range(k)) + [
        round(rng.uniform(0, 0.5), 3) for _ in range(n - k)
    ]


@pytest.mark.parametrize(
    "style", ["random", "sorted", "plateau", "identical", "sorted_with_tail"]
)
def test_matches_the_scan_it_replaced(style):
    rng = random.Random(hash(style) % 10_000)
    for _ in range(2_000):
        a = np.asarray(_array(rng, style, rng.randint(1, 25)), dtype=float)
        i = rng.randrange(len(a))
        # Half the probes land exactly on a stored value, half fall between or outside.
        x = rng.choice(list(a)) if rng.random() < 0.5 else rng.uniform(-0.2, 1.2)
        mn, mx = sorted((rng.uniform(-0.2, 1.2), rng.uniform(-0.2, 1.2)))
        if rng.random() < 0.3:  # every entry in range
            mn, mx = a.min() - 1, a.max() + 1
        elif rng.random() < 0.15:  # no entry in range
            mn, mx = 5.0, 6.0
        eps = rng.choice([EPS, 2 * EPS, 1e-3, 1.0])
        assert _nearest_around(x, a, i, mn, mx, eps=eps) == scan(x, a, i, mn, mx, eps)


def test_falls_back_to_the_given_index_when_nothing_is_in_range():
    """The clamping behaviour the presorting inverters rely on to avoid stalling."""
    a = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0])
    assert _nearest_around(0.5, a, 2, 5.0, 6.0, eps=EPS) == 2


def test_an_exact_hit_far_from_the_starting_index_is_still_found():
    """The search covers the whole array, not a window around `i`."""
    a = np.asarray([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    assert _nearest_around(0.9, a, 0, 0.0, 1.0, eps=EPS) == 9


def test_is_far_cheaper_than_the_scan_on_a_large_array():
    """The point of vectorizing: the same O(n) work, at a much smaller constant.

    This is a constant-factor fix, not a complexity one -- the search still has to
    consider every entry, because `a` is not sorted end to end. What changed is that the
    per-entry work moved out of the Python interpreter.
    """
    import time

    n = 200_000
    a = np.linspace(0.0, 1.0, n)
    probes = [k / 50 for k in range(50)]

    def elapsed(f):
        t = time.perf_counter()
        for x in probes:
            f(x, a, n // 2, 0.0, 1.0, eps=EPS)
        return time.perf_counter() - t

    fast, slow = elapsed(_nearest_around), elapsed(scan)
    assert fast * 5 < slow, f"vectorized {fast:.4f}s vs scan {slow:.4f}s"
