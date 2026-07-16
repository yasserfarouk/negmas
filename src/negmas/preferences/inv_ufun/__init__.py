"""Utility function inverters.

An inverse utility function (`InverseUFun`, see `negmas.preferences.protocols`) can
be used to find one or more outcomes with utility values in some given range.  This
package provides several implementations that trade off **accuracy**, **speed**,
**memory usage**, and **diversity** of the returned outcomes.  Each implementation
lives in its own module.

Available implementations
--------------------------

- `sampling` → `SamplingInverseUtilityFunction`:
    Inverts by drawing random outcomes from the outcome space and keeping those
    whose utility falls in the requested range.  Zero init cost, constant memory,
    but accuracy degrades severely for large or sparse spaces.

- `presorting` → `PresortingInverseUtilityFunction` (**recommended default**):
    Enumerates (or, for continuous spaces, samples) the full outcome space once,
    sorts all outcomes by utility in ``init()``, then answers every query with a
    binary search in ``O(log n)``.  Exact for discrete spaces; limited to
    spaces that fit in memory (~10^6 outcomes on typical hardware).

- `presorting_legacy` → `PresortingLegacyInverseUtilityFunction`:
    Functionally identical to `PresortingInverseUtilityFunction` but additionally
    narrows the binary-search window using a small set of pre-computed "waypoints"
    before the final bisection.  Kept for backwards compatibility only; empirically
    no faster (actually ~2× *slower*) than the default — see
    ``coding_agents/benchmark_presorting_waypoints.py``.

- `presorting_bruteforce` → `PresortingInverseUtilityFunctionBruteForce`:
    Like `PresortingInverseUtilityFunction` but uses simple linear scans instead
    of binary search.  Intentionally slow and simple — used as an exact ground
    truth for testing other inverters on tiny outcome spaces.

- `bids` → `BIDSInverseUtilityFunction`:
    Implements BIDS (Bidding using Diversified Search, Koça et al. 2024).
    Exploits the additive structure of `LinearAdditiveUtilityFunction` (and its
    subclass `LinearUtilityFunction`) to build a dynamic-programming table in
    ``init()`` and answer each ``one_in()`` query in *O(1)*.
    **Only works for additive utility functions** (raises ``TypeError`` otherwise).
    Approximate (error ≤ ``n_issues × 10^-precision``), but the only inverter that
    scales to spaces with 10^200+ outcomes.

- `adaptive` → `AdaptiveInverseUtilityFunction`:
    Automatically selects the best inverter based on ufun type and outcome-space
    size.  Use this when you do not want to couple your code to a specific
    implementation.

All classes implement the `InverseUFun` protocol from `negmas.preferences.protocols`.


Performance comparison
-----------------------

Benchmark on a MacBook Pro (Apple M-series), random `LinearAdditiveUtilityFunction`,
10 values per issue.  Times are median over 50 queries.  Accuracy = mean
``|normalised_utility(returned) − target|``.  See
``coding_agents/benchmark_all_inverters.py`` to reproduce.

.. code-block:: text

    Outcomes   Inverter      init (ms)  one_in (ms)  some (ms)    err   Notes
    ─────────────────────────────────────────────────────────────────────────────────
    25         BruteForce          0.1        0.001      0.002  0.159  exact (ground truth)
    25         Presorting          0.1        0.003      0.002  0.097  exact
    25         Legacy              0.1        0.005      0.003  0.098  exact
    25         Sampling            0.0        0.027      0.012  0.092  approx (2 issues only)
    25         BIDS-p3             0.2        0.005      0.018  0.322  approx (too few outcomes for grid)

    100        BruteForce          0.1        0.002      0.003  0.005  exact
    100        Presorting          0.2        0.003      0.004  0.005  exact
    100        Legacy              0.1        0.006      0.005  0.005  exact
    100        Sampling            0.0        0.034      0.013  0.004  approx
    100        BIDS-p3             0.2        0.004      0.018  0.004  approx

    10^3       BruteForce          0.8        0.009      0.014  0.006  exact
    10^3       Presorting          1.0        0.003      0.024  0.005  exact
    10^3       Legacy              1.0        0.006      0.023  0.005  exact
    10^3       Sampling            0.0        0.040      0.018  0.006  approx
    10^3       BIDS-p3             0.2        0.006      0.028  0.001  approx

    10^4       BruteForce          9.3        0.093      0.199  0.005  exact
    10^4       Presorting         11.7        0.003      0.291  0.005  exact
    10^4       Legacy             11.4        0.006      0.258  0.005  exact
    10^4       Sampling            0.0        0.056      0.022  0.027  approx
    10^4       BIDS-p3             0.3        0.009      0.036  0.000  approx

    10^5       Presorting        126.2        0.003      2.990  0.006  exact
    10^5       Legacy            126.2        0.006      2.488  0.005  exact
    10^5       Sampling            0.0        0.103      0.026  0.006  approx
    10^5       BIDS-p3             0.4        0.011      0.047  0.000  approx

    10^6       Presorting       1424.9        0.004     91.131  0.005  exact (some() scans all)
    10^6       Legacy           1408.8        0.008     76.587  0.006  exact (some() scans all)
    10^6       Sampling            0.0        0.182      0.029  0.007  approx
    10^6       BIDS-p3             0.4        0.013      0.055  0.000  approx

    10^10      Sampling            0.0        0.206      0.046  0.023  approx
    10^10      BIDS-p3             0.5        0.021      0.091  0.000  approx (additive only)

    10^100     BIDS-p3             4.5        0.189      0.909  0.045  approx (additive only)

    10^200     BIDS-p3             8.9        0.386      1.849  0.072  approx (additive only)

Key observations:

* **Presorting / Legacy** are exact and have sub-millisecond ``one_in()`` queries
  after init, but ``init()`` costs scale as ``O(n log n)`` with the space size
  (~1.4 s at 10^6 outcomes) and ``some()`` scans the sorted array linearly (~91 ms
  at 10^6).  They cannot handle spaces beyond ~5×10^6 due to memory constraints.

* **BIDS** is the clear winner for large additive spaces: ``init()`` is under 10 ms
  even at 10^200 outcomes, ``one_in()`` stays below 0.4 ms, and accuracy is
  excellent (err < 0.001 at 10^3 – 10^10 outcomes, < 0.1 at 10^100+).  Note that
  accuracy can be poor for very small discrete spaces (err ≈ 0.32 at 25 outcomes)
  because the 1001-point grid is too coarse relative to the 25-outcome space —
  `PresortingInverseUtilityFunction` is preferable there.

* **Sampling** has zero init cost and works on any ufun type, but accuracy
  degrades rapidly beyond 10^4 outcomes and the inverter is completely non-functional
  for very large spaces that cannot be enumerated.

* **BruteForce** is the exact ground truth, suitable only for tiny spaces
  (< 50 K outcomes).

Choosing an inverter
---------------------

In most cases, use `AdaptiveInverseUtilityFunction`, which applies the following
policy automatically:

1. If ``cardinality ≤ max_presorting_outcomes`` (default 10^6):
   → `PresortingInverseUtilityFunction` (exact).
2. Else if the ufun is `LinearAdditiveUtilityFunction` / `LinearUtilityFunction`:
   → `BIDSInverseUtilityFunction` (approximate, scalable).
3. Else (large, non-additive ufun):
   → `PresortingInverseUtilityFunction` with ``max_cache_size`` sampling.

If you know your ufun is additive and the space is large, instantiate
`BIDSInverseUtilityFunction` directly and tune ``precision`` (higher = smaller
error but larger table) and ``n_samples`` (more samples = better ``best_in``/
``worst_in`` quality at the cost of more ``one_in()`` calls).

Reference
---------
Koça, T., de Jonge, D., & Baarslag, T. (2024).
*Search algorithms for automated negotiation in large domains.*
Algorithms 17(5), 200.
"""

from __future__ import annotations

from .adaptive import AdaptiveInverseUtilityFunction
from .bids import BIDSInverseUtilityFunction
from .presorting import PresortingInverseUtilityFunction
from .presorting_bruteforce import PresortingInverseUtilityFunctionBruteForce
from .presorting_legacy import PresortingLegacyInverseUtilityFunction
from .sampling import SamplingInverseUtilityFunction

# NOTE: PresortingInverseUtilityFunctionBruteForce is intentionally not part of
# __all__ (matching the pre-refactor public API) since it is meant as a testing
# ground-truth reference rather than a class end users should build strategies
# around. It remains fully importable directly from this package or its own
# submodule (`negmas.preferences.inv_ufun.presorting_bruteforce`).
__all__ = [
    "AdaptiveInverseUtilityFunction",
    "BIDSInverseUtilityFunction",
    "SamplingInverseUtilityFunction",
    "PresortingInverseUtilityFunction",
    "PresortingLegacyInverseUtilityFunction",
]
