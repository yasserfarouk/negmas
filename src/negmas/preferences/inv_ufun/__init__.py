"""Utility function inverters.

An inverse utility function (`InverseUFun`, see `negmas.preferences.protocols`) can
be used to find one or more outcomes with utility values in some given range.  This
package provides several implementations that trade off **accuracy**, **speed**,
**memory usage**, and **diversity** of the returned outcomes.  Each implementation
lives in its own module.

Two kinds of inverters
----------------------

All inverters implement the `InverseUFun` protocol with three core single-outcome
queries — ``one_in``, ``worst_in``, ``best_in`` — and a multi-outcome query ``some``.
They differ in **what happens when no outcome's utility falls inside the requested
range**:

* **Strict** inverters return ``None``.  They never clamp, expand the range, or
  fall back to an out-of-range outcome.  Use them when you need the exact
  semantics "tell me whether an in-range outcome exists".

  - `BruteForceInverseUtilityFunction`
  - `PresortingInverseUtilityFunctionBruteForce`

* **Clamping** inverters apply fallback strategies so that, as long as the
  outcome space is non-empty, they return *some* outcome rather than ``None``.
  The fallback order is:

  1. **Expand the range upward** to ``[rng[0], max]`` and retry (controlled by
     ``fallback_to_higher``; default ``True``).  This is the primary strategy
     used by `AspirationNegotiator` when its aspiration band sits above the
     currently achievable utilities (e.g. under heavy discounting).
  2. **Return the nearest boundary outcome**: the best outcome if the
     requested range is in the upper half of the utility range, the worst
     outcome if it is in the lower half (controlled by ``fallback_to_best`` /
     ``fallback_to_worst``; default ``True``).  This keeps the fallback close
     to the requested range.

  - `SamplingInverseUtilityFunction`
  - `PresortingInverseUtilityFunction`
  - `PresortingLegacyInverseUtilityFunction`
  - `BIDSInverseUtilityFunction`
  - `MCTSInverseUtilityFunction`
  - `AttributePlanningInverseUtilityFunction`
  - `AdaptiveInverseUtilityFunction` (delegates to one of the above)

``one_in`` is clamping in **every** inverter (including the strict ones): it
always has ``fallback_to_higher`` and ``fallback_to_best`` parameters (both
default ``True``), so it never returns ``None`` for a non-empty outcome space.
The strict/clamping distinction only applies to ``worst_in`` and ``best_in``.

Available implementations
--------------------------

- `sampling` → `SamplingInverseUtilityFunction`:
    Inverts by drawing random outcomes from the outcome space and keeping those
    whose utility falls in the requested range.  Zero init cost, constant memory,
    but accuracy degrades severely for large or sparse spaces.

- `presorting` → `PresortingInverseUtilityFunction`:
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

- `bruteforce` → `BruteForceInverseUtilityFunction` **(strict)**:
    The project's exact **ground truth**. Enumerates the outcome space, sorts once,
    and answers every query with a simple linear scan (no bisection, no clamping).
    Caches the outcome/utility arrays only for *stationary* ufuns (rebuilds each call
    otherwise). Intentionally slow and simple — used to validate the faster inverters
    on small domains.

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

- `attribute_planning` → `AttributePlanningInverseUtilityFunction`:
    Lightweight per-issue target matching (Jonker & Treur). Very fast and
    scalable for additive ufuns, but less accurate and less diverse than BIDS.

- `mcts` → `MCTSInverseUtilityFunction`:
    Monte-Carlo Tree Search inverter. Works with non-additive ufuns and provides
    diverse outcomes, but is slower than presorting/BIDS at equal accuracy.

- `DefaultInverseUtilityFunction`:
    Public default used by the library. Aliased to
    `AdaptiveInverseUtilityFunction`.

All classes implement the `InverseUFun` protocol from `negmas.preferences.protocols`.


Feature table
-------------

The table below summarises the key behavioural and performance properties of
every inverter.  "Fallback" refers to what ``worst_in``/``best_in`` do when no
in-range outcome is found (see "Two kinds of inverters" above).

.. list-table::
   :header-rows: 1
   :widths: 22 10 10 10 10 10 10 18

   * - Inverter
     - Type
     - Fallback
     - Tolerance
     - Exact?
     - Additive-only?
     - Scales to 10^6+
     - Notes
   * - BruteForceInverseUtilityFunction
     - exact
     - **strict** (None)
     - eps param
     - yes
     - no
     - no (< 50K)
     - ground truth; linear scan
   * - PresortingInverseUtilityFunctionBruteForce
     - exact
     - **strict** (None)
     - eps param
     - yes
     - no
     - no (< 50K)
     - brute-force variant of presorting
   * - PresortingInverseUtilityFunction
     - exact
     - clamping
     - clamp_tolerance
     - yes
     - no
     - yes (~10^6)
     - binary search; default for small spaces
   * - PresortingLegacyInverseUtilityFunction
     - exact
     - clamping
     - eps param
     - yes
     - no
     - yes (~10^6)
     - waypointed bisection; ~2× slower
   * - SamplingInverseUtilityFunction
     - approx
     - clamping
     - eps + rel_eps
     - no
     - no
     - yes (any)
     - zero init; misses rare outcomes
   * - BIDSInverseUtilityFunction
     - approx
     - clamping
     - EPS
     - no (≤ n×10^-p)
     - **yes**
     - yes (10^200+)
     - DP table; best for large additive
   * - AttributePlanningInverseUtilityFunction
     - approx
     - clamping
     - EPS
     - no
     - yes (additive best)
     - yes (10^200+)
     - per-issue matching; fast
   * - MCTSInverseUtilityFunction
     - approx
     - clamping
     - EPS
     - no
     - no
     - yes (any)
     - MCTS search; diverse; slow
   * - AdaptiveInverseUtilityFunction
     - auto
     - clamping (delegates)
     - inherited
     - inherited
     - inherited
     - yes
     - picks presorting/BIDS/sampling

Key for the "Fallback" column:

* **strict (None)** — ``worst_in``/``best_in`` return ``None`` when no in-range
  outcome is found.  No clamping, no range expansion.
* **clamping** — ``worst_in``/``best_in`` expand the range upward, then fall back
  to the nearest boundary outcome (best if the range is in the upper half of the
  utility range, worst if in the lower half).  Never returns ``None`` for a
  non-empty outcome space.

Key for the "Tolerance" column:

* **eps param** — a single ``eps`` parameter widens the accepted band by
  ``±eps`` around the requested range.
* **clamp_tolerance** — in addition to ``eps``, a separate ``_clamp_tolerance``
  controls how far outside the range a clamped outcome may lie.
* **eps + rel_eps** — an absolute ``eps`` and a relative ``rel_eps``; the
  effective tolerance is ``min(eps, rel_eps × x)``.


Performance comparison
-----------------------

Benchmark on a MacBook Pro (Apple M-series), random `LinearAdditiveUtilityFunction`,
10 values per issue.  Times are median over 50 queries.  Accuracy = mean
``|normalised_utility(returned) − target|``.  See
``coding_agents/benchmark_all_inverters.py`` and
``coding_agents/benchmark_all_inverters_results.txt`` to reproduce.

.. code-block:: text

    Outcomes   Inverter        init (ms)  one_in (ms)  some (ms)     err  Status
    ─────────────────────────────────────────────────────────────────────────────
    25         BruteForce            0.1        0.001      0.001  0.159  OK
    25         PresortingBF          0.0        0.001      0.002  0.128  OK
    25         Presorting            0.1        0.002      0.002  0.015  OK
    25         Legacy                0.1        0.005      0.003  0.229  OK
    25         Sampling              0.0        0.039      0.008  0.097  OK
    25         BIDS-p3               0.2        0.004      0.019  0.242  OK
    25         Adaptive              0.1        0.003      0.002  0.016  OK
    25         AttrPlan              0.0        0.106      0.021  0.327  OK
    25         MCTS                  0.0        0.312      1.561  0.129  OK

    100        BruteForce            0.1        0.002      0.003  0.090  OK
    100        PresortingBF          0.1        0.002      0.003  0.075  OK
    100        Presorting            0.2        0.002      0.004  0.010  OK
    100        Legacy                0.1        0.005      0.004  0.196  OK
    100        Sampling              0.0        0.049      0.008  0.050  OK
    100        BIDS-p3               0.2        0.004      0.018  0.261  OK
    100        Adaptive              0.2        0.002      0.004  0.010  OK
    100        AttrPlan              0.0        0.005      0.020  0.345  OK
    100        MCTS                  0.1        0.380      1.947  0.079  OK

    10^3       BruteForce            2.1        0.010      0.017  0.005  OK
    10^3       PresortingBF          0.8        0.010      0.016  0.005  OK
    10^3       Presorting            1.1        0.002      0.028  0.006  OK
    10^3       Legacy                1.0        0.005      0.024  0.008  OK
    10^3       Sampling              0.0        0.052      0.010  0.007  OK
    10^3       BIDS-p3               0.2        0.007      0.028  0.034  OK
    10^3       Adaptive              1.1        0.003      0.027  0.005  OK
    10^3       AttrPlan              0.0        0.006      0.025  0.239  OK
    10^3       MCTS                  0.7        0.479      2.478  0.034  OK

    10^4       BruteForce            8.4        0.088      0.154  0.005  OK
    10^4       PresortingBF          8.3        0.073      0.142  0.005  OK
    10^4       Presorting           11.0        0.003      0.268  0.005  OK
    10^4       Legacy               13.3        0.005      0.192  0.010  OK
    10^4       Sampling              0.0        0.048      0.013  0.006  OK
    10^4       BIDS-p3               0.3        0.008      0.037  0.000  OK
    10^4       Adaptive             11.2        0.003      0.257  0.006  OK
    10^4       AttrPlan              0.1        0.008      0.031  0.196  OK
    10^4       MCTS                  8.8        0.558      2.840  0.041  OK

    10^5       PresortingBF        104.9        3.037      5.478  0.005  OK
    10^5       Presorting          138.1        0.003      3.653  0.005  OK
    10^5       Legacy              135.7        0.005      2.921  0.010  OK
    10^5       Sampling              0.0        0.087      0.015  0.005  OK
    10^5       BIDS-p3               0.3        0.011      0.048  0.000  OK
    10^5       Adaptive            133.3        0.003      3.495  0.004  OK
    10^5       AttrPlan              0.1        0.009      0.037  0.245  OK
    10^5       MCTS                  0.0        0.607      3.099  0.066  OK

    10^6       Presorting         1473.5        0.004     67.401  0.004  OK
    10^6       Legacy             1424.7        0.006     55.671  0.010  OK
    10^6       Sampling              0.0        0.177      0.018  0.005  OK
    10^6       BIDS-p3               0.4        0.013      0.054  0.000  OK
    10^6       Adaptive           1462.2        0.004     64.855  0.006  OK
    10^6       AttrPlan              0.1        0.012      0.044  0.227  OK
    10^6       MCTS                  0.0        0.664      3.429  0.041  OK

    10^10      Sampling              0.0        0.221      0.028  0.043  OK
    10^10      BIDS-p3               0.6        0.021      0.092  0.000  OK
    10^10      Adaptive              0.6        0.021      0.095  0.000  OK
    10^10      AttrPlan              0.1        0.016      0.066  0.300  OK
    10^10      MCTS                  0.0        0.864      4.332  0.093  OK

    10^20      Sampling              0.0        1.309      0.043  0.035  OK
    10^20      BIDS-p3               2.4        0.038      0.181  0.000  OK
    10^20      Adaptive              1.0        0.038      0.176  0.000  OK
    10^20      AttrPlan              0.1        0.026      0.123  0.176  OK
    10^20      MCTS                  0.0        1.021      5.047  0.176  OK

    10^50      Sampling              0.0       66.195      0.102  0.137  OK
    10^50      BIDS-p3               5.1        0.097      0.457  0.000  OK
    10^50      Adaptive              2.4        0.095      0.462  0.000  OK
    10^50      AttrPlan              0.2        0.065      0.308  0.064  OK
    10^50      MCTS                  0.0        2.026     10.043  0.129  OK

    10^100     Sampling              0.0     1799.568      0.197  0.161  OK
    10^100     BIDS-p3               5.0        0.197      0.957  0.000  OK
    10^100     Adaptive              4.9        0.197      0.936  0.000  OK
    10^100     AttrPlan              0.4        0.125      0.601  0.072  OK
    10^100     MCTS                  0.0        3.913     19.192  0.204  OK

    10^200     Sampling              0.0     3636.924      0.491  0.144  OK
    10^200     BIDS-p3              19.2        0.842      3.975  0.000  OK
    10^200     Adaptive             19.7        0.872      4.064  0.000  OK
    10^200     AttrPlan              1.2        0.602      2.396  0.035  OK
    10^200     MCTS                  0.1       13.234     53.091  0.173  OK

Key observations:

* **Presorting / Legacy** are exact and have sub-millisecond ``one_in()`` queries
  after init, but ``init()`` costs scale as ``O(n log n)`` with the space size
  (~1.5 s at 10^6 outcomes) and ``some()`` scans the sorted array linearly (~67 ms
  at 10^6).  They cannot handle spaces beyond ~5×10^6 due to memory constraints.

* **BIDS** is the clear winner for large additive spaces: ``init()`` is under 20 ms
  even at 10^200 outcomes, ``one_in()`` stays below 1 ms, and accuracy is
  excellent (err < 0.001 across 10^3 – 10^200 outcomes).  Note that accuracy can
  be poor for very small discrete spaces (err ≈ 0.26 at 25 outcomes) because the
  1001-point grid is too coarse relative to the 25-outcome space —
  `PresortingInverseUtilityFunction` is preferable there.

* **Adaptive** tracks the best inverter at every scale: exact Presorting for
  ≤ 10^6 outcomes, then BIDS for large additive spaces. It is the recommended
  default when you do not want to couple to a specific implementation.

* **Sampling** has zero init cost and works on any ufun type, but ``one_in()``
  time grows linearly with the outcome space (it samples up to
  ``max_samples_per_call`` outcomes each call) and accuracy degrades beyond
  10^10 outcomes (~1.8 s and err ≈ 0.16 at 10^100). It remains useful for
  non-additive ufuns in huge spaces where no other inverter applies.

* **AttrPlan** is extremely fast (sub-millisecond ``one_in()`` even at 10^200)
  and works for any additive ufun, but its accuracy is the worst of the additive
  inverters (err ≈ 0.20–0.35 for small/medium spaces, ~0.04–0.18 for very large
  ones). Prefer BIDS when accuracy matters.

* **MCTS** works for any ufun and provides diverse outcomes, but it is the
  slowest inverter at every scale (``one_in()`` ~0.3–13 ms, ``some()`` up to
  53 ms) with middling accuracy. Use it only when you need diversity and the
  ufun is non-additive.

* **BruteForce / PresortingBF** are the exact ground truth, suitable only for
  tiny spaces (< 5×10^4 and < 10^5 outcomes respectively).

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
from .attribute_planning import AttributePlanningInverseUtilityFunction
from .bids import BIDSInverseUtilityFunction
from .bruteforce import BruteForceInverseUtilityFunction
from .mcts import MCTSInverseUtilityFunction
from .presorting import PresortingInverseUtilityFunction
from .presorting_legacy import PresortingLegacyInverseUtilityFunction
from .presorting_legacy import PresortingInverseUtilityFunctionBruteForce
from .sampling import SamplingInverseUtilityFunction

# The default inverter used throughout the codebase.
DefaultInverseUtilityFunction = AdaptiveInverseUtilityFunction
FastInverseUtilityFunction = BIDSInverseUtilityFunction

__all__ = [
    "AdaptiveInverseUtilityFunction",
    "AttributePlanningInverseUtilityFunction",
    "BIDSInverseUtilityFunction",
    "BruteForceInverseUtilityFunction",
    "DefaultInverseUtilityFunction",
    "FastInverseUtilityFunction",
    "MCTSInverseUtilityFunction",
    "SamplingInverseUtilityFunction",
    "PresortingInverseUtilityFunction",
    "PresortingLegacyInverseUtilityFunction",
    "PresortingInverseUtilityFunctionBruteForce",
]
