"""Pareto sampler components for trade-off queries in automated negotiation.

A ``ParetoSampler`` wraps both the agent's own utility function and an estimate of
the opponent's utility function to answer *trade-off queries*: finding outcomes that
are approximately Pareto-optimal with respect to both parties' utilities.

This contrasts with ``InverseUFun`` (which only requires the agent's own utility
function) and implements the third class of search queries identified in:

    Koça, T., de Jonge, D., & Baarslag, T. (2024).
    *Search algorithms for automated negotiation in large domains.*
    Algorithms 17(5), 200.

The opponent's utility function is accessed via ``Negotiator.opponent_ufun``,
which returns ``None`` when no estimate is available (e.g. early in a negotiation).
Implementations should handle ``None`` gracefully.

The ``ParetoSampler`` protocol itself lives in ``negmas.preferences.protocols``
alongside ``InverseUFun``.  It is re-exported here for convenience.

By default, ``BaseUtilityFunction.make_pareto_sampler()`` uses
``IPSParetoSampler`` unless another implementation is explicitly requested.

Available implementations
--------------------------
- ``IPSParetoSampler``: IPS (Iterative Pareto Search), exploits the additive
  structure of both ufuns to iteratively build an approximate Pareto frontier.
  Requires both ufuns to be ``LinearAdditiveUtilityFunction`` instances.
- ``NB3ParetoSampler``: Branch-and-Bound Pareto sampler (De Jonge & Sierra 2017).
  Anytime algorithm with per-agent upper-bound pruning.
- ``MOBANOSParetoSampler``: Exact iterative Pareto construction (De Jonge et al.
  2018/2019).  No rounding — more accurate than IPS but exponential in worst case.
- ``BruteForceParetoSampler``: Exact Pareto frontier via full enumeration and
  ``negmas.preferences.ops.pareto_frontier``. Ground truth on tiny spaces only.
"""

from __future__ import annotations

from negmas.preferences.protocols import ParetoSampler

from .bruteforce import BruteForceParetoSampler
from .ips import IPSParetoSampler
from .mobanos import MOBANOSParetoSampler
from .nb3 import NB3ParetoSampler

__all__ = [
    "ParetoSampler",
    "BruteForceParetoSampler",
    "IPSParetoSampler",
    "NB3ParetoSampler",
    "MOBANOSParetoSampler",
]
