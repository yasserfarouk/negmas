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

Available implementations
--------------------------
- ``IPSParetoSampler``: IPS (Iterative Pareto Search), exploits the additive
  structure of both ufuns to iteratively build an approximate Pareto frontier.
  Requires both ufuns to be ``LinearAdditiveUtilityFunction`` instances.
"""

from __future__ import annotations

from ._protocol import ParetoSampler
from .ips import IPSParetoSampler

__all__ = [
    "ParetoSampler",
    "IPSParetoSampler",
]
