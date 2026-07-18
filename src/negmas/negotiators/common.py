"""Common data structures and utilities."""

from __future__ import annotations

from typing import Any, NamedTuple

__all__ = ["NegotiatorEntry"]


class NegotiatorEntry(NamedTuple):
    """Pairs an active negotiator with its negotiation context.

    The return type of the ``negotiators`` member of `Controller`.
    """

    negotiator: Any
    context: Any
