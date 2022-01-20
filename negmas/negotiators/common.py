from __future__ import annotations

from collections import namedtuple

__all__ = ["NegotiatorInfo"]

NegotiatorInfo = namedtuple("NegotiatorInfo", ["negotiator", "context"])
"""The return type of `negotiators` member of `Controller`."""
