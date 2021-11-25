"""
Common datastructures used in the outcomes module.
"""
from __future__ import annotations

from typing import Any, Mapping

__all__ = [
    "Outcome",
    "PartialOutcome",
    "OutcomeRange",
]

Outcome = tuple
"""An outcome is a tuple of issue values."""

PartialOutcome = Mapping[int, Any]
"""A partial outcome is a simple mapping between issue INDEX and its value. Both a `tuple` and a `dict[int, Any]` satisfy this definition."""

OutcomeRange = Mapping[int, Any]
"""An outcome range is a a mapping between issue INDEX and either a value, a list of values or a Tuple with two values"""
