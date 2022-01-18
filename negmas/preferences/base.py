from __future__ import annotations

from typing import Callable, Mapping, Union

from negmas.helpers import Distribution
from negmas.outcomes.common import Outcome

__all__ = ["INVALID_UTILITY", "Distribution", "UtilityValue"]

INVALID_UTILITY = float("-inf")


UtilityValue = Union[Distribution, float]
"""
Either a probability distribution or an exact value.
"""

OutcomeUtilityMapping = Union[
    Callable[[Outcome], UtilityValue], Mapping[Outcome, UtilityValue]
]
"""A mapping from an outcome to its utility value"""
