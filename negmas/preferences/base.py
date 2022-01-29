from __future__ import annotations

from typing import Callable, Mapping, Union

from negmas.common import Value
from negmas.helpers.prob import Distribution
from negmas.outcomes.common import Outcome

__all__ = ["INVALID_UTILITY", "Distribution", "Value", "UtilityValue"]

INVALID_UTILITY = float("-inf")

UtilityValue = Value
"""
An alias for `Value` which is either a probability `Distribution` or a `float`
"""

OutcomeUtilityMapping = Union[Callable[[Outcome], Value], Mapping[Outcome, Value]]
"""A mapping from an outcome to its utility value"""
