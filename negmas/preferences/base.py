from typing import Callable, Mapping, Union

from negmas.helpers import DistributionLike
from negmas.outcomes import Outcome

__all__ = ["INVALID_UTILITY", "DistributionLike", "UtilityValue"]

INVALID_UTILITY = float("-inf")


UtilityValue = Union[DistributionLike, float]
"""
Either a probability distribution or an exact value.
"""

OutcomeUtilityMapping = Union[
    Callable[[Outcome], UtilityValue], Mapping[Outcome, UtilityValue]
]
"""A mapping from an outcome to its utility value"""
