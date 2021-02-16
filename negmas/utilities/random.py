from typing import List, Optional, Type
import numpy as np
from negmas.outcomes import Outcome
from negmas.utilities.nonlinear import MappingUtilityFunction

__all__ = ["RandomUtilityFunction"]


class RandomUtilityFunction(MappingUtilityFunction):
    """A random utility function for a discrete outcome space"""

    def __init__(
        self,
        outcomes: List[Outcome],
        reserved_value=float("-inf"),
        outcome_type: Optional[Type] = None,
        **kwargs,
    ):
        if len(outcomes) < 1:
            raise ValueError("Cannot create a random utility function without outcomes")
        if isinstance(outcomes[0], tuple):
            pass
        else:
            outcomes = [tuple(o.values()) for o in outcomes]
        super().__init__(
            mapping=dict(zip(outcomes, np.random.rand(len(outcomes)))),
            reserved_value=reserved_value,
            outcome_type=outcome_type,
            **kwargs,
        )
