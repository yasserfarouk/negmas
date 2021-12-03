from __future__ import annotations

from typing import TYPE_CHECKING, List

import numpy as np

from .mapping import MappingUtilityFunction

if TYPE_CHECKING:
    from negmas.outcomes import Outcome

__all__ = ["RandomUtilityFunction"]


class RandomUtilityFunction(MappingUtilityFunction):
    """A random utility function for a discrete outcome space"""

    def __init__(
        self,
        outcomes: List[Outcome],
        reserved_value=float("-inf"),
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
            **kwargs,
        )
