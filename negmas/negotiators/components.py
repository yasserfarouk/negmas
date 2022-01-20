from __future__ import annotations

import math
from abc import abstractmethod
from typing import Protocol, Union

__all__ = [
    "Aspiration",
    "PolyAspiration",
]


class Aspiration(Protocol):
    @abstractmethod
    def aspiration(self, t: float) -> float:
        pass


class PolyAspiration(Aspiration):
    """Adds aspiration level calculation. This Mixin MUST be used with a `Negotiator` class."""

    def __init__(
        self,
        max_aspiration: float,
        aspiration_type: Union[str, int, float],
        above_reserved_value=True,
    ):
        """

        Args:
            max_aspiration: The aspiration level to start from (usually 1.0)
            aspiration_type: The aspiration type. Can be a string ("boulware", "linear", "conceder") or a number giving the exponent of the aspiration curve.
            above_reserved_value: If False, the lowest value for the aspiration curve will be set to zero instead of the reserved_value
        """
        self.max_aspiration = max_aspiration
        self.aspiration_type = aspiration_type
        self.exponent = 1.0
        if isinstance(aspiration_type, int):
            self.exponent = float(aspiration_type)
        elif isinstance(aspiration_type, float):
            self.exponent = aspiration_type
        elif aspiration_type == "boulware":
            self.exponent = 4.0
        elif aspiration_type == "linear":
            self.exponent = 1.0
        elif aspiration_type == "conceder":
            self.exponent = 0.25
        else:
            raise ValueError(f"Unknown aspiration type {aspiration_type}")
        self.above_reserved = above_reserved_value

    def aspiration(self, t: float) -> float:
        """
        The aspiration level

        Args:
            t: relative time (a number between zero and one)

        Returns:
            aspiration level
        """
        if t is None:
            raise ValueError(
                f"Aspiration negotiators cannot be used in negotiations with no time or #steps limit!!"
            )
        return self.max_aspiration * (1.0 - math.pow(t, self.exponent))
