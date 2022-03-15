from __future__ import annotations

import math
from abc import abstractmethod
from typing import Literal, Protocol, runtime_checkable

__all__ = [
    "TimeCurve",
    "Aspiration",
    "PolyAspiration",
    "ExpAspiration",
]


@runtime_checkable
class TimeCurve(Protocol):
    """
    Models a time-curve mapping relative timge (going from 0.0 to 1.0) to a utility range to use
    """

    @abstractmethod
    def utility_range(self, t: float) -> tuple[float, float]:
        pass


@runtime_checkable
class Aspiration(TimeCurve, Protocol):
    """
    A monotonically decreasing time-curve
    """

    @abstractmethod
    def utility_at(self, t: float) -> float:
        pass

    def utility_range(self, t: float) -> tuple[float, float]:
        return self.utility_at(t), 1.0


class ExpAspiration(Aspiration):
    """
    An exponential conceding curve

    Args:
        max_aspiration: The aspiration level to start from (usually 1.0)
        aspiration_type: The aspiration type. Can be a string ("boulware", "linear", "conceder") or a number giving the exponent of the aspiration curve.
    """

    def __init__(
        self,
        max_aspiration: float,
        aspiration_type: Literal["boulware"]
        | Literal["conceder"]
        | Literal["linear"]
        | float,
    ):
        self.max_aspiration = max_aspiration
        self.aspiration_type = aspiration_type
        self.exponent = 1.0
        if isinstance(aspiration_type, int):
            self.exponent = float(aspiration_type)
        elif isinstance(aspiration_type, float):
            self.exponent = aspiration_type
        elif aspiration_type == "boulware":
            self.exponent = 0.125
        elif aspiration_type == "linear":
            self.exponent = 0.725
        elif aspiration_type == "conceder":
            self.exponent = 4.0
        else:
            raise ValueError(f"Unknown aspiration type {aspiration_type}")
        self._denominator = math.exp(1) - 1

    def utility_at(self, t: float) -> float:
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
        return (
            self.max_aspiration
            * (math.exp(math.pow(1 - t, self.exponent)) - 1)
            / self._denominator
        )


class PolyAspiration(Aspiration):
    """
    A polynomially conceding curve

    Args:
        max_aspiration: The aspiration level to start from (usually 1.0)
        aspiration_type: The aspiration type. Can be a string ("boulware", "linear", "conceder") or a number giving the exponent of the aspiration curve.
    """

    def __init__(
        self,
        max_aspiration: float,
        aspiration_type: Literal["boulware"]
        | Literal["conceder"]
        | Literal["linear"]
        | Literal["hardheaded"]
        | float,
    ):
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
        elif aspiration_type == "hardheaded":
            self.exponent = float("inf")
        else:
            raise ValueError(f"Unknown aspiration type {aspiration_type}")

    def utility_at(self, t: float) -> float:
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
