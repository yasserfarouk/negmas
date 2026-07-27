"""Time-based aspiration and utility curve helper classes for negotiators."""

from __future__ import annotations

import math
from abc import abstractmethod
from typing import Literal, Protocol, runtime_checkable

__all__ = [
    "TimeCurve",
    "Aspiration",
    "PolyAspiration",
    "ExpAspiration",
    "POLY_ASPIRATION_EXPONENTS",
    "EXP_ASPIRATION_EXPONENTS",
]


@runtime_checkable
class TimeCurve(Protocol):
    """
    Models a time-curve mapping relative timge (going from 0.0 to 1.0) to a utility range to use
    """

    @abstractmethod
    def utility_range(self, t: float) -> tuple[float, float]:
        """Utility range.

        Args:
            t: T.

        Returns:
            tuple[float, float]: The result.
        """
        raise NotImplementedError(
            "utility_range is not implemented in a TimeCurve class"
        )


@runtime_checkable
class Aspiration(TimeCurve, Protocol):
    """
    A monotonically decreasing time-curve
    """

    @abstractmethod
    def utility_at(self, t: float) -> float:
        """Utility at.

        Args:
            t: T.

        Returns:
            float: The result.
        """
        raise NotImplementedError("utility_at is not implemented in a TimeCurve class")

    def utility_range(self, t: float) -> tuple[float, float]:
        """Utility range.

        Args:
            t: T.

        Returns:
            tuple[float, float]: The result.
        """
        return self.utility_at(t), 1.0


EXP_ASPIRATION_EXPONENTS: dict[str, float] = {
    "boulware": 0.125,
    "linear": 0.725,
    "conceder": 4.0,
}
"""Default exponent for each named `ExpAspiration` curve type."""

POLY_ASPIRATION_EXPONENTS: dict[str, float] = {
    "boulware": 4.0,
    "linear": 1.0,
    "conceder": 0.25,
    "hardheaded": float("inf"),
}
"""Default exponent for each named `PolyAspiration` curve type."""


class ExpAspiration(Aspiration):
    """
    An exponential conceding curve

    Args:
        max_aspiration: The aspiration level to start from (usually 1.0)
        aspiration_type: The aspiration type. Can be a string ("boulware", "linear", "conceder") or a number giving the exponent of the aspiration curve.
        exponents: Maps each named ``aspiration_type`` to its exponent. Defaults
            to `EXP_ASPIRATION_EXPONENTS`. Pass a modified mapping (e.g.
            ``{"boulware": 0.2, ...}``) to tune what the names mean.
    """

    def __init__(
        self,
        max_aspiration: float,
        aspiration_type: Literal["boulware"]
        | Literal["conceder"]
        | Literal["linear"]
        | float,
        exponents: dict[str, float] | None = None,
    ):
        """Initialize the instance.

        Args:
            max_aspiration: Max aspiration.
            aspiration_type: Aspiration type.
            exponents: Named-curve to exponent mapping (see class docstring).
        """
        self.max_aspiration = max_aspiration
        self.aspiration_type = aspiration_type
        self.exponents = (
            dict(EXP_ASPIRATION_EXPONENTS) if exponents is None else dict(exponents)
        )
        self.exponent = 1.0
        if isinstance(aspiration_type, int):
            self.exponent = float(aspiration_type)
        elif isinstance(aspiration_type, float):
            self.exponent = aspiration_type
        elif aspiration_type in self.exponents:
            self.exponent = self.exponents[aspiration_type]
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
                "Aspiration negotiators cannot be used in negotiations with no time or #steps limit!!"
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
        aspiration_type: The aspiration type. Can be a string ("boulware", "linear", "conceder", "hardheaded") or a number giving the exponent of the aspiration curve.
        exponents: Maps each named ``aspiration_type`` to its exponent. Defaults
            to `POLY_ASPIRATION_EXPONENTS`. Pass a modified mapping (e.g.
            ``{"boulware": 3.0, ...}``) to tune what the names mean.
    """

    def __init__(
        self,
        max_aspiration: float,
        aspiration_type: Literal["boulware"]
        | Literal["conceder"]
        | Literal["linear"]
        | Literal["hardheaded"]
        | float,
        exponents: dict[str, float] | None = None,
    ):
        """Initialize the instance.

        Args:
            max_aspiration: Max aspiration.
            aspiration_type: Aspiration type.
            exponents: Named-curve to exponent mapping (see class docstring).
        """
        self.max_aspiration = max_aspiration
        self.aspiration_type = aspiration_type
        self.exponents = (
            dict(POLY_ASPIRATION_EXPONENTS) if exponents is None else dict(exponents)
        )
        self.exponent = 1.0
        if isinstance(aspiration_type, int):
            self.exponent = float(aspiration_type)
        elif isinstance(aspiration_type, float):
            self.exponent = aspiration_type
        elif aspiration_type in self.exponents:
            self.exponent = self.exponents[aspiration_type]
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
                "Aspiration negotiators cannot be used in negotiations with no time or #steps limit!!"
            )
        return self.max_aspiration * (1.0 - math.pow(t, self.exponent))
