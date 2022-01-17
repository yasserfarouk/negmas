from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Optional

from negmas.helpers import get_full_type_name
from negmas.outcomes import Outcome
from negmas.preferences.protocols import IndIssues, StationaryCrisp
from negmas.serialization import PYTHON_CLASS_IDENTIFIER, deserialize, serialize

from .base import UtilityValue
from .ufun import BaseUtilityFunction, UtilityFunction

__all__ = ["ComplexWeightedUtilityFunction", "ComplexNonlinearUtilityFunction"]


class ComplexWeightedUtilityFunction(UtilityFunction, IndIssues, StationaryCrisp):
    """A utility function composed of linear aggregation of other utility functions

    Args:
        ufuns: An iterable of utility functions
        weights: Weights used for combination. If not given all weights are assumed to equal 1.
        name: Utility function name

    """

    def __init__(
        self,
        ufuns: Iterable[BaseUtilityFunction],
        weights: Optional[Iterable[float]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.values: list[BaseUtilityFunction] = list(ufuns)
        if weights is None:
            weights = [1.0] * len(self.values)
        self.weights = list(weights)

    def is_stationary(self) -> bool:
        return any(_.is_stationary() for _ in self.values)

    def eval(self, offer: "Outcome") -> float:
        """Calculate the utility_function value for a given outcome.

        Args:
            offer: The offer to be evaluated.


        Remarks:
            - You cannot return None from overriden apply() functions but raise an exception (ValueError) if it was
              not possible to calculate the UtilityValue.
            - Return A UtilityValue not a float for real-valued utilities for the benefit of inspection code.

        Returns:
            UtilityValue: The utility_function value which may be a distribution. If `None` it means the utility_function value cannot be
            calculated.
        """
        if offer is None:
            return self.reserved_value
        u = float(0.0)
        for f, w in zip(self.values, self.weights):
            util = f(offer)
            if util is not None:
                u += util * w
            else:
                raise ValueError(f"Cannot calculate ufility for {offer}")
        return u

    def to_dict(self) -> Dict[str, Any]:
        d = {PYTHON_CLASS_IDENTIFIER: get_full_type_name(type(self))}
        d.update(super().to_dict())
        return dict(
            **d,
            ufuns=[serialize(_) for _ in self.values],
            weights=self.weights,
        )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]):
        d.pop(PYTHON_CLASS_IDENTIFIER, None)
        d["ufuns"] = [deserialize(_) for _ in d["ufuns"]]
        return cls(**d)


class ComplexNonlinearUtilityFunction(UtilityFunction, StationaryCrisp):
    """A utility function composed of nonlinear aggregation of other utility functions

    Args:
        ufuns: An iterable of utility functions
        combination_function: The function used to combine results of ufuns
        name: Utility function name

    """

    def __init__(
        self,
        ufuns: Iterable[BaseUtilityFunction],
        combination_function=Callable[[Iterable[UtilityValue]], UtilityValue],
        name=None,
        reserved_value: float = float("-inf"),
        id: str = None,
    ):
        super().__init__(
            name=name,
            reserved_value=reserved_value,
            id=id,
        )
        self.ufuns = list(ufuns)
        self.combination_function = combination_function

    def is_stationary(self) -> bool:
        return any(_.is_stationary() for _ in self.ufuns)

    def to_dict(self) -> Dict[str, Any]:
        d = {PYTHON_CLASS_IDENTIFIER: get_full_type_name(type(self))}
        d.update(super().to_dict())
        return dict(
            ufuns=serialize(self.ufuns),
            combination_function=serialize(self.combination_function),
        )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]):
        d.pop(PYTHON_CLASS_IDENTIFIER, None)
        d["ufuns"] = deserialize(d["ufuns"])
        d["combination_function"] = deserialize(d["combination_function"])
        return cls(**d)

    def eval(self, offer: "Outcome") -> UtilityValue:
        """Calculate the utility_function value for a given outcome.

        Args:
            offer: The offer to be evaluated.


        Remarks:
            - You cannot return None from overriden apply() functions but raise an exception (ValueError) if it was
              not possible to calculate the UtilityValue.
            - Return A UtilityValue not a float for real-valued utilities for the benefit of inspection code.

        Returns:
            UtilityValue: The utility_function value which may be a distribution. If `None` it means the utility_function value cannot be
            calculated.
        """
        if offer is None:
            return self.reserved_value
        return self.combination_function([f(offer) for f in self.ufuns])
