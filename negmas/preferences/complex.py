from __future__ import annotations

import random
from typing import Any, Callable, Dict, Iterable, List, Optional, Type

from negmas.common import NegotiatorMechanismInterface
from negmas.helpers import get_full_type_name, make_range
from negmas.outcomes import Issue, Outcome
from negmas.serialization import PYTHON_CLASS_IDENTIFIER, deserialize, serialize

from .base import UtilityValue
from .base_crisp import UtilityFunction
from .static import StaticPreferences

__all__ = [
    "ComplexWeightedUtilityFunction",
    "ComplexNonlinearUtilityFunction",
]


class ComplexWeightedUtilityFunction(StaticPreferences, UtilityFunction):
    """A utility function composed of linear aggregation of other utility functions

    Args:
        ufuns: An iterable of utility functions
        weights: Weights used for combination. If not given all weights are assumed to equal 1.
        name: Utility function name

    """

    def __init__(
        self,
        ufuns: Iterable[UtilityFunction],
        weights: Optional[Iterable[float]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ufuns = list(ufuns)
        if weights is None:
            weights = [1.0] * len(self.ufuns)
        self.weights = list(weights)

    def is_dymanic(self) -> bool:
        return any(_.is_dymanic() for _ in self.ufuns)

    @UtilityFunction.ami.setter
    def ami(self, value):
        UtilityFunction.ami.fset(self, value)
        for ufun in self.ufuns:
            if hasattr(ufun, "ami"):
                ufun.ami = value

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
        u = float(0.0)
        failure = False
        for f, w in zip(self.ufuns, self.weights):
            util = f(offer)
            if util is not None:
                u += w * util
            else:
                failure = True
        return u if not failure else None

    def xml(self, issues: List[Issue]) -> str:
        output = ""
        # @todo implement weights. Here I assume they are always 1.0
        for f, _ in zip(self.ufuns, self.weights):
            this_output = f.xml(issues)
            if this_output:
                output += this_output
            else:
                output += str(vars(f))
        return output

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ufuns": [serialize(_) for _ in self.ufuns],
            "weights": self.weights,
            "id": self.id,
            "name": self.name,
            "reserved_value": self.reserved_value,
            PYTHON_CLASS_IDENTIFIER: get_full_type_name(type(self)),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]):
        d.pop(PYTHON_CLASS_IDENTIFIER, None)
        d["ufuns"] = [deserialize(_) for _ in d["ufuns"]]
        return cls(**d)


class ComplexNonlinearUtilityFunction(UtilityFunction):
    """A utility function composed of nonlinear aggregation of other utility functions

    Args:
        ufuns: An iterable of utility functions
        combination_function: The function used to combine results of ufuns
        name: Utility function name

    """

    def __init__(
        self,
        ufuns: Iterable[UtilityFunction],
        combination_function=Callable[[Iterable[UtilityValue]], UtilityValue],
        name=None,
        reserved_value: UtilityValue = float("-inf"),
        ami: NegotiatorMechanismInterface = None,
        id: str = None,
    ):
        super().__init__(
            name=name,
            reserved_value=reserved_value,
            ami=ami,
            id=id,
        )
        self.ufuns = list(ufuns)
        self.combination_function = combination_function

    def is_dymanic(self) -> bool:
        return any(_.is_dymanic() for _ in self.ufuns)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ufuns": [serialize(_) for _ in self.ufuns],
            "combination_function": serialize(self.combination_function),
            "id": self.id,
            "name": self.name,
            "reserved_value": self.reserved_value,
            PYTHON_CLASS_IDENTIFIER: get_full_type_name(type(self)),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]):
        d.pop(PYTHON_CLASS_IDENTIFIER, None)
        d["ufuns"] = [deserialize(_) for _ in d["ufuns"]]
        d["combination_function"] = deserialize(d["combination_function"])
        return cls(**d)

    @UtilityFunction.ami.setter
    def ami(self, value):
        UtilityFunction.ami.fset(self, value)
        for ufun in self.ufuns:
            if hasattr(ufun, "ami"):
                ufun.ami = value

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

    def xml(self, issues: List[Issue]) -> str:
        raise NotImplementedError(f"Cannot convert {self.__class__.__name__} to xml")
