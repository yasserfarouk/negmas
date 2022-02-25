from __future__ import annotations

import random
from typing import Any, Callable, Iterable

from negmas.helpers import get_full_type_name
from negmas.helpers.numeric import get_one_int
from negmas.outcomes import Outcome
from negmas.serialization import PYTHON_CLASS_IDENTIFIER, deserialize, serialize

from .base import Value
from .base_ufun import BaseUtilityFunction
from .crisp.linear import LinearAdditiveUtilityFunction

__all__ = ["WeightedUtilityFunction", "ComplexNonlinearUtilityFunction"]


class _DependenceMixin:
    """Used to set dependence properties based on the object's `values` ."""

    def is_session_dependent(self):  # type: ignore
        return any(_.is_session_dependent() for _ in self.values)  # type: ignore

    def is_stationary(self) -> bool:  # type: ignore
        return any(_.is_stationary() for _ in self.values)  # type: ignore

    def is_volatile(self) -> bool:  # type: ignore
        return any(_.is_volatile() for _ in self.values)  # type: ignore

    def is_state_dependent(self) -> bool:  # type: ignore
        return any(_.is_state_dependent() for _ in self.values)  # type: ignore


class WeightedUtilityFunction(_DependenceMixin, BaseUtilityFunction):
    """A utility function composed of linear aggregation of other utility functions

    Args:
        ufuns: An iterable of utility functions
        weights: Weights used for combination. If not given all weights are assumed to equal 1.
        name: Utility function name

    """

    def __init__(
        self,
        ufuns: Iterable[BaseUtilityFunction],
        weights: Iterable[float] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.values: list[BaseUtilityFunction] = list(ufuns)
        if weights is None:
            weights = [1.0] * len(self.values)
        self.weights = list(weights)

    def to_stationary(self):
        return WeightedUtilityFunction(
            ufuns=[_.to_stationary() for _ in self.values],
            weights=self.weights,
            name=self.name,
            id=self.id,
        )

    @classmethod
    def random(
        cls,
        outcome_space,
        reserved_value,
        normalized=True,
        n_ufuns=(1, 4),
        ufun_types=(LinearAdditiveUtilityFunction,),
        **kwargs,
    ) -> WeightedUtilityFunction:
        """Generates a random ufun of the given type"""
        n = get_one_int(n_ufuns)
        ufuns = [
            random.choice(ufun_types).random(outcome_space, 0, normalized)
            for _ in range(n)
        ]
        weights = [random.random() for _ in range(n)]
        return WeightedUtilityFunction(
            reserved_value=reserved_value,
            ufuns=ufuns,
            weights=weights,
            outcome_space=outcome_space,
            **kwargs,
        )

    def eval(self, offer: Outcome) -> Value:
        """Calculate the utility_function value for a given outcome.

        Args:
            offer: The offer to be evaluated.


        Remarks:
            - You cannot return None from overriden apply() functions but raise an exception (ValueError) if it was
              not possible to calculate the Value.
            - Return A Value not a float for real-valued utilities for the benefit of inspection code.

        Returns:
            Value: The utility_function value which may be a distribution. If `None` it means the utility_function value cannot be
            calculated.
        """
        if offer is None:
            return self.reserved_value
        u = float(0.0)
        for f, w in zip(self.values, self.weights):
            util = f(offer)
            if util is None or w is None:
                raise ValueError(
                    f"Cannot calculate utility for {offer}\n\t UFun {str(f)}\n\t with vars\n{vars(f)}"
                )
            u += util * w  # type: ignore
        return u

    def to_dict(self) -> dict[str, Any]:
        d = {PYTHON_CLASS_IDENTIFIER: get_full_type_name(type(self))}
        d.update(super().to_dict())
        return dict(
            **d,
            ufuns=[serialize(_) for _ in self.values],
            weights=self.weights,
        )

    @classmethod
    def from_dict(cls, d: dict[str, Any]):
        d.pop(PYTHON_CLASS_IDENTIFIER, None)
        d["ufuns"] = [deserialize(_) for _ in d["ufuns"]]
        return cls(**d)


class ComplexNonlinearUtilityFunction(_DependenceMixin, BaseUtilityFunction):
    """A utility function composed of nonlinear aggregation of other utility functions

    Args:
        ufuns: An iterable of utility functions
        combination_function: The function used to combine results of ufuns
        name: Utility function name

    """

    def __init__(
        self,
        ufuns: Iterable[BaseUtilityFunction],
        combination_function=Callable[[Iterable[Value]], Value],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ufuns = list(ufuns)
        self.combination_function = combination_function

    def to_stationary(self):
        return ComplexNonlinearUtilityFunction(
            ufuns=[_.to_stationary() for _ in self.ufuns],
            combination_function=self.combination_function,
            name=self.name,
            id=self.id,
        )

    @classmethod
    def random(
        cls,
        outcome_space,
        reserved_value,
        normalized=True,
        n_ufuns=(1, 4),
        ufun_types=(LinearAdditiveUtilityFunction,),
        **kwargs,
    ) -> ComplexNonlinearUtilityFunction:
        """Generates a random ufun of the given type"""
        n = get_one_int(n_ufuns)
        ufuns = [
            random.choice(ufun_types).random(outcome_space, 0, normalized)
            for _ in range(n)
        ]
        weights = [random.random() for _ in range(n)]
        return ComplexNonlinearUtilityFunction(
            reserved_value=reserved_value,
            ufuns=ufuns,
            combination_function=lambda vals: sum(w * v for w, v in zip(weights, vals)),
            outcome_space=outcome_space,
            **kwargs,
        )

    def to_dict(self) -> dict[str, Any]:
        d = {PYTHON_CLASS_IDENTIFIER: get_full_type_name(type(self))}
        d.update(super().to_dict())
        return dict(
            ufuns=serialize(self.ufuns),
            combination_function=serialize(self.combination_function),
        )

    @classmethod
    def from_dict(cls, d: dict[str, Any]):
        d.pop(PYTHON_CLASS_IDENTIFIER, None)
        d["ufuns"] = deserialize(d["ufuns"])
        d["combination_function"] = deserialize(d["combination_function"])
        return cls(**d)

    def eval(self, offer: Outcome) -> Value:
        """Calculate the utility_function value for a given outcome.

        Args:
            offer: The offer to be evaluated.


        Remarks:
            - You cannot return None from overriden apply() functions but raise an exception (ValueError) if it was
              not possible to calculate the Value.
            - Return A Value not a float for real-valued utilities for the benefit of inspection code.

        Returns:
            Value: The utility_function value which may be a distribution. If `None` it means the utility_function value cannot be
            calculated.
        """
        if offer is None:
            return self.reserved_value
        return self.combination_function([f(offer) for f in self.ufuns])
