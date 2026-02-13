"""Composite utility functions that combine multiple utility functions."""

from __future__ import annotations

import functools
import random
from typing import Any, Callable, Iterable

from negmas.helpers import get_full_type_name
from negmas.helpers.numeric import get_one_int
from negmas.outcomes import Outcome
from negmas.serialization import PYTHON_CLASS_IDENTIFIER, deserialize, serialize

from .base import Value
from .base_ufun import BaseUtilityFunction
from .crisp.linear import LinearAdditiveUtilityFunction
from .stability import STABLE_DIFF_RATIOS, STABLE_ORDERING, STATIONARY, Stability

__all__ = ["WeightedUtilityFunction", "ComplexNonlinearUtilityFunction"]


def _and_stability(ufuns: Iterable[BaseUtilityFunction]) -> Stability:
    """Compute combined stability by ANDing stability of all ufuns.

    When combining utility functions, the combined stability can only be
    as strong as the weakest link. For example, if one ufun has volatile
    ordering, the combined ufun also has volatile ordering.

    Args:
        ufuns: Iterable of utility functions to combine.

    Returns:
        Combined stability flags (intersection of all stability flags).
    """
    return functools.reduce(lambda a, b: Stability(a & b.stability), ufuns, STATIONARY)


class WeightedUtilityFunction(BaseUtilityFunction):
    """A utility function composed of linear aggregation of other utility functions.

    This combines multiple utility functions using weighted sum:
        u(o) = sum(w_i * u_i(o))

    where w_i are the weights and u_i are the component utility functions.

    Stability Properties:
        - Inherits stability by ANDing all component ufuns' stability
        - Additionally preserves STABLE_ORDERING and STABLE_DIFF_RATIOS
          (weighted sum preserves relative ordering and ratios)

    Args:
        ufuns: An iterable of utility functions
        weights: Weights used for combination. If not given all weights are assumed to equal 1.
        name: Utility function name

    Example:
        >>> from negmas.preferences import LinearUtilityFunction
        >>> from negmas.outcomes import make_issue
        >>> issues = [make_issue(5)]
        >>> u1 = LinearUtilityFunction(weights=[1.0], issues=issues)
        >>> u2 = LinearUtilityFunction(weights=[0.5], issues=issues)
        >>> combined = WeightedUtilityFunction(ufuns=[u1, u2], weights=[0.6, 0.4])
        >>> combined.is_stationary()
        True
    """

    def __init__(
        self,
        ufuns: Iterable[BaseUtilityFunction],
        weights: Iterable[float] | None = None,
        **kwargs,
    ):
        """Initialize the instance.

        Args:
            ufuns: Collection of component utility functions to combine.
            weights: Coefficients for the weighted sum (defaults to equal weights if not provided).
            **kwargs: Additional keyword arguments passed to the base class.
        """
        self.values: list[BaseUtilityFunction] = list(ufuns)
        if weights is None:
            weights = [1.0] * len(self.values)
        self.weights = list(weights)

        # Compute stability: AND of all component ufuns + linear preserves ordering/diff_ratios
        combined_stability = _and_stability(self.values)
        # Linear aggregation preserves ordering and diff ratios
        combined_stability |= STABLE_ORDERING | STABLE_DIFF_RATIOS
        # But we can only claim these if ALL components have them
        combined_stability &= _and_stability(self.values)

        # Allow user to override if needed, otherwise use computed
        if "stability" not in kwargs:
            kwargs["stability"] = combined_stability

        super().__init__(**kwargs)

    def to_stationary(self):
        """Returns a stationary version with all component utility functions converted to stationary."""
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

    def to_dict(
        self, python_class_identifier=PYTHON_CLASS_IDENTIFIER
    ) -> dict[str, Any]:
        """To dict.

        Args:
            python_class_identifier: Python class identifier.

        Returns:
            dict[str, Any]: The result.
        """
        d = {python_class_identifier: get_full_type_name(type(self))}
        d.update(super().to_dict(python_class_identifier=python_class_identifier))
        return dict(
            **d,
            ufuns=[
                serialize(_, python_class_identifier=python_class_identifier)
                for _ in self.values
            ],
            weights=self.weights,
        )

    @classmethod
    def from_dict(
        cls, d: dict[str, Any], python_class_identifier=PYTHON_CLASS_IDENTIFIER
    ):
        """From dict.

        Args:
            d: D.
            python_class_identifier: Python class identifier.
        """
        d.pop(python_class_identifier, None)
        d["ufuns"] = [
            deserialize(_, python_class_identifier=python_class_identifier)
            for _ in d["ufuns"]
        ]
        return cls(**d)


class ComplexNonlinearUtilityFunction(BaseUtilityFunction):
    """A utility function composed of nonlinear aggregation of other utility functions.

    This combines multiple utility functions using an arbitrary combination function:
        u(o) = f(u_1(o), u_2(o), ..., u_n(o))

    where f is the combination_function.

    Stability Properties:
        - Inherits stability by ANDing all component ufuns' stability
        - Does NOT assume STABLE_ORDERING or STABLE_DIFF_RATIOS since the
          combination function is arbitrary (could be non-monotonic)
        - If you know your combination function preserves these properties,
          you can explicitly pass them via the `stability` parameter

    Args:
        ufuns: An iterable of utility functions
        combination_function: The function used to combine results of ufuns
        name: Utility function name

    Example:
        >>> from negmas.preferences import LinearUtilityFunction, STATIONARY
        >>> from negmas.preferences.stability import STABLE_ORDERING, STABLE_DIFF_RATIOS
        >>> from negmas.outcomes import make_issue
        >>> issues = [make_issue(5)]
        >>> u1 = LinearUtilityFunction(weights=[1.0], issues=issues)
        >>> u2 = LinearUtilityFunction(weights=[0.5], issues=issues)
        >>> # Product combination - by default does not assume ordering preservation
        >>> combined = ComplexNonlinearUtilityFunction(
        ...     ufuns=[u1, u2], combination_function=lambda vals: vals[0] * vals[1]
        ... )
        >>> combined.is_stationary()  # False because ordering not guaranteed
        False
        >>> # If you know your function preserves ordering, specify it explicitly:
        >>> combined_explicit = ComplexNonlinearUtilityFunction(
        ...     ufuns=[u1, u2],
        ...     combination_function=lambda vals: vals[0] * vals[1],
        ...     stability=STATIONARY,
        ... )
        >>> combined_explicit.is_stationary()
        True
    """

    def __init__(
        self,
        ufuns: Iterable[BaseUtilityFunction],
        combination_function: Callable[[Iterable[Value]], Value],
        **kwargs,
    ):
        """Initialize the instance.

        Args:
            ufuns: Collection of component utility functions to combine.
            combination_function: Function that takes an iterable of utility values and returns the combined utility.
            **kwargs: Additional keyword arguments passed to the base class.
        """
        self.ufuns = list(ufuns)
        self.combination_function = combination_function

        # Compute stability: AND of all component ufuns
        # We do NOT add STABLE_ORDERING or STABLE_DIFF_RATIOS since the
        # combination function is arbitrary (could reverse ordering, etc.)
        combined_stability = _and_stability(self.ufuns)
        # Clear STABLE_ORDERING and STABLE_DIFF_RATIOS unless explicitly provided
        if "stability" not in kwargs:
            combined_stability = Stability(
                combined_stability & ~STABLE_ORDERING & ~STABLE_DIFF_RATIOS
            )
            kwargs["stability"] = combined_stability

        super().__init__(**kwargs)

    def to_stationary(self):
        """Returns a stationary version with all component utility functions converted to stationary."""
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
            combination_function=lambda vals: sum(v * w for w, v in zip(weights, vals)),  # type: ignore
            outcome_space=outcome_space,
            **kwargs,
        )

    def to_dict(
        self, python_class_identifier=PYTHON_CLASS_IDENTIFIER
    ) -> dict[str, Any]:
        """To dict.

        Args:
            python_class_identifier: Python class identifier.

        Returns:
            dict[str, Any]: The result.
        """
        d = {python_class_identifier: get_full_type_name(type(self))}
        d.update(super().to_dict(python_class_identifier=python_class_identifier))
        return dict(
            ufuns=serialize(
                self.ufuns, python_class_identifier=python_class_identifier
            ),
            combination_function=serialize(
                self.combination_function,
                python_class_identifier=python_class_identifier,
            ),
        )

    @classmethod
    def from_dict(
        cls, d: dict[str, Any], python_class_identifier=PYTHON_CLASS_IDENTIFIER
    ):
        """From dict.

        Args:
            d: D.
            python_class_identifier: Python class identifier.
        """
        d.pop(python_class_identifier, None)
        d["ufuns"] = deserialize(
            d["ufuns"], python_class_identifier=python_class_identifier
        )
        d["combination_function"] = deserialize(
            d["combination_function"], python_class_identifier=python_class_identifier
        )
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
