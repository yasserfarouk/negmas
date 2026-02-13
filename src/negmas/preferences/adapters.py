"""Utility function adapters that wrap and transform other utility functions.

This module provides adapter classes that wrap utility functions to add
constraints, transformations, or interface adaptations while properly
propagating stability information.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from negmas.helpers import get_full_type_name
from negmas.outcomes import Outcome
from negmas.serialization import PYTHON_CLASS_IDENTIFIER, deserialize, serialize
from .base import Value

from .base_ufun import BaseUtilityFunction
from .stability import (
    STABLE_DIFF_RATIOS,
    STABLE_IRRATIONAL_OUTCOMES,
    STABLE_ORDERING,
    STABLE_RATIONAL_OUTCOMES,
    Stability,
)

if TYPE_CHECKING:
    from negmas.outcomes.protocols import OutcomeSpace

__all__ = ["UtilityFunctionAdapter", "UFunConstraint"]


class UtilityFunctionAdapter(BaseUtilityFunction):
    """Base class for utility function adapters that wrap a single utility function.

    This class provides a foundation for creating utility functions that wrap
    and transform another utility function. It automatically inherits:
    - Stability from the inner ufun (ANDed with any additional stability)
    - Outcome space from the inner ufun (if not explicitly provided)
    - Reserved value from the inner ufun (if not explicitly provided)

    Subclasses should override `eval()` to implement their transformation logic.

    Attributes:
        ufun: The wrapped utility function.

    Example:
        >>> from negmas.preferences import LinearUtilityFunction
        >>> from negmas.outcomes import make_issue
        >>> issues = [make_issue(5), make_issue(3)]
        >>> base_ufun = LinearUtilityFunction.random(issues=issues)
        >>> adapter = UtilityFunctionAdapter(ufun=base_ufun)
        >>> adapter.stability == base_ufun.stability
        True
    """

    def __init__(
        self,
        ufun: BaseUtilityFunction,
        *,
        name: str | None = None,
        outcome_space: OutcomeSpace | None = None,
        reserved_value: float | None = None,
        stability: Stability | int | None = None,
        **kwargs,
    ):
        """Initialize the adapter with a wrapped utility function.

        Args:
            ufun: The utility function to wrap.
            name: Optional name for this adapter.
            outcome_space: Outcome space (defaults to inner ufun's outcome space).
            reserved_value: Reserved value (defaults to inner ufun's reserved value).
            stability: Additional stability flags to AND with inner ufun's stability.
                If None, inherits full stability from inner ufun.
            **kwargs: Additional arguments passed to parent class.
        """
        # Use inner ufun's values as defaults
        if outcome_space is None:
            outcome_space = ufun.outcome_space
        if reserved_value is None:
            reserved_value = ufun.reserved_value

        super().__init__(
            name=name,
            outcome_space=outcome_space,
            reserved_value=reserved_value,
            **kwargs,
        )

        self._ufun = ufun

        # Compute stability by ANDing with inner ufun's stability
        if stability is not None:
            self._stability = Stability(stability) & ufun.stability
        else:
            self._stability = ufun.stability

    @property
    def ufun(self) -> BaseUtilityFunction:
        """The wrapped utility function."""
        return self._ufun

    def eval(self, offer: Outcome) -> Value:
        """Evaluate the utility of an offer.

        Default implementation delegates to the inner ufun.
        Subclasses should override this to implement their transformation.

        Args:
            offer: The outcome to evaluate.

        Returns:
            The utility value.
        """
        return self._ufun.eval(offer)

    def to_stationary(self) -> UtilityFunctionAdapter:
        """Returns a stationary version of this adapter.

        Creates a new adapter wrapping the stationary version of the inner ufun.
        """
        return type(self)(
            ufun=self._ufun.to_stationary(),
            name=self.name,
            outcome_space=self.outcome_space,
            reserved_value=self.reserved_value,
        )

    def to_dict(
        self, python_class_identifier: str = PYTHON_CLASS_IDENTIFIER
    ) -> dict[str, Any]:
        """Serialize to dictionary.

        Args:
            python_class_identifier: Key for storing class type.

        Returns:
            Dictionary representation.
        """
        d = {python_class_identifier: get_full_type_name(type(self))}
        d.update(super().to_dict(python_class_identifier=python_class_identifier))
        d["ufun"] = serialize(
            self._ufun, python_class_identifier=python_class_identifier
        )
        return d

    @classmethod
    def from_dict(
        cls, d: dict[str, Any], python_class_identifier: str = PYTHON_CLASS_IDENTIFIER
    ) -> UtilityFunctionAdapter:
        """Deserialize from dictionary.

        Args:
            d: Dictionary representation.
            python_class_identifier: Key for class type.

        Returns:
            New instance.
        """
        d = d.copy()
        d.pop(python_class_identifier, None)
        d["ufun"] = deserialize(
            d["ufun"], python_class_identifier=python_class_identifier
        )
        return cls(**d)

    def __getattr__(self, item: str) -> Any:
        """Delegate attribute access to the inner utility function.

        This allows the adapter to transparently expose attributes of the
        wrapped utility function.
        """
        # Prevent infinite recursion during deepcopy/pickle
        if item == "_ufun":
            raise AttributeError(item)
        return getattr(self._ufun, item)


class UFunConstraint(UtilityFunctionAdapter):
    """A utility function adapter that applies a constraint.

    This adapter wraps a utility function and applies a constraint predicate.
    If the constraint is satisfied (predicate returns True), the utility value
    is returned unchanged. If the constraint is violated (predicate returns False),
    negative infinity is returned.

    This is useful for enforcing constraints like budget limits, capacity constraints,
    or other feasibility requirements.

    Attributes:
        constraint: A callable that takes an outcome and returns True if valid.

    Example:
        >>> from negmas.preferences import LinearUtilityFunction
        >>> from negmas.outcomes import make_issue
        >>> issues = [make_issue(5), make_issue(5)]
        >>> base_ufun = LinearUtilityFunction(weights=[1.0, 1.0], issues=issues)
        >>> # Constraint: sum of values must be <= 6
        >>> constrained = UFunConstraint(
        ...     ufun=base_ufun, constraint=lambda o: sum(o) <= 6
        ... )
        >>> constrained((2, 3))  # sum=5, valid
        5.0
        >>> constrained((4, 4))  # sum=8, invalid
        -inf
    """

    def __init__(
        self,
        ufun: BaseUtilityFunction,
        constraint: Callable[[Outcome], bool],
        *,
        name: str | None = None,
        outcome_space: OutcomeSpace | None = None,
        reserved_value: float | None = None,
        stability: Stability | int | None = None,
        **kwargs,
    ):
        """Initialize the constrained utility function.

        Args:
            ufun: The utility function to wrap.
            constraint: A callable that takes an outcome and returns True if
                the outcome satisfies the constraint, False otherwise.
            name: Optional name for this utility function.
            outcome_space: Outcome space (defaults to inner ufun's outcome space).
            reserved_value: Reserved value (defaults to inner ufun's reserved value).
            stability: Additional stability constraints. By default, constraints
                preserve ordering, diff ratios, and rational/irrational outcomes
                of the inner ufun (constrained outcomes become -inf which preserves
                the relative ordering of valid outcomes).
            **kwargs: Additional arguments passed to parent class.
        """
        # Constraints preserve ordering and diff ratios among valid outcomes
        # They also preserve rational/irrational status (invalid -> -inf is always irrational)
        if stability is None:
            stability = (
                STABLE_ORDERING
                | STABLE_DIFF_RATIOS
                | STABLE_RATIONAL_OUTCOMES
                | STABLE_IRRATIONAL_OUTCOMES
            )

        super().__init__(
            ufun=ufun,
            name=name,
            outcome_space=outcome_space,
            reserved_value=reserved_value,
            stability=stability,
            **kwargs,
        )

        self._constraint = constraint

    @property
    def constraint(self) -> Callable[[Outcome], bool]:
        """The constraint predicate."""
        return self._constraint

    def eval(self, offer: Outcome) -> Value:
        """Evaluate the constrained utility of an offer.

        Args:
            offer: The outcome to evaluate.

        Returns:
            The utility value if constraint is satisfied, -inf otherwise.
        """
        if offer is None:
            return self.reserved_value
        if not self._constraint(offer):
            return float("-inf")
        return self._ufun.eval(offer)

    def __call__(self, offer: Outcome | None) -> Value:
        """Evaluate the constrained utility of an offer.

        Args:
            offer: The outcome to evaluate, or None for reserved value.

        Returns:
            The utility value if constraint is satisfied, -inf otherwise.
        """
        if offer is None:
            return self.reserved_value
        if not self._constraint(offer):
            return float("-inf")
        return self._ufun(offer)

    def to_stationary(self) -> UFunConstraint:
        """Returns a stationary version of this constrained utility function."""
        return UFunConstraint(
            ufun=self._ufun.to_stationary(),
            constraint=self._constraint,
            name=self.name,
            outcome_space=self.outcome_space,
            reserved_value=self.reserved_value,
        )

    def to_dict(
        self, python_class_identifier: str = PYTHON_CLASS_IDENTIFIER
    ) -> dict[str, Any]:
        """Serialize to dictionary.

        Args:
            python_class_identifier: Key for storing class type.

        Returns:
            Dictionary representation.
        """
        d = super().to_dict(python_class_identifier=python_class_identifier)
        d["constraint"] = serialize(
            self._constraint, python_class_identifier=python_class_identifier
        )
        return d

    @classmethod
    def from_dict(
        cls, d: dict[str, Any], python_class_identifier: str = PYTHON_CLASS_IDENTIFIER
    ) -> UFunConstraint:
        """Deserialize from dictionary.

        Args:
            d: Dictionary representation.
            python_class_identifier: Key for class type.

        Returns:
            New instance.
        """
        d = d.copy()
        d.pop(python_class_identifier, None)
        d["ufun"] = deserialize(
            d["ufun"], python_class_identifier=python_class_identifier
        )
        d["constraint"] = deserialize(
            d["constraint"], python_class_identifier=python_class_identifier
        )
        return cls(**d)
