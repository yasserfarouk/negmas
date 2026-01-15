"""Discounted utility function implementations for time-sensitive negotiations."""

from __future__ import annotations

import random
from typing import Any, Callable

from negmas.common import MechanismState, NegotiatorMechanismInterface
from negmas.helpers import get_class
from negmas.helpers.numeric import make_range
from negmas.outcomes import Issue, Outcome
from negmas.serialization import PYTHON_CLASS_IDENTIFIER, deserialize, serialize

from .base import Value
from .base_ufun import BaseUtilityFunction
from .mixins import StateDependentUFunMixin

__all__ = ["LinDiscountedUFun", "ExpDiscountedUFun", "DiscountedUtilityFunction"]


class DiscountedUtilityFunction(StateDependentUFunMixin, BaseUtilityFunction):
    """Base class for all discounted ufuns"""

    def __init__(self, ufun: BaseUtilityFunction, **kwargs):
        """Initialize the discounted utility function.

        Args:
            ufun: The base utility function to apply discounting to.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(**kwargs)
        self.ufun = ufun

    def is_state_dependent(self):
        """Check if state dependent."""
        return True

    def to_stationary(self):
        """Returns the underlying stationary (non-discounted) utility function."""
        return self.ufun.to_stationary()


class ExpDiscountedUFun(DiscountedUtilityFunction):
    """A discounted utility function based on some factor of the negotiation

    Args:
        ufun: The utility function that is being discounted
        discount: discount factor
        factor: str -> The name of the AgentMechanismInterface variable based on which discounting operate
        callable -> must receive a mechanism info object and returns a float representing the factor

    """

    def __init__(
        self,
        ufun: BaseUtilityFunction,
        discount: float | None = None,
        factor: str | Callable[[MechanismState], float] = "step",
        name=None,
        reserved_value: Value = float("-inf"),
        dynamic_reservation=True,
        id=None,
        **kwargs,
    ):
        """Initialize an exponentially discounted utility function.

        Args:
            ufun: The base utility function to discount.
            discount: Discount factor applied exponentially (e.g., 0.9 means 90% retained per unit).
            factor: State attribute name or callable returning the discount exponent.
            name: Optional name for this utility function.
            reserved_value: Utility value when no agreement is reached.
            dynamic_reservation: If True, apply discounting to reserved value too.
            id: Optional unique identifier.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(
            ufun=ufun, name=name, reserved_value=reserved_value, id=id, **kwargs
        )
        self.ufun = ufun
        self.discount = discount
        self.factor = factor
        self.dynamic_reservation = dynamic_reservation

    def minmax(
        self,
        outcome_space=None,
        issues=None,
        outcomes=None,
        max_cardinality=10_000,
        above_reserve=False,
    ) -> tuple[float, float]:
        """Finds minimum and maximum utility values over the outcome space.

        Args:
            outcome_space: The outcome space to search. Uses self.outcome_space if None.
            issues: Alternative way to specify outcomes via issues.
            outcomes: Explicit list of outcomes to evaluate.
            max_cardinality: Maximum outcomes to sample for large spaces.
            above_reserve: If True, only consider outcomes above reserved value.

        Returns:
            Tuple of (minimum utility, maximum utility).
        """
        # Fix: use self.outcome_space as fallback when outcome_space is None
        if outcome_space is None:
            outcome_space = self.outcome_space
        return self.ufun.minmax(
            outcome_space,
            issues,
            outcomes,
            max_cardinality,
            above_reserve=above_reserve,
        )

    def shift_by(self, offset: float, shift_reserved: bool = True) -> ExpDiscountedUFun:
        """Returns a new utility function with all values shifted by offset.

        Args:
            offset: Amount to add to all utility values.
            shift_reserved: If True, also shift the reserved value.

        Returns:
            New ExpDiscountedUFun with shifted values.
        """
        return ExpDiscountedUFun(
            outcome_space=self.outcome_space,
            ufun=self.ufun.shift_by(offset, shift_reserved),
            discount=self.discount,
            factor=self.factor,
            name=self.name,
            reserved_value=self.reserved_value + offset
            if shift_reserved
            else self.reserved_value,
            dynamic_reservation=self.dynamic_reservation,
        )

    def scale_by(self, scale: float, scale_reserved: bool = True) -> ExpDiscountedUFun:
        """Returns a new utility function with all values scaled by a factor.

        Args:
            scale: Multiplier for all utility values.
            scale_reserved: If True, also scale the reserved value.

        Returns:
            New ExpDiscountedUFun with scaled values.
        """
        return ExpDiscountedUFun(
            outcome_space=self.outcome_space,
            ufun=self.ufun.scale_by(scale, scale_reserved),
            discount=self.discount,
            factor=self.factor,
            name=self.name,
            reserved_value=self.reserved_value + scale
            if scale_reserved
            else self.reserved_value,
            dynamic_reservation=self.dynamic_reservation,
        )

    def to_dict(
        self, python_class_identifier=PYTHON_CLASS_IDENTIFIER
    ) -> dict[str, Any]:
        """Serializes this utility function to a dictionary.

        Args:
            python_class_identifier: Key used to store the class type.

        Returns:
            Dictionary representation suitable for JSON serialization.
        """
        d = super().to_dict(python_class_identifier=python_class_identifier)
        return dict(
            **d,
            ufun=serialize(self.ufun, python_class_identifier=python_class_identifier),
            discount=self.discount,
            factor=self.factor,
            dynamic_reservation=self.dynamic_reservation,
        )

    @classmethod
    def from_dict(
        cls, d: dict[str, Any], python_class_identifier=PYTHON_CLASS_IDENTIFIER
    ):
        """Creates an instance from a dictionary representation.

        Args:
            d: Dictionary containing serialized utility function data.
            python_class_identifier: Key used to identify the class type.
        """
        d.pop(python_class_identifier, None)
        d["ufun"] = deserialize(
            d["ufun"], python_class_identifier=python_class_identifier
        )
        return cls(**d)

    @classmethod
    def random(
        cls,
        issues,
        reserved_value=(0.0, 1.0),
        normalized=True,
        discount_range=(0.8, 1.0),
        base_preferences_type: (
            str | type[BaseUtilityFunction]
        ) = "negmas.LinearAdditiveUtilityFunction",
        **kwargs,
    ) -> ExpDiscountedUFun:
        """Generates a random ufun of the given type"""
        reserved_value = make_range(reserved_value)
        discount_range = make_range(discount_range)
        kwargs["discount"] = (
            random.random() * (discount_range[1] - discount_range[0])
            + discount_range[0]
        )
        kwargs["reserved_value"] = (
            random.random() * (reserved_value[1] - reserved_value[0])
            + reserved_value[0]
        )
        return cls(
            get_class(base_preferences_type).random(  # type: ignore
                issues, reserved_value=reserved_value, normalized=normalized
            ),
            **kwargs,
        )

    # @lru_cache(100)
    # def eval_normalized(self, offer: Outcome | None, above_reserve: bool = True, expected_limits: bool = True) -> Value:
    #     """
    #     Caches the top 100 values as the  normalized value for exponentially discounted ufun does not change with time.
    #     """
    #     return super().eval_normalized(offer, above_reserve, expected_limits)

    def eval_on_state(
        self,
        offer: Outcome,
        nmi: NegotiatorMechanismInterface | None = None,
        state: MechanismState | None = None,
    ):
        """Evaluates discounted utility for an offer given the current negotiation state.

        Args:
            offer: The outcome to evaluate.
            nmi: Negotiator-mechanism interface (optional).
            state: Current mechanism state used to compute the discount factor.
        """
        if offer is None and not self.dynamic_reservation:
            return self.reserved_value
        u = self.ufun(offer)
        if not self.discount or self.discount == 1.0 or state is None:
            return u
        if isinstance(self.factor, str):
            factor = getattr(state, self.factor)
        else:
            factor = self.factor(state)
        return (self.discount**factor) * u

    def xml(self, issues: list[Issue]) -> str:
        """Serializes this utility function to XML format (Genius compatibility).

        Args:
            issues: List of issues defining the negotiation domain.

        Returns:
            XML string representation.
        """
        if not hasattr(self.ufun, "xml"):
            raise ValueError(
                f"Cannot serialize because my internal ufun of type {self.ufun.type} is not serializable"
            )
        output = self.ufun.xml(issues)  # type: ignore
        output += "</objective>\n"
        factor = None
        if self.factor is not None:
            factor = str(self.factor)
        if self.discount is not None:
            output += f'<discount_factor value="{self.discount}" '
            if factor is not None and factor != "step":
                output += f' variable="{factor}" '
            output += "/>\n"
        return output

    @property
    def base_type(self):
        """Returns the type name of the underlying utility function."""
        return self.ufun.type

    @property
    def type(self):
        """Type."""
        return self.ufun.type + "_exponentially_discounted"

    def __getattr__(self, item):
        """getattr  .

        Args:
            item: Item.
        """
        # Prevent infinite recursion during deepcopy/pickle when ufun is not yet set
        if item == "ufun":
            raise AttributeError(item)
        return getattr(self.ufun, item)

    def __str__(self):
        """str  ."""
        return f"{self.ufun.type}-cost:{self.discount} based on {self.factor}"


class LinDiscountedUFun(DiscountedUtilityFunction):
    """A utility function with linear discounting based on some factor of the negotiation

    Args:

        ufun: The utility function that is being discounted
        cost: discount factor
        factor: str -> The name of the AgentMechanismInterface variable based on which discounting operate
        callable -> must receive a mechanism info object and returns a float representing the factor
        power: A power to raise the total cost to before discounting it from the utility_function value

    """

    def __init__(
        self,
        ufun: BaseUtilityFunction,
        cost: float | None = None,
        factor: str | Callable[[MechanismState], float] = "current_step",
        power: float | None = 1.0,
        name=None,
        reserved_value: Value = float("-inf"),
        dynamic_reservation=True,
        id=None,
        **kwargs,
    ):
        """Initialize a linearly discounted utility function.

        Args:
            ufun: The base utility function to discount.
            cost: Cost per unit of the factor (subtracted from utility).
            factor: State attribute name or callable returning the cost multiplier.
            power: Exponent applied to (factor * cost) before subtraction.
            name: Optional name for this utility function.
            reserved_value: Utility value when no agreement is reached.
            dynamic_reservation: If True, apply discounting to reserved value too.
            id: Optional unique identifier.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(
            ufun=ufun, name=name, reserved_value=reserved_value, id=id, **kwargs
        )
        if power is None:
            power = 1.0
        self.ufun = ufun
        self.cost = cost
        self.factor = factor
        self.power = power
        self.dynamic_reservation = dynamic_reservation

    def to_dict(
        self, python_class_identifier=PYTHON_CLASS_IDENTIFIER
    ) -> dict[str, Any]:
        """Serializes this utility function to a dictionary.

        Args:
            python_class_identifier: Key used to store the class type.

        Returns:
            Dictionary representation suitable for JSON serialization.
        """
        d = super().to_dict(python_class_identifier=python_class_identifier)
        return dict(
            **d,
            ufun=serialize(self.ufun, python_class_identifier=python_class_identifier),
            cost=self.cost,
            power=self.power,
            factor=self.factor,
            dynamic_reservation=self.dynamic_reservation,
        )

    @classmethod
    def from_dict(
        cls, d: dict[str, Any], python_class_identifier=PYTHON_CLASS_IDENTIFIER
    ):
        """Creates an instance from a dictionary representation.

        Args:
            d: Dictionary containing serialized utility function data.
            python_class_identifier: Key used to identify the class type.
        """
        d.pop(python_class_identifier, None)
        d["ufun"] = deserialize(
            d["ufun"], python_class_identifier=python_class_identifier
        )
        return cls(**d)

    def eval_on_state(
        self,
        offer: Outcome,
        nmi: NegotiatorMechanismInterface | None = None,
        state: MechanismState | None = None,
    ):
        """Evaluates discounted utility for an offer given the current negotiation state.

        Args:
            offer: The outcome to evaluate.
            nmi: Negotiator-mechanism interface (optional).
            state: Current mechanism state used to compute the discount factor.
        """
        if offer is None and not self.dynamic_reservation:
            return self.reserved_value
        u = self.ufun(offer)
        if not self.cost or self.cost == 0.0 or state is None:
            return u
        if isinstance(self.factor, str):
            factor = getattr(state, self.factor)
        else:
            factor = self.factor(state)
        return u - ((factor * self.cost) ** self.power)

    def xml(self, issues: list[Issue]) -> str:
        """Serializes this utility function to XML format (Genius compatibility).

        Args:
            issues: List of issues defining the negotiation domain.

        Returns:
            XML string representation.
        """
        if not hasattr(self.ufun, "xml"):
            raise ValueError(
                f"Cannot serialize because my internal ufun of type {self.ufun.type} is not serializable"
            )
        output = self.ufun.xml(issues)  # type: ignore
        output += "</objective>\n"
        factor = None
        if self.factor is not None:
            factor = str(self.factor)
        if self.cost is not None:
            output += f'<cost value="{self.cost}" '
            if factor is not None and factor != "step":
                output += f' variable="{factor}" '
            if self.power is not None and self.power != 1.0:
                output += f' power="{self.power}" '
            output += "/>\n"

        return output

    @property
    def base_type(self):
        """Returns the type name of the underlying utility function."""
        return self.ufun.type

    @property
    def type(self):
        """Type."""
        return self.ufun.type + "_linearly_discounted"

    @classmethod
    def random(
        cls,
        issues,
        reserved_value=(0.0, 1.0),
        normalized=True,
        cost_range=(0.8, 1.0),
        power_range=(0.0, 1.0),
        base_preferences_type: type[BaseUtilityFunction]
        | str = "negmas.LinearAdditiveUtilityFunction",
        **kwargs,
    ) -> LinDiscountedUFun:
        """Generates a random ufun of the given type"""
        reserved_value = make_range(reserved_value)
        cost_range = make_range(cost_range)
        power_range = make_range(power_range)
        kwargs["cost"] = (
            random.random() * (cost_range[1] - cost_range[0]) + cost_range[0]
        )
        kwargs["power"] = (
            random.random() * (power_range[1] - power_range[0]) + power_range[0]
        )
        kwargs["reserved_value"] = (
            random.random() * (reserved_value[1] - reserved_value[0])
            + reserved_value[0]
        )
        return cls(
            get_class(base_preferences_type).random(  # type: ignore
                issues, reserved_value=kwargs["reserved_value"], normalized=normalized
            ),
            **kwargs,
        )

    def __getattr__(self, item):
        """getattr  .

        Args:
            item: Item.
        """
        # Prevent infinite recursion during deepcopy/pickle when ufun is not yet set
        if item == "ufun":
            raise AttributeError(item)
        return getattr(self.ufun, item)

    def __str__(self):
        """str  ."""
        return f"{self.ufun.type}-cost:{self.cost} raised to {self.power} based on {self.factor}"

    def minmax(
        self,
        outcome_space=None,
        issues=None,
        outcomes=None,
        max_cardinality=10_000,
        above_reserve=False,
    ) -> tuple[float, float]:
        """Finds minimum and maximum utility values over the outcome space.

        Args:
            outcome_space: The outcome space to search. Uses self.outcome_space if None.
            issues: Alternative way to specify outcomes via issues.
            outcomes: Explicit list of outcomes to evaluate.
            max_cardinality: Maximum outcomes to sample for large spaces.
            above_reserve: If True, only consider outcomes above reserved value.

        Returns:
            Tuple of (minimum utility, maximum utility).
        """
        # Fix: use self.outcome_space as fallback when outcome_space is None
        if outcome_space is None:
            outcome_space = self.outcome_space
        return self.ufun.minmax(
            outcome_space,
            issues,
            outcomes,
            max_cardinality,
            above_reserve=above_reserve,
        )
