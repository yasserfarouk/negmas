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
        super().__init__(**kwargs)
        self.ufun = ufun

    def is_state_dependent(self):
        return True

    def to_stationary(self):
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
        super().__init__(
            ufun=ufun, name=name, reserved_value=reserved_value, id=id, **kwargs
        )
        self.ufun = ufun
        self.discount = discount
        self.factor = factor
        self.dynamic_reservation = dynamic_reservation

    def minmax(
        self, outcome_space=None, issues=None, outcomes=None, max_cardinality=10_000
    ) -> tuple[float, float]:
        return self.ufun.minmax(outcome_space, issues, outcomes, max_cardinality)

    def shift_by(self, offset: float, shift_reserved: bool = True) -> ExpDiscountedUFun:
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

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        return dict(
            **d,
            ufun=serialize(self.ufun),
            discount=self.discount,
            factor=self.factor,
            dynamic_reservation=self.dynamic_reservation,
        )

    @classmethod
    def from_dict(cls, d: dict[str, Any]):
        d.pop(PYTHON_CLASS_IDENTIFIER, None)
        d["ufun"] = deserialize(d["ufun"])
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
        return self.ufun.type

    @property
    def type(self):
        return self.ufun.type + "_exponentially_discounted"

    def __getattr__(self, item):
        return getattr(self.ufun, item)

    def __str__(self):
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

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        return dict(
            **d,
            ufun=serialize(self.ufun),
            cost=self.cost,
            power=self.power,
            factor=self.factor,
            dynamic_reservation=self.dynamic_reservation,
        )

    @classmethod
    def from_dict(cls, d: dict[str, Any]):
        d.pop(PYTHON_CLASS_IDENTIFIER, None)
        d["ufun"] = deserialize(d["ufun"])
        return cls(**d)

    def eval_on_state(
        self,
        offer: Outcome,
        nmi: NegotiatorMechanismInterface | None = None,
        state: MechanismState | None = None,
    ):
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
        return self.ufun.type

    @property
    def type(self):
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
        return getattr(self.ufun, item)

    def __str__(self):
        return f"{self.ufun.type}-cost:{self.cost} raised to {self.power} based on {self.factor}"
