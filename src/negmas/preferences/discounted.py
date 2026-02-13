"""Discounted utility function implementations for time-sensitive negotiations."""

from __future__ import annotations

import random
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Callable

from negmas.common import MechanismState, NegotiatorMechanismInterface, Value
from negmas.helpers import get_class
from negmas.helpers.numeric import make_range
from negmas.outcomes import Issue, Outcome
from negmas.serialization import PYTHON_CLASS_IDENTIFIER, deserialize

from .adapters import UtilityFunctionAdapter
from .base_ufun import BaseUtilityFunction
from .stability import (
    FIXED_RESERVED_VALUE,
    SESSION_INDEPENDENT,
    STABLE_DIFF_RATIOS,
    STABLE_IRRATIONAL_OUTCOMES,
    STABLE_ORDERING,
    STABLE_RATIONAL_OUTCOMES,
    STABLE_RESERVED_VALUE,
    STATE_INDEPENDENT,
    Stability,
)

if TYPE_CHECKING:
    from negmas.negotiators import Negotiator

__all__ = ["LinDiscountedUFun", "ExpDiscountedUFun", "DiscountedUtilityFunction"]


class DiscountedUtilityFunction(UtilityFunctionAdapter):
    """Base class for all discounted ufuns.

    Discounted utility functions wrap another utility function and apply a
    time/state-dependent discount to the utility values. They preserve:
    - Ordering of outcomes (STABLE_ORDERING)
    - Difference ratios between utilities (STABLE_DIFF_RATIOS)

    Stability Properties:
        Discounted ufuns are ALWAYS state-dependent (they use state.step or state.time
        to compute the discount factor), so they clear the STATE_INDEPENDENT flag.

        However, discounted ufuns inherit SESSION_INDEPENDENT from the inner ufun.
        Discounting does not depend on NMI parameters (like n_negotiators), only on
        state variables (like time/step). So if the inner ufun is session-independent,
        the discounted ufun will also be session-independent.

        Note: State independence and session independence are orthogonal concepts:
        - STATE_INDEPENDENT: Does not depend on MechanismState (time, step, etc.)
        - SESSION_INDEPENDENT: Does not depend on NMI (n_negotiators, mechanism params)

        A discounted ufun is a typical example of a ufun that is session-independent
        but state-dependent.

    Subclasses must implement `eval_on_state()` to define their discounting logic.
    """

    def __init__(self, ufun: BaseUtilityFunction, **kwargs):
        """Initialize the discounted utility function.

        Args:
            ufun: The base utility function to apply discounting to.
            **kwargs: Additional arguments passed to parent class.
        """
        # Compute base stability for discounted ufuns
        # They preserve ordering and diff ratios
        base_stability = STABLE_ORDERING | STABLE_DIFF_RATIOS

        # Inherit session independence from inner ufun
        if ufun.stability & SESSION_INDEPENDENT:
            base_stability |= SESSION_INDEPENDENT

        # Pass computed stability to parent (will be ANDed with inner ufun)
        kwargs.setdefault("stability", base_stability)

        super().__init__(ufun=ufun, **kwargs)

        # Ensure outcome space is synchronized
        if self.outcome_space:
            self._ufun.outcome_space = self.outcome_space
        else:
            self.outcome_space = self._ufun.outcome_space

    @property
    def ufun(self) -> BaseUtilityFunction:
        """The wrapped utility function (alias for _ufun for backward compatibility)."""
        return self._ufun

    @ufun.setter
    def ufun(self, value: BaseUtilityFunction) -> None:
        """Set the wrapped utility function."""
        self._ufun = value

    def extract_base_ufun(self, deep: bool = False) -> BaseUtilityFunction:
        """Extracts the underlying base utility function without discounting."""
        self._ufun.outcome_space = self.outcome_space
        ufun = self
        if not deep:
            return ufun
        while isinstance(ufun, DiscountedUtilityFunction):
            ufun = ufun._ufun
            ufun.outcome_space = self.outcome_space
        return ufun

    def is_state_dependent(self):
        """Returns True since discounted utility functions depend on negotiation state."""
        return True

    def to_stationary(self):
        """Returns the underlying stationary (non-discounted) utility function."""
        return self._ufun.to_stationary()

    @abstractmethod
    def eval_on_state(
        self,
        offer: Outcome,
        nmi: NegotiatorMechanismInterface | None = None,
        state: MechanismState | None = None,
    ) -> Value:
        """Evaluates the offer given a session and state.

        Subclasses must implement this to define their discounting logic.

        Args:
            offer: The outcome to evaluate.
            nmi: Negotiator-mechanism interface (optional).
            state: Current mechanism state used to compute the discount factor.

        Returns:
            The discounted utility value.
        """
        ...

    def eval(self, offer: Outcome) -> Value:
        """Evaluate the utility of an offer using current negotiation state.

        This method retrieves the current state from the owner's NMI (if available)
        and delegates to `eval_on_state()`.

        Args:
            offer: The outcome to evaluate.

        Returns:
            The discounted utility value.
        """
        if not self.owner or not self.owner.nmi:
            return self.eval_on_state(offer, None, None)
        self.owner: Negotiator
        return self.eval_on_state(offer, self.owner.nmi, self.owner.nmi.state)


class ExpDiscountedUFun(DiscountedUtilityFunction):
    """A utility function with exponential discounting based on negotiation state.

    The discounted utility is computed as: u_discounted = (discount ^ factor) * u_base

    Where:
    - discount: The discount rate (typically 0 < discount <= 1)
    - factor: A value from the mechanism state (e.g., step, time)
    - u_base: The utility from the underlying utility function

    Stability Properties:
        - Always STATE-DEPENDENT: Clears STATE_INDEPENDENT (depends on state.step/time)
        - Inherits SESSION_INDEPENDENT from inner ufun (discounting doesn't use NMI)
        - Always preserves ordering (STABLE_ORDERING) - multiplication by positive factor
        - Always preserves difference ratios (STABLE_DIFF_RATIOS) - proportional scaling
        - If dynamic_reservation=True: STABLE_RESERVED_VALUE, STABLE_RATIONAL_OUTCOMES,
          STABLE_IRRATIONAL_OUTCOMES (everything scales together)
        - If dynamic_reservation=False: FIXED_RESERVED_VALUE, plus:
          - If discount > 1: STABLE_RATIONAL_OUTCOMES (utilities increase, rational stays rational)
          - If discount <= 1: STABLE_IRRATIONAL_OUTCOMES (utilities decrease, irrational stays irrational)
        - If discount == 1 or None: Inherits full stability from inner ufun (no effective discounting)

    Note on Independence:
        State independence and session independence are orthogonal. This ufun is
        state-dependent (uses state.step) but can be session-independent (doesn't
        use NMI parameters like n_negotiators).

    Args:
        ufun: The utility function that is being discounted
        discount: Discount factor (e.g., 0.9 means 90% retained per unit)
        factor: str -> The name of the MechanismState variable for discounting
                callable -> receives a MechanismState and returns a float
        dynamic_reservation: If True, apply discounting to reserved value too
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
        # Compute stability based on discount and dynamic_reservation
        stability = self._compute_exp_stability(ufun, discount, dynamic_reservation)

        super().__init__(
            ufun=ufun,
            name=name,
            reserved_value=reserved_value,
            id=id,
            stability=stability,
            **kwargs,
        )
        self.discount = discount
        self.factor = factor
        self.dynamic_reservation = dynamic_reservation

    @staticmethod
    def _compute_exp_stability(
        ufun: BaseUtilityFunction, discount: float | None, dynamic_reservation: bool
    ) -> Stability:
        """Compute stability flags for exponential discounting.

        Args:
            ufun: The inner utility function.
            discount: The discount factor.
            dynamic_reservation: Whether discounting applies to reserved value.

        Returns:
            Computed stability flags ANDed with inner ufun's stability.
        """
        # Special case: no effective discounting
        if discount is None or discount == 1.0:
            return ufun.stability

        # Base stability: preserves ordering and diff ratios
        stability = STABLE_ORDERING | STABLE_DIFF_RATIOS

        # Inherit session independence from inner ufun
        if ufun.stability & SESSION_INDEPENDENT:
            stability |= SESSION_INDEPENDENT

        # Reserved value and rational/irrational stability
        if dynamic_reservation:
            # Everything scales together - rational/irrational status preserved
            stability |= (
                STABLE_RESERVED_VALUE
                | STABLE_RATIONAL_OUTCOMES
                | STABLE_IRRATIONAL_OUTCOMES
            )
        else:
            # Reserved value is fixed (not discounted)
            stability |= FIXED_RESERVED_VALUE
            if discount > 1.0:
                # Utilities increase over time, rational outcomes stay rational
                # but irrational may become rational
                stability |= STABLE_RATIONAL_OUTCOMES
            else:
                # Utilities decrease over time (discount < 1), irrational stay irrational
                # but rational may become irrational
                stability |= STABLE_IRRATIONAL_OUTCOMES

        # AND with inner ufun's stability, but clear STATE_INDEPENDENT
        # since discounting depends on state (time/step)
        return (stability & ufun.stability) & ~STATE_INDEPENDENT

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
        if outcome_space is None:
            outcome_space = self.outcome_space
        return self._ufun.minmax(
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
            ufun=self._ufun.shift_by(offset, shift_reserved),
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
            ufun=self._ufun.scale_by(scale, scale_reserved),
            discount=self.discount,
            factor=self.factor,
            name=self.name,
            reserved_value=self.reserved_value * scale
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
        d = d.copy()
        d.pop(python_class_identifier, None)
        # Remove stability - it's computed from the inner ufun in __init__
        d.pop("stability", None)
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

    def eval_on_state(
        self,
        offer: Outcome,
        nmi: NegotiatorMechanismInterface | None = None,
        state: MechanismState | None = None,
    ) -> Value:
        """Evaluates discounted utility for an offer given the current negotiation state.

        Computes: (discount ^ factor) * u_base

        Args:
            offer: The outcome to evaluate.
            nmi: Negotiator-mechanism interface (optional).
            state: Current mechanism state used to compute the discount factor.

        Returns:
            The discounted utility value.
        """
        if offer is None and not self.dynamic_reservation:
            return self.reserved_value
        u = self._ufun(offer)
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
        if not hasattr(self._ufun, "xml"):
            raise ValueError(
                f"Cannot serialize because my internal ufun of type {self._ufun.type} is not serializable"
            )
        output = self._ufun.xml(issues)  # type: ignore
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
        return self._ufun.type

    @property
    def type(self):
        """Returns the type name with exponential discounting suffix."""
        return self._ufun.type + "_exponentially_discounted"

    def __str__(self):
        """Returns a human-readable string representation."""
        return f"{self._ufun.type}-discount:{self.discount} based on {self.factor}"


class LinDiscountedUFun(DiscountedUtilityFunction):
    """A utility function with linear discounting based on negotiation state.

    The discounted utility is computed as: u_discounted = u_base - ((factor * cost) ^ power)

    Where:
    - cost: The cost per unit of the factor
    - factor: A value from the mechanism state (e.g., step, time)
    - power: Exponent applied to the total cost
    - u_base: The utility from the underlying utility function

    Stability Properties:
        - Always STATE-DEPENDENT: Clears STATE_INDEPENDENT (depends on state.step/time)
        - Inherits SESSION_INDEPENDENT from inner ufun (discounting doesn't use NMI)
        - Always preserves ordering (STABLE_ORDERING) - subtracting same value from all
        - Always preserves difference ratios (STABLE_DIFF_RATIOS) - constant subtraction
        - If dynamic_reservation=True: STABLE_RESERVED_VALUE, STABLE_RATIONAL_OUTCOMES,
          STABLE_IRRATIONAL_OUTCOMES (everything reduces together)
        - If dynamic_reservation=False: FIXED_RESERVED_VALUE, plus:
          - If cost < 0: STABLE_RATIONAL_OUTCOMES (utilities increase, rational stays rational)
          - If cost >= 0: STABLE_IRRATIONAL_OUTCOMES (utilities decrease, irrational stays irrational)
        - If cost == 0 or None: Inherits full stability from inner ufun (no effective discounting)

    Note on Independence:
        State independence and session independence are orthogonal. This ufun is
        state-dependent (uses state.step) but can be session-independent (doesn't
        use NMI parameters like n_negotiators).

    Args:
        ufun: The utility function that is being discounted
        cost: Cost per unit of the factor (subtracted from utility)
        factor: str -> The name of the MechanismState variable for discounting
                callable -> receives a MechanismState and returns a float
        power: Exponent applied to (factor * cost) before subtraction
        dynamic_reservation: If True, apply discounting to reserved value too
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
        if power is None:
            power = 1.0

        # Compute stability based on cost and dynamic_reservation
        stability = self._compute_lin_stability(ufun, cost, dynamic_reservation)

        super().__init__(
            ufun=ufun,
            name=name,
            reserved_value=reserved_value,
            id=id,
            stability=stability,
            **kwargs,
        )
        self.cost = cost
        self.factor = factor
        self.power = power
        self.dynamic_reservation = dynamic_reservation

    @staticmethod
    def _compute_lin_stability(
        ufun: BaseUtilityFunction, cost: float | None, dynamic_reservation: bool
    ) -> Stability:
        """Compute stability flags for linear discounting.

        Args:
            ufun: The inner utility function.
            cost: The cost factor.
            dynamic_reservation: Whether discounting applies to reserved value.

        Returns:
            Computed stability flags ANDed with inner ufun's stability.
        """
        # Special case: no effective discounting
        if cost is None or cost == 0.0:
            return ufun.stability

        # Base stability: preserves ordering and diff ratios
        stability = STABLE_ORDERING | STABLE_DIFF_RATIOS

        # Inherit session independence from inner ufun
        if ufun.stability & SESSION_INDEPENDENT:
            stability |= SESSION_INDEPENDENT

        # Reserved value and rational/irrational stability
        if dynamic_reservation:
            # Everything reduces together - rational/irrational status preserved
            stability |= (
                STABLE_RESERVED_VALUE
                | STABLE_RATIONAL_OUTCOMES
                | STABLE_IRRATIONAL_OUTCOMES
            )
        else:
            # Reserved value is fixed (not discounted)
            stability |= FIXED_RESERVED_VALUE
            if cost < 0:
                # Utilities increase over time (negative cost), rational stays rational
                # but irrational may become rational
                stability |= STABLE_RATIONAL_OUTCOMES
            else:
                # Utilities decrease over time (positive cost), irrational stay irrational
                # but rational may become irrational
                stability |= STABLE_IRRATIONAL_OUTCOMES

        # AND with inner ufun's stability, but clear STATE_INDEPENDENT
        # since discounting depends on state (time/step)
        return (stability & ufun.stability) & ~STATE_INDEPENDENT

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
        d = d.copy()
        d.pop(python_class_identifier, None)
        # Remove stability - it's computed from the inner ufun in __init__
        d.pop("stability", None)
        d["ufun"] = deserialize(
            d["ufun"], python_class_identifier=python_class_identifier
        )
        return cls(**d)

    def eval_on_state(
        self,
        offer: Outcome,
        nmi: NegotiatorMechanismInterface | None = None,
        state: MechanismState | None = None,
    ) -> Value:
        """Evaluates discounted utility for an offer given the current negotiation state.

        Computes: u_base - ((factor * cost) ^ power)

        Args:
            offer: The outcome to evaluate.
            nmi: Negotiator-mechanism interface (optional).
            state: Current mechanism state used to compute the discount factor.

        Returns:
            The discounted utility value.
        """
        if offer is None and not self.dynamic_reservation:
            return self.reserved_value
        u = self._ufun(offer)
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
        if not hasattr(self._ufun, "xml"):
            raise ValueError(
                f"Cannot serialize because my internal ufun of type {self._ufun.type} is not serializable"
            )
        output = self._ufun.xml(issues)  # type: ignore
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
        return self._ufun.type

    @property
    def type(self):
        """Returns the type name with linear discounting suffix."""
        return self._ufun.type + "_linearly_discounted"

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

    def __str__(self):
        """Returns a human-readable string representation."""
        return f"{self._ufun.type}-cost:{self.cost} raised to {self.power} based on {self.factor}"

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
        if outcome_space is None:
            outcome_space = self.outcome_space
        return self._ufun.minmax(
            outcome_space,
            issues,
            outcomes,
            max_cardinality,
            above_reserve=above_reserve,
        )
