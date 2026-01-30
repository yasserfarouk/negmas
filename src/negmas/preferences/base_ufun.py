"""Preferences base classes."""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Sequence, TypeVar

from negmas import warnings
from negmas.common import Distribution, Value
from negmas.helpers.prob import Real, make_distribution
from negmas.helpers.types import get_full_type_name
from negmas.outcomes import Issue, Outcome, dict2outcome
from negmas.outcomes.common import check_one_at_most, os_or_none
from negmas.outcomes.issue_ops import issues_from_geniusweb_json
from negmas.outcomes.outcome_space import make_os
from negmas.outcomes.protocols import IndependentIssuesOS, OutcomeSpace
from negmas.preferences.value_fun import TableFun
from negmas.serialization import PYTHON_CLASS_IDENTIFIER, deserialize, serialize
from negmas.warnings import warn_if_slow

from .preferences import Preferences
from .protocols import InverseUFun
from .value_fun import make_fun_from_xml

if TYPE_CHECKING:
    from negmas.preferences import (
        ConstUtilityFunction,
        ProbUtilityFunction,
        UtilityFunction,
        WeightedUtilityFunction,
    )

__all__ = ["BaseUtilityFunction"]


MAX_CARDINALITY = 10_000_000_000
T = TypeVar("T", bound="BaseUtilityFunction")


# PartiallyScalable,
# HasRange,
# HasReservedValue,
# StationaryConvertible,
# OrdinalRanking,
# CardinalRanking,
# BasePref,
class BaseUtilityFunction(Preferences, ABC):
    """
    Base class for all utility functions in negmas
    """

    def __init__(
        self,
        *args,
        reserved_value: Value = float("-inf"),
        invalid_value: float | None = None,
        **kwargs,
    ):
        """Initialize the instance.

        Args:
            reserved_value: The utility value when no agreement is reached. Can be a float
                or a Distribution. When read back via the reserved_value property, always
                returns a float (mean if Distribution). Use reserved_distribution to get
                the full distribution.
            invalid_value: The value to return for invalid outcomes.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self._reserved_value: Value = reserved_value
        self._cached_inverse: InverseUFun | None = None
        self._cached_inverse_type: type[InverseUFun] | None = None
        self._invalid_value = invalid_value

    @property
    def reserved_value(self) -> float:
        """Returns the reserved value as a float.

        If the underlying value is a Distribution, returns its mean.
        If it's already a numeric type (int/float), returns it as-is for backward compatibility.
        """
        if isinstance(self._reserved_value, Distribution):
            return float(self._reserved_value.mean())
        return self._reserved_value  # type: ignore (preserves int/float)

    @reserved_value.setter
    def reserved_value(self, value: Value) -> None:
        """Sets the reserved value.

        Args:
            value: Can be a float or a Distribution.
        """
        self._reserved_value = value

    @property
    def reserved_distribution(self) -> Distribution:
        """Returns the reserved value as a Distribution.

        If the underlying value is a float, returns a delta distribution at that value.
        """
        return make_distribution(self._reserved_value)

    @abstractmethod
    def eval(self, offer: Outcome) -> Value:
        """Evaluate the utility of an offer and return its value."""
        ...

    def to_stationary(self: T) -> T:
        """Convert this utility function to a stationary (time-independent) version."""
        raise NotImplementedError(
            f"I do not know how to convert a ufun of type {self.type_name} to a stationary ufun."
        )

    def extreme_outcomes(
        self,
        outcome_space: OutcomeSpace | None = None,
        issues: Iterable[Issue] | None = None,
        outcomes: Iterable[Outcome] | None = None,
        max_cardinality=100_000,
    ) -> tuple[Outcome, Outcome]:
        """Find the worst and best outcomes by evaluating utilities.

        Args:
            outcome_space: The outcome space to search. If None, uses the ufun's outcome space.
            issues: Alternative to outcome_space - list of issues defining the space.
            outcomes: Alternative to outcome_space - explicit iterable of outcomes to evaluate.
            max_cardinality: Maximum number of outcomes to enumerate or sample.

        Returns:
            A tuple of (worst_outcome, best_outcome).

        Raises:
            ValueError: If no outcomes can be found to evaluate.
        """
        check_one_at_most(outcome_space, issues, outcomes)
        outcome_space = os_or_none(outcome_space, issues, outcomes)
        if not outcome_space:
            outcome_space = self.outcome_space
        if outcome_space and not outcomes:
            outcomes = outcome_space.enumerate_or_sample(
                max_cardinality=max_cardinality
            )
        if not outcomes:
            raise ValueError("Cannot find outcomes to use for finding extremes")
        mn, mx = float("inf"), float("-inf")
        worst, best = None, None
        warn_if_slow(len(list(outcomes)), "Extreme Outcomes too Slow")
        for o in outcomes:
            u = self(o)
            if u < mn:
                worst, mn = o, u
            if u > mx:
                best, mx = o, u
        if worst is None or best is None:
            raise ValueError(f"Cound not find worst and best outcomes for {self}")
        return worst, best

    def minmax(
        self,
        outcome_space: OutcomeSpace | None = None,
        issues: Sequence[Issue] | None = None,
        outcomes: Sequence[Outcome] | None = None,
        max_cardinality=1000,
        above_reserve=False,
    ) -> tuple[float, float]:
        """Finds the range of the given utility function for the given outcomes

        Args:
            self: The utility function
            issues: List of issues (optional)
            outcomes: A collection of outcomes (optional)
            max_cardinality: the maximum number of outcomes to try sampling (if sampling is used and outcomes are not given)
            above_reserve: If given, the minimum and maximum will be set to reserved value if they were less than it.

        Returns:
            (lowest, highest) utilities in that order

        """
        (worst, best) = self.extreme_outcomes(
            outcome_space, issues, outcomes, max_cardinality
        )
        w, b = self(worst), self(best)
        if isinstance(w, Distribution):
            w = w.min
        if isinstance(b, Distribution):
            b = b.max
        if above_reserve:
            r = self.reserved_value
            if r is None:
                return w, b
            if b < r:
                b, w = r, r
            elif w < r:
                w = r
        return w, b

    def max(self) -> Value:
        """Return the maximum utility value over the outcome space."""
        _, mx = self.minmax()
        return mx

    def min(self) -> Value:
        """Return the minimum utility value over the outcome space."""
        mn, _ = self.minmax()
        return mn

    def best(self) -> Outcome:
        """Return the outcome with the highest utility value."""
        _, mx = self.extreme_outcomes()
        return mx

    def worst(self) -> Outcome:
        """Return the outcome with the lowest utility value."""
        mn, _ = self.extreme_outcomes()
        return mn

    def eval_normalized(
        self,
        offer: Outcome | None,
        above_reserve: bool = True,
        expected_limits: bool = True,
    ) -> Value:
        """
        Evaluates the ufun normalizing the result between zero and one

        Args:
            offer (Outcome | None): offer
            above_reserve (bool): If True, zero corresponds to the reserved value not the minimum
            expected_limits (bool): If True, the expectation of the utility limits will be used for normalization instead of the maximum range and minimum lowest limit

        Remarks:
            - If the maximum and the minium are equal, finite and above reserve, will return 1.0.
            - If the maximum and the minium are equal, initinte or below reserve, will return 0.0.
            - For probabilistic ufuns, a distribution will still be returned.
            - The minimum and maximum will be evaluated freshly every time. If they are already caached in the ufun, the cache will be used.

        """
        r = self.reserved_value
        u = self.eval(offer) if offer else r
        mn, mx = self.minmax()
        if above_reserve:
            if mx < r:
                mx = mn = float("-inf")
            elif mn < r:
                mn = r
        d = mx - mn
        if isinstance(d, Distribution):
            d = float(d) if expected_limits else d.max
        if isinstance(mn, Distribution):
            mn = float(mn) if expected_limits else mn.min
        if d < 1e-5:
            warnings.warn(
                f"Ufun has equal max and min. The outcome will be normalized to zero if they were finite otherwise 1.0: {mn=}, {mx=}, {r=}, {u=}"
            )
            return 1.0 if math.isfinite(mx) else 0.0
        d = 1 / d
        return (u - mn) * d

    def invert(
        self, inverter: type[InverseUFun] | None = None, **kwargs
    ) -> InverseUFun:
        """
        Inverts the ufun, initializes it and caches the result.
        """
        from .inv_ufun import PresortingInverseUtilityFunction

        if self._cached_inverse and (
            inverter is None or self._cached_inverse_type == inverter
        ):
            return self._cached_inverse
        if inverter is None:
            inverter = PresortingInverseUtilityFunction
        self._cached_inverse_type = inverter
        self._cached_inverse = inverter(self, **kwargs)
        self._cached_inverse.init()
        return self._cached_inverse

    def forget_inverter(self):
        """Deletes the cached inverter."""
        self._cached_inverse = None

    def is_volatile(self) -> bool:
        """Return True if this utility function can change between calls."""
        return True

    def is_session_dependent(self) -> bool:
        """Return True if this utility function depends on the negotiation session."""
        return True

    def is_state_dependent(self) -> bool:
        """Return True if this utility function depends on the negotiation state."""
        return True

    def scale_by(
        self: T, scale: float, scale_reserved=True
    ) -> WeightedUtilityFunction | T:
        """Return a new ufun with all utility values scaled by the given factor.

        Args:
            scale: The scaling factor to multiply all utility values by. Must be non-negative.
            scale_reserved: If True, also scale the reserved value by the same factor.

        Returns:
            A new WeightedUtilityFunction wrapping this ufun with the given scale applied.

        Raises:
            ValueError: If scale is negative.
        """
        if scale < 0:
            raise ValueError(f"Cannot scale with a negative multiplier ({scale})")
        from negmas.preferences.complex import WeightedUtilityFunction

        r = (scale * self.reserved_value) if scale_reserved else self.reserved_value
        return WeightedUtilityFunction(
            ufuns=[self], weights=[scale], name=self.name, reserved_value=r
        )

    def scale_min_for(
        self: T,
        to: float,
        outcome_space: OutcomeSpace | None = None,
        issues: Sequence[Issue] | None = None,
        outcomes: Sequence[Outcome] | None = None,
        rng: tuple[float, float] | None = None,
    ) -> T:
        """Scale utilities so the minimum value equals the target over the given space.

        Args:
            to: The target value for the minimum utility.
            outcome_space: The outcome space to compute minimum over.
            issues: Alternative to outcome_space - list of issues defining the space.
            outcomes: Alternative to outcome_space - explicit list of outcomes.
            rng: Pre-computed (min, max) range to avoid recomputation.

        Returns:
            A new scaled utility function.
        """
        if rng is None:
            mn, _ = self.minmax(outcome_space, issues, outcomes)
        else:
            mn, _ = rng
        scale = to / mn
        return self.scale_by(scale)

    def scale_min(self: T, to: float, rng: tuple[float, float] | None = None) -> T:
        """Scale utilities so the minimum value equals the target over the ufun's outcome space.

        Args:
            to: The target value for the minimum utility.
            rng: Pre-computed (min, max) range to avoid recomputation.

        Returns:
            A new scaled utility function.
        """
        return self.scale_min_for(to, outcome_space=self.outcome_space, rng=rng)

    def scale_max_for(
        self: T,
        to: float,
        outcome_space: OutcomeSpace | None = None,
        issues: Sequence[Issue] | None = None,
        outcomes: Sequence[Outcome] | None = None,
        rng: tuple[float, float] | None = None,
    ) -> T:
        """Scale utilities so the maximum value equals the target over the given space.

        Args:
            to: The target value for the maximum utility.
            outcome_space: The outcome space to compute maximum over.
            issues: Alternative to outcome_space - list of issues defining the space.
            outcomes: Alternative to outcome_space - explicit list of outcomes.
            rng: Pre-computed (min, max) range to avoid recomputation.

        Returns:
            A new scaled utility function.
        """
        if rng is None:
            _, mx = self.minmax(outcome_space, issues, outcomes)
        else:
            _, mx = rng
        scale = to / mx
        return self.scale_by(scale)

    def scale_max(self: T, to: float, rng: tuple[float, float] | None = None) -> T:
        """Scale utilities so the maximum value equals the target over the ufun's outcome space.

        Args:
            to: The target value for the maximum utility.
            rng: Pre-computed (min, max) range to avoid recomputation.

        Returns:
            A new scaled utility function.
        """
        return self.scale_max_for(to, outcome_space=self.outcome_space, rng=rng)

    def normalize_for(
        self: T,
        to: tuple[float, float] = (0.0, 1.0),
        outcome_space: OutcomeSpace | None = None,
        guarantee_max: bool = True,
        guarantee_min: bool = True,
        max_cardinality: int = MAX_CARDINALITY,
        normalize_reserved_values: bool = False,
        reserved_value_penalty: float | None = None,
    ) -> T | ConstUtilityFunction:
        """Normalize utilities to the target range over the given outcome space.

        Args:
            to: The target (min, max) range for normalized utilities. Defaults to (0.0, 1.0).
            outcome_space: The outcome space to normalize over. If None, uses the ufun's
                outcome space.
            guarantee_max: If True, ensures maximum utility equals to[1]. Defaults to True.
                **Active for:** LinearAdditiveUtilityFunction only.
                **Ignored by:** BaseUtilityFunction (this implementation).
            guarantee_min: If True, ensures minimum utility equals to[0]. Defaults to True.
                **Active for:** LinearAdditiveUtilityFunction only.
                **Ignored by:** BaseUtilityFunction (this implementation).
            max_cardinality: Maximum number of outcomes to consider when normalizing.
                Defaults to 10 billion.
                **Active for:** All implementations (used in minmax()).
            normalize_reserved_values: If True, corrects non-finite reserved values (None, inf, -inf, NaN).
                Defaults to False.
            reserved_value_penalty: Penalty to subtract from ufun.min() when correcting reserved values.
                If None, uses DEFAULT_RESERVED_VALUE_PENALTY from negmas.common.

        Returns:
            A new normalized utility function. Returns a ConstUtilityFunction if the
            original range is too small (< 1e-7).

        Raises:
            ValueError: If no outcome space is provided or defined for the ufun.

        Remarks:
            - This base implementation always guarantees both min and max, so the
              guarantee_max and guarantee_min parameters are ignored.
            - Subclasses like LinearAdditiveUtilityFunction use these parameters to
              control whether strict guarantees are enforced at the cost of potentially
              non-uniform weight scaling.
            - If normalize_reserved_values is True, any non-finite reserved value will be
              corrected to ufun.min() - penalty before normalization.
        """
        # Correct reserved value if requested
        if normalize_reserved_values:
            from negmas.preferences.ops import correct_reserved_value

            corrected_rv, was_corrected = correct_reserved_value(
                self.reserved_value, self, eps=reserved_value_penalty, warn=True
            )
            if was_corrected:
                self.reserved_value = corrected_rv

        if not outcome_space:
            outcome_space = self.outcome_space
        if not outcome_space:
            raise ValueError(
                "Cannot find the outcome-space to normalize for. "
                "You must pass outcome_space, issues or outcomes or have the ufun being constructed with one of them"
            )
        mn, mx = self.minmax(outcome_space, max_cardinality=max_cardinality)

        d = float(mx - mn)
        if d < 1e-7:
            from negmas.preferences.crisp.const import ConstUtilityFunction

            return ConstUtilityFunction(
                to[0] if mx < self.reserved_value else to[1],
                outcome_space=self.outcome_space,
                name=self.name,
                reserved_value=to[1] if mn < self.reserved_value else to[0],
            )

        scale = float(to[1] - to[0]) / d

        u = self.scale_by(scale, scale_reserved=True)
        return u.shift_by(to[0] - scale * mn, shift_reserved=True)

    def normalize(
        self: T,
        to: tuple[float, float] = (0.0, 1.0),
        normalize_weights: bool = False,
        normalize_reserved_values: bool = False,
        reserved_value_penalty: float | None = None,
    ) -> T | ConstUtilityFunction:
        """Normalize utilities to the target range over the ufun's outcome space.

        Args:
            to: The target (min, max) range for normalized utilities. Defaults to (0.0, 1.0).
            normalize_weights: Currently unused, kept for API compatibility.
            normalize_reserved_values: If True, corrects non-finite reserved values (None, inf, -inf, NaN).
                Defaults to False.
            reserved_value_penalty: Penalty to subtract from ufun.min() when correcting reserved values.
                If None, uses DEFAULT_RESERVED_VALUE_PENALTY from negmas.common.

        Returns:
            A new normalized utility function. Returns a ConstUtilityFunction if the
            original range is too small (< 1e-8).

        Raises:
            ValueError: If the ufun has no outcome space defined.

        Remarks:
            - If normalize_reserved_values is True, any non-finite reserved value will be
              corrected to ufun.min() - penalty before normalization.
        """
        _ = normalize_weights

        # Correct reserved value if requested
        if normalize_reserved_values:
            from negmas.preferences.ops import correct_reserved_value

            corrected_rv, was_corrected = correct_reserved_value(
                self.reserved_value, self, eps=reserved_value_penalty, warn=True
            )
            if was_corrected:
                self.reserved_value = corrected_rv

        from negmas.preferences import ConstUtilityFunction

        if not self.outcome_space:
            raise ValueError("Cannot normalize a ufun without an outcome-space")
        mn, mx = self.minmax(self.outcome_space, max_cardinality=MAX_CARDINALITY)

        d = float(mx - mn)
        if d < 1e-8:
            return ConstUtilityFunction(
                to[0] if mx < self.reserved_value else to[1],
                name=self.name,
                reserved_value=to[1] if mn < self.reserved_value else to[0],
            )

        scale = float(to[1] - to[0]) / d

        # u = self.shift_by(-mn, shift_reserved=True)
        u = self.scale_by(scale, scale_reserved=True)
        return u.shift_by(to[0] - scale * mn, shift_reserved=True)

    @classmethod
    def normalize_all_for(
        cls: type[T],
        ufuns: tuple[T, ...],
        to: tuple[float | None, float | None] = (0.0, 1.0),
        max_cardinality: int = MAX_CARDINALITY,
        outcome_space: OutcomeSpace | None = None,
        guarantee_max: bool = True,
        guarantee_min: bool = False,
    ) -> tuple[T | ConstUtilityFunction, ...]:
        """Normalize multiple utility functions to a common scale.

        This method normalizes a collection of utility functions together so that
        they share the same scale. This is critical for multi-agent scenarios where
        utility values need to be comparable across agents.

        Args:
            ufuns: Tuple of utility functions to normalize together.
            to: Target range (min, max) for normalized utilities. If either is None,
                only the other bound is enforced.
            max_cardinality: Maximum number of outcomes to consider when computing
                min/max values for outcome spaces.
            outcome_space: The outcome space to normalize over. If None, uses each
                ufun's own outcome space.
            guarantee_max: If True, ensures at least one ufun reaches exactly to[1].
            guarantee_min: If True, ensures at least one ufun reaches exactly to[0].

        Returns:
            Tuple of normalized utility functions on a common scale.

        Raises:
            ValueError: If both to[0] and to[1] are None.

        Example:
            >>> from negmas import make_issue
            >>> from negmas.preferences import LinearUtilityFunction
            >>> issues = [make_issue([0, 5, 10], "x"), make_issue([5, 10, 15], "y")]
            >>> # Agent 1: utilities range [0, 10]
            >>> u1 = LinearUtilityFunction(weights=[1.0, 0.0], issues=issues)
            >>> # Agent 2: utilities range [5, 15]
            >>> u2 = LinearUtilityFunction(weights=[0.0, 1.0], issues=issues)
            >>> # Normalize to common scale
            >>> n1, n2 = LinearUtilityFunction.normalize_all_for(
            ...     (u1, u2), to=(0.0, 1.0)
            ... )
            >>> # Now n1 range [0, 0.67] and n2 range [0.33, 1.0]
            >>> # Utilities are comparable across agents

        Note:
            - This method handles discounted utility functions by recursively
              normalizing their base utility functions.
            - When guarantee_max=True, the agent with the highest utility will
              have its maximum map to exactly to[1].
            - When guarantee_min=True, the agent with the lowest utility will
              have its minimum map to exactly to[0].
        """
        from negmas.preferences.crisp.const import ConstUtilityFunction
        from negmas.preferences.discounted import DiscountedUtilityFunction

        epsilon = 1e-8

        # Handle discounted utility functions recursively
        roots, parents, children = [], [], []

        def get_base_ufun(
            u: BaseUtilityFunction,
        ) -> tuple[
            BaseUtilityFunction, BaseUtilityFunction | None, BaseUtilityFunction
        ]:
            root, parent, child = u, None, u
            while isinstance(child, DiscountedUtilityFunction):
                parent, child = child, child.ufun  # type: ignore
            return root, parent, child

        for u in ufuns:
            r, p, c = get_base_ufun(u)
            parents.append(p)
            roots.append(r)
            children.append(c)

        # If any ufun is discounted, normalize the base ufuns and reconstruct
        if any(_ is not None for _ in parents):
            normalized = cls.normalize_all_for(
                tuple(children),  # type: ignore
                to=to,
                max_cardinality=max_cardinality,
                outcome_space=outcome_space,
                guarantee_max=guarantee_max,
                guarantee_min=guarantee_min,
            )
            new_ufuns = []
            for root, parent, new_child in zip(roots, parents, normalized, strict=True):
                if parent is not None:
                    parent.ufun = new_child  # type: ignore
                new_ufuns.append(root)
            return tuple(new_ufuns)

        # If no bounds specified, return unchanged
        if not to or (to[0] is None and to[1] is None):
            return ufuns

        # Find global min and max across all ufuns
        mn, mx = float("inf"), float("-inf")
        minmaxs = []
        for u in ufuns:
            current_mn, current_mx = u.minmax(
                outcome_space or u.outcome_space, max_cardinality=max_cardinality
            )
            minmaxs.append((current_mn, current_mx))
            mn = min(current_mn, mn)
            mx = max(current_mx, mx)

        # If only max is given, just shift to make max equal to[1]
        if to[1] is None:
            return tuple(u.shift_by(to[0] - mn, shift_reserved=True) for u in ufuns)  # type: ignore

        # If only min is given, just shift to make min equal to[0]
        if to[0] is None:
            return tuple(u.shift_by(to[1] - mx, shift_reserved=True) for u in ufuns)

        # Find common scale
        d = float(mx - mn)
        if d < epsilon:
            # All ufuns are constant - map to to[0] or to[1] based on reserved value
            return tuple(
                ConstUtilityFunction(
                    to[0] if mx < u.reserved_value else to[1],
                    outcome_space=u.outcome_space,
                    name=u.name,
                    reserved_value=to[1] if mx < u.reserved_value else to[0],
                )
                for u in ufuns
            )

        results = []
        always_scale = guarantee_max and guarantee_min

        def within_limits(
            x: float, limit: float, limits: tuple[float, float], guarantee: bool
        ) -> bool:
            if always_scale:
                return False
            if guarantee:
                return abs(x - limit) <= epsilon
            return limits[0] - epsilon <= x <= limits[1] + epsilon

        for (mymn, mymx), u in zip(minmaxs, ufuns):
            myto = (to[0], to[1])

            # Skip if already normalized exactly
            if abs(mymn - to[0]) < epsilon and abs(mymx - to[1]) < epsilon:
                results.append(u)
                continue

            # Skip if within limits and guarantees are satisfied
            if within_limits(mymn, myto[0], myto, guarantee_min) and within_limits(
                mymx, myto[1], myto, guarantee_max
            ):
                results.append(u)
                continue

            # If the minimum is negative, shift everything up to make the minimum zero
            if mymn < 0:
                u, mymn = u.shift_by(-mymn, shift_reserved=True), 0.0

            # Scale everything to have a range of myto[1] - myto[0]
            scale = float(myto[1] - myto[0]) / d
            u = u.scale_by(scale, scale_reserved=True)

            # Finally shift to align with target range
            if guarantee_max:
                # Align maximum with to[1]
                results.append(u.shift_by(myto[1] - scale * mymx, shift_reserved=True))
            else:
                # Align minimum with to[0]
                results.append(u.shift_by(myto[0] - scale * mymn, shift_reserved=True))

        return tuple(results)

    def shift_by(
        self: T, offset: float, shift_reserved=True
    ) -> WeightedUtilityFunction | T:
        """Return a new ufun with all utility values shifted by the given offset.

        Args:
            offset: The amount to add to all utility values.
            shift_reserved: If True, also shift the reserved value by the same offset.

        Returns:
            A new WeightedUtilityFunction combining this ufun with a constant offset.
        """
        from negmas.preferences.complex import WeightedUtilityFunction
        from negmas.preferences.crisp.const import ConstUtilityFunction

        r = (self.reserved_value + offset) if shift_reserved else self.reserved_value
        return WeightedUtilityFunction(
            ufuns=[self, ConstUtilityFunction(offset)],
            weights=[1, 1],
            name=self.name,
            reserved_value=r,
        )

    def shift_min_for(
        self: T,
        to: float,
        outcome_space: OutcomeSpace | None = None,
        issues: Sequence[Issue] | None = None,
        outcomes: Sequence[Outcome] | None = None,
        rng: tuple[float, float] | None = None,
    ) -> T:
        """Shift utilities so the minimum value equals the target over the given space.

        Args:
            to: The target value for the minimum utility.
            outcome_space: The outcome space to compute minimum over.
            issues: Alternative to outcome_space - list of issues defining the space.
            outcomes: Alternative to outcome_space - explicit list of outcomes.
            rng: Pre-computed (min, max) range to avoid recomputation.

        Returns:
            A new shifted utility function.
        """
        if rng is None:
            mn, _ = self.minmax(outcome_space, issues, outcomes)
        else:
            mn, _ = rng
        offset = to - mn
        return self.shift_by(offset)

    def shift_max_for(
        self: T,
        to: float,
        outcome_space: OutcomeSpace | None = None,
        issues: Sequence[Issue] | None = None,
        outcomes: Sequence[Outcome] | None = None,
        rng: tuple[float, float] | None = None,
    ) -> T:
        """Shift utilities so the maximum value equals the target over the given space.

        Args:
            to: The target value for the maximum utility.
            outcome_space: The outcome space to compute maximum over.
            issues: Alternative to outcome_space - list of issues defining the space.
            outcomes: Alternative to outcome_space - explicit list of outcomes.
            rng: Pre-computed (min, max) range to avoid recomputation.

        Returns:
            A new shifted utility function.
        """
        if rng is None:
            _, mx = self.minmax(outcome_space, issues, outcomes)
        else:
            _, mx = rng
        offset = to - mx
        return self.shift_by(offset)

    def _do_rank(self, vals, descending):
        vals = sorted(vals, key=lambda x: x[1], reverse=descending)
        if not vals:
            return []
        ranks = [([vals[0][0]], vals[0][1])]
        for w, v in vals[1:]:
            if v == ranks[-1][1]:
                ranks[-1][0].append(w)
                continue
            ranks.append(([w], v))
        return ranks

    def argrank_with_weights(
        self, outcomes: Sequence[Outcome | None], descending=True
    ) -> list[tuple[list[Outcome | None], float]]:
        """
        Ranks the given list of outcomes with weights. None stands for the null outcome.

        Returns:

            - A list of tuples each with two values:
                - an list of integers giving the index in the input array (outcomes) of an outcome (at the given utility level)
                - the weight of that outcome
            - The list is sorted by weights descendingly

        """
        vals = zip(range(len(list(outcomes))), (self(_) for _ in outcomes))
        return self._do_rank(vals, descending)

    def argrank(
        self, outcomes: Sequence[Outcome | None], descending=True
    ) -> list[list[Outcome | None]]:
        """
        Ranks the given list of outcomes with weights. None stands for the null outcome.

        Returns:
            A list of lists of integers giving the outcome index in the input. The list is sorted by utlity value

        """
        ranks = self.argrank_with_weights(outcomes, descending)
        return [_[0] for _ in ranks]

    def rank_with_weights(
        self, outcomes: Sequence[Outcome | None], descending=True
    ) -> list[tuple[list[Outcome | None], float]]:
        """
        Ranks the given list of outcomes with weights. None stands for the null outcome.

        Returns:

            - A list of tuples each with two values:
                - an list of integers giving the index in the input array (outcomes) of an outcome (at the given utility level)
                - the weight of that outcome
            - The list is sorted by weights descendingly

        """
        vals = zip(outcomes, (self(_) for _ in outcomes))
        return self._do_rank(vals, descending)

    def rank(
        self, outcomes: Sequence[Outcome | None], descending=True
    ) -> list[list[Outcome | None]]:
        """
        Ranks the given list of outcomes with weights. None stands for the null outcome.

        Returns:
            A list of lists of integers giving the outcome index in the input. The list is sorted by utlity value

        """
        ranks = self.rank_with_weights(outcomes, descending)
        return [_[0] for _ in ranks]

    def eu(self, offer: Outcome | None) -> float:
        """
        calculates the **expected** utility value of the input outcome
        """
        return float(self(offer))

    def to_crisp(self) -> UtilityFunction:
        """Convert this utility function to a crisp (deterministic) utility function."""
        from negmas.preferences.crisp_ufun import CrispAdapter

        return CrispAdapter(self)

    def to_prob(self) -> ProbUtilityFunction:
        """Convert this utility function to a probabilistic utility function."""
        from negmas.preferences.prob_ufun import ProbAdapter

        return ProbAdapter(self)

    def to_dict(
        self, python_class_identifier=PYTHON_CLASS_IDENTIFIER
    ) -> dict[str, Any]:
        """Serialize this utility function to a dictionary.

        Args:
            python_class_identifier: The key used to store the Python class name
                in the serialized dictionary. Defaults to PYTHON_CLASS_IDENTIFIER.

        Returns:
            A dictionary containing the serialized utility function data including
            the outcome space, reserved value, name, and id.
        """
        d = {python_class_identifier: get_full_type_name(type(self))}
        return dict(
            **d,
            outcome_space=serialize(
                self.outcome_space, python_class_identifier=python_class_identifier
            ),
            reserved_value=self.reserved_value,
            name=self.name,
            id=self.id,
        )

    @classmethod
    def from_dict(cls, d, python_class_identifier=PYTHON_CLASS_IDENTIFIER):
        """Deserialize a utility function from a dictionary.

        Args:
            d: The dictionary containing serialized utility function data.
            python_class_identifier: The key used to identify the Python class name
                in the dictionary. Defaults to PYTHON_CLASS_IDENTIFIER.

        Returns:
            A new utility function instance reconstructed from the dictionary.
        """
        d.pop(python_class_identifier, None)
        d["outcome_space"] = deserialize(
            d.get("outcome_space", None),
            python_class_identifier=python_class_identifier,
        )
        return cls(**d)

    def sample_outcome_with_utility(
        self,
        rng: tuple[float, float],
        outcome_space: OutcomeSpace | None = None,
        issues: Sequence[Issue] | None = None,
        outcomes: Sequence[Outcome] | None = None,
        n_trials: int = 100,
    ) -> Outcome | None:
        """
        Samples an outcome in the given utiltity range or return None if not possible

        Args:
            rng (Tuple[float, float]): rng
            outcome_space (OutcomeSpace | None): outcome_space
            issues (Sequence[Issue] | None): issues
            outcomes (Sequence[Outcome] | None): outcomes
            n_trials (int): n_trials

        Returns:
            Optional["Outcome"]:
        """
        if rng[0] is None:
            rng = (float("-inf"), rng[1])
        if rng[1] is None:
            rng = (rng[0], float("inf"))
        outcome_space = os_or_none(outcome_space, issues, outcomes)
        if not outcome_space:
            outcome_space = self.outcome_space
        if not outcome_space:
            raise ValueError("No outcome-space is given or defined for the ufun")
        if outcome_space.cardinality < n_trials:
            n_trials = outcome_space.cardinality  # type: ignore I know that it is an int (see the if)
        for o in outcome_space.sample(n_trials, with_replacement=False):
            if o is None:
                continue
            assert o in outcome_space, (
                f"Sampled outcome {o} which is not in the outcome-space {outcome_space}"
            )
            if rng[0] - 1e-6 <= float(self(o)) <= rng[1] + 1e-6:
                return o
        return None

    @classmethod
    def from_xml_str(
        cls,
        xml_str: str,
        issues: Iterable[Issue] | Sequence[Issue],
        safe_parsing=True,
        ignore_discount=False,
        ignore_reserved=False,
        name: str | None = None,
    ) -> tuple[BaseUtilityFunction | None, float | None]:
        """Imports a utility function from a GENIUS XML string.

        Args:

            xml_str (str): The string containing GENIUS style XML utility function definition
            issues (Sequence[Issue] | None): Optional issue space to confirm that the utility function is valid
            product of all issues in the input
            safe_parsing (bool): Turn on extra checks

        Returns:

            A utility function object (depending on the input file)


        Examples:

            >>> from negmas.preferences import UtilityFunction
            >>> import pkg_resources
            >>> from negmas.inout import load_genius_domain
            >>> domain = load_genius_domain(
            ...     pkg_resources.resource_filename(
            ...         "negmas", resource_name="tests/data/Laptop/Laptop-C-domain.xml"
            ...     )
            ... )
            >>> with open(
            ...     pkg_resources.resource_filename(
            ...         "negmas", resource_name="tests/data/Laptop/Laptop-C-prof1.xml"
            ...     ),
            ...     "r",
            ... ) as ff:
            ...     u, _ = UtilityFunction.from_xml_str(ff.read(), issues=domain.issues)
            >>> with open(
            ...     pkg_resources.resource_filename(
            ...         "negmas", resource_name="tests/data/Laptop/Laptop-C-prof1.xml"
            ...     ),
            ...     "r",
            ... ) as ff:
            ...     u, _ = UtilityFunction.from_xml_str(ff.read(), issues=domain.issues)
            >>> assert (
            ...     abs(u(("Dell", "60 Gb", "19'' LCD")) - 21.987727736172488)
            ...     < 0.000001
            ... )
            >>> assert (
            ...     abs(u(("HP", "80 Gb", "20'' LCD")) - 22.68559475583014) < 0.000001
            ... )


        """
        from negmas.preferences.complex import WeightedUtilityFunction
        from negmas.preferences.crisp.linear import (
            AffineUtilityFunction,
            LinearAdditiveUtilityFunction,
        )
        from negmas.preferences.crisp.nonlinear import HyperRectangleUtilityFunction

        root = ET.fromstring(xml_str)
        if safe_parsing and root.tag != "utility_space":
            raise ValueError(f"Root tag is {root.tag}: Expected utility_space")

        issues = list(issues)
        ordered_issues: list[Issue] = []
        domain_issues_dict: dict[str, Issue] | None = None
        ordered_issues = issues
        domain_issues_dict = dict(zip([_.name for _ in issues], issues))
        # issue_indices = dict(zip([_.name for _ in issues], range(len(issues))))
        objective = None
        reserved_value = 0.0
        discount_factor = 0.0
        for child in root:
            if child.tag == "objective":
                objective = child
            elif child.tag == "reservation":
                reserved_value = float(child.attrib["value"])
            elif child.tag == "discount_factor":
                discount_factor = float(child.attrib["value"])

        if objective is None:
            objective = root
        weights = {}
        found_issues = {}
        issue_info = {}
        issue_keys = {}
        rects, rect_utils = [], []
        all_numeric = True
        global_bias = 0

        def _get_hyperrects(ufun, max_utility, utiltype=float):
            utype = ufun.attrib.get("type", "none")
            uweight = float(ufun.attrib.get("weight", 1))
            uagg = ufun.attrib.get("aggregation", "sum")
            if uagg != "sum":
                raise ValueError(
                    f"Hypervolumes combined using {uagg} are not supported (only sum is supported)"
                )
            total_util = utiltype(0)
            rects = []
            rect_utils = []
            if utype == "PlainUfun":
                for rect in ufun:
                    util = utiltype(rect.attrib.get("utility", 0))
                    total_util += util if util > 0 else 0
                    ranges = {}
                    rect_utils.append(util * uweight)
                    for r in rect:
                        ii = int(r.attrib["index"]) - 1
                        # key = issue_keys[ii]
                        ranges[ii] = (
                            utiltype(r.attrib["min"]),
                            utiltype(r.attrib["max"]),
                        )
                    rects.append(ranges)
            else:
                raise ValueError(f"Unknown ufun type {utype}")
            total_util = total_util if not max_utility else max_utility
            return rects, rect_utils

        for child in objective:
            if child.tag == "weight":
                indx = int(child.attrib["index"]) - 1
                if indx < 0 or indx >= len(issues):
                    global_bias += float(child.attrib["value"])
                    continue
                weights[issues[indx].name] = float(child.attrib["value"])
            elif child.tag == "utility_function" or child.tag == "utility":
                utility_tag = child
                max_utility = child.attrib.get("maxutility", None)
                if max_utility is not None:
                    max_utility = float(max_utility)
                ufun_found = False
                for ufun in utility_tag:
                    if ufun.tag == "ufun":
                        ufun_found = True
                        _r, _u = _get_hyperrects(ufun, max_utility)
                        rects += _r
                        rect_utils += _u
                if not ufun_found:
                    raise ValueError(
                        "Cannot find ufun tag inside a utility_function tag"
                    )
            elif child.tag == "issue":
                indx = int(child.attrib["index"]) - 1
                issue_key = child.attrib["name"]
                if (
                    domain_issues_dict is not None
                    and issue_key not in domain_issues_dict.keys()
                ):
                    raise ValueError(
                        f"Issue {issue_key} is not in the input issue names ({domain_issues_dict.keys()})"
                    )
                issue_info[issue_key] = {"name": issue_key, "index": indx}
                issue_keys[indx] = issue_key
                info = {"type": "discrete", "etype": "discrete", "vtype": "discrete"}
                for a in ("type", "etype", "vtype"):
                    info[a] = child.attrib.get(a, info[a])
                issue_info[issue_key].update(info)
                mytype = info["type"]
                info["vtype"]
                if domain_issues_dict is None:
                    raise ValueError("unknown domain-issue-dict!!!")

                current_issue = domain_issues_dict[issue_key]

                if mytype == "discrete":
                    found_issues[issue_key] = dict()
                    if current_issue.is_continuous():
                        raise ValueError(
                            f"Got a {mytype} issue but expected a continuous valued issue"
                        )
                elif mytype in ("integer", "real"):
                    lower = current_issue.min_value
                    upper = current_issue.max_value
                    lower, upper = (
                        child.attrib.get("lowerbound", lower),
                        child.attrib.get("upperbound", upper),
                    )
                    for rng_child in child:
                        if rng_child.tag == "range":
                            lower, upper = (
                                rng_child.attrib.get("lowerbound", lower),
                                rng_child.attrib.get("upperbound", upper),
                            )
                    if mytype == "integer":
                        if current_issue.is_continuous():
                            raise ValueError(
                                f"Got a {mytype} issue but expected a continuous valued issue"
                            )
                        lower, upper = int(lower), int(upper)  # type: ignore
                    else:
                        lower, upper = float(lower), float(upper)  # type: ignore
                    if (
                        lower < current_issue.min_value
                        or upper > current_issue.max_value
                    ):  # type: ignore
                        raise ValueError(
                            f"Bounds ({lower}, {upper}) are invalid for issue {issue_key} with bounds: "
                            f"{current_issue.values}"
                        )
                else:
                    raise ValueError(f"Unknown type: {mytype}")
                # now we found ranges for range issues and will find values for all issues
                found_values = False
                for item in child:
                    if item.tag == "item":
                        if mytype != "discrete":
                            raise ValueError(
                                f"cannot specify item utilities for not-discrete type: {mytype}"
                            )
                        all_numeric = False
                        item_indx = int(item.attrib["index"]) - 1
                        item_name = item.attrib.get("value", None)
                        if item_name is None:
                            warnings.warn(
                                f"An item without a value at index {item_indx} for issue {issue_key}",
                                warnings.NegmasIOWarning,
                            )
                            continue
                        # may be I do not need this
                        if current_issue.is_integer():
                            item_name = int(item_name)
                        if current_issue.is_float():
                            item_name = float(item_name)
                        if not current_issue.is_valid(item_name):
                            raise ValueError(
                                f"Value {item_name} is not in the domain issue values: "
                                f"{current_issue.values}"
                            )
                        val = item.attrib.get("evaluation", None)
                        if val is None:
                            raise ValueError(
                                f"Item {item_name} of issue {issue_key} has no evaluation attribute!!"
                            )
                        float(val)
                        found_issues[issue_key][item_name] = float(val)
                        found_values = True
                        issue_info[issue_key]["map_type"] = "dict"
                    elif item.tag == "evaluator":
                        _f, _name = make_fun_from_xml(item)
                        found_issues[issue_key] = _f
                        issue_info[issue_key]["map_type"] = _name
                        found_values = True
                if not found_values and issue_key in found_issues.keys():
                    found_issues.pop(issue_key, None)

        # add utilities specified not as hyper-rectangles
        if not all_numeric and all(_.is_numeric() for _ in issues):
            raise ValueError(
                "Some found issues are not numeric but all input issues are"
            )
        u = None
        if len(found_issues) > 0:
            if all_numeric:
                slopes, biases, ws = [], [], []
                for key in (_.name for _ in issues):
                    if key in found_issues:
                        slopes.append(found_issues[key].slope)
                        biases.append(found_issues[key].bias)
                    else:
                        slopes.append(0.0)
                        biases.append(0.0)
                    ws.append(weights.get(key, 1.0))
                bias = 0.0
                for b, w in zip(biases, ws):
                    bias += b * w
                for i, s in enumerate(slopes):
                    ws[i] *= s

                u = AffineUtilityFunction(
                    weights=ws,
                    outcome_space=make_os(ordered_issues),
                    bias=bias + global_bias,
                )
            else:
                u = LinearAdditiveUtilityFunction(
                    values=found_issues,
                    weights=weights,
                    outcome_space=make_os(ordered_issues),
                    bias=global_bias,
                )

        if len(rects) > 0:
            uhyper = HyperRectangleUtilityFunction(
                outcome_ranges=rects,
                utilities=rect_utils,
                name=name,
                outcome_space=make_os(ordered_issues),
                bias=global_bias,
            )
            if u is None:
                u = uhyper
            else:
                u = WeightedUtilityFunction(
                    ufuns=[u, uhyper],
                    weights=[1.0, 1.0],
                    name=name,
                    outcome_space=make_os(ordered_issues),
                )
        if u is None:
            raise ValueError("No issues found")
        if not ignore_reserved:
            u.reserved_value = reserved_value
        u.name = name
        # if not ignore_discount and discount_factor != 0.0:
        #     from negmas.preferences.discounted import ExpDiscountedUFun
        #     u = ExpDiscountedUFun(ufun=u, discount=discount_factor, name=name)
        if ignore_discount:
            discount_factor = None
        return u, discount_factor

    @classmethod
    def from_genius(
        cls, file_name: PathLike | str, **kwargs
    ) -> tuple[BaseUtilityFunction | None, float | None]:
        """Imports a utility function from a GENIUS XML file.

        Args:

            file_name (str): File name to import from

        Returns:

            A utility function object (depending on the input file)


        Examples:

            >>> from negmas.preferences import UtilityFunction
            >>> import pkg_resources
            >>> from negmas.inout import load_genius_domain
            >>> domain = load_genius_domain(
            ...     pkg_resources.resource_filename(
            ...         "negmas", resource_name="tests/data/Laptop/Laptop-C-domain.xml"
            ...     )
            ... )
            >>> u, d = UtilityFunction.from_genius(
            ...     file_name=pkg_resources.resource_filename(
            ...         "negmas", resource_name="tests/data/Laptop/Laptop-C-prof1.xml"
            ...     ),
            ...     issues=domain.issues,
            ... )
            >>> u.__class__.__name__
            'LinearAdditiveUtilityFunction'
            >>> u.reserved_value
            0.0
            >>> d
            1.0

        Remarks:
            See ``from_xml_str`` for all the parameters

        """
        if "name" not in kwargs:
            kwargs["name"] = str(Path(file_name).stem)
        with open(file_name) as f:
            xml_str = f.read()
        u, x = cls.from_xml_str(xml_str=xml_str, **kwargs)
        if u is not None:
            u.path = file_name
        return u, x

    @classmethod
    def from_geniusweb_json_str(
        cls,
        json_str: str | dict,
        safe_parsing=True,
        issues: Iterable[Issue] | Sequence[Issue] | None = None,
        ignore_discount=False,
        ignore_reserved=False,
        use_reserved_outcome=False,
        name: str | None = None,
    ) -> tuple[BaseUtilityFunction | None, float | None]:
        """Imports a utility function from a GeniusWeb JSON string.

        Args:

            json_str (str): The string containing GENIUS style XML utility function definition
            issues (Sequence[Issue] | None): Optional issue space to confirm that the utility function is valid
            product of all issues in the input
            safe_parsing (bool): Turn on extra checks

        Returns:

            A utility function object (depending on the input file)

        """
        from negmas.preferences.crisp.linear import LinearAdditiveUtilityFunction

        _ = safe_parsing

        if isinstance(json_str, str):
            d = json.loads(json_str)
        else:
            d = json_str
        reserved_outcome, discount_factor, u = None, 1.0, None
        if "LinearAdditiveUtilitySpace" in d.keys():
            udict = d["LinearAdditiveUtilitySpace"]
            domain = (
                make_os(issues_from_geniusweb_json(udict["domain"])[0])
                if "domain" in udict.keys()
                else None
            )
            if domain is None and issues is not None:
                domain = make_os(tuple(issues))
            discount_factor = (
                udict.get("discount_factor", 1.0) if not ignore_discount else 1.0
            )
            reserved_value = udict.get("reserved_value", None)
            uname = udict.get("name", name)
            reserved_dict = udict.get("reservationBid", dict()).get("issuevalues", None)
            if reserved_dict and not ignore_reserved:
                reserved_outcome = dict2outcome(reserved_dict, issues=domain.issues)  # type: ignore
            weights = udict.get("issueWeights", None)
            utils = udict.get("issueUtilities", dict())
            values = dict()
            for iname, idict in utils.items():
                vals = idict.get("discreteutils", dict()).get("valueUtilities", dict())
                values[iname] = TableFun(vals)
            u = LinearAdditiveUtilityFunction(
                values=values,
                weights=weights,
                bias=0.0,
                name=uname,
                reserved_outcome=None,
                reserved_value=None,
                outcome_space=domain,
            )
            if not ignore_reserved:
                if reserved_outcome and not use_reserved_outcome:
                    reserved_value = u(reserved_outcome)
                if use_reserved_outcome:
                    u.reserved_outcome = reserved_outcome  # type: ignore
                u.reserved_value = reserved_value  # type: ignore

        return u, discount_factor

    @classmethod
    def from_geniusweb(
        cls, file_name: PathLike | str, **kwargs
    ) -> tuple[BaseUtilityFunction | None, float | None]:
        """Imports a utility function from a GeniusWeb json file.

        Args:

            file_name (str): File name to import from

        Returns:

            A utility function object (depending on the input file)

        Remarks:
            See ``from_geniusweb_json_str`` for all the parameters

        """
        kwargs["name"] = str(file_name)
        with open(file_name) as f:
            xml_str = f.read()
        u, x = cls.from_geniusweb_json_str(json_str=xml_str, **kwargs)
        if u is not None:
            u.path = Path(file_name)
        return u, x

    def to_xml_str(
        self, issues: Iterable[Issue] | None = None, discount_factor=None
    ) -> str:
        """
        Exports a utility function to a well formatted string
        """
        if not hasattr(self, "xml"):
            raise ValueError(
                f"ufun of type {self.__class__.__name__} has no xml() member and cannot be saved to XML string\nThe ufun params: {self.to_dict()}"
            )
        if issues is None:
            if not isinstance(self.outcome_space, IndependentIssuesOS):
                raise ValueError(
                    "Cannot convert to xml because the outcome-space of the ufun is not a cartesian outcome space"
                )
            issues = self.outcome_space.issues
            n_issues = 0
        else:
            issues = list(issues)
            n_issues = len(issues)
        output = (
            f'<utility_space number_of_issues="{n_issues}">\n'
            f'<objective index="1" etype="objective" type="objective" description="" name="any">\n'
        )

        output += self.xml(issues=issues)  # type: ignore
        if "</objective>" not in output:
            output += "</objective>\n"
            if discount_factor is not None:
                output += f'<discount_factor value="{discount_factor}" />\n'

        # Handle reserved value - correct problematic values before writing to XML
        rv = self.reserved_value
        write_reservation = False
        if "<reservation value" not in output:
            from negmas.preferences.ops import correct_reserved_value

            # Check and correct reserved value if needed (using eps=0.0 for Genius)
            try:
                corrected_rv, was_corrected = correct_reserved_value(
                    rv, self, eps=0.0, warn=True
                )
                if was_corrected:
                    rv = corrected_rv
                    write_reservation = True
                elif rv is not None and rv != float("-inf"):
                    # Normal finite value (and not -inf which Genius uses as default)
                    write_reservation = True
            except Exception as e:
                if rv is not None:
                    warnings.warn(
                        f"Utility function has problematic reserved value ({rv}) but could not correct it: {e}. "
                        f"Skipping reservation value in XML export.",
                        warnings.NegmasUnexpectedValueWarning,
                    )

        if write_reservation:
            output += f'<reservation value="{rv}" />\n'

        if "</utility_space>" not in output:
            output += "</utility_space>\n"
        return output

    def to_genius(
        self, file_name: PathLike | str, issues: Iterable[Issue] | None = None, **kwargs
    ):
        """
        Exports a utility function to a GENIUS XML file.

        Args:

            file_name (str): File name to export to
            u: utility function
            issues: The issues being considered as defined in the domain

        Returns:

            None


        Examples:

            >>> from negmas.preferences import UtilityFunction
            >>> from negmas.inout import load_genius_domain
            >>> import pkg_resources
            >>> domain = load_genius_domain(
            ...     domain_file_name=pkg_resources.resource_filename(
            ...         "negmas", resource_name="tests/data/Laptop/Laptop-C-domain.xml"
            ...     )
            ... )
            >>> u, d = UtilityFunction.from_genius(
            ...     file_name=pkg_resources.resource_filename(
            ...         "negmas", resource_name="tests/data/Laptop/Laptop-C-prof1.xml"
            ...     ),
            ...     issues=domain.issues,
            ... )
            >>> u.to_genius(
            ...     discount_factor=d,
            ...     file_name=pkg_resources.resource_filename(
            ...         "negmas",
            ...         resource_name="tests/data/LaptopConv/Laptop-C-prof1.xml",
            ...     ),
            ...     issues=domain.issues,
            ... )

        Remarks:
            See ``to_xml_str`` for all the parameters

        """
        file_name = Path(file_name).absolute()
        if file_name.suffix == "":
            file_name = file_name.parent / f"{file_name.stem}.xml"
        with open(file_name, "w") as f:
            f.write(self.to_xml_str(issues=issues, **kwargs))

    def difference_prob(
        self, first: Outcome | None, second: Outcome | None
    ) -> Distribution:
        """
        Returns a numeric difference between the utility of the two given outcomes
        """
        f, s = self(first), self(second)
        if not isinstance(f, Distribution):
            f = Real(f)
        if not isinstance(s, Distribution):
            s = Real(s)
        return f - s

    def is_not_worse(self, first: Outcome | None, second: Outcome | None) -> bool:
        """Check if the first outcome is at least as good as the second.

        Args:
            first: The first outcome to compare (None represents no agreement).
            second: The second outcome to compare (None represents no agreement).

        Returns:
            True if the utility of first is greater than or equal to the utility of second.
        """
        return self.difference_prob(first, second) >= 0.0

    def difference(self, first: Outcome | None, second: Outcome | None) -> float:
        """
        Returns a numeric difference between the utility of the two given outcomes
        """
        return float(self(first)) - float(self(second))

    def __call__(self, offer: Outcome | None) -> Value:
        """
        Calculate the utility for a given outcome at the given negotiation state.

        Args:
            offer: The offer to be evaluated.


        Remarks:

            - It calls the abstract method `eval` after opationally adjusting the
              outcome type.
            - It is preferred to override eval instead of directly overriding this method
            - You cannot return None from overriden eval() functions but raise an exception (ValueError) if it was
              not possible to calculate the Value.
            - Return a float from your `eval` implementation.
            - Return the reserved value if the offer was None

        Returns:
            The utility of the given outcome
        """
        if offer is None:
            return self.reserved_value  # type: ignore I know that concrete subclasses will be returning the correct type
        if (
            self._invalid_value is not None
            and self.outcome_space
            and offer not in self.outcome_space
        ):
            return self._invalid_value
        return self.eval(offer)


class _FullyStatic:
    """
    Used internally to indicate that the ufun can **NEVER** change due to anything.
    """

    def is_session_dependent(self) -> bool:
        """Return False since fully static ufuns never depend on the negotiation session."""
        return False

    def is_volatile(self) -> bool:
        """Return False since fully static ufuns never change between calls."""
        return False

    def is_state_dependent(self) -> bool:
        """Return False since fully static ufuns never depend on the negotiation state."""
        return False

    def is_stationary(self) -> bool:
        """Return True since fully static ufuns are always stationary."""
        return True


class _ExtremelyDynamic:
    """
    Used internally to indicate that the ufun can change due to anything.
    """

    def is_session_dependent(self) -> bool:
        """Return True since extremely dynamic ufuns depend on the negotiation session."""
        return True

    def is_volatile(self) -> bool:
        """Return True since extremely dynamic ufuns can change between calls."""
        return True

    def is_state_dependent(self) -> bool:
        """Return True since extremely dynamic ufuns depend on the negotiation state."""
        return True

    def is_stationary(self) -> bool:
        """Return False since extremely dynamic ufuns are never stationary."""
        return False
