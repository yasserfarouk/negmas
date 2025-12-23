"""Module for Polynomial Aggregation Utility Function."""

from __future__ import annotations

import random
from functools import partial
from math import pow
from typing import TYPE_CHECKING, Iterable, Sequence

from negmas.helpers import get_full_type_name
from negmas.helpers.numeric import make_range
from negmas.outcomes import Issue, Outcome
from negmas.outcomes.base_issue import DiscreteIssue
from negmas.outcomes.outcome_space import CartesianOutcomeSpace
from negmas.outcomes.protocols import IndependentIssuesOS
from negmas.preferences.protocols import SingleIssueFun
from negmas.serialization import PYTHON_CLASS_IDENTIFIER, deserialize, serialize

from ..crisp_ufun import UtilityFunction
from ..mixins import StationaryMixin
from ..value_fun import IdentityFun, LambdaFun, TableFun, BaseFun

if TYPE_CHECKING:
    pass

__all__ = ["PAUtilityFunction"]


def _rand_mapping_normalized(x, mx, mn, r):
    if mx == mn:
        return mx
    return r * (x - mn) / (mx - mn)


def _rand_mapping(x, r):
    return (r - 0.5) * x


def _random_mapping(issue: Issue, normalized=False):
    r = random.random()
    if issue.is_numeric():
        return (
            partial(
                _rand_mapping_normalized, mx=issue.max_value, mn=issue.min_value, r=r
            )
            if normalized
            else partial(_rand_mapping, r=r)
        )
    if isinstance(issue, DiscreteIssue):
        return dict(
            zip(
                issue.all,
                [
                    random.uniform(0.1, 1.0) if normalized else random.random() - 0.5
                    for _ in range(issue.cardinality)
                ],
            )
        )
    return (
        partial(_rand_mapping_normalized, mx=issue.max_value, mn=issue.min_value, r=r)
        if normalized
        else partial(_rand_mapping, r=r)
    )


class PAUtilityFunction(StationaryMixin, UtilityFunction):
    r"""A Polynomial Aggregation Utility Function.

    Like `LinearAdditiveUtilityFunction`, this has one value function per issue.
    However, instead of linearly aggregating the value function outputs, this class
    uses polynomial expansion where each term is a product of value function outputs
    raised to specified powers.

    Mathematical definition:

    .. math::

        u(\omega) = b + \sum_{k} c_k \prod_{i=1}^{n} v_i(\omega_i)^{p_{ki}}

    where:
        - :math:`b` is the bias
        - :math:`c_k` is the coefficient for term :math:`k`
        - :math:`v_i` is the value function for issue :math:`i`
        - :math:`\omega_i` is the value of issue :math:`i` in outcome :math:`\omega`
        - :math:`p_{ki}` is the power of issue :math:`i`'s value in term :math:`k`

    Args:
        values: Value functions for individual issues (one per issue).
        terms: A list of polynomial terms, where each term is a tuple of:
            - coefficient: The coefficient for this term
            - powers: A tuple of powers for each value function (length must match number of issues)
        bias: A constant bias term added to the final utility
        name: Name of the utility function

    Examples:
        >>> from negmas.outcomes import make_issue
        >>> issues = [make_issue(10, "A"), make_issue(10, "B"), make_issue(10, "C")]

        Define value functions (one per issue):

        >>> values = [
        ...     lambda x: float(x),  # v_A = A
        ...     lambda x: float(x),  # v_B = B
        ...     lambda x: float(x),  # v_C = C
        ... ]

        Define polynomial: u = v_A^3 + v_B^2 * v_C + v_A * v_B * v_C^2

        >>> terms = [
        ...     (1.0, (3, 0, 0)),  # v_A^3
        ...     (1.0, (0, 2, 1)),  # v_B^2 * v_C
        ...     (1.0, (1, 1, 2)),  # v_A * v_B * v_C^2
        ... ]
        >>> f = PAUtilityFunction(values=values, terms=terms, issues=issues)

        Evaluate: for outcome (2, 3, 4): v_A=2, v_B=3, v_C=4
        u = 2^3 + 3^2*4 + 2*3*4^2 = 8 + 36 + 96 = 140

        >>> f((2, 3, 4))
        140.0

    Note:
        - This is a generalization of LinearAdditiveUtilityFunction where
          linear aggregation corresponds to terms with single non-zero powers of 1.
        - For linear behavior, use terms like [(w1, (1,0,0)), (w2, (0,1,0)), ...]
    """

    def __init__(
        self,
        values: (
            dict[str, SingleIssueFun]
            | tuple[SingleIssueFun, ...]
            | list[SingleIssueFun]
        ),
        terms: Sequence[tuple[float, tuple[int, ...]]],
        bias: float = 0.0,
        *args,
        **kwargs,
    ) -> None:
        """Initialize the PAUtilityFunction.

        Args:
            values: Value functions for individual issues.
            terms: List of (coefficient, powers) tuples defining polynomial terms.
            bias: Constant bias term.
            *args: Additional positional arguments for parent class.
            **kwargs: Additional keyword arguments for parent class.
        """
        super().__init__(*args, **kwargs)

        if self.outcome_space and not isinstance(
            self.outcome_space, IndependentIssuesOS
        ):
            raise ValueError(
                f"Cannot create {self.type} ufun with an outcome space without "
                f"independent issues.\nGiven OS: {self.outcome_space} of type "
                f"{type(self.outcome_space)}\nGiven args {kwargs}"
            )

        self.issues: list[Issue] | None = (
            list(self.outcome_space.issues) if self.outcome_space else None  # type: ignore
        )

        # Process values (same as LinearAdditiveUtilityFunction)
        if isinstance(values, dict):
            if self.issues is None:
                raise ValueError("Must specify issues when passing `values` as a dict")
            values = [
                values.get(_, IdentityFun())  # type: ignore
                for _ in [i.name if isinstance(i, Issue) else i for i in self.issues]
            ]
        else:
            values = list(values)

        self._values: list[BaseFun] = []
        for i, v in enumerate(values):
            if isinstance(v, BaseFun):
                self._values.append(v)
            elif isinstance(v, dict):
                self._values.append(TableFun(v))
            elif callable(v):
                self._values.append(LambdaFun(v))
            elif isinstance(v, Iterable):
                if (
                    not self.issues
                    or len(self.issues) < i + 1
                    or not self.issues[i].is_discrete()
                ):
                    raise TypeError(
                        "When passing an iterable as the value function for an issue, "
                        "the issue MUST be discrete"
                    )
                d = dict(zip(self.issues[i].enumerate(), v))  # type: ignore
                self._values.append(TableFun(d))
            else:
                raise TypeError(
                    f"Value {v} is not supported: It is of type ({type(v)}) "
                    f"but we only support BaseFun, Dict or Callable"
                )

        # Validate and store terms
        n_issues = len(self._values)
        self._terms: list[tuple[float, tuple[int, ...]]] = []
        for coef, powers in terms:
            if len(powers) != n_issues:
                raise ValueError(
                    f"Term powers length ({len(powers)}) must match number of issues ({n_issues})"
                )
            self._terms.append((float(coef), tuple(powers)))

        self._bias = bias

    @property
    def values(self) -> list[BaseFun]:
        """Return the value functions."""
        return self._values

    @property
    def terms(self) -> list[tuple[float, tuple[int, ...]]]:
        """Return the polynomial terms."""
        return self._terms

    @property
    def bias(self) -> float:
        """Return the bias term."""
        return self._bias

    def eval(self, offer: Outcome | None) -> float:
        """Evaluate the utility of an outcome.

        Args:
            offer: The outcome to evaluate.

        Returns:
            The utility value.
        """
        if offer is None:
            return self.reserved_value

        # First, evaluate all value functions
        value_outputs: list[float] = []
        for i, vfun in enumerate(self._values):
            try:
                v = vfun(offer[i])
                if v is None:
                    return float("nan")
                value_outputs.append(v)
            except (KeyError, TypeError, FloatingPointError):
                return float("nan")

        # Now evaluate the polynomial
        u = self._bias
        for coef, powers in self._terms:
            term_value = coef
            for i, p in enumerate(powers):
                if p != 0:
                    term_value *= pow(value_outputs[i], p)
            u += term_value

        return u

    def xml(self, issues: list[Issue] | None = None) -> str:
        """Generate an XML string representing the utility function.

        Raises:
            NotImplementedError: PA utility functions cannot be represented in GENIUS XML format.
        """
        raise NotImplementedError(
            f"Cannot convert {self.__class__.__name__} to XML. "
            "GENIUS XML format does not support polynomial aggregation."
        )

    def to_dict(self, python_class_identifier=PYTHON_CLASS_IDENTIFIER) -> dict:
        """Convert to dictionary for serialization."""
        d = {python_class_identifier: get_full_type_name(type(self))}
        d.update(super().to_dict(python_class_identifier=python_class_identifier))

        return dict(
            **d,
            values=serialize(
                self._values, python_class_identifier=python_class_identifier
            ),
            terms=self._terms,
            bias=self._bias,
        )

    @classmethod
    def from_dict(
        cls, d: dict, python_class_identifier=PYTHON_CLASS_IDENTIFIER
    ) -> PAUtilityFunction:
        """Create from dictionary."""
        if isinstance(d, cls):
            return d

        d = dict(d)
        d.pop(python_class_identifier, None)

        # Backup terms before general deserialization
        terms_backup = d.pop("terms", None)
        d = deserialize(  # type: ignore[assignment]
            d,
            deep=True,
            remove_type_field=True,
            python_class_identifier=python_class_identifier,
        )
        if not isinstance(d, dict):
            raise TypeError(f"Expected dict after deserialization, got {type(d)}")
        if terms_backup is not None:
            d["terms"] = [(c, tuple(p)) for c, p in terms_backup]

        return cls(**d)

    @classmethod
    def random(
        cls,
        outcome_space: CartesianOutcomeSpace | None = None,
        issues: list[Issue] | tuple[Issue, ...] | None = None,
        reserved_value: tuple[float, float] = (0.0, 1.0),
        normalized: bool = True,
        n_terms: int | tuple[int, int] = (2, 5),
        max_power: int = 3,
        **kwargs,
    ) -> PAUtilityFunction:
        """Generate a random PA utility function.

        Args:
            outcome_space: The outcome space.
            issues: The issues (used if outcome_space is not provided).
            reserved_value: Range for random reserved value.
            normalized: Whether to normalize coefficients.
            n_terms: Number of polynomial terms or range (min, max).
            max_power: Maximum power for any value in a term.
            **kwargs: Additional arguments passed to the constructor.

        Returns:
            A random PAUtilityFunction.
        """
        if not issues and outcome_space:
            issues = outcome_space.issues
        if not issues:
            raise ValueError("Cannot generate a random ufun without knowing the issues")

        issues = list(issues)
        n_issues = len(issues)
        reserved_value = make_range(reserved_value)

        # Determine number of terms
        if isinstance(n_terms, tuple):
            num_terms = random.randint(n_terms[0], n_terms[1])
        else:
            num_terms = n_terms

        # Create random value functions (one per issue)
        values = [_random_mapping(issue, normalized) for issue in issues]

        # Create random polynomial terms
        terms: list[tuple[float, tuple[int, ...]]] = []
        for _ in range(num_terms):
            coef = random.uniform(-1.0, 1.0)
            # Random powers, but ensure at least one issue has power > 0
            powers = [random.randint(0, max_power) for _ in range(n_issues)]
            if all(p == 0 for p in powers):
                powers[random.randint(0, n_issues - 1)] = random.randint(1, max_power)
            terms.append((coef, tuple(powers)))

        # Normalize coefficients if requested
        if normalized and terms:
            total = sum(abs(c) for c, _ in terms)
            if total > 0:
                terms = [(c / total, p) for c, p in terms]

        r = (
            random.random() * (reserved_value[1] - reserved_value[0])
            + reserved_value[0]
        )

        return cls(
            values=values,  # type: ignore
            terms=terms,
            bias=0.0,
            issues=issues,
            reserved_value=r,
            **kwargs,
        )

    def shift_by(self, offset: float, shift_reserved: bool = True) -> PAUtilityFunction:
        """Create a shifted version of this utility function."""
        return PAUtilityFunction(
            values=self._values,  # type: ignore
            terms=list(self._terms),
            bias=self._bias + offset,
            outcome_space=self.outcome_space,
            name=self.name,
            reserved_value=(
                self.reserved_value + offset if shift_reserved else self.reserved_value
            ),
        )

    def scale_by(self, scale: float, scale_reserved: bool = True) -> PAUtilityFunction:
        """Create a scaled version of this utility function."""
        if scale < 0:
            raise ValueError(f"Cannot have a negative scale: {scale}")

        new_terms = [(c * scale, p) for c, p in self._terms]

        return PAUtilityFunction(
            values=self._values,  # type: ignore
            terms=new_terms,
            bias=self._bias * scale,
            outcome_space=self.outcome_space,
            name=self.name,
            reserved_value=(
                self.reserved_value * scale if scale_reserved else self.reserved_value
            ),
        )

    def __str__(self) -> str:
        """Return string representation."""
        value_strs = []
        for i, vfun in enumerate(self._values):
            value_strs.append(f"    v{i}: {type(vfun).__name__}")
        values_str = "\n".join(value_strs)

        term_strs = []
        for coef, powers in self._terms:
            parts = []
            for i, p in enumerate(powers):
                if p > 0:
                    if p == 1:
                        parts.append(f"v{i}")
                    else:
                        parts.append(f"v{i}^{p}")
            term_str = " * ".join(parts) if parts else "1"
            term_strs.append(f"    {coef:+.3f} * {term_str}")
        terms_str = "\n".join(term_strs)

        return (
            f"PAUtilityFunction(\n"
            f"  bias={self._bias},\n"
            f"  values=[\n{values_str}\n  ],\n"
            f"  terms=[\n{terms_str}\n  ]\n)"
        )
