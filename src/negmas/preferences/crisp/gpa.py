"""Module for Generalized Polynomial Aggregation Utility Function."""

from __future__ import annotations

import itertools
from math import pow
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

from negmas.helpers import get_full_type_name
from negmas.outcomes import Issue, Outcome
from negmas.outcomes.outcome_space import CartesianOutcomeSpace
from negmas.outcomes.protocols import IndependentIssuesOS
from negmas.serialization import PYTHON_CLASS_IDENTIFIER, deserialize, serialize

from ..crisp_ufun import UtilityFunction
from ..mixins import StationaryMixin
from ..value_fun import (
    AffineFun,
    AffineMultiFun,
    BaseFun,
    BaseMultiFun,
    LambdaFun,
    LambdaMultiFun,
    TableFun,
    TableMultiFun,
)

if TYPE_CHECKING:
    pass

__all__ = ["GPAUtilityFunction"]

# Type alias for factor functions - can be single or multi-issue functions
FactorFun = (
    BaseFun | BaseMultiFun | Callable[[Any], float] | Mapping[Any, float] | float
)


class GPAUtilityFunction(StationaryMixin, UtilityFunction):
    r"""A Generalized Polynomial Aggregation Utility Function.

    Like `GLAUtilityFunction`, each factor can operate on any subset of issues.
    However, instead of linearly aggregating factor results, this class uses
    polynomial expansion where each term is a product of factor values raised
    to specified powers.

    Mathematical definition:

    .. math::

        u(\omega) = b + \sum_{k} c_k \prod_{j} f_j(\omega_{I_j})^{p_{kj}}

    where:
        - :math:`b` is the bias
        - :math:`c_k` is the coefficient for term :math:`k`
        - :math:`f_j` is the factor function for factor :math:`j`
        - :math:`\omega_{I_j}` are the issue values for factor :math:`j`
        - :math:`p_{kj}` is the power of factor :math:`j` in term :math:`k`

    Args:
        factors: A list of tuples, where each tuple contains:
            - issue_indices: A tuple of issue indices (or names) that the factor operates on
            - function: The factor function (can be BaseFun, Callable, Mapping, or float)
        terms: A list of polynomial terms, where each term is a tuple of:
            - coefficient: The coefficient for this term
            - powers: A tuple of powers for each factor (length must match number of factors)
        bias: A constant bias term added to the final utility
        name: Name of the utility function

    Examples:
        >>> from negmas.outcomes import make_issue
        >>> from negmas.preferences.value_fun import AffineFun
        >>> issues = [make_issue(10, "A"), make_issue(10, "B"), make_issue(10, "C")]

        Create factors a, b, c based on issues:

        >>> factors = [
        ...     (("A",), AffineFun(slope=1.0, bias=0.0)),  # a = A
        ...     (("B",), AffineFun(slope=1.0, bias=0.0)),  # b = B
        ...     (("C",), AffineFun(slope=1.0, bias=0.0)),  # c = C
        ... ]

        Define polynomial: u = a^3 + b^2*c + a*b*c^2

        >>> terms = [
        ...     (1.0, (3, 0, 0)),  # a^3
        ...     (1.0, (0, 2, 1)),  # b^2 * c
        ...     (1.0, (1, 1, 2)),  # a * b * c^2
        ... ]
        >>> f = GPAUtilityFunction(factors=factors, terms=terms, issues=issues)

        Evaluate: for outcome (2, 3, 4): a=2, b=3, c=4
        u = 2^3 + 3^2*4 + 2*3*4^2 = 8 + 36 + 96 = 140

        >>> f((2, 3, 4))
        140.0

    Remarks:
        - Factor functions can be any of the types supported by GLAUtilityFunction
        - Powers of 0 exclude a factor from a term (factor^0 = 1)
        - For linear aggregation (like GLAUtilityFunction), use powers of (1, 0, 0, ...),
          (0, 1, 0, ...), etc. with appropriate coefficients as weights
    """

    def __init__(
        self,
        factors: Sequence[tuple[tuple[int | str, ...], FactorFun]],
        terms: Sequence[tuple[float, tuple[int, ...]]],
        bias: float = 0.0,
        *args,
        **kwargs,
    ) -> None:
        """Initialize the GPAUtilityFunction.

        Args:
            factors: List of (issue_indices, function) tuples defining factor functions.
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

        # Build issue name to index mapping
        self._name_to_index: dict[str, int] = {}
        if self.issues:
            for i, issue in enumerate(self.issues):
                self._name_to_index[issue.name] = i

        # Process factors
        self._factors: list[tuple[tuple[int, ...], BaseFun | BaseMultiFun]] = []

        for factor_spec in factors:
            issue_spec, func = factor_spec

            # Convert issue names to indices
            indices = self._resolve_issue_indices(issue_spec)

            # Convert function to appropriate type
            processed_func = self._process_factor_function(func, len(indices))

            self._factors.append((indices, processed_func))

        # Validate and store terms
        n_factors = len(self._factors)
        self._terms: list[tuple[float, tuple[int, ...]]] = []
        for coef, powers in terms:
            if len(powers) != n_factors:
                raise ValueError(
                    f"Term powers length ({len(powers)}) must match number of factors ({n_factors})"
                )
            self._terms.append((float(coef), tuple(powers)))

        self._bias = bias

    def _resolve_issue_indices(
        self, issue_spec: tuple[int | str, ...]
    ) -> tuple[int, ...]:
        """Convert issue names/indices to a tuple of integer indices."""
        indices = []
        for spec in issue_spec:
            if isinstance(spec, str):
                if spec not in self._name_to_index:
                    raise ValueError(
                        f"Issue name '{spec}' not found. "
                        f"Available issues: {list(self._name_to_index.keys())}"
                    )
                indices.append(self._name_to_index[spec])
            else:
                indices.append(int(spec))
        return tuple(indices)

    def _process_factor_function(
        self, func: FactorFun, n_issues: int
    ) -> BaseFun | BaseMultiFun:
        """Process a factor function into the appropriate internal type."""
        if isinstance(func, (BaseFun, BaseMultiFun)):
            return func

        if isinstance(func, (float, int)):
            if n_issues == 1:
                return AffineFun(slope=0.0, bias=float(func))
            else:
                return AffineMultiFun(
                    slope=tuple(0.0 for _ in range(n_issues)), bias=float(func)
                )

        if isinstance(func, Mapping):
            if n_issues == 1:
                return TableFun(mapping=dict(func))
            else:
                return TableMultiFun(mapping=dict(func))

        if callable(func):
            if n_issues == 1:
                return LambdaFun(f=func)
            else:
                return LambdaMultiFun(f=func)

        raise TypeError(
            f"Unsupported factor function type: {type(func)}. "
            f"Expected BaseFun, Callable, Mapping, or float."
        )

    @property
    def factors(self) -> list[tuple[tuple[int, ...], BaseFun | BaseMultiFun]]:
        """Return the list of factors."""
        return self._factors

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

        # First, evaluate all factors
        factor_values: list[float] = []
        for indices, func in self._factors:
            if len(indices) == 1:
                values = offer[indices[0]]
            else:
                values = tuple(offer[i] for i in indices)

            try:
                fv = func(values)
                if fv is None:
                    return float("nan")
                factor_values.append(fv)
            except (KeyError, TypeError, FloatingPointError):
                return float("nan")

        # Now evaluate the polynomial
        u = self._bias
        for coef, powers in self._terms:
            term_value = coef
            for j, p in enumerate(powers):
                if p != 0:
                    term_value *= pow(factor_values[j], p)
            u += term_value

        return u

    def xml(self, issues: list[Issue] | None = None) -> str:
        """Generate an XML string representing the utility function.

        Raises:
            NotImplementedError: GPA utility functions cannot be represented in GENIUS XML format.
        """
        raise NotImplementedError(
            f"Cannot convert {self.__class__.__name__} to XML. "
            "GENIUS XML format does not support polynomial aggregation."
        )

    def to_dict(self, python_class_identifier=PYTHON_CLASS_IDENTIFIER) -> dict:
        """Convert to dictionary for serialization."""
        d = {python_class_identifier: get_full_type_name(type(self))}
        d.update(super().to_dict(python_class_identifier=python_class_identifier))

        # Serialize factors
        serialized_factors = []
        for indices, func in self._factors:
            serialized_factors.append(
                (
                    indices,
                    serialize(func, python_class_identifier=python_class_identifier),
                )
            )

        return dict(**d, factors=serialized_factors, terms=self._terms, bias=self._bias)

    @classmethod
    def from_dict(
        cls, d: dict, python_class_identifier=PYTHON_CLASS_IDENTIFIER
    ) -> GPAUtilityFunction:
        """Create from dictionary."""
        if isinstance(d, cls):
            return d

        d = dict(d)
        d.pop(python_class_identifier, None)

        # Deserialize factors
        if "factors" in d:
            factors = []
            for factor_spec in d["factors"]:
                indices, func_dict = factor_spec
                func = deserialize(
                    func_dict, python_class_identifier=python_class_identifier
                )
                factors.append((tuple(indices), func))
            d["factors"] = factors

        # Deserialize remaining fields
        factors_backup = d.pop("factors", None)
        terms_backup = d.pop("terms", None)
        d = deserialize(  # type: ignore[assignment]
            d,
            deep=True,
            remove_type_field=True,
            python_class_identifier=python_class_identifier,
        )
        if not isinstance(d, dict):
            raise TypeError(f"Expected dict after deserialization, got {type(d)}")
        if factors_backup is not None:
            d["factors"] = factors_backup
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
        n_factors: int | tuple[int, int] = (2, 4),
        n_terms: int | tuple[int, int] = (2, 5),
        max_factor_issues: int = 2,
        max_power: int = 3,
        **kwargs,
    ) -> GPAUtilityFunction:
        """Generate a random GPA utility function.

        Args:
            outcome_space: The outcome space.
            issues: The issues (used if outcome_space is not provided).
            reserved_value: Range for random reserved value.
            normalized: Whether to normalize coefficients.
            n_factors: Number of factors or range (min, max).
            n_terms: Number of polynomial terms or range (min, max).
            max_factor_issues: Maximum number of issues per factor.
            max_power: Maximum power for any factor in a term.
            **kwargs: Additional arguments passed to the constructor.

        Returns:
            A random GPAUtilityFunction.
        """
        import random

        from negmas.helpers.numeric import make_range

        if not issues and outcome_space:
            issues = outcome_space.issues
        if not issues:
            raise ValueError("Cannot generate a random ufun without knowing the issues")

        issues = list(issues)
        n_issues = len(issues)
        reserved_value = make_range(reserved_value)

        # Determine number of factors
        if isinstance(n_factors, tuple):
            num_factors = random.randint(n_factors[0], n_factors[1])
        else:
            num_factors = n_factors

        # Determine number of terms
        if isinstance(n_terms, tuple):
            num_terms = random.randint(n_terms[0], n_terms[1])
        else:
            num_terms = n_terms

        # Create random factors
        factors: list[tuple[tuple[int, ...], BaseFun | BaseMultiFun]] = []

        for _ in range(num_factors):
            factor_n_issues = random.randint(1, min(max_factor_issues, n_issues))
            selected_indices = tuple(
                sorted(random.sample(range(n_issues), factor_n_issues))
            )

            func: BaseFun | BaseMultiFun
            if factor_n_issues == 1:
                issue = issues[selected_indices[0]]
                if issue.is_numeric():
                    slope = random.uniform(0.1, 1.0)  # Positive to avoid negative bases
                    func = AffineFun(slope=slope, bias=0.0)
                else:
                    values = list(issue.all)
                    mapping = {v: random.uniform(0.1, 1.0) for v in values}
                    func = TableFun(mapping=mapping)
            else:
                all_numeric = all(issues[i].is_numeric() for i in selected_indices)
                if all_numeric:
                    slope = tuple(random.uniform(0.1, 1.0) for _ in selected_indices)
                    func = AffineMultiFun(slope=slope, bias=0.0)
                else:
                    selected_issues = [issues[i] for i in selected_indices]
                    all_discrete = all(
                        issue.is_discrete() and issue.cardinality <= 10
                        for issue in selected_issues
                    )
                    if all_discrete:
                        combos = list(
                            itertools.product(
                                *[list(issue.all) for issue in selected_issues]
                            )
                        )
                        if len(combos) <= 100:
                            mapping = {c: random.uniform(0.1, 1.0) for c in combos}
                            func = TableMultiFun(mapping=mapping)
                        else:
                            func = LambdaMultiFun(
                                f=lambda x: abs(sum(hash(v) % 100 for v in x)) / 100.0
                                + 0.1
                            )
                    else:
                        func = LambdaMultiFun(
                            f=lambda x: abs(
                                sum(
                                    float(v)
                                    if isinstance(v, (int, float))
                                    else hash(v) % 10
                                    for v in x
                                )
                            )
                            / len(x)
                            + 0.1
                        )

            factors.append((selected_indices, func))

        # Create random polynomial terms
        terms: list[tuple[float, tuple[int, ...]]] = []
        for _ in range(num_terms):
            coef = random.uniform(-1.0, 1.0)
            # Random powers, but ensure at least one factor has power > 0
            powers = [random.randint(0, max_power) for _ in range(num_factors)]
            if all(p == 0 for p in powers):
                powers[random.randint(0, num_factors - 1)] = random.randint(
                    1, max_power
                )
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
            factors=factors,
            terms=terms,
            bias=0.0,
            issues=issues,
            reserved_value=r,
            **kwargs,
        )

    def shift_by(
        self, offset: float, shift_reserved: bool = True
    ) -> GPAUtilityFunction:
        """Create a shifted version of this utility function."""
        return GPAUtilityFunction(
            factors=list(self._factors),
            terms=list(self._terms),
            bias=self._bias + offset,
            outcome_space=self.outcome_space,
            name=self.name,
            reserved_value=(
                self.reserved_value + offset if shift_reserved else self.reserved_value
            ),
        )

    def scale_by(self, scale: float, scale_reserved: bool = True) -> GPAUtilityFunction:
        """Create a scaled version of this utility function."""
        if scale < 0:
            raise ValueError(f"Cannot have a negative scale: {scale}")

        new_terms = [(c * scale, p) for c, p in self._terms]

        return GPAUtilityFunction(
            factors=list(self._factors),
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
        factor_strs = []
        for i, (indices, func) in enumerate(self._factors):
            factor_strs.append(f"    f{i}: {indices} -> {type(func).__name__}")
        factors_str = "\n".join(factor_strs)

        term_strs = []
        for coef, powers in self._terms:
            parts = []
            for j, p in enumerate(powers):
                if p > 0:
                    if p == 1:
                        parts.append(f"f{j}")
                    else:
                        parts.append(f"f{j}^{p}")
            term_str = " * ".join(parts) if parts else "1"
            term_strs.append(f"    {coef:+.3f} * {term_str}")
        terms_str = "\n".join(term_strs)

        return (
            f"GPAUtilityFunction(\n"
            f"  bias={self._bias},\n"
            f"  factors=[\n{factors_str}\n  ],\n"
            f"  terms=[\n{terms_str}\n  ]\n)"
        )
