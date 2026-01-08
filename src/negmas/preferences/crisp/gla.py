"""Module for Generalized Linear Aggregation Utility Function."""

from __future__ import annotations

import itertools
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

__all__ = ["GLAUtilityFunction"]

# Type alias for factor functions - can be single or multi-issue functions
FactorFun = (
    BaseFun | BaseMultiFun | Callable[[Any], float] | Mapping[Any, float] | float
)


class GLAUtilityFunction(StationaryMixin, UtilityFunction):
    r"""A Generalized Linear Aggregation Utility Function.

    Unlike `LinearAdditiveUtilityFunction` which has one function per issue,
    this class allows each factor to operate on any subset of issues.

    Args:
        factors: A list of tuples, where each tuple contains:
            - issue_indices: A tuple of issue indices (or names) that the factor operates on
            - function: The factor function (can be BaseFun subclass,
              Callable, Mapping/dict, or a constant float)
        weights: Optional weights for combining factors (defaults to 1.0 for each)
        bias: A constant bias term added to the final utility
        name: Name of the utility function

    Notes:
        The utility value is calculated as:

        .. math::

            u = b + \sum_{k=0}^{n_{factors}-1} {w_k * f_k(\omega_{I_k})}

        where $b$ is the bias, $w_k$ is the weight for factor $k$, $f_k$ is the
        factor function, and $\omega_{I_k}$ represents the values of issues in
        the index set $I_k$ for outcome $\omega$.

    Examples:
        >>> from negmas.outcomes import make_issue
        >>> from negmas.preferences.value_fun import LinearMultiFun
        >>> issues = [
        ...     make_issue(10, "A"),
        ...     make_issue(5, "B"),
        ...     make_issue(["x", "y", "z"], "C"),
        ...     make_issue((0.0, 1.0), "D"),
        ...     make_issue(4, "E"),
        ... ]

        Create a GLA utility function with various factor types:

        >>> f = GLAUtilityFunction(
        ...     factors=[
        ...         # Factor on issue A (affine function)
        ...         (("A",), AffineFun(slope=2.0, bias=1.0)),
        ...         # Factor on issues A and B (dict mapping)
        ...         (("A", "B"), {(0, 0): 1.0, (0, 1): 2.0, (1, 0): 3.0, (1, 1): 4.0}),
        ...         # Factor on issues A, C, E (lambda function)
        ...         (("A", "C", "E"), lambda x: x[0] + (1 if x[1] == "x" else 0) + x[2]),
        ...         # Factor on issue D only
        ...         (("D",), lambda x: x * 10),
        ...         # Factor on issues B and C (multivariate polynomial-like)
        ...         (("B", "C"), AffineMultiFun(slope=(1.0, 0.0), bias=0.5)),
        ...         # Constant factor (interaction term for D and A)
        ...         (("D", "A"), LinearMultiFun(slope=(1.0, 0.5))),
        ...     ],
        ...     weights=[1.0, 0.5, 2.0, 1.0, 0.3, 0.2],
        ...     issues=issues,
        ... )

        Evaluate the utility of an outcome:

        >>> outcome = (2, 1, "x", 0.5, 3)  # A=2, B=1, C="x", D=0.5, E=3
        >>> u = f(outcome)
        >>> isinstance(u, float)
        True

    Remarks:
        - Factor functions can be:
            - Single-issue functions: AffineFun, LinearFun, TableFun, LambdaFun, etc.
            - Multi-issue functions: TableMultiFun, AffineMultiFun, LambdaMultiFun, etc.
            - `Callable`: Any callable taking a value (single issue) or tuple (multi-issue)
            - `Mapping`/`dict`: A mapping from values/tuples to utilities
            - `float`: A constant value (useful for bias terms on specific issue combinations)
        - Issue indices can be integers or issue names (strings)
        - The outcome space must have independent issues (CartesianOutcomeSpace)
    """

    def __init__(
        self,
        factors: Sequence[
            tuple[tuple[int | str, ...], FactorFun]
            | tuple[tuple[int | str, ...], FactorFun, float]
        ],
        weights: Sequence[float] | None = None,
        bias: float = 0.0,
        *args,
        **kwargs,
    ) -> None:
        """Initialize the GLAUtilityFunction.

        Args:
            factors: List of (issue_indices, function) or (issue_indices, function, weight) tuples.
            weights: Optional weights for each factor. If None, all weights are 1.0.
                     If factors include weights as third element, those override this.
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
        self._weights: list[float] = []

        default_weights = weights if weights is not None else [1.0] * len(factors)

        for i, factor_spec in enumerate(factors):
            if len(factor_spec) == 3:
                issue_spec, func, weight = factor_spec  # type: ignore
            else:
                issue_spec, func = factor_spec  # type: ignore
                weight = default_weights[i] if i < len(default_weights) else 1.0

            # Convert issue names to indices
            indices = self._resolve_issue_indices(issue_spec)

            # Convert function to appropriate type
            processed_func = self._process_factor_function(func, len(indices))

            self._factors.append((indices, processed_func))
            self._weights.append(weight)

        self._bias = bias

    def _resolve_issue_indices(
        self, issue_spec: tuple[int | str, ...]
    ) -> tuple[int, ...]:
        """Convert issue names/indices to a tuple of integer indices.

        Args:
            issue_spec: Tuple of issue names or indices.

        Returns:
            Tuple of integer indices.
        """
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
        """Process a factor function into the appropriate internal type.

        Args:
            func: The factor function to process.
            n_issues: Number of issues this factor operates on.

        Returns:
            A BaseFun instance (single or multi-issue).
        """
        if isinstance(func, (BaseFun, BaseMultiFun)):
            return func

        if isinstance(func, (float, int)):
            # Constant function
            if n_issues == 1:
                return AffineFun(slope=0.0, bias=float(func))
            else:
                return AffineMultiFun(
                    slope=tuple(0.0 for _ in range(n_issues)), bias=float(func)
                )

        if isinstance(func, Mapping):
            # Dictionary/mapping
            if n_issues == 1:
                return TableFun(mapping=dict(func))
            else:
                # For multi-issue, keys should be tuples
                return TableMultiFun(mapping=dict(func))

        if callable(func):
            # Lambda or callable
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
    def weights(self) -> list[float]:
        """Return the weights."""
        return self._weights

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

        u = self._bias

        for (indices, func), weight in zip(self._factors, self._weights):
            # Extract the relevant values from the outcome
            if len(indices) == 1:
                values = offer[indices[0]]
            else:
                values = tuple(offer[i] for i in indices)

            # Evaluate the factor function
            try:
                factor_value = func(values)
                if factor_value is None:
                    return float("nan")
                u += weight * factor_value
            except (KeyError, TypeError, FloatingPointError):
                return float("nan")

        return u

    def xml(self, issues: list[Issue] | None = None) -> str:
        """Generate an XML string representing the utility function.

        Args:
            issues: The issues (uses self.issues if not provided).

        Returns:
            XML string representation.

        Raises:
            NotImplementedError: GLA utility functions cannot be fully represented in GENIUS XML format.
        """
        raise NotImplementedError(
            f"Cannot convert {self.__class__.__name__} to XML. "
            "GENIUS XML format does not support generalized linear aggregation with "
            "multi-issue factors. Consider using LinearAdditiveUtilityFunction instead."
        )

    def to_dict(self, python_class_identifier=PYTHON_CLASS_IDENTIFIER) -> dict:
        """Convert to dictionary for serialization.

        Args:
            python_class_identifier: The key to use for the Python class name.

        Returns:
            Dictionary representation.
        """
        d = {python_class_identifier: get_full_type_name(type(self))}
        d.update(super().to_dict(python_class_identifier=python_class_identifier))

        # Serialize factors
        serialized_factors = []
        for (indices, func), weight in zip(self._factors, self._weights):
            serialized_factors.append(
                (
                    indices,
                    serialize(func, python_class_identifier=python_class_identifier),
                    weight,
                )
            )

        return dict(**d, factors=serialized_factors, bias=self._bias)

    @classmethod
    def from_dict(
        cls, d: dict, python_class_identifier=PYTHON_CLASS_IDENTIFIER
    ) -> GLAUtilityFunction:
        """Create from dictionary.

        Args:
            d: Dictionary representation.
            python_class_identifier: The key used for the Python class name.

        Returns:
            GLAUtilityFunction instance.
        """
        if isinstance(d, cls):
            return d

        d = dict(d)  # Make a copy
        d.pop(python_class_identifier, None)

        # Deserialize factors
        if "factors" in d:
            factors = []
            for factor_spec in d["factors"]:
                indices, func_dict, weight = factor_spec
                func = deserialize(
                    func_dict, python_class_identifier=python_class_identifier
                )
                factors.append((tuple(indices), func, weight))
            d["factors"] = factors

        # Handle weights separately if present (for backward compatibility)
        d.pop("weights", None)

        # Deserialize remaining fields but keep factors as-is
        factors_backup = d.pop("factors", None)
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

        return cls(**d)

    @classmethod
    def random(
        cls,
        outcome_space: CartesianOutcomeSpace | None = None,
        issues: list[Issue] | tuple[Issue, ...] | None = None,
        reserved_value: tuple[float, float] = (0.0, 1.0),
        normalized: bool = True,
        n_factors: int | tuple[int, int] = (1, 5),
        max_factor_issues: int = 3,
        **kwargs,
    ) -> GLAUtilityFunction:
        """Generate a random GLA utility function.

        Args:
            outcome_space: The outcome space.
            issues: The issues (used if outcome_space is not provided).
            reserved_value: Range for random reserved value.
            normalized: Whether to normalize weights.
            n_factors: Number of factors or range (min, max).
            max_factor_issues: Maximum number of issues per factor.
            **kwargs: Additional arguments passed to the constructor.

        Returns:
            A random GLAUtilityFunction.
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

        factors: list[tuple[tuple[int, ...], BaseFun | BaseMultiFun]] = []
        weights: list[float] = []

        for _ in range(num_factors):
            # Random number of issues for this factor (1 to max_factor_issues)
            factor_n_issues = random.randint(1, min(max_factor_issues, n_issues))

            # Random selection of issues
            selected_indices = tuple(
                sorted(random.sample(range(n_issues), factor_n_issues))
            )

            # Create a random factor function
            func: BaseFun | BaseMultiFun
            if factor_n_issues == 1:
                issue = issues[selected_indices[0]]
                if issue.is_numeric():
                    # Random affine function
                    slope = random.uniform(-1.0, 1.0)
                    bias = random.uniform(-0.5, 0.5) if not normalized else 0.0
                    func = AffineFun(slope=slope, bias=bias)
                else:
                    # Random table function for discrete issues
                    values = list(issue.all)
                    mapping = {v: random.random() for v in values}
                    func = TableFun(mapping=mapping)
            else:
                # For multi-issue factors, use random affine or table
                all_numeric = all(issues[i].is_numeric() for i in selected_indices)
                if all_numeric:
                    slope = tuple(random.uniform(-1.0, 1.0) for _ in selected_indices)
                    bias = random.uniform(-0.5, 0.5) if not normalized else 0.0
                    func = AffineMultiFun(slope=slope, bias=bias)
                else:
                    # Create a table for combinations (limited to avoid explosion)
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
                            mapping = {c: random.random() for c in combos}
                            func = TableMultiFun(mapping=mapping)
                        else:
                            # Fall back to lambda
                            func = LambdaMultiFun(
                                f=lambda x: sum(hash(v) % 100 for v in x) / 100.0
                            )
                    else:
                        # Use a simple lambda
                        func = LambdaMultiFun(
                            f=lambda x: sum(
                                float(v)
                                if isinstance(v, (int, float))
                                else hash(v) % 10
                                for v in x
                            )
                            / len(x)
                        )

            factors.append((selected_indices, func))
            weights.append(random.random())

        # Normalize weights if requested
        if normalized and weights:
            total = sum(weights)
            if total > 0:
                weights = [w / total for w in weights]

        r = (
            random.random() * (reserved_value[1] - reserved_value[0])
            + reserved_value[0]
        )

        return cls(
            factors=list(
                zip([f[0] for f in factors], [f[1] for f in factors], weights)
            ),
            bias=0.0,
            issues=issues,
            reserved_value=r,
            **kwargs,
        )

    def shift_by(
        self, offset: float, shift_reserved: bool = True
    ) -> GLAUtilityFunction:
        """Create a shifted version of this utility function.

        Args:
            offset: The amount to shift by.
            shift_reserved: Whether to also shift the reserved value.

        Returns:
            A new shifted GLAUtilityFunction.
        """
        return GLAUtilityFunction(
            factors=list(
                zip(
                    [f[0] for f in self._factors],
                    [f[1] for f in self._factors],
                    self._weights,
                )
            ),
            bias=self._bias + offset,
            outcome_space=self.outcome_space,
            name=self.name,
            reserved_value=(
                self.reserved_value + offset if shift_reserved else self.reserved_value
            ),
        )

    def scale_by(self, scale: float, scale_reserved: bool = True) -> GLAUtilityFunction:
        """Create a scaled version of this utility function.

        Args:
            scale: The scaling factor.
            scale_reserved: Whether to also scale the reserved value.

        Returns:
            A new scaled GLAUtilityFunction.
        """
        if scale < 0:
            raise ValueError(f"Cannot have a negative scale: {scale}")

        new_weights = [w * scale for w in self._weights]

        return GLAUtilityFunction(
            factors=list(
                zip(
                    [f[0] for f in self._factors],
                    [f[1] for f in self._factors],
                    new_weights,
                )
            ),
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
        for (indices, func), weight in zip(self._factors, self._weights):
            factor_strs.append(f"  {indices}: {type(func).__name__} (w={weight:.3f})")
        factors_str = "\n".join(factor_strs)
        return f"GLAUtilityFunction(\n  bias={self._bias},\n  factors=[\n{factors_str}\n  ]\n)"
