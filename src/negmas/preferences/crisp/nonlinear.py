"""Module for nonlinear functionality."""

from __future__ import annotations

import math
from typing import Callable, Iterable

from negmas.generics import GenericMapping, gmap, ikeys
from negmas.helpers import get_full_type_name
from negmas.outcomes import Issue, Outcome, OutcomeRange, outcome_in_range
from negmas.outcomes.outcome_space import CartesianOutcomeSpace
from negmas.serialization import PYTHON_CLASS_IDENTIFIER, deserialize, serialize

from ..base import OutcomeUtilityMapping
from ..crisp_ufun import UtilityFunction

__all__ = [
    "NonLinearAggregationUtilityFunction",
    "HyperRectangleUtilityFunction",
    "NonlinearHyperRectangleUtilityFunction",
]


class NonLinearAggregationUtilityFunction(UtilityFunction):
    r"""A nonlinear utility function.

    Allows for the modeling of a single nonlinear utility function that combines the utilities of different issues.

    Args:
        values: A set of mappings from issue values to utility functions. These are generic mappings so
                        `Callable` (s) and `Mapping` (s) are both accepted
        f: A nonlinear function mapping from a dict of utility_function-per-issue to a float
        name: name of the utility function. If None a random name will be generated.

    Notes:

        The utility is calculated as:

        .. math::

                u = f\left(u_0\left(i_0\right), u_1\left(i_1\right), ..., u_n\left(i_n\right)\right)

        where :math:`u_j()` is the utility function for issue :math:`j` and :math:`i_j` is value of issue :math:`j` in the
        evaluated outcome.


    Examples:

        >>> from negmas.outcomes import make_issue
        >>> from negmas.preferences.crisp.mapping import MappingUtilityFunction
        >>> issues = [
        ...     make_issue((10.0, 20.0), "price"),
        ...     make_issue(["delivered", "not delivered"], "delivery"),
        ...     make_issue(5, "quality"),
        ... ]
        >>> print(list(map(str, issues)))
        ['price: (10.0, 20.0)', "delivery: ['delivered', 'not delivered']", 'quality: (0, 4)']
        >>> g = NonLinearAggregationUtilityFunction(
        ...     {
        ...         "price": lambda x: 2.0 * x,
        ...         "delivery": {"delivered": 10, "not delivered": -10},
        ...         "quality": MappingUtilityFunction(lambda x: x - 3),
        ...     },
        ...     f=lambda u: u[0] + 2.0 * u[-1],
        ...     issues=issues,
        ... )
        >>> g((14.0, "delivered", 2)) - ((2.0 * 14.0) + 2.0 * (2 - 3))
        0.0

        You must pass a value for each issue in the outcome. If some issues are not used for the ufun, you can pass them as any value that is acceptable to the corresponding value function

        >>> g = NonLinearAggregationUtilityFunction(
        ...     {
        ...         "price": lambda x: 2.0 * x,
        ...         "delivery": {"delivered": 10, "not delivered": -10},
        ...     },
        ...     f=lambda u: 2.0 * u[0],
        ...     issues=issues[:2],
        ... )
        >>> g((14.0, "delivered")) - (2.0 * (2.0 * 14))
        0.0

    """

    def __init__(
        self,
        values: dict[str, GenericMapping] | list[GenericMapping] | None,
        f: Callable[[tuple[float]], float],
        *args,
        **kwargs,
    ) -> None:
        """Initialize the nonlinear aggregation utility function.

        Args:
            values: Mappings from issue values to utility values, either as a dict
                keyed by issue name or a list ordered by issue index.
            f: Nonlinear aggregation function that combines per-issue utilities into
                a single utility value.
            *args: Additional positional arguments passed to parent class.
            **kwargs: Additional keyword arguments passed to parent class.
        """
        super().__init__(*args, **kwargs)
        if isinstance(values, dict):
            if not isinstance(self.outcome_space, CartesianOutcomeSpace):
                raise ValueError(
                    f"Cannot create a {self.__class__.__name__} with an outcome-space that is not Cartesian while passing values as a dict"
                )
            if not self.outcome_space.issues:
                raise ValueError(
                    "Cannot initializes with dict of values if you are not specifying issues"
                )
            values = [
                values[_]
                for _ in [
                    i.name if isinstance(i, Issue) else i
                    for i in self.outcome_space.issues
                ]
            ]
        self.values = values
        self.f = f

    def xml(self, issues: list[Issue]) -> str:
        """Generate an XML representation of the utility function.

        Args:
            issues: The issues defining the outcome space.

        Returns:
            XML string representation of the utility function.

        Raises:
            NotImplementedError: This function cannot be converted to XML.
        """
        raise NotImplementedError(f"Cannot convert {self.__class__.__name__} to xml")

    def to_dict(self, python_class_identifier=PYTHON_CLASS_IDENTIFIER):
        """Convert to a dictionary for serialization.

        Args:
            python_class_identifier: Key used to store the class type identifier.

        Returns:
            Dictionary representation suitable for JSON serialization.
        """
        d = {python_class_identifier: get_full_type_name(type(self))}
        return dict(
            **d,
            values=serialize(
                self.values, python_class_identifier=python_class_identifier
            ),
            f=serialize(self.f),
        )

    @classmethod
    def from_dict(cls, d, python_class_identifier=PYTHON_CLASS_IDENTIFIER):
        """Create an instance from a dictionary.

        Args:
            d: Dictionary containing the serialized utility function.
            python_class_identifier: Key used to identify the class type.

        Returns:
            A new NonLinearAggregationUtilityFunction instance.
        """
        d.pop(python_class_identifier, None)
        for k in ("values", "f"):
            d[k] = deserialize(
                d.get(k, None), python_class_identifier=python_class_identifier
            )
        return cls(**d)

    def eval(self, offer: Outcome | None) -> float:
        """Evaluate the utility of an outcome.

        Computes u = f(u_0(omega_0), u_1(omega_1), ..., u_n(omega_n)).

        Args:
            offer: The outcome to evaluate. If None, returns the reserved value.

        Returns:
            The utility value for the given outcome.

        Raises:
            ValueError: If no issue utilities (values) have been set.
        """
        if offer is None:
            return self.reserved_value
        if self.values is None:
            raise ValueError("No issue utilities were set.")

        u = tuple(gmap(v, w) for w, v in zip(offer, self.values))
        return self.f(u)


def _coerce_ranges(outcome_range):
    """Restore ``(min, max)`` range *tuples* inside an outcome-range dict.

    Serialization formats such as YAML/JSON turn tuples into lists, which
    ``outcome_in_range`` would misread as a *discrete value set* rather than a
    numeric range (``[4.0, 9.0]`` -> "value must be 4 or 9" instead of
    "4 <= value <= 9"). This converts any two-element numeric list back to a
    tuple (recursing into lists of sub-ranges), leaving genuine value sets and
    scalars untouched.
    """
    if outcome_range is None:
        return None

    def as_range(v):
        if isinstance(v, (list, tuple)):
            if len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
                return (v[0], v[1])
            return [as_range(x) for x in v]
        return v

    return {k: as_range(outcome_range[k]) for k in outcome_range}


class HyperRectangleUtilityFunction(UtilityFunction):
    r"""A utility function defined as a set of hyper-volumes.

    The utility function that is calulated by combining linearly a set of *probably nonlinear* functions applied in
    predefined hyper-volumes of the outcome space.

    Args:
         outcome_ranges: The outcome_ranges for which the `mappings` are defined
         weights: The *optional* weights to use for combining the outputs of the `mappings`
         ignore_issues_not_in_input: If a hyper-volumne local function is defined for some issue
         that is not in the outcome being evaluated ignore it.
         ignore_failing_range_utilities: If a hyper-volume local function fails, just assume it
         did not exist for this outcome.
         name: name of the utility function. If None a random name will be generated.

    Notes:

        The utility value is calculated as:

        .. math::

            u(\omega) = b + \sum_{k: \omega \in R_k} w_k \cdot f_k(\omega)

        where :math:`b` is the bias term, :math:`R_k` are hyper-rectangular regions of the
        outcome space, :math:`w_k` are the weights for each region, and :math:`f_k` is either
        a constant or a function that maps outcomes to utility values within region :math:`R_k`.

     Examples:
         We will use the following issue space of cardinality :math:`10 \times 5 \times 4`:

         >>> from negmas.outcomes import make_issue
         >>> issues = [make_issue(10), make_issue(5), make_issue(4)]

         Now create the utility function with

         >>> f = HyperRectangleUtilityFunction(
         ...     outcome_ranges=[
         ...         {0: (1.0, 2.0), 1: (1.0, 2.0)},
         ...         {0: (1.4, 2.0), 2: (2.0, 3.0)},
         ...     ],
         ...     utilities=[2.0, lambda x: 2 * x[2] + x[0]],
         ... )
         >>> g = HyperRectangleUtilityFunction(
         ...     outcome_ranges=[
         ...         {0: (1.0, 2.0), 1: (1.0, 2.0)},
         ...         {0: (1.4, 2.0), 2: (2.0, 3.0)},
         ...     ],
         ...     utilities=[2.0, lambda x: 2 * x[2] + x[0]],
         ...     ignore_issues_not_in_input=True,
         ... )
         >>> h = HyperRectangleUtilityFunction(
         ...     outcome_ranges=[
         ...         {0: (1.0, 2.0), 1: (1.0, 2.0)},
         ...         {0: (1.4, 2.0), 2: (2.0, 3.0)},
         ...     ],
         ...     utilities=[2.0, lambda x: 2 * x[2] + x[0]],
         ...     ignore_failing_range_utilities=True,
         ... )

         We can now calcualte the utility_function of some outcomes:

         An outcome that belongs to the both outcome_ranges:

         >>> [
         ...     f({0: 1.5, 1: 1.5, 2: 2.5}),
         ...     g({0: 1.5, 1: 1.5, 2: 2.5}),
         ...     h({0: 1.5, 1: 1.5, 2: 2.5}),
         ... ]
         [8.5, 8.5, 8.5]

         An outcome that belongs to the first hypervolume only:

         >>> [
         ...     f({0: 1.5, 1: 1.5, 2: 1.0}),
         ...     g({0: 1.5, 1: 1.5, 2: 1.0}),
         ...     h({0: 1.5, 1: 1.5, 2: 1.0}),
         ... ]
         [2.0, 2.0, 2.0]

         An outcome that belongs to and has the first hypervolume only:

         >>> [f({0: 1.5}), g({0: 1.5}), h({0: 1.5})]
         [nan, 0.0, nan]

         An outcome that belongs to the second hypervolume only:

         >>> [f({0: 1.5, 2: 2.5}), g({0: 1.5, 2: 2.5}), h({0: 1.5, 2: 2.5})]
         [nan, 6.5, nan]

         An outcome that has and belongs to the second hypervolume only:

         >>> [f({2: 2.5}), g({2: 2.5}), h({2: 2.5})]
         [nan, 0.0, nan]

         An outcome that belongs to no outcome_ranges:

         >>> [
         ...     f({0: 11.5, 1: 11.5, 2: 12.5}),
         ...     g({0: 11.5, 1: 11.5, 2: 12.5}),
         ...     h({0: 11.5, 1: 11.5, 2: 12.5}),
         ... ]
         [0.0, 0.0, 0.0]


     Remarks:
         - The number of outcome_ranges, mappings, and weights must be the same
         - if no weights are given they are all assumed to equal unity
         - mappings can either by an `OutcomeUtilityMapping` or a constant.

    """

    def adjust_params(self):
        """Initialize default weights if not provided.

        Sets all weights to 1.0 if no weights were specified.
        """
        if self.weights is None:
            self.weights = [1.0] * len(self.outcome_ranges)

    def __init__(
        self,
        outcome_ranges: Iterable[OutcomeRange | None],
        utilities: list[float] | list[OutcomeUtilityMapping],
        weights: list[float] | None = None,
        ignore_issues_not_in_input=False,
        ignore_failing_range_utilities=False,
        bias: float = 0.0,
        *args,
        **kwargs,
    ) -> None:
        """Initialize the hyper-rectangle utility function.

        Args:
            outcome_ranges: Hyper-rectangular regions defining subsets of the outcome space.
            utilities: Utility values or functions for each region.
            weights: Multipliers for combining each region's utility contribution.
            ignore_issues_not_in_input: Skip regions with issues missing from the outcome.
            ignore_failing_range_utilities: Continue if a region's utility computation fails.
            bias: Constant offset added to the final utility.
            *args: Additional positional arguments passed to parent class.
            **kwargs: Additional keyword arguments passed to parent class.
        """
        super().__init__(*args, **kwargs)
        self.outcome_ranges = [_coerce_ranges(r) for r in outcome_ranges]
        self.mappings = list(utilities)
        self.weights = list(weights) if weights else ([1.0] * len(self.outcome_ranges))
        self.ignore_issues_not_in_input = ignore_issues_not_in_input
        self.ignore_failing_range_utilities = ignore_failing_range_utilities
        self.bias = bias
        self.adjust_params()

    def xml(self, issues: list[Issue]) -> str:
        """Represents the function as XML

        Args:
            issues: The issues defining the negotiation domain structure.

        Examples:

            >>> from negmas.outcomes import make_issue
            >>> f = HyperRectangleUtilityFunction(
            ...     outcome_ranges=[
            ...         {0: (1.0, 2.0), 1: (1.0, 2.0)},
            ...         {0: (1.4, 2.0), 2: (2.0, 3.0)},
            ...     ],
            ...     utilities=[2.0, 9.0 + 4.0],
            ... )
            >>> print(
            ...     f.xml(
            ...         [
            ...             make_issue((0.0, 4.0), name="0"),
            ...             make_issue((0.0, 9.0), name="1"),
            ...             make_issue((0.0, 9.0), name="2"),
            ...         ]
            ...     ).strip()
            ... )
            <issue index="1" name="0" vtype="real" type="real" etype="real">
                <range lowerbound="0.0" upperbound="4.0"></range>
            </issue><issue index="2" name="1" vtype="real" type="real" etype="real">
                <range lowerbound="0.0" upperbound="9.0"></range>
            </issue><issue index="3" name="2" vtype="real" type="real" etype="real">
                <range lowerbound="0.0" upperbound="9.0"></range>
            </issue><utility_function maxutility="-1.0">
                <ufun type="PlainUfun" weight="1" aggregation="sum">
                    <hyperRectangle utility_function="2.0">
                        <INCLUDES index="1" min="1.0" max="2.0" />
                        <INCLUDES index="2" min="1.0" max="2.0" />
                    </hyperRectangle>
                    <hyperRectangle utility_function="13.0">
                        <INCLUDES index="1" min="1.4" max="2.0" />
                        <INCLUDES index="3" min="2.0" max="3.0" />
                    </hyperRectangle>
                </ufun>
            </utility_function>

        """
        output = ""
        for i, issue in enumerate(issues):
            name = issue.name
            if isinstance(issue.values, tuple):
                output += (
                    f'<issue index="{i + 1}" name="{name}" vtype="real" type="real" etype="real">\n'
                    f'    <range lowerbound="{issue.values[0]}" upperbound="{issue.values[1]}"></range>\n'
                    f"</issue>"
                )
            elif isinstance(issue.values, int):
                output += (
                    f'<issue index="{i + 1}" name="{name}" vtype="integer" type="integer" etype="integer" '
                    f'lowerbound="0" upperbound="{issue.values - 1}"/>\n'
                )
            else:
                output += (
                    f'<issue index="{i + 1}" name="{name}" vtype="integer" type="integer" etype="integer" '
                    f'lowerbound="{min(issue.values)}" upperbound="{max(issue.values)}"/>\n'
                )
        # todo find the real maxutility
        output += '<utility_function maxutility="-1.0">\n    <ufun type="PlainUfun" weight="1" aggregation="sum">\n'
        for rect, u, w in zip(self.outcome_ranges, self.mappings, self.weights):
            if not isinstance(u, float):
                raise ValueError(
                    "Only hyper-rectangles with constant utility per rectangle can be convereted to xml"
                )
            output += f'        <hyperRectangle utility_function="{u * w}">\n'
            if not rect:
                continue
            for key in rect.keys():
                # indx = [i for i, _ in enumerate(issues) if _.name == key][0] + 1
                indx = key + 1
                values = rect.get(key, None)
                if values is None:
                    continue
                if isinstance(values, float) or isinstance(values, int):
                    mn, mx = values, values
                elif isinstance(values, tuple):
                    mn, mx = values
                else:
                    mn, mx = min(values), max(values)
                output += (
                    f'            <INCLUDES index="{indx}" min="{mn}" max="{mx}" />\n'
                )
            output += "        </hyperRectangle>\n"
        output += "    </ufun>\n</utility_function>"
        return output

    def to_stationary(self):
        """Return a stationary version of this utility function.

        Returns:
            Self, as HyperRectangleUtilityFunction is already stationary.
        """
        return self

    @classmethod
    def random(
        cls, outcome_space, reserved_value, normalized=True, rectangles=(1, 4), **kwargs
    ) -> HyperRectangleUtilityFunction:
        """Generate a random hyper-rectangle utility function.

        Args:
            outcome_space: The outcome space to generate the utility function for.
            reserved_value: The reserved value (utility of disagreement).
            normalized: Whether to normalize the utility values.
            rectangles: Range (min, max) for the number of rectangles.
            **kwargs: Additional arguments.

        Raises:
            NotImplementedError: Random generation is not yet implemented.
        """
        raise NotImplementedError("random hyper-rectangle ufuns are not implemented")

    def to_dict(self, python_class_identifier=PYTHON_CLASS_IDENTIFIER):
        """Convert to a dictionary for serialization.

        Args:
            python_class_identifier: Key used to store the class type identifier.

        Returns:
            Dictionary representation suitable for JSON serialization.
        """
        d = {python_class_identifier: get_full_type_name(type(self))}
        d.update(super().to_dict(python_class_identifier=python_class_identifier))
        return dict(
            **d,
            outcome_ranges=serialize(
                self.outcome_ranges, python_class_identifier=python_class_identifier
            ),
            utilities=serialize(
                self.mappings, python_class_identifier=python_class_identifier
            ),
            weights=self.weights,
            bias=self.bias,
            ignore_issues_not_in_input=self.ignore_issues_not_in_input,
            ignore_failing_range_utilities=self.ignore_failing_range_utilities,
        )

    @classmethod
    def from_dict(cls, d, python_class_identifier=PYTHON_CLASS_IDENTIFIER):
        """Create an instance from a dictionary.

        Args:
            d: Dictionary containing the serialized utility function.
            python_class_identifier: Key used to identify the class type.

        Returns:
            A new HyperRectangleUtilityFunction instance.
        """
        d.pop(python_class_identifier, None)
        for k in ("outcome_ranges", "utilities"):
            if k in d:
                d[k] = deserialize(
                    d.get(k, None), python_class_identifier=python_class_identifier
                )
        return cls(**d)

    def eval(self, offer: Outcome | None) -> float:
        """Evaluate the utility of an outcome.

        Computes u = bias + sum(w_k * f_k(omega)) for all regions k containing omega.

        Args:
            offer: The outcome to evaluate. If None, returns the reserved value.

        Returns:
            The utility value for the given outcome. Returns NaN if the outcome
            contains issues not covered by hyper-rectangles and
            ignore_issues_not_in_input is False.
        """
        if offer is None:
            return self.reserved_value
        u = self.bias
        for weight, outcome_range, mapping in zip(
            self.weights, self.outcome_ranges, self.mappings
        ):
            # fail on any outcome_range that constrains issues not in the presented outcome
            if (
                outcome_range is not None
                and set(ikeys(outcome_range)) - set(ikeys(offer)) != set()
            ):
                if self.ignore_issues_not_in_input:
                    continue

                return float("nan")

            elif outcome_range is None or outcome_in_range(offer, outcome_range):
                if isinstance(mapping, float) or isinstance(mapping, int):
                    u += weight * mapping
                else:
                    # fail if any outcome_range utility_function cannot be calculated from the input
                    try:
                        u += weight * gmap(mapping, offer)
                    except KeyError:
                        if self.ignore_failing_range_utilities:
                            continue

                        return float("nan")

        return u

    # -- range normalization without enumerating the outcome space ------------
    def _box_bounds(self) -> list[dict]:
        """Each rectangle as ``{issue_index: (lo, hi)}`` (scalars -> (v, v))."""
        boxes = []
        for rect in self.outcome_ranges:
            b = {}
            if rect:
                for k in ikeys(rect):
                    v = rect[k]
                    if isinstance(v, (tuple, list)) and len(v) == 2:
                        b[k] = (min(v), max(v))
                    else:
                        b[k] = (v, v)
            boxes.append(b)
        return boxes

    def _range_extremes(self) -> tuple[float, float]:
        """Exact ``(min, max)`` utility over the whole outcome space.

        The utility an outcome receives depends only on which axis-aligned
        rectangles contain it. By Helly's theorem for boxes, a set of rectangles
        has a common outcome iff they pairwise overlap, so the achievable utility
        offsets are exactly the total weights of *cliques* in the rectangle-overlap
        graph. The maximum is therefore ``bias + max-weight clique`` (over
        positive-contribution rectangles) and the minimum ``bias + max-weight
        clique`` over negative ones -- found without enumerating outcomes.

        Requires constant per-rectangle utilities; raises ``ValueError`` otherwise.
        """
        import networkx as nx

        contribs = []
        for w, m in zip(self.weights, self.mappings):
            if not isinstance(m, (int, float)):
                raise ValueError(
                    "Range-normalization requires constant per-rectangle utilities"
                )
            contribs.append(float(w) * float(m))
        boxes = self._box_bounds()
        n = len(boxes)

        def compatible(i: int, j: int) -> bool:
            a, b = boxes[i], boxes[j]
            for k in a.keys() & b.keys():
                if max(a[k][0], b[k][0]) > min(a[k][1], b[k][1]):
                    return False
            return True

        scale = 10**6

        def best_offset(sign: int) -> float:
            nodes = [i for i in range(n) if sign * contribs[i] > 0]
            if not nodes:
                return 0.0
            g = nx.Graph()
            for i in nodes:
                g.add_node(i, weight=max(1, round(sign * contribs[i] * scale)))
            for a_ in range(len(nodes)):
                for b_ in range(a_ + 1, len(nodes)):
                    if compatible(nodes[a_], nodes[b_]):
                        g.add_edge(nodes[a_], nodes[b_])
            clique, _ = nx.max_weight_clique(g, weight="weight")
            return sum(contribs[i] for i in clique)

        return self.bias + best_offset(-1), self.bias + best_offset(+1)

    def minmax(
        self,
        outcome_space=None,
        issues=None,
        outcomes=None,
        max_cardinality=1000,
        above_reserve=False,
    ) -> tuple[float, float]:
        """Exact ``(min, max)`` via the rectangle-overlap clique (no enumeration).

        Falls back to the base implementation when an explicit set of ``outcomes``
        is supplied or the per-rectangle utilities are non-constant.
        """
        if outcomes is None:
            try:
                w, b = self._range_extremes()
            except ValueError:
                return super().minmax(
                    outcome_space, issues, outcomes, max_cardinality, above_reserve
                )
            if above_reserve and self.reserved_value is not None:
                r = self.reserved_value
                if b < r:
                    b = w = r
                elif w < r:
                    w = r
            return w, b
        return super().minmax(
            outcome_space, issues, outcomes, max_cardinality, above_reserve
        )

    def normalize_for(
        self,
        to: tuple[float, float] = (0.0, 1.0),
        outcome_space=None,
        guarantee_max: bool = True,
        guarantee_min: bool = True,
        max_cardinality: int = 10_000_000_000,
        normalize_reserved_values: bool = False,
        reserved_value_penalty=None,
    ):
        """Affinely range-normalize this hyper-rectangle ufun to ``to``.

        Uses :meth:`_range_extremes` (exact, no enumeration) to build the affine
        map ``u' = a*u + c``, applied by scaling the per-rectangle weights and the
        bias; the reserved value undergoes the same map. This preserves the shape
        of the ufun and maps it onto ``to`` (conditions 1-4). It is **not** the
        linear-additive Method 3: a non-linear ufun has no per-issue weights/value
        functions, so conditions 5-8 do not apply.
        """
        _ = (
            max_cardinality,
            guarantee_max,
            guarantee_min,
            normalize_reserved_values,
            reserved_value_penalty,
        )
        from negmas.preferences.crisp.const import ConstUtilityFunction

        os_ = outcome_space if outcome_space is not None else self.outcome_space
        eps = 1e-9
        mn, mx = self._range_extremes()
        if abs(mx - mn) < eps:
            return ConstUtilityFunction(
                (to[0] + to[1]) / 2.0,
                outcome_space=os_,
                name=self.name,
                reserved_value=self.reserved_value,
            )
        a = (to[1] - to[0]) / (mx - mn)
        c = to[0] - a * mn
        r = self.reserved_value
        new_r = a * r + c if (r is not None and math.isfinite(r)) else r
        return HyperRectangleUtilityFunction(
            outcome_ranges=self.outcome_ranges,
            utilities=self.mappings,
            weights=[a * w for w in self.weights],
            bias=a * self.bias + c,
            ignore_issues_not_in_input=self.ignore_issues_not_in_input,
            ignore_failing_range_utilities=self.ignore_failing_range_utilities,
            outcome_space=os_,
            name=self.name,
            reserved_value=new_r,
        )

    def normalize(
        self,
        to: tuple[float, float] = (0.0, 1.0),
        normalize_weights: bool = False,
        normalize_reserved_values: bool = False,
        reserved_value_penalty=None,
        guarantee_max: bool = True,
        guarantee_min: bool = True,
        max_cardinality: int = 10_000_000_000,
    ):
        _ = (normalize_weights, normalize_reserved_values, reserved_value_penalty)
        return self.normalize_for(
            to,
            outcome_space=self.outcome_space,
            guarantee_max=guarantee_max,
            guarantee_min=guarantee_min,
            max_cardinality=max_cardinality,
        )


class NonlinearHyperRectangleUtilityFunction(UtilityFunction):
    r"""A utility function combining hyper-rectangles with nonlinear aggregation.

    Similar to HyperRectangleUtilityFunction, but uses a nonlinear function to combine
    the utilities from each active hyper-rectangle region.

    Args:
        hypervolumes: The hyper-rectangular regions defining the outcome space partitions.
        mappings: Functions mapping outcomes to utility values within each region.
        f: A nonlinear function to aggregate the utilities from active regions.
        name: Name of the utility function. If None, a random name will be generated.
        reserved_value: The utility value for disagreement (None offer).

    Notes:

        The utility value is calculated as:

        .. math::

            u(\omega) = f\left(\left[g_k(\omega) : \omega \in H_k\right]\right)

        where :math:`H_k` are hypervolumes (regions), :math:`g_k` is the mapping for region
        :math:`k`, and :math:`f` is the nonlinear aggregation function that receives the
        list of utilities from all regions containing :math:`\omega`.
    """

    def __init__(
        self,
        hypervolumes: Iterable[OutcomeRange],
        mappings: list[OutcomeUtilityMapping],
        f: Callable[[list[float]], float],
        name: str | None = None,
        reserved_value: float = float("-inf"),
        id=None,
    ) -> None:
        """Initialize the nonlinear hyper-rectangle utility function.

        Args:
            hypervolumes: Hyper-rectangular regions partitioning the outcome space.
            mappings: Utility functions applied within each corresponding region.
            f: Nonlinear aggregation function combining utilities from active regions.
            name: Optional name for this utility function.
            reserved_value: Utility value returned when no agreement is reached.
            id: Optional unique identifier for this utility function.
        """
        super().__init__(name=name, reserved_value=reserved_value, id=id)
        self.hypervolumes = hypervolumes
        self.mappings = mappings
        self.f = f

    def xml(self, issues: list[Issue]) -> str:
        """Generate an XML representation of the utility function.

        Args:
            issues: The issues defining the outcome space.

        Returns:
            XML string representation of the utility function.

        Raises:
            NotImplementedError: This function cannot be converted to XML.
        """
        raise NotImplementedError(f"Cannot convert {self.__class__.__name__} to xml")

    def to_dict(self, python_class_identifier=PYTHON_CLASS_IDENTIFIER):
        """Convert to a dictionary for serialization.

        Args:
            python_class_identifier: Key used to store the class type identifier.

        Returns:
            Dictionary representation suitable for JSON serialization.
        """
        d = {python_class_identifier: get_full_type_name(type(self))}
        d.update(super().to_dict(python_class_identifier=python_class_identifier))
        return dict(
            **d,
            hypervolumes=serialize(
                self.hypervolumes, python_class_identifier=python_class_identifier
            ),
            mappings=serialize(
                self.mappings, python_class_identifier=python_class_identifier
            ),
            f=serialize(self.f, python_class_identifier=python_class_identifier),
        )

    @classmethod
    def from_dict(cls, d, python_class_identifier=PYTHON_CLASS_IDENTIFIER):
        """Create an instance from a dictionary.

        Args:
            d: Dictionary containing the serialized utility function.
            python_class_identifier: Key used to identify the class type.

        Returns:
            A new NonlinearHyperRectangleUtilityFunction instance.
        """
        d.pop(python_class_identifier, None)
        for k in ("hypervolumes", "mapoints", "f"):
            d[k] = deserialize(
                d.get(k, None), python_class_identifier=python_class_identifier
            )
        return cls(**d)

    def eval(self, offer: Outcome | None) -> float:
        """Evaluate the utility of an outcome.

        Computes u = f([g_k(omega) for all k where omega in H_k]).

        Args:
            offer: The outcome to evaluate. If None, returns the reserved value.

        Returns:
            The utility value for the given outcome.

        Raises:
            ValueError: If hypervolumes have not been set.
        """
        if offer is None:
            return self.reserved_value
        if not isinstance(self.hypervolumes, Iterable):
            raise ValueError(
                "Hypervolumes are not set. Call set_params() or pass them through the constructor."
            )

        u = []
        for hypervolume, mapping in zip(self.hypervolumes, self.mappings):
            if outcome_in_range(offer, hypervolume):
                u.append(gmap(mapping, offer))
        return self.f(u)
