from __future__ import annotations

from typing import Callable, Iterable

from negmas.generics import GenericMapping, gmap, ikeys
from negmas.helpers import get_full_type_name
from negmas.outcomes import Issue, Outcome, OutcomeRange, outcome_in_range
from negmas.outcomes.outcome_space import CartesianOutcomeSpace
from negmas.serialization import PYTHON_CLASS_IDENTIFIER, deserialize, serialize

from ..base import OutcomeUtilityMapping
from ..crisp_ufun import UtilityFunction
from ..mixins import StationaryMixin

__all__ = [
    "NonLinearAggregationUtilityFunction",
    "HyperRectangleUtilityFunction",
    "NonlinearHyperRectangleUtilityFunction",
]


class NonLinearAggregationUtilityFunction(StationaryMixin, UtilityFunction):
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
        >>> issues = [make_issue((10.0, 20.0), 'price'), make_issue(['delivered', 'not delivered'], 'delivery')
        ...           , make_issue(5, 'quality')]
        >>> print(list(map(str, issues)))
        ['price: (10.0, 20.0)', "delivery: ['delivered', 'not delivered']", 'quality: (0, 4)']
        >>> g = NonLinearAggregationUtilityFunction({'price': lambda x: 2.0*x
        ...                                         , 'delivery': {'delivered': 10, 'not delivered': -10}
        ...                                         , 'quality': MappingUtilityFunction(lambda x: x-3)}
        ...         , f=lambda u: u[0]  + 2.0 * u[-1], issues=issues)
        >>> g((14.0, 'delivered', 2)) - ((2.0*14.0)+2.0*(2-3))
        0.0

        You must pass a value for each issue in the outcome. If some issues are not used for the ufun, you can pass them as any value that is acceptable to the corresponding value function

        >>> g = NonLinearAggregationUtilityFunction({'price'    : lambda x: 2.0*x
        ...                                         , 'delivery': {'delivered': 10, 'not delivered': -10}}
        ...         , f=lambda u: 2.0 * u[0], issues=issues[:2])
        >>> g((14.0, 'delivered')) - (2.0*(2.0*14))
        0.0

    """

    def __init__(
        self,
        values: dict[str, GenericMapping] | list[GenericMapping] | None,
        f: Callable[[tuple[float]], float],
        *args,
        **kwargs,
    ) -> None:
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
        raise NotImplementedError(f"Cannot convert {self.__class__.__name__} to xml")

    def to_dict(self):
        d = {PYTHON_CLASS_IDENTIFIER: get_full_type_name(type(self))}
        return dict(
            **d,
            values=serialize(self.values),
            f=serialize(self.f),
        )

    @classmethod
    def from_dict(cls, d):
        d.pop(PYTHON_CLASS_IDENTIFIER, None)
        for k in ("values", "f"):
            d[k] = deserialize(d.get(k, None))
        return cls(**d)

    def eval(self, offer: Outcome | None) -> float | None:
        if offer is None:
            return self.reserved_value
        if self.values is None:
            raise ValueError("No issue utilities were set.")

        u = tuple(gmap(v, w) for w, v in zip(offer, self.values))
        return self.f(u)


class HyperRectangleUtilityFunction(StationaryMixin, UtilityFunction):
    """A utility function defined as a set of hyper-volumes.

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

     Examples:
         We will use the following issue space of cardinality :math:`10 \times 5 \times 4`:

         >>> from negmas.outcomes import make_issue
         >>> issues = [make_issue(10), make_issue(5), make_issue(4)]

         Now create the utility function with

         >>> f = HyperRectangleUtilityFunction(outcome_ranges=[
         ...                                        {0: (1.0, 2.0), 1: (1.0, 2.0)},
         ...                                        {0: (1.4, 2.0), 2: (2.0, 3.0)}]
         ...                                , utilities= [2.0, lambda x: 2 * x[2] + x[0]])
         >>> g = HyperRectangleUtilityFunction(outcome_ranges=[
         ...                                        {0: (1.0, 2.0), 1: (1.0, 2.0)},
         ...                                        {0: (1.4, 2.0), 2: (2.0, 3.0)}]
         ...                                , utilities= [2.0, lambda x: 2 * x[2] + x[0]]
         ...                                , ignore_issues_not_in_input=True)
         >>> h = HyperRectangleUtilityFunction(outcome_ranges=[
         ...                                        {0: (1.0, 2.0), 1: (1.0, 2.0)},
         ...                                        {0: (1.4, 2.0), 2: (2.0, 3.0)}]
         ...                                , utilities= [2.0, lambda x: 2 * x[2] + x[0]]
         ...                                , ignore_failing_range_utilities=True)

         We can now calcualte the utility_function of some outcomes:

         * An outcome that belongs to the both outcome_ranges:
         >>> [f({0: 1.5,1: 1.5, 2: 2.5}), g({0: 1.5,1: 1.5, 2: 2.5}), h({0: 1.5,1: 1.5, 2: 2.5})]
         [8.5, 8.5, 8.5]

         * An outcome that belongs to the first hypervolume only:
         >>> [f({0: 1.5,1: 1.5, 2: 1.0}), g({0: 1.5,1: 1.5, 2: 1.0}), h({0: 1.5,1: 1.5, 2: 1.0})]
         [2.0, 2.0, 2.0]

         * An outcome that belongs to and has the first hypervolume only:
         >>> [f({0: 1.5}), g({0: 1.5}), h({0: 1.5})]
         [None, 0.0, None]

         * An outcome that belongs to the second hypervolume only:
         >>> [f({0: 1.5,2: 2.5}), g({0: 1.5,2: 2.5}), h({0: 1.5,2: 2.5})]
         [None, 6.5, None]

         * An outcome that has and belongs to the second hypervolume only:
         >>> [f({2: 2.5}), g({2: 2.5}), h({2: 2.5})]
         [None, 0.0, None]

         * An outcome that belongs to no outcome_ranges:
         >>> [f({0: 11.5,1: 11.5, 2: 12.5}), g({0: 11.5,1: 11.5, 2: 12.5}), h({0: 11.5,1: 11.5, 2: 12.5})]
         [0.0, 0.0, 0.0]


     Remarks:
         - The number of outcome_ranges, mappings, and weights must be the same
         - if no weights are given they are all assumed to equal unity
         - mappings can either by an `OutcomeUtilityMapping` or a constant.

    """

    def adjust_params(self):
        if self.weights is None:
            self.weights = [1.0] * len(self.outcome_ranges)

    def __init__(
        self,
        outcome_ranges: Iterable[OutcomeRange],
        utilities: list[float] | list[OutcomeUtilityMapping],
        weights: list[float] | None = None,
        ignore_issues_not_in_input=False,
        ignore_failing_range_utilities=False,
        bias: float = 0.0,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.outcome_ranges = list(outcome_ranges)
        self.mappings = list(utilities)
        self.weights = list(weights) if weights else ([1.0] * len(self.outcome_ranges))
        self.ignore_issues_not_in_input = ignore_issues_not_in_input
        self.ignore_failing_range_utilities = ignore_failing_range_utilities
        self.bias = bias
        self.adjust_params()

    def xml(self, issues: list[Issue]) -> str:
        """Represents the function as XML

        Args:
            issues:

        Examples:

            >>> from negmas.outcomes import make_issue
            >>> f = HyperRectangleUtilityFunction(outcome_ranges=[
            ...                                        {0: (1.0, 2.0), 1: (1.0, 2.0)},
            ...                                        {0: (1.4, 2.0), 2: (2.0, 3.0)}]
            ...                                , utilities= [2.0, 9.0 + 4.0])
            >>> print(f.xml([make_issue((0.0, 4.0), name='0'), make_issue((0.0, 9.0), name='1')
            ... , make_issue((0.0, 9.0), name='2')]).strip())
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
                    f'<issue index="{i+1}" name="{name}" vtype="real" type="real" etype="real">\n'
                    f'    <range lowerbound="{issue.values[0]}" upperbound="{issue.values[1]}"></range>\n'
                    f"</issue>"
                )
            elif isinstance(issue.values, int):
                output += (
                    f'<issue index="{i+1}" name="{name}" vtype="integer" type="integer" etype="integer" '
                    f'lowerbound="0" upperbound="{issue.values - 1}"/>\n'
                )
            else:
                output += (
                    f'<issue index="{i+1}" name="{name}" vtype="integer" type="integer" etype="integer" '
                    f'lowerbound="{min(issue.values)}" upperbound="{max(issue.values)}"/>\n'
                )
        # todo find the real maxutility
        output += '<utility_function maxutility="-1.0">\n    <ufun type="PlainUfun" weight="1" aggregation="sum">\n'
        for rect, u, w in zip(self.outcome_ranges, self.mappings, self.weights):
            if not isinstance(u, float):
                raise ValueError(
                    f"Only hyper-rectangles with constant utility per rectangle can be convereted to xml"
                )
            output += f'        <hyperRectangle utility_function="{u * w}">\n'
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
            output += f"        </hyperRectangle>\n"
        output += "    </ufun>\n</utility_function>"
        return output

    def to_stationary(self):
        return self

    @classmethod
    def random(
        cls,
        outcome_space,
        reserved_value,
        normalized=True,
        rectangles=(1, 4),
        **kwargs,
    ) -> HyperRectangleUtilityFunction:
        """Generates a random ufun of the given type"""
        raise NotImplementedError("random hyper-rectangle ufuns are not implemented")

    def to_dict(self):
        d = {PYTHON_CLASS_IDENTIFIER: get_full_type_name(type(self))}
        d.update(super().to_dict())
        return dict(
            **d,
            outcome_ranges=serialize(self.outcome_ranges),
            utilities=serialize(self.mappings),
            weights=self.weights,
            ignore_issues_not_in_input=self.ignore_issues_not_in_input,
            ignore_failing_range_utilities=self.ignore_failing_range_utilities,
        )

    @classmethod
    def from_dict(cls, d):
        d.pop(PYTHON_CLASS_IDENTIFIER, None)
        for k in ("oucome_ranges", "utilities"):
            d[k] = deserialize(d.get(k, None))
        return cls(**d)

    def eval(self, offer: Outcome | None) -> float | None:
        if offer is None:
            return self.reserved_value
        u = self.bias
        for weight, outcome_range, mapping in zip(
            self.weights, self.outcome_ranges, self.mappings
        ):  # type: ignore
            # fail on any outcome_range that constrains issues not in the presented outcome
            if (
                outcome_range is not None
                and set(ikeys(outcome_range)) - set(ikeys(offer)) != set()
            ):
                if self.ignore_issues_not_in_input:
                    continue

                return None

            elif outcome_range is None or outcome_in_range(offer, outcome_range):
                if isinstance(mapping, float):
                    u += weight * mapping
                else:
                    # fail if any outcome_range utility_function cannot be calculated from the input
                    try:
                        # noinspection PyTypeChecker
                        u += weight * gmap(mapping, offer)
                    except KeyError:
                        if self.ignore_failing_range_utilities:
                            continue

                        return None

        return u


class NonlinearHyperRectangleUtilityFunction(StationaryMixin, UtilityFunction):
    """A utility function defined as a set of outcome_ranges.


    Args:
           hypervolumes: see `HyperRectangleUtilityFunction`
           mappings: see `HyperRectangleUtilityFunction`
           f: A nonlinear function to combine the results of `mappings`
           name: name of the utility function. If None a random name will be generated
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
        super().__init__(
            name=name,
            reserved_value=reserved_value,
            id=id,
        )
        self.hypervolumes = hypervolumes
        self.mappings = mappings
        self.f = f

    def xml(self, issues: list[Issue]) -> str:
        raise NotImplementedError(f"Cannot convert {self.__class__.__name__} to xml")

    def to_dict(self):
        d = {PYTHON_CLASS_IDENTIFIER: get_full_type_name(type(self))}
        d.update(super().to_dict())
        return dict(
            **d,
            hypervolumes=serialize(self.hypervolumes),
            mappings=serialize(self.mappings),
            f=serialize(self.f),
        )

    @classmethod
    def from_dict(cls, d):
        d.pop(PYTHON_CLASS_IDENTIFIER, None)
        for k in ("hypervolumes", "mapoints", "f"):
            d[k] = deserialize(d.get(k, None))
        return cls(**d)

    def eval(self, offer: Outcome | None) -> float | None:
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
