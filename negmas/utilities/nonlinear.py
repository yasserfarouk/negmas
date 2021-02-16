from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    MutableMapping,
    Optional,
    Type,
    Union,
)


from negmas.common import AgentMechanismInterface
from negmas.generics import GenericMapping, ienumerate, iget, ivalues
from negmas.helpers import Floats, gmap, ikeys
from negmas.outcomes import (
    Issue,
    OutcomeRange,
    outcome_as_tuple,
    outcome_in_range,
    Outcome,
)
from negmas.utilities.base import (
    UtilityFunction,
    UtilityValue,
    OutcomeUtilityMapping,
    OutcomeUtilityMappings,
    ExactUtilityValue,
)

__all__ = [
    "MappingUtilityFunction",
    "NonLinearUtilityAggregationFunction",
    "HyperRectangleUtilityFunction",
    "NonlinearHyperRectangleUtilityFunction",
]


class MappingUtilityFunction(UtilityFunction):
    """Outcome mapping utility function.

    This is the simplest possible utility function and it just maps a set of `Outcome`s to a set of
    `UtilityValue`(s). It is only usable with single-issue negotiations. It can be constructed with wither a mapping
    (e.g. a dict) or a callable function.

    Args:
            mapping: Either a callable or a mapping from `Outcome` to `UtilityValue`.
            default: value returned for outcomes causing exception (e.g. invalid outcomes).
            name: name of the utility function. If None a random name will be generated.
            reserved_value: The reserved value (utility of not getting an agreement = utility(None) )
            ami: an `AgentMechnismInterface` that is used mostly for setting the outcome-type in methods returning
                 outcomes.
            outcome_type: The type that should be returned from methods returning `Outcome` which can be tuple, dict or
                          any `OutcomeType`.

    Examples:

        Single issue outcome case:

        >>> issue =Issue(values=['to be', 'not to be'], name='THE problem')
        >>> print(str(issue))
        THE problem: ['to be', 'not to be']
        >>> f = MappingUtilityFunction({'to be':10.0, 'not to be':0.0})
        >>> print(list(map(f, ['to be', 'not to be'])))
        [10.0, 0.0]
        >>> f = MappingUtilityFunction(mapping={'to be':-10.0, 'not to be':10.0})
        >>> print(list(map(f, ['to be', 'not to be'])))
        [-10.0, 10.0]
        >>> f = MappingUtilityFunction(lambda x: float(len(x)))
        >>> print(list(map(f, ['to be', 'not to be'])))
        [5.0, 9.0]

        Multi issue case:

        >>> issues = [Issue((10.0, 20.0), 'price'), Issue(['delivered', 'not delivered'], 'delivery')
        ...           , Issue(5, 'quality')]
        >>> print(list(map(str, issues)))
        ['price: (10.0, 20.0)', "delivery: ['delivered', 'not delivered']", 'quality: (0, 4)']
        >>> f = MappingUtilityFunction(lambda x: x['price'] if x['delivery'] == 'delivered' else -1.0)
        >>> g = MappingUtilityFunction(lambda x: x['price'] if x['delivery'] == 'delivered' else -1.0
        ...     , default=-1000 )
        >>> f({'price': 16.0}) is None
        True
        >>> g({'price': 16.0})
        -1000
        >>> f({'price': 16.0, 'delivery':  'delivered'})
        16.0
        >>> f({'price': 16.0, 'delivery':  'not delivered'})
        -1.0

    Remarks:
        - If the mapping used failed on the outcome (for example because it is not a valid outcome), then the
        ``default`` value given to the constructor (which defaults to None) will be returned.

    """

    def __init__(
        self,
        mapping: OutcomeUtilityMapping,
        default=None,
        name: str = None,
        reserved_value: UtilityValue = float("-inf"),
        ami: AgentMechanismInterface = None,
        outcome_type: Optional[Type] = None,
        id = None,
    ) -> None:
        super().__init__(
            name=name, outcome_type=outcome_type, reserved_value=reserved_value, ami=ami, id=id,
        )
        self.mapping = mapping
        self.default = default

    def eval(self, offer: Optional[Outcome]) -> Optional[UtilityValue]:
        # noinspection PyBroadException
        if offer is None:
            return self.reserved_value
        try:
            if isinstance(offer, dict) and isinstance(self.mapping, dict):
                m = gmap(self.mapping, outcome_as_tuple(offer))
            else:
                m = gmap(self.mapping, offer)
        except Exception:
            return self.default

        return m

    def xml(self, issues: List[Issue]) -> str:
        """

        Examples:

            >>> issue =Issue(values=['to be', 'not to be'], name='THE problem')
            >>> print(str(issue))
            THE problem: ['to be', 'not to be']
            >>> f = MappingUtilityFunction({'to be':10.0, 'not to be':0.0})
            >>> print(list(map(f, ['to be', 'not to be'])))
            [10.0, 0.0]
            >>> print(f.xml([issue]))
            <issue index="1" etype="discrete" type="discrete" vtype="discrete" name="THE problem">
                <item index="1" value="to be"  cost="0"  evaluation="10.0" description="to be">
                </item>
                <item index="2" value="not to be"  cost="0"  evaluation="0.0" description="not to be">
                </item>
            </issue>
            <weight index="1" value="1.0">
            </weight>
            <BLANKLINE>
        """
        if len(issues) > 1:
            raise ValueError(
                "Cannot call xml() on a mapping utility function with more than one issue"
            )
        if issues is not None:
            issue_names = [_.name for _ in issues]
            key = issue_names[0]
        else:
            key = "0"
        output = f'<issue index="1" etype="discrete" type="discrete" vtype="discrete" name="{key}">\n'
        if isinstance(self.mapping, Callable):
            for i, k in enumerate(issues[key].all):
                if isinstance(k, tuple) or isinstance(k, list):
                    k = "-".join([str(_) for _ in k])
                output += (
                    f'    <item index="{i+1}" value="{k}"  cost="0"  evaluation="{self(k)}" description="{k}">\n'
                    f"    </item>\n"
                )
        else:
            for i, (k, v) in enumerate(ienumerate(self.mapping)):
                if isinstance(k, tuple) or isinstance(k, list):
                    k = "-".join([str(_) for _ in k])
                output += (
                    f'    <item index="{i+1}" value="{k}"  cost="0"  evaluation="{v}" description="{k}">\n'
                    f"    </item>\n"
                )
        output += "</issue>\n"
        output += '<weight index="1" value="1.0">\n</weight>\n'
        return output

    def __str__(self) -> str:
        return f"mapping: {self.mapping}\ndefault: {self.default}"

    @classmethod
    def random(
        cls,
        issues: List["Issue"],
        reserved_value=(0.0, 1.0),
        normalized=True,
        max_n_outcomes: int = 10000,
    ):
        outcomes = (
            Issue.enumerate(issues)
            if Issue.num_outcomes(issues) <= max_n_outcomes
            else Issue.sample(
                issues, max_n_outcomes, with_replacement=False, fail_if_not_enough=False
            )
        )
        return UtilityFunction.generate_random(1, outcomes)[0]


class NonLinearUtilityAggregationFunction(UtilityFunction):
    r"""A nonlinear utility function.

    Allows for the modeling of a single nonlinear utility function that combines the utilities of different issues.

    Args:
        issue_utilities: A set of mappings from issue values to utility functions. These are generic mappings so
                        \ `Callable`\ (s) and \ `Mapping`\ (s) are both accepted
        f: A nonlinear function mapping from a dict of utility_function-per-issue to a float
        name: name of the utility function. If None a random name will be generated.

    Notes:

        The utility is calculated as:

        .. math::

                u = f\\left(u_0\\left(i_0\\right), u_1\\left(i_1\\right), ..., u_n\\left(i_n\\right)\\right)

        where :math:`u_j()` is the utility function for issue :math:`j` and :math:`i_j` is value of issue :math:`j` in the
        evaluated outcome.


    Examples:
        >>> issues = [Issue((10.0, 20.0), 'price'), Issue(['delivered', 'not delivered'], 'delivery')
        ...           , Issue(5, 'quality')]
        >>> print(list(map(str, issues)))
        ['price: (10.0, 20.0)', "delivery: ['delivered', 'not delivered']", 'quality: (0, 4)']
        >>> g = NonLinearUtilityAggregationFunction({ 'price': lambda x: 2.0*x
        ...                                         , 'delivery': {'delivered': 10, 'not delivered': -10}
        ...                                         , 'quality': MappingUtilityFunction(lambda x: x-3)}
        ...         , f=lambda u: u['price']  + 2.0 * u['quality'])
        >>> float(g({'quality': 2, 'price': 14.0, 'delivery': 'delivered'})) - ((2.0*14)+2.0*(2.0-3.0))
        0.0
        >>> g = NonLinearUtilityAggregationFunction({'price'    : lambda x: 2.0*x
        ...                                         , 'delivery': {'delivered': 10, 'not delivered': -10}}
        ...         , f=lambda u: 2.0 * u['price'] )
        >>> float(g({'price': 14.0, 'delivery': 'delivered'})) - (2.0*(2.0*14))
        0.0

    """

    def xml(self, issues: List[Issue]) -> str:
        raise NotImplementedError(f"Cannot convert {self.__class__.__name__} to xml")

    def __init__(
        self,
        issue_utilities: MutableMapping[Any, GenericMapping],
        f: Callable[[Dict[Any, UtilityValue]], UtilityValue],
        name: Optional[str] = None,
        reserved_value: UtilityValue = float("-inf"),
        ami: AgentMechanismInterface = None,
        outcome_type: Optional[Type] = None,
        id = None,
    ) -> None:
        super().__init__(
            name=name, outcome_type=outcome_type, reserved_value=reserved_value, ami=ami, id=id,
        )
        self.issue_utilities = issue_utilities
        self.f = f

    def eval(self, offer: Optional["Outcome"]) -> Optional[UtilityValue]:
        if offer is None:
            return self.reserved_value
        if self.issue_utilities is None:
            raise ValueError(
                "No issue utilities were set. Call set_params() or use the constructor"
            )

        u = {}
        for k in ikeys(self.issue_utilities):
            v = iget(offer, k)
            u[k] = gmap(iget(self.issue_utilities, k), v)
        return self.f(u)

    @UtilityFunction.outcome_type.setter
    def outcome_type(self, value: AgentMechanismInterface):
        UtilityFunction.outcome_type.fset(self, value)
        if isinstance(self.issue_utilities, dict):
            for k, v in self.issue_utilities.items():
                if isinstance(v, UtilityFunction):
                    v.outcome_type = value
        else:
            for v in self.issue_utilities:
                if isinstance(v, UtilityFunction):
                    v.outcome_type = value

    @UtilityFunction.ami.setter
    def ami(self, value: AgentMechanismInterface):
        UtilityFunction.ami.fset(self, value)
        if isinstance(self.issue_utilities, dict):
            for k, v in self.issue_utilities.items():
                if isinstance(v, UtilityFunction):
                    v.ami = value
        else:
            for v in self.issue_utilities:
                if isinstance(v, UtilityFunction):
                    v.ami = value


class HyperRectangleUtilityFunction(UtilityFunction):
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

         >>> issues = [Issue(10), Issue(5), Issue(4)]

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

    def xml(self, issues: List[Issue]) -> str:
        """Represents the function as XML

        Args:
            issues:

        Examples:

            >>> f = HyperRectangleUtilityFunction(outcome_ranges=[
            ...                                        {0: (1.0, 2.0), 1: (1.0, 2.0)},
            ...                                        {0: (1.4, 2.0), 2: (2.0, 3.0)}]
            ...                                , utilities= [2.0, 9.0 + 4.0])
            >>> print(f.xml([Issue((0.0, 4.0), name='0'), Issue((0.0, 9.0), name='1')
            ... , Issue((0.0, 9.0), name='2')]).strip())
            <issue index="1" name="0" vtype="real" type="real" etype="real">
                <range lowerbound="0.0" upperbound="4.0"></range>
            </issue><issue index="2" name="1" vtype="real" type="real" etype="real">
                <range lowerbound="0.0" upperbound="9.0"></range>
            </issue><issue index="3" name="2" vtype="real" type="real" etype="real">
                <range lowerbound="0.0" upperbound="9.0"></range>
            </issue><utility_function maxutility="-1.0">
                <ufun type="PlainUfun" weight="1" aggregation="sum">
                    <hyperRectangle utility_function="2.0">
                        <INCLUDES index="0" min="1.0" max="2.0" />
                        <INCLUDES index="1" min="1.0" max="2.0" />
                    </hyperRectangle>
                    <hyperRectangle utility_function="13.0">
                        <INCLUDES index="0" min="1.4" max="2.0" />
                        <INCLUDES index="2" min="2.0" max="3.0" />
                    </hyperRectangle>
                </ufun>
            </utility_function>

        """
        output = ""
        for i, issue in enumerate(ivalues(issues)):
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
            output += f'        <hyperRectangle utility_function="{u * w}">\n'
            for indx in ikeys(rect):
                values = iget(rect, indx, None)
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

    def __init__(
        self,
        outcome_ranges: Iterable[OutcomeRange],
        utilities: Union[Floats, OutcomeUtilityMappings],
        weights: Optional[Floats] = None,
        *,
        ignore_issues_not_in_input=False,
        ignore_failing_range_utilities=False,
        name: Optional[str] = None,
        reserved_value: UtilityValue = float("-inf"),
        ami: AgentMechanismInterface = None,
        outcome_type: Optional[Type] = None,
        id = None,
    ) -> None:
        super().__init__(
            name=name, outcome_type=outcome_type, reserved_value=reserved_value, ami=ami, id=id,
        )
        self.outcome_ranges = outcome_ranges
        self.mappings = utilities
        self.weights = weights
        self.ignore_issues_not_in_input = ignore_issues_not_in_input
        self.ignore_failing_range_utilities = ignore_failing_range_utilities
        self.adjust_params()

    def adjust_params(self):
        if self.weights is None:
            self.weights = [1.0] * len(self.outcome_ranges)

    def eval(self, offer: Optional["Outcome"]) -> Optional[UtilityValue]:
        if offer is None:
            return self.reserved_value
        u = ExactUtilityValue(0.0)
        for weight, outcome_range, mapping in zip(
            self.weights, self.outcome_ranges, self.mappings
        ):  # type: ignore
            # fail on any outcome_range that constrains issues not in the presented outcome
            if outcome_range is not None and set(ikeys(outcome_range)) - set(
                ikeys(offer)
            ) != set([]):
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


class NonlinearHyperRectangleUtilityFunction(UtilityFunction):
    """A utility function defined as a set of outcome_ranges.


    Args:
           hypervolumes: see `HyperRectangleUtilityFunction`
           mappings: see `HyperRectangleUtilityFunction`
           f: A nonlinear function to combine the results of `mappings`
           name: name of the utility function. If None a random name will be generated
    """

    def xml(self, issues: List[Issue]) -> str:
        raise NotImplementedError(f"Cannot convert {self.__class__.__name__} to xml")

    def __init__(
        self,
        hypervolumes: Iterable[OutcomeRange],
        mappings: OutcomeUtilityMappings,
        f: Callable[[List[UtilityValue]], UtilityValue],
        name: Optional[str] = None,
        reserved_value: UtilityValue = float("-inf"),
        ami: AgentMechanismInterface = None,
        outcome_type: Optional[Type] = None,
        id=None,
    ) -> None:
        super().__init__(
            name=name, outcome_type=outcome_type, reserved_value=reserved_value, ami=ami, id=id,
        )
        self.hypervolumes = hypervolumes
        self.mappings = mappings
        self.f = f

    def eval(self, offer: Optional["Outcome"]) -> Optional[UtilityValue]:
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
