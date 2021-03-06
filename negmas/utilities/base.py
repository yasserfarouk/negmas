import warnings
import itertools
from math import sqrt
import random
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from functools import reduce
from operator import mul
from typing import (
    Callable,
    Collection,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    Any,
)

import numpy as np

from negmas.common import AgentMechanismInterface, NamedObject
from negmas.generics import ienumerate, ivalues
from negmas.helpers import Distribution, PATH, ikeys, snake_case, get_full_type_name
from negmas.outcomes import (
    Issue,
    Outcome,
    outcome_as,
    outcome_as_dict,
    outcome_as_tuple,
    outcome_is_valid,
    sample_outcomes,
)
from negmas.serialization import serialize, deserialize

__all__ = [
    "UtilityDistribution",
    "UtilityValue",
    "UtilityDistribution",
    "UtilityFunction",
    "INVALID_UTILITY",
    "OutcomeUtilityMapping",
    "OutcomeUtilityMappings",
    "ExactUtilityValue",
]

INVALID_UTILITY = float("-inf")


# Helper Types just used to make type hinting more readable
OutcomeUtilityMapping = Union[
    Callable[[Union["Outcome", int, str, float]], "UtilityValue"],
    Mapping[Union[Sequence, Mapping, int, str, float], "UtilityValue"],
]
"""A mapping from an outcome to its utility value"""


OutcomeUtilityMappings = List[OutcomeUtilityMapping]
"""Maps from multi-issue or single-issue outcomes to Negotiator values."""


IssueUtilityFunctionMapping = Union[
    Callable[["Issue"], "UtilityFunction"], Mapping["Issue", "UtilityFunction"],
]
"""A mapping from issues to utility functions"""


IssueUtilityFunctionMappings = List[IssueUtilityFunctionMapping]
"""A list of `IssueUtilityFunctionMapping`"""


UtilityDistribution = Distribution
"""A probability distribution over utility values"""


class ExactUtilityValue(float):
    """Encapsulates a single offerable_outcomes utility_function value."""


UtilityValue = Union[UtilityDistribution, float]
"""
Either a utility_function distribution or an exact offerable_outcomes
utility_function value.

`UtilityFunction`s always return a `UtilityValue` which makes it easier to
implement algorithms relying  on probabilistic modeling of utility functions.
"""


class UtilityFunction(ABC, NamedObject):
    """The abstract base class for all utility functions.

    A utility function encapsulates a mapping from outcomes to UtilityValue(s).
    This is a generalization of standard
    utility functions that are expected to always return a real-value.
    This generalization is useful for modeling cases
    in which only partial knowledge of the utility function is available.

    To define a new utility function, you have to override one of the following
    methods:

        - `eval` to define a standard ufun mapping outcomes to utility values.
        - `is_better` to define prferences by a partial ordering over outcomes
          implemented through bilateral comparisons
        - `rank` to define the preferences by partial ordering defined by
          a ranking of outcomes.

    Args:
        name: Name of the utility function. If None, a random name will
              be given.
        reserved_value: The value to return if the input offer to
                        `__call__` is None
        outcome_type: The type to use when evauating utilities.
                      It can be tuple, dict, or any `OutcomeType`
        issue_names: The names of issues. Only needed if outcome_type is not tuple
        issues: The list of issues for which the ufun is defined (optional)
        ami: The `AgentMechanismInterface` for a mechanism for which the ufun
             is defined (optinoal)
        id: An optional system-wide unique identifier. You should not change
            the default value except in special circumstances like during
            serialization and should always guarantee system-wide uniquness
            if you set this value explicitly

    Remarks:
        - If ami is given, it overrides outcome_type, issues and issue_names
        - If issues is given, it overrides issue_names
        - One of `eval`, `is_better`, `rank` **MUST** be overriden, otherwise
          calling any of them will lead to an infinite loop which is very hard
          to debug.

    """

    def __init__(
        self,
        name: Optional[str] = None,
        reserved_value: UtilityValue = float("-inf"),
        outcome_type: Optional[Type] = None,
        issue_names: Optional[List[str]] = None,
        issues: List["Issue"] = None,
        ami: AgentMechanismInterface = None,
        id: str = None,
    ) -> None:
        super().__init__(name, id=id)
        self.reserved_value = reserved_value
        self._ami = ami
        self._inverse_initialzed = False
        self._issues = issues if ami is None or ami.issues is None else ami.issues
        self._outcome_type = outcome_type if ami is None else ami.outcome_type
        self.issue_names = (
            issue_names if self._issues is None else [_.name for _ in self._issues]
        )
        if self._outcome_type is not None and (
            not issubclass(self._outcome_type, tuple) and self.issue_names is None
        ):
            raise ValueError(
                "You must specify issue_names because you are using a non-tuple outcome type"
            )

    @property
    def ami(self):
        return self._ami

    @ami.setter
    def ami(self, value: AgentMechanismInterface):
        self._ami = value

    @property
    def issues(self):
        return self._issues

    @issues.setter
    def issues(self, value: List[Issue]):
        self._issues = value
        self.issue_names = [_.name for _ in value] if value else value

    @property
    def outcome_type(self):
        if not hasattr(self, "_outcome_type"):
            return None
        return self._outcome_type

    @outcome_type.setter
    def outcome_type(self, value: Type):
        self._outcome_type = value

    @property
    def is_dynamic(self):
        """
        Whether the utility function can potentially depend on negotiation
        state (mechanism information).

        - If this property is `False`,  the ufun can safely be assumed to be
          static (not dependent on negotiation state).
        - If this property is `True`, the ufun may depend on negotiation state
          but it may also not depend on it.
        """
        # TODO: fix this. Now we always attach ami when the ufun joins a neg.
        #       so having no ami is not a good proxy for being dynamic
        return self.ami is None

    @classmethod
    def from_genius(cls, file_name: PATH, **kwargs):
        """Imports a utility function from a GENIUS XML file.

        Args:

            file_name (str): File name to import from

        Returns:

            A utility function object (depending on the input file)


        Examples:

            >>> from negmas.utilities import UtilityFunction
            >>> import pkg_resources
            >>> u, d = UtilityFunction.from_genius(file_name = pkg_resources.resource_filename('negmas'
            ...                                      , resource_name='tests/data/Laptop/Laptop-C-prof1.xml'))
            >>> u.__class__.__name__
            'LinearUtilityAggregationFunction'
            >>> u.reserved_value
            0.0
            >>> d
            1.0

        Remarks:
            See ``from_xml_str`` for all the parameters

        """
        with open(file_name, "r") as f:
            xml_str = f.read()
            return cls.from_xml_str(xml_str=xml_str, **kwargs)

    @classmethod
    def to_genius(
        cls, u: "UtilityFunction", issues: List[Issue], file_name: PATH, **kwargs
    ):
        """Exports a utility function to a GENIUS XML file.

        Args:

            file_name (str): File name to export to
            u: utility function
            issues: The issues being considered as defined in the domain

        Returns:

            None


        Examples:

            >>> from negmas.utilities import UtilityFunction
            >>> from negmas.inout import load_genius_domain
            >>> import pkg_resources
            >>> _, _, issues = load_genius_domain(domain_file_name=pkg_resources.resource_filename('negmas'
            ...                                             , resource_name='tests/data/Laptop/Laptop-C-domain.xml')
            ...             , keep_issue_names=False)
            >>> u, discount = UtilityFunction.from_genius(file_name=pkg_resources.resource_filename('negmas'
            ...                                             , resource_name='tests/data/Laptop/Laptop-C-prof1.xml')
            ...             , keep_issue_names=False)
            >>> UtilityFunction.to_genius(u=u, issues=issues, discount_factor=discount
            ...     , file_name = pkg_resources.resource_filename('negmas'
            ...                                             , resource_name='tests/data/LaptopConv/Laptop-C-prof1.xml'))

        Remarks:
            See ``to_xml_str`` for all the parameters

        """
        with open(file_name, "w") as f:
            f.write(cls.to_xml_str(u=u, issues=issues, **kwargs))

    @classmethod
    def to_xml_str(
        cls, u: "UtilityFunction", issues: List[Issue], discount_factor=None
    ) -> str:
        """Exports a utility function to a well formatted string"""
        if issues is not None:
            n_issues = len(issues)
        else:
            n_issues = 0
        output = (
            f'<utility_space type="any" number_of_issues="{n_issues}">\n'
            f'<objective index="1" etype="objective" type="objective" description="" name="any">\n'
        )

        output += u.xml(issues=issues)
        if "</objective>" not in output:
            output += "</objective>\n"
            if discount_factor is not None:
                output += f'<discount_factor value="{discount_factor}" />\n'
        if u.reserved_value != float("-inf") and "<reservation value" not in output:
            output += f'<reservation value="{u.reserved_value}" />\n'
        if "</utility_space>" not in output:
            output += "</utility_space>\n"
        return output

    @classmethod
    def from_xml_str(
        cls,
        xml_str: str,
        domain_issues: Optional[List[Issue]] = None,
        force_single_issue=False,
        force_numeric=False,
        keep_issue_names=True,
        keep_value_names=True,
        safe_parsing=True,
        normalize_utility=False,
        normalize_max_only=False,
        max_n_outcomes: int = 1_000_000,
        ignore_discount=False,
        ignore_reserved=False,
    ):
        """Imports a utility function from a GENIUS XML string.

        Args:

            xml_str (str): The string containing GENIUS style XML utility function definition
            domain_issues (List[Issue]): Optional issue space to confirm that the utility function is valid
            force_single_issue (bool): Tries to generate a MappingUtility function with a single issue which is the
            product of all issues in the input
            keep_issue_names (bool): Keep names of issues
            keep_value_names (bool): Keep names of values
            safe_parsing (bool): Turn on extra checks
            normalize_utility (bool): Normalize the output utilities to the range from 0 to 1
            normalize_max_only (bool): If True ensures that max(utility) = 1 but does not ensure that min(utility) = 0. and
                                      if false, ensures both max(utility) = 1 and min(utility) = 0
            max_n_outcomes (int): Maximum number of outcomes allowed (effective only if force_single_issue is True)

        Returns:

            A utility function object (depending on the input file)


        Examples:

            >>> import pkg_resources
            >>> u, _ = UtilityFunction.from_xml_str(open(pkg_resources.resource_filename('negmas'
            ...                                      , resource_name='tests/data/Laptop/Laptop-C-prof1.xml')
            ...                                      , 'r').read(), force_single_issue=False
            ...                                     , normalize_utility=True
            ...                                     , keep_issue_names=False, keep_value_names=True)

            >>> u, _ = UtilityFunction.from_xml_str(open(pkg_resources.resource_filename('negmas'
            ...                                      , resource_name='tests/data/Laptop/Laptop-C-prof1.xml')
            ...                                      , 'r').read()
            ...                                      , force_single_issue=True, normalize_utility=False)
            >>> assert abs(u(("Dell+60 Gb+19'' LCD",)) - 21.987727736172488) < 0.000001
            >>> assert abs(u(("HP+80 Gb+20'' LCD",)) - 22.68559475583014) < 0.000001

            >>> u, _ = UtilityFunction.from_xml_str(open(pkg_resources.resource_filename('negmas'
            ...                                      , resource_name='tests/data/Laptop/Laptop-C-prof1.xml')
            ...                                      , 'r').read(), force_single_issue=True
            ... , keep_issue_names=False, keep_value_names=False, normalize_utility=False)
            >>> assert abs(u((0,)) - 21.987727736172488) < 0.000001

            >>> u, _ = UtilityFunction.from_xml_str(open(pkg_resources.resource_filename('negmas'
            ...                                      , resource_name='tests/data/Laptop/Laptop-C-prof1.xml')
            ...                 , 'r').read(), force_single_issue=False, normalize_utility=False)
            >>> assert abs(u({'Laptop': 'Dell', 'Harddisk': '60 Gb', 'External Monitor': "19'' LCD"}) - 21.987727736172488) < 0.000001
            >>> assert abs(u({'Laptop': 'HP', 'Harddisk': '80 Gb', 'External Monitor': "20'' LCD"}) - 22.68559475583014) < 0.000001

            >>> u, _ = UtilityFunction.from_xml_str(open(pkg_resources.resource_filename('negmas'
            ...                                      , resource_name='tests/data/Laptop/Laptop-C-prof1.xml')
            ...                                      , 'r').read()
            ...                                      , force_single_issue=True, normalize_utility=True)

            >>> u, _ = UtilityFunction.from_xml_str(open(pkg_resources.resource_filename('negmas'
            ...                                      , resource_name='tests/data/Laptop/Laptop-C-prof1.xml')
            ...                                      , 'r').read(), force_single_issue=True
            ... , keep_issue_names=False, keep_value_names=False, normalize_utility=True)

            >>> u, _ = UtilityFunction.from_xml_str(open(pkg_resources.resource_filename('negmas'
            ...                                      , resource_name='tests/data/Laptop/Laptop-C-prof1.xml')
            ...         , 'r').read(), force_single_issue=False, normalize_utility=True)

        """
        from negmas.utilities.linear import LinearUtilityAggregationFunction
        from negmas.utilities.nonlinear import MappingUtilityFunction
        from negmas.utilities.complex import ComplexWeightedUtilityFunction
        from negmas.utilities.nonlinear import HyperRectangleUtilityFunction

        root = ET.fromstring(xml_str)
        if safe_parsing and root.tag != "utility_space":
            raise ValueError(f"Root tag is {root.tag}: Expected utility_space")

        if domain_issues is not None:
            if isinstance(domain_issues, list):
                domain_issues: Dict[str, Issue] = dict(
                    zip([_.name for _ in domain_issues], domain_issues)
                )
            elif isinstance(domain_issues, Issue) and force_single_issue:
                domain_issues = dict(zip([domain_issues.name], [domain_issues]))
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
            if safe_parsing:
                pass
                # raise ValueError(f'No objective child was found in the root')
            objective = root
        weights = {}
        issues = {}
        real_issues = {}
        issue_info = {}
        issue_keys = {}
        rects, rect_utils = [], []

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
                        key = issue_keys[int(r.attrib["index"]) - 1]
                        ranges[key] = (
                            utiltype(r.attrib["min"]),
                            utiltype(r.attrib["max"]),
                        )
                    rects.append(ranges)
            else:
                raise ValueError(f"Unknown ufun type {utype}")
            total_util = total_util if not max_utility else max_utility
            if normalize_utility:
                for i, u in enumerate(rect_utils):
                    rect_utils[i] = u / total_util
            return rects, rect_utils

        for child in objective:
            if child.tag == "weight":
                indx = int(child.attrib["index"]) - 1
                weights[indx] = float(child.attrib["value"])
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
                        f"Cannot find ufun tag inside a utility_function tag"
                    )

            elif child.tag == "issue":
                indx = int(child.attrib["index"]) - 1
                myname = child.attrib["name"]
                issue_key = myname if keep_issue_names else indx
                if domain_issues is not None and myname not in domain_issues.keys():
                    raise ValueError(
                        f"Issue {myname} is not in the input issue names ({domain_issues.keys()})"
                    )
                issue_info[issue_key] = {"name": myname, "index": indx}
                issue_keys[indx] = issue_key
                info = {"type": "discrete", "etype": "discrete", "vtype": "discrete"}
                for a in ("type", "etype", "vtype"):
                    info[a] = child.attrib.get(a, info[a])
                mytype = info["type"]
                value_scale = None
                value_shift = None
                if mytype == "discrete":
                    issues[issue_key] = {}
                    if (
                        domain_issues is not None
                        and domain_issues[myname].is_uncountable()
                    ):
                        raise ValueError(
                            f"Got a {mytype} issue but expected a continuous valued issue"
                        )
                    # issues[indx]['items'] = {}
                elif mytype in ("integer", "real"):
                    lower, upper = (
                        child.attrib.get("lowerbound", None),
                        child.attrib.get("upperbound", None),
                    )
                    for rng_child in child:
                        if rng_child.tag == "range":
                            lower, upper = (
                                rng_child.attrib.get("lowerbound", lower),
                                rng_child.attrib.get("upperbound", upper),
                            )
                    if mytype == "integer":
                        issues[issue_key] = {}
                        if (
                            domain_issues is not None
                            and domain_issues[myname].is_uncountable()
                        ):
                            raise ValueError(
                                f"Got a {mytype} issue but expected a continuous valued issue"
                            )
                        # issues[indx]['items'] = {}
                        lower, upper = int(lower), int(upper)
                        for i in range(lower, upper + 1):
                            if domain_issues is not None and not outcome_is_valid(
                                (i,), [domain_issues[myname]]
                            ):
                                raise ValueError(
                                    f"Value {i} is not in the domain issue values: "
                                    f"{domain_issues[myname].values}"
                                )
                            issues[issue_key][i] = i if keep_value_names else i - lower
                    else:
                        lower, upper = float(lower), float(upper)
                        if (
                            domain_issues is not None
                            and not domain_issues[myname].is_uncountable()
                        ):
                            n_steps = domain_issues[myname].cardinality
                            delta = (n_steps - 1) / (upper - lower)
                            value_shift = -lower * delta
                            value_scale = delta
                            lower, upper = 0, n_steps - 1
                            issues[issue_key] = {}
                            for i in range(lower, upper + 1):
                                issues[issue_key][i] = (
                                    str(i) if keep_value_names else i - lower
                                )
                        else:
                            real_issues[issue_key] = {}
                            real_issues[issue_key]["range"] = (lower, upper)
                            real_issues[issue_key]["key"] = issue_key
                else:
                    raise ValueError(f"Unknown type: {mytype}")
                if mytype in "discrete" or "integer" or "real":
                    found_values = False
                    for item in child:
                        if item.tag == "item":
                            if mytype == "real":
                                raise ValueError(
                                    f"cannot specify item utilities for real type"
                                )
                            item_indx = int(item.attrib["index"]) - 1
                            item_name: str = item.attrib.get("value", None)
                            if item_name is None:
                                continue
                            item_key = (
                                item_name
                                if keep_value_names
                                and item_name is not None
                                and not force_numeric
                                else item_indx
                            )
                            if domain_issues is not None:
                                domain_all = list(domain_issues[myname].all)
                                if len(domain_all) > 0 and isinstance(
                                    domain_all[0], int
                                ):
                                    item_key = int(item_key)
                                if len(domain_all) > 0 and isinstance(
                                    domain_all[0], int
                                ):
                                    item_name = int(item_name)
                                if item_name not in domain_all:
                                    raise ValueError(
                                        f"Value {item_name} is not in the domain issue values: "
                                        f"{domain_issues[myname].values}"
                                    )
                                if len(domain_all) > 0 and isinstance(
                                    domain_all[0], int
                                ):
                                    item_name = str(item_name)
                            # TODO check that this casting is correct
                            if mytype == "integer":
                                item_key = int(item_key)
                            val = item.attrib.get("evaluation", None)
                            if val is None:
                                raise ValueError(
                                    f"Item {item_key} of issue {item_name} has not evaluation attribute!!"
                                )
                            issues[issue_key][item_key] = float(val)
                            found_values = True
                        elif item.tag == "evaluator":
                            if item.attrib["ftype"] == "linear":
                                offset = item.attrib.get(
                                    "offset", item.attrib.get("parameter0", 0.0)
                                )
                                slope = item.attrib.get(
                                    "slope", item.attrib.get("parameter1", 1.0)
                                )
                                offset, slope = float(offset), float(slope)
                                if value_scale is None:
                                    fun = lambda x: offset + slope * float(x)
                                else:
                                    fun = lambda x: offset + slope * (
                                        value_scale * float(x) + value_shift
                                    )
                            elif item.attrib["ftype"] == "triangular":
                                strt = item.attrib.get("parameter0", 0.0)
                                end = item.attrib.get("parameter1", 1.0)
                                middle = item.attrib.get("parameter2", 1.0)
                                strt, end, middle = (
                                    float(strt),
                                    float(end),
                                    float(middle),
                                )
                                offset1, slope1 = strt, (middle - strt)
                                offset2, slope2 = middle, (middle - end)
                                if value_scale is None:
                                    fun = (
                                        lambda x: offset1 + slope1 * float(x)
                                        if x < middle
                                        else offset2 + slope2 * float(x)
                                    )
                                else:
                                    fun = (
                                        lambda x: offset1
                                        + slope1
                                        * (value_scale * float(x) + value_shift)
                                        if x < middle
                                        else offset2
                                        + slope2
                                        * (value_scale * float(x) + value_shift)
                                    )
                            else:
                                raise ValueError(
                                    f'Unknown ftype {item.attrib["ftype"]}'
                                )
                            if mytype == "real" and value_scale is None:
                                real_issues[issue_key]["fun"] = fun
                            else:
                                for item_key, value in issues[issue_key].items():
                                    issues[issue_key][item_key] = fun(value)
                                found_values = True
                    if not found_values and issue_key in issues.keys():
                        issues.pop(issue_key, None)
                else:
                    """Here goes the code for real-valued issues"""

        if not keep_issue_names:
            issues = [issues[_] for _ in issues.keys()]
            real_issues = [real_issues[_] for _ in sorted(real_issues.keys())]
            for i, issue in enumerate(issues):
                issues[i] = [issue[_] for _ in issue.keys()]

        if safe_parsing and (
            len(weights) > 0
            and len(weights) != len(issues) + len(real_issues)
            and len(weights) != len(issues)
        ):
            raise ValueError(
                f"Got {len(weights)} weights for {len(issues)} issues and {len(real_issues)} real issues"
            )

        if force_single_issue and (
            len(rects) > 0
            or len(real_issues) > 1
            or (len(real_issues) > 0 and len(issues) > 0)
        ):
            raise ValueError(
                f"Cannot force single issue with a hyper-volumes based function"
            )

        # add utilities specified not as hyper-rectangles
        u = None
        if len(issues) > 0:
            if force_single_issue:
                if len(weights) > 0:
                    for key, issue in zip(ikeys(issues), ivalues(issues)):
                        try:
                            w = weights[issue_info[key]["index"]]
                        except:
                            w = 1.0
                        for item_key in ikeys(issue):
                            issue[item_key] *= w
                n_outcomes = None
                if max_n_outcomes is not None:
                    n_items = [len(_) for _ in ivalues(issues)]
                    n_outcomes = reduce(mul, n_items, 1)
                    if n_outcomes > max_n_outcomes:
                        return None, reserved_value, discount_factor
                if keep_value_names:
                    names = itertools.product(
                        *[
                            [
                                str(item_key).replace("&", "-")
                                for item_key in ikeys(items)
                            ]
                            for issue_key, items in zip(ikeys(issues), ivalues(issues))
                        ]
                    )
                    names = map(lambda items: ("+".join(items),), names)
                else:
                    if n_outcomes is None:
                        n_items = [len(_) for _ in ivalues(issues)]
                        n_outcomes = reduce(mul, n_items, 1)
                    names = [(_,) for _ in range(n_outcomes)]
                utils = itertools.product(
                    *[
                        [item_utility for item_utility in ivalues(items)]
                        for issue_key, items in zip(ikeys(issues), ivalues(issues))
                    ]
                )
                utils = map(lambda vals: sum(vals), utils)
                if normalize_utility:
                    utils = list(utils)
                    umax, umin = max(utils), (0.0 if normalize_max_only else min(utils))
                    if umax != umin:
                        utils = [(_ - umin) / (umax - umin) for _ in utils]
                if keep_issue_names:
                    u = MappingUtilityFunction(dict(zip(names, utils)))
                else:
                    u = MappingUtilityFunction(dict(zip(names, utils)))
            else:
                utils = None
                if normalize_utility:
                    utils = itertools.product(
                        *[
                            [
                                item_utility * weights[issue_info[issue_key]["index"]]
                                for item_utility in ivalues(items)
                            ]
                            for issue_key, items in zip(ikeys(issues), ivalues(issues))
                        ]
                    )
                    if len(weights) > 0:
                        ws = dict()
                        for key, issue in zip(ikeys(issues), ivalues(issues)):
                            try:
                                ws[key] = weights[issue_info[key]["index"]]
                            except:
                                ws[key] = 1.0
                        wsum = sum(weights.values())
                    else:
                        ws = [1.0] * len(issues)
                        wsum = len(issues)

                    utils = list(map(sum, utils))
                    umax, umin = max(utils), (0.0 if normalize_max_only else min(utils))
                    factor = umax - umin
                    if factor > 1e-8:
                        offset = umin / (wsum * factor)
                    else:
                        offset = 0.0
                        factor = umax if umax > 1e-8 else 1.0
                    for key, issue in ienumerate(issues):
                        for item_key in ikeys(issue):
                            issues[key][item_key] = (
                                issues[key][item_key] / factor - offset
                            )
                if len(issues) > 1:
                    ws = dict()
                    if len(weights) > 0:
                        for key, issue in zip(ikeys(issues), ivalues(issues)):
                            try:
                                ws[key] = weights[issue_info[key]["index"]]
                            except:
                                ws[key] = 1.0

                    if isinstance(issues, list):
                        ws = [ws[i] for i in range(len(issues))]

                    u = LinearUtilityAggregationFunction(
                        issue_utilities=issues, weights=ws
                    )
                else:
                    if len(weights) > 0:
                        for key, issue in zip(ikeys(issues), ivalues(issues)):
                            try:
                                w = weights[issue_info[key]["index"]]
                            except:
                                w = 1.0
                            for item_key in ikeys(issue):
                                issue[item_key] *= w
                    first_key = list(ikeys(issues))[0]
                    if utils is None:
                        utils = ivalues(issues[first_key])
                    if keep_issue_names:
                        u = MappingUtilityFunction(
                            dict(zip([(_,) for _ in ikeys(issues[first_key])], utils))
                        )
                    else:
                        u = MappingUtilityFunction(
                            dict(zip([(_,) for _ in range(len(utils))], utils))
                        )

        # add real_valued issues
        if len(real_issues) > 0:
            if len(weights) > 0:
                for key, issue in zip(ikeys(real_issues), ivalues(real_issues)):
                    try:
                        w = weights[issue_info[key]["index"]]
                    except:
                        w = 1.0
                    issue["fun_final"] = lambda x: w * issue["fun"](x)
            if normalize_utility:
                n_items_to_test = 10
                utils = itertools.product(
                    *[
                        [
                            issue["fun"](_)
                            for _ in np.linspace(
                                issue["range"][0],
                                issue["range"][1],
                                num=n_items_to_test,
                                endpoint=True,
                            )
                        ]
                        for key, issue in zip(ikeys(real_issues), ivalues(real_issues))
                    ]
                )
                if len(weights) > 0:
                    ws = dict()
                    for key, issue in zip(ikeys(issues), ivalues(issues)):
                        try:
                            ws[key] = weights[issue_info[key]["index"]]
                        except:
                            ws[key] = 1.0
                    wsum = sum(weights.values())
                else:
                    ws = [1.0] * len(issues)
                    wsum = len(issues)

                utils = list(map(lambda vals: sum(vals), utils))
                umax, umin = max(utils), (0.0 if normalize_max_only else min(utils))
                factor = umax - umin
                if factor > 1e-8:
                    offset = (umin / wsum) / factor
                else:
                    offset = 0.0
                    factor = 1.0
                for key, issue in real_issues.items():
                    issue["fun_final"] = lambda x: w * issue["fun"](x) / factor - offset
            u_real = LinearUtilityAggregationFunction(
                issue_utilities={_["key"]: _["fun_final"] for _ in real_issues.values()}
            )
            if u is None:
                u = u_real
            else:
                u = ComplexWeightedUtilityFunction(
                    ufuns=[u, u_real], weights=[1.0, 1.0]
                )

        # add hyper rectangles issues
        if len(rects) > 0:
            uhyper = HyperRectangleUtilityFunction(
                outcome_ranges=rects, utilities=rect_utils
            )
            if u is None:
                u = uhyper
            else:
                u = ComplexWeightedUtilityFunction(
                    ufuns=[u, uhyper], weights=[1.0, 1.0]
                )
        if not ignore_reserved and u is not None:
            u.reserved_value = reserved_value
        if ignore_discount:
            discount_factor = None
        return u, discount_factor

    def __getitem__(self, offer: "Outcome") -> Optional[UtilityValue]:
        """Overrides [] operator to call the ufun allowing it to act as a mapping"""
        return self(offer)

    def __call__(self, offer: "Outcome") -> UtilityValue:
        """Calculate the utility_function value for a given outcome.

        Args:
            offer: The offer to be evaluated.


        Remarks:

            - It calls the abstract method `eval` after opationally adjusting the
              outcome type.
            - It is preferred to override eval instead of directly overriding this method
            - You cannot return None from overriden eval() functions but raise an exception (ValueError) if it was
              not possible to calculate the UtilityValue.
            - Return A UtilityValue not a float for real-valued utilities for the benefit of inspection code.
            - Return the reserved value if the offer was None

        Returns:
            UtilityValue: The utility_function value which may be a distribution. If `None` it means the
                          utility_function value cannot be calculated.
        """
        if not self.issues and self.ami:
            self.issues = self.ami.issues
        if offer is None:
            return self.reserved_value
        if self.outcome_type is None:
            pass
        elif issubclass(self.outcome_type, tuple):
            offer = outcome_as_tuple(offer)
        elif issubclass(self.outcome_type, dict):
            offer = outcome_as_dict(offer, self.issue_names)
        else:
            if isinstance(offer, dict):
                offer = self.outcome_type(**offer)
            elif isinstance(offer, tuple):
                offer = self.outcome_type(*offer)
            else:
                offer = self.outcome_type(offer)
        return self.eval(offer)

    def eval(self, offer: "Outcome") -> UtilityValue:
        """Calculate the utility value for a given outcome.

        Args:
            offer: The offer to be evaluated.

        Returns:
            UtilityValue: The utility_function value which may be a distribution.
                          If `None` it means the utility_function value cannot
                          be calculated.

        Remarks:
            - You cannot return None from overriden eval() functions but
              raise an exception (ValueError) if it was
              not possible to calculate the UtilityValue.
            - Typehint the return type as a `UtilityValue` instead of a float
              for the benefit of inspection code.
            - Return the reserved value if the offer was None
            - *NEVER* call the baseclass using super() when overriding this
              method. Calling super will lead to an infinite loop.
            - The default implementation assumes that `is_better` is defined
              and uses it to do the evaluation. Note that the default
              implementation of `is_better` does assume that `eval` is defined
              and uses it. This means that failing to define both leads to
              an infinite loop.
        """
        raise NotImplementedError("Could not calculate the utility value.")

    def eval_all(self, outcomes: List["Outcome"]) -> Iterable[UtilityValue]:
        """
        Calculates the utility value of a list of outcomes and returns their
        utility values

        Args:
            outcomes: A list of offers

        Returns:
            An iterable with the utility values of the given outcomes in order

        Remarks:
            - The default implementation just iterates over the outcomes
              calling the ufun for each of them. In a distributed environment,
              it is possible to do this in parallel using a thread-pool for example.
        """
        return [self(_) for _ in outcomes]

    @classmethod
    def approximate(
        cls,
        ufuns: List["UtilityFunction"],
        issues: Iterable["Issue"],
        n_outcomes: int,
        min_per_dim=5,
        force_single_issue=False,
    ) -> Tuple[List["MappingUtilityFunction"], List["Outcome"], List["Issue"]]:
        """
        Approximates a list of ufuns with a list of mapping discrete ufuns

        Args:
            ufuns: The list of ufuns to approximate
            issues: The issues
            n_outcomes: The number of outcomes to use in the approximation
            min_per_dim: Minimum number of levels per continuous dimension
            force_single_issue: Force the output to have a single issue

        Returns:

        """
        issues = list(issues)
        issue_names = [_.name for _ in issues]
        outcomes = sample_outcomes(
            issues=issues,
            astype=tuple,
            min_per_dim=min_per_dim,
            expansion_policy="null",
            n_outcomes=n_outcomes,
        )
        if force_single_issue:
            output_outcomes = [(_,) for _ in range(n_outcomes)]
            output_issues = [Issue(values=len(output_outcomes))]
        else:
            output_outcomes = outcomes
            issue_values = []
            for i in range(len(issues)):
                vals = np.unique(np.array([_[i] for _ in outcomes])).tolist()
                issue_values.append(vals)
            output_issues = [
                Issue(name=issue.name, values=issue_vals)
                for issue, issue_vals in zip(issues, issue_values)
            ]

        utils = []
        for ufun in ufuns:
            u = [ufun(outcome_as(o, ufun.outcome_type, issue_names)) for o in outcomes]
            utils.append(MappingUtilityFunction(mapping=dict(zip(output_outcomes, u))))

        return utils, output_outcomes, output_issues

    def compare_real(self, o1: "Outcome", o2: "Outcome", method="mean") -> float:
        """
        Compares the two outcomes and returns a measure of the difference
        between their utilities.

        Args:
            o1: First outcome
            o2: Second outcome
            method: The comparison method if one of the two outcomes result in
                    a distribution. Acceptable values are:

                        - mean: Compares the means of the two distributions
                        - min: Compares minimum values with nonzero probability
                        - max: Compares maximum values with nonzero probability
                        - int: Calculates :math:`int (u_1-u_2) du_1du_2`.
                        - Callable: The callable is given u(o1), u(o2) and
                                    should return the comparison.

        """
        u1, u2 = self(o1), self(o2)
        if isinstance(u1, float) and isinstance(u2, float):
            return u1 - u2
        if isinstance(method, Callable):
            return method(u1, u2)
        if isinstance(u1, float):
            u1 = UtilityDistribution(dtype="uniform", loc=u1, scale=1e-10)
        if isinstance(u2, float):
            u2 = UtilityDistribution(dtype="uniform", loc=u2, scale=1e-10)
        if method == "mean":
            return u1.mean() - u2.mean()
        if method == "min":
            return u1.min() - u2.min()
        if method == "max":
            return u1.max() - u2.max()
        if method == "int":
            if u1.scale <= 1e-9:
                return u1 - u2.mean()
            if u2.scale <= 1e-9:
                return u1.mean() - u2
        raise NotImplementedError(
            "Should calculate the integration [(u1-u2) du1 du2] but not implemented yet."
        )

    def rank_with_weights(
        self, outcomes: List[Optional[Outcome]], descending=True
    ) -> List[Tuple[int, float]]:
        """Ranks the given list of outcomes with weights. None stands for the null outcome.

        Returns:

            - A list of tuples each with two values:
                - an integer giving the index in the input array (outcomes) of an outcome
                - the weight of that outcome
            - The list is sorted by weights descendingly

        """
        return sorted(
            zip(list(range(len(outcomes))), [float(self(o)) for o in outcomes]),
            key=lambda x: x[1],
            reverse=descending,
        )

    def argsort(self, outcomes: List[Optional[Outcome]], descending=True) -> List[int]:
        """Finds the rank of each outcome as an integer"""
        return [_[0] for _ in self.rank_with_weights(outcomes, descending=descending)]

    def sort(
        self, outcomes: List[Optional[Outcome]], descending=True
    ) -> List[Optional[Outcome]]:
        """Sorts the given outcomes in place in ascending or descending order of utility value.

        Returns:
            Returns the input list after being sorted. Notice that the array is sorted in-place

        """
        outcomes.sort(key=self, reverse=descending)
        return outcomes

    rank = argsort
    """Ranks the given list of outcomes. None stands for the null outcome"""

    def is_better(
        self, first: "Outcome", second: "Outcome", epsilon=1e-10
    ) -> Optional[bool]:
        """
        Compares two offers using the `ufun` returning whether the first is better than the second

        Args:
            first: First outcome to be compared
            second: Second outcome to be compared
            epsilon: comparison threshold. If the utility difference within the range [-epsilon, epsilon] the two
                     outcomes are assumed to be compatible

        Returns:
            True if utility(first) > utility(second) + epsilon
            None if |utility(first) - utility(second)| <= epsilon
            False if utility(first) < utility(second) - epsilon
        """
        u1, u2 = self(first), self(second)
        if u1 is None or u2 is None or abs(u1 - u2) <= epsilon:
            return None
        return float(u1) > float(u2)

    @abstractmethod
    def xml(self, issues: List[Issue]) -> str:
        """Converts the function into a well formed XML string preferrably in GENIUS format.

        If the output has with </objective> then discount factor and reserved value should also be included
        If the output has </utility_space> it will not be appended in `to_xml_str`

        """

    @property
    def type(self) -> str:
        """Returns the utility_function type.

        Each class inheriting from this ``UtilityFunction`` class will have its own type. The default type is the empty
        string.

        Examples:
            >>> from negmas.utilities import *
            >>> print(LinearUtilityAggregationFunction({1:lambda x:x, 2:lambda x:x}).type)
            linear_aggregation
            >>> print(MappingUtilityFunction(lambda x: x).type)
            mapping
            >>> print(NonLinearUtilityAggregationFunction({1:lambda x:x}, f=lambda x: x).type)
            non_linear_aggregation

        Returns:
            str: utility_function type
        """
        return snake_case(
            self.__class__.__name__.replace("Function", "").replace("Utility", "")
        )

    @property
    def base_type(self) -> str:
        """Returns the utility_function base type ignoring discounting and similar wrappings."""
        return self.type

    def eu(self, offer: "Outcome") -> Optional[float]:
        """Calculate the expected utility value.

        Args:
            offer: The offer to be evaluated.

        Returns:
            float: The expected utility value for UFuns that return a distribution and just utility value for real-valued utilities.

        """
        v = self(offer)
        return float(v) if v is not None else None

    @classmethod
    def generate_bilateral(
        cls,
        outcomes: Union[int, List[Outcome]],
        conflict_level: float = 0.5,
        conflict_delta=0.005,
        win_win=0.5,
    ) -> Tuple["UtilityFunction", "UtilityFunction"]:
        """Generates a couple of utility functions

        Args:

            n_outcomes (int): number of outcomes to use
            conflict_level: How conflicting are the two ufuns to generate.
                            1.0 means maximum conflict.
            conflict_delta: How variable is the conflict at different outcomes.
            win_win: How much are their opportunities for win-win situations.

        Examples:

            >>> u1, u2 = UtilityFunction.generate_bilateral(outcomes=10, conflict_level=0.0
            ...                                             , conflict_delta=0.0, win_win=0.0)
            >>> print(UtilityFunction.conflict_level(u1, u2, outcomes=10))
            0.0

            >>> u1, u2 = UtilityFunction.generate_bilateral(outcomes=10, conflict_level=1.0
            ...                                             , conflict_delta=0.0, win_win=0.0)
            >>> print(UtilityFunction.conflict_level(u1, u2, outcomes=10))
            1.0

            >>> u1, u2 = UtilityFunction.generate_bilateral(outcomes=10, conflict_level=0.5
            ...                                             , conflict_delta=0.0, win_win=1.0)
            >>> 0.0 <= UtilityFunction.conflict_level(u1, u2, outcomes=10) <= 1.0
            True


        """
        from negmas.utilities.nonlinear import MappingUtilityFunction

        if isinstance(outcomes, int):
            outcomes = [(_,) for _ in range(outcomes)]
        n_outcomes = len(outcomes)
        u1 = np.random.random(n_outcomes)
        rand = np.random.random(n_outcomes)
        if conflict_level > 0.5:
            conflicting = 1.0 - u1 + conflict_delta * np.random.random(n_outcomes)
            u2 = conflicting * conflict_level + rand * (1 - conflict_level)
        elif conflict_level < 0.5:
            same = u1 + conflict_delta * np.random.random(n_outcomes)
            u2 = same * (1 - conflict_level) + rand * conflict_level
        else:
            u2 = rand

        # todo implement win_win correctly. Order the ufun then make outcomes with good outcome even better and vice
        # versa
        # u2 += u2 * win_win
        # u2 += np.random.random(n_outcomes) * conflict_delta
        u1 -= u1.min()
        u2 -= u2.min()
        u1 = u1 / u1.max()
        u2 = u2 / u2.max()
        if random.random() > 0.5:
            u1, u2 = u2, u1
        return (
            MappingUtilityFunction(dict(zip(outcomes, u1))),
            MappingUtilityFunction(dict(zip(outcomes, u2))),
        )

    @classmethod
    def generate_random_bilateral(
        cls, outcomes: Union[int, List[Outcome]]
    ) -> Tuple["UtilityFunction", "UtilityFunction"]:
        """Generates a couple of utility functions

        Args:

            n_outcomes (int): number of outcomes to use
            conflict_level: How conflicting are the two ufuns to generate. 1.0 means maximum conflict.
            conflict_delta: How variable is the conflict at different outcomes.
            zero_summness: How zero-sum like are the two ufuns.


        """
        from negmas.utilities.nonlinear import MappingUtilityFunction

        if isinstance(outcomes, int):
            outcomes = [(_,) for _ in range(outcomes)]
        n_outcomes = len(outcomes)
        u1 = np.random.random(n_outcomes)
        u2 = np.random.random(n_outcomes)
        u1 -= u1.min()
        u2 -= u2.min()
        u1 /= u1.max()
        u2 /= u2.max()
        return (
            MappingUtilityFunction(
                dict(zip(outcomes, u1)), outcome_type=type(outcomes[0])
            ),
            MappingUtilityFunction(
                dict(zip(outcomes, u2)), outcome_type=type(outcomes[0])
            ),
        )

    @classmethod
    def generate_random(
        cls, n: int, outcomes: Union[int, List[Outcome]], normalized: bool = True
    ) -> List["UtilityFunction"]:
        """Generates N mapping utility functions

        Args:
            n: number of utility functions to generate
            outcomes: number of outcomes to use
            normalized: if true, the resulting ufuns will be normlized between zero and one.


        """
        from negmas.utilities.nonlinear import MappingUtilityFunction

        if isinstance(outcomes, int):
            outcomes = [(_,) for _ in range(outcomes)]
        n_outcomes = len(outcomes)
        ufuns = []
        for _ in range(n):
            u1 = np.random.random(n_outcomes)
            if normalized:
                u1 -= u1.min()
                u1 /= u1.max()
            ufuns.append(MappingUtilityFunction(dict(zip(outcomes, u1))))
        return ufuns

    @classmethod
    def opposition_level(
        cls,
        ufuns=List["UtilityFunction"],
        max_utils: Union[float, Tuple[float, float]] = 1.0,
        outcomes: Union[int, List["Outcome"]] = None,
        issues: List["Issue"] = None,
        max_tests: int = 10000,
    ) -> float:
        """
        Finds the opposition level of the two ufuns defined as the minimum distance to outcome (1, 1)

        Args:
            ufuns: A list of utility functions to use.
            max_utils: A list of maximum utility value for each ufun (or a single number if they are equal).
            outcomes: A list of outcomes (should be the complete issue space) or an integer giving the number
                     of outcomes. In the later case, ufuns should expect a tuple of a single integer.
            issues: The issues (only used if outcomes is None).
            max_tests: The maximum number of outcomes to use. Only used if issues is given and has more
                       outcomes than this value.


        Examples:


            - Opposition level of the same ufun repeated is always 0
            >>> from negmas.utilities.nonlinear import MappingUtilityFunction
            >>> u1, u2 = lambda x: x[0], lambda x: x[0]
            >>> UtilityFunction.opposition_level([u1, u2], outcomes=10, max_utils=9)
            0.0

            - Opposition level of two ufuns that are zero-sum
            >>> u1, u2 = MappingUtilityFunction(lambda x: x[0]), MappingUtilityFunction(lambda x: 9 - x[0])
            >>> UtilityFunction.opposition_level([u1, u2], outcomes=10, max_utils=9)
            0.7114582486036499

        """
        if outcomes is None and issues is None:
            raise ValueError("You must either give outcomes or issues")
        if outcomes is None:
            outcomes = Issue.enumerate(issues, max_n_outcomes=max_tests, astype=tuple)
        if isinstance(outcomes, int):
            outcomes = [(_,) for _ in range(outcomes)]
        if not isinstance(max_utils, Iterable):
            max_utils = [max_utils] * len(ufuns)
        if len(ufuns) != len(max_utils):
            raise ValueError(
                f"Cannot use {len(ufuns)} ufuns with only {len(max_utils)} max. utility values"
            )

        nearest_val = float("inf")
        assert not any(abs(_) < 1e-7 for _ in max_utils), f"max-utils : {max_utils}"
        for outcome in outcomes:
            v = sum(
                (1.0 - u(outcome) / max_util) ** 2
                for max_util, u in zip(max_utils, ufuns)
            )
            if v == float("inf"):
                warnings.warn(
                    f"u is infinity: {outcome}, {[_(outcome) for _ in ufuns]}, max_utils"
                )
            if v < nearest_val:
                nearest_val = v
        return sqrt(nearest_val)

    @classmethod
    def conflict_level(
        cls,
        u1: "UtilityFunction",
        u2: "UtilityFunction",
        outcomes: Union[int, List["Outcome"]],
        max_tests: int = 10000,
    ) -> float:
        """
        Finds the conflict level in these two ufuns

        Args:
            u1: first utility function
            u2: second utility function

        Examples:
            - A nonlinear strictly zero sum case
            >>> from negmas.utilities.nonlinear import MappingUtilityFunction
            >>> outcomes = [(_,) for _ in range(10)]
            >>> u1 = MappingUtilityFunction(dict(zip(outcomes,
            ... np.random.random(len(outcomes)))))
            >>> u2 = MappingUtilityFunction(dict(zip(outcomes,
            ... 1.0 - np.array(list(u1.mapping.values())))))
            >>> print(UtilityFunction.conflict_level(u1=u1, u2=u2, outcomes=outcomes))
            1.0

            - The same ufun
            >>> print(UtilityFunction.conflict_level(u1=u1, u2=u1, outcomes=outcomes))
            0.0

            - A linear strictly zero sum case
            >>> outcomes = [(i,) for i in range(10)]
            >>> u1 = MappingUtilityFunction(dict(zip(outcomes,
            ... np.linspace(0.0, 1.0, len(outcomes), endpoint=True))))
            >>> u2 = MappingUtilityFunction(dict(zip(outcomes,
            ... np.linspace(1.0, 0.0, len(outcomes), endpoint=True))))
            >>> print(UtilityFunction.conflict_level(u1=u1, u2=u2, outcomes=outcomes))
            1.0
        """
        if isinstance(outcomes, int):
            outcomes = [(_,) for _ in range(outcomes)]
        n_outcomes = len(outcomes)
        points = np.array([[u1(o), u2(o)] for o in outcomes])
        order = np.random.permutation(np.array(range(n_outcomes)))
        p1, p2 = points[order, 0], points[order, 1]
        signs = []
        trial = 0
        for i in range(n_outcomes - 1):
            for j in range(i + 1, n_outcomes):
                if trial >= max_tests:
                    break
                trial += 1
                o11, o12 = p1[i], p1[j]
                o21, o22 = p2[i], p2[j]
                if o12 == o11 and o21 == o22:
                    continue
                signs.append(
                    int((o12 > o11 and o21 > o22) or (o12 < o11 and o21 < o22))
                )
        signs = np.array(signs)
        if len(signs) == 0:
            return None
        return signs.mean()

    @classmethod
    def winwin_level(
        cls,
        u1: "UtilityFunction",
        u2: "UtilityFunction",
        outcomes: Union[int, List["Outcome"]],
        max_tests: int = 10000,
    ) -> float:
        """
        Finds the win-win level in these two ufuns

        Args:
            u1: first utility function
            u2: second utility function

        Examples:
            - A nonlinear same ufun case
            >>> from negmas.utilities.nonlinear import MappingUtilityFunction
            >>> outcomes = [(_,) for _ in range(10)]
            >>> u1 = MappingUtilityFunction(dict(zip(outcomes,
            ... np.linspace(1.0, 0.0, len(outcomes), endpoint=True))))

            - A linear strictly zero sum case
            >>> outcomes = [(_,) for _ in range(10)]
            >>> u1 = MappingUtilityFunction(dict(zip(outcomes,
            ... np.linspace(0.0, 1.0, len(outcomes), endpoint=True))))
            >>> u2 = MappingUtilityFunction(dict(zip(outcomes,
            ... np.linspace(1.0, 0.0, len(outcomes), endpoint=True))))


        """
        if isinstance(outcomes, int):
            outcomes = [(_,) for _ in range(outcomes)]
        n_outcomes = len(outcomes)
        points = np.array([[u1(o), u2(o)] for o in outcomes])
        order = np.random.permutation(np.array(range(n_outcomes)))
        p1, p2 = points[order, 0], points[order, 1]
        signed_diffs = []
        for trial, (i, j) in enumerate(
            zip(range(n_outcomes - 1), range(1, n_outcomes))
        ):
            if trial >= max_tests:
                break
            o11, o12 = p1[i], p1[j]
            o21, o22 = p2[i], p2[j]
            if o11 == o12:
                if o21 == o22:
                    continue
                else:
                    win = abs(o22 - o21)
            elif o11 < o12:
                if o21 == o22:
                    win = o12 - o11
                else:
                    win = (o12 - o11) + (o22 - o21)
            else:
                if o21 == o22:
                    win = o11 - o12
                else:
                    win = (o11 - o12) + (o22 - o21)
            signed_diffs.append(win)
        signed_diffs = np.array(signed_diffs)
        if len(signed_diffs) == 0:
            return None
        return signed_diffs.mean()

    def outcome_with_utility(
        self,
        rng: Tuple[Optional[float], Optional[float]],
        issues: List[Issue] = None,
        outcomes: List[Outcome] = None,
        n_trials: int = 100,
    ) -> Optional["Outcome"]:
        """
        Gets one outcome within the given utility range or None on failure

        Args:
            self: The utility function
            rng: The utility range
            issues: The issues the utility function is defined on
            outcomes: The outcomes to sample from
            n_trials: The maximum number of trials

        Returns:

            - Either issues, or outcomes should be given but not both

        """
        if outcomes is None:
            outcomes = Issue.sample(
                issues=issues,
                n_outcomes=n_trials,
                astype=self.outcome_type,
                with_replacement=False,
                fail_if_not_enough=False,
            )
        n = min(len(outcomes), n_trials)
        mn, mx = rng
        if mn is None:
            mn = float("-inf")
        if mx is None:
            mx = float("inf")
        for i in range(n):
            o = outcomes[i]
            if mn <= self(o) <= mx:
                return o
        return None

    def utility_range(
        self,
        issues: List[Issue] = None,
        outcomes: Collection[Outcome] = None,
        infeasible_cutoff: Optional[float] = None,
        return_outcomes=False,
        max_n_outcomes=1000,
        ami: Optional["AgentMechnismInterface"] = None,
    ) -> Union[
        Tuple[UtilityValue, UtilityValue],
        Tuple[UtilityValue, UtilityValue, Outcome, Outcome],
    ]:
        """Finds the range of the given utility function for the given outcomes

        Args:
            self: The utility function
            issues: List of issues (optional)
            outcomes: A collection of outcomes (optional)
            infeasible_cutoff: A value under which any utility is considered infeasible and is not used in calculation
            return_outcomes: If true, will also return an outcome for min and max utils
            max_n_outcomes: the maximum number of outcomes to try sampling (if sampling is used and outcomes are not
                            given)
            ami: Optional AMI to use

        Returns:
            UtilityFunction: A utility function that is guaranteed to be normalized for the set of given outcomes

        """

        if outcomes is None:
            outcomes = Issue.sample(
                issues,
                n_outcomes=max_n_outcomes,
                with_replacement=True,
                fail_if_not_enough=False,
                astype=self.outcome_type,
            )
        outcomes = [_ for _ in outcomes if _ is not None]
        utils = [self(o) for o in outcomes]
        errors = [i for i, u in enumerate(utils) if u is None]
        if len(errors) > 0:
            raise ValueError(
                f"UFun returnd None for {len(errors) / len(utils):03%} outcomes\n"
                # f"outcomes {[outcomes[e] for e in errors]}\n"
            )
        # if there are no outcomes return zeros for utils
        if len(utils) == 0:
            if return_outcomes:
                return 0.0, 0.0, None, None
            return 0.0, 0.0

        # make sure the utility value is converted to float
        utils = [float(_) for _ in utils]

        # if there is an infeasible_cutoff, apply it
        if infeasible_cutoff is not None:
            if return_outcomes:
                outcomes = [o for o, _ in zip(outcomes, utils) if _ > infeasible_cutoff]
            utils = np.array([_ for _ in utils if _ > infeasible_cutoff])

        if return_outcomes:
            minloc, maxloc = np.argmin(utils), np.argmax(utils)
            return (
                utils[minloc],
                utils[maxloc],
                outcomes[minloc],
                outcomes[maxloc],
            )
        return float(np.min(utils)), float(np.max(utils))

    def init_inverse(
        self,
        issues: Optional[List["Issue"]] = None,
        outcomes: Optional[List["Outcome"]] = None,
        n_trials: int = 10000,
        max_cache_size: int = 10000,
    ) -> None:
        """
        Initializes the inverse ufun used to map utilities to outcomes.

        Args:
            issues: The issue space to be searched for inverse. If not given, the issue space of the mechanism may be used.
            outcomes: The outcomes to consider. This takes precedence over `issues`
            n_trials: Used for constraining computation if necessary
            max_cache_size: The maximum allowed number of outcomes to cache

        """
        if issues is None:
            issues = self.issues
        self._min, self._max, self._worst, self._best = self.utility_range(
            issues, outcomes, return_outcomes=True, max_n_outcomes=n_trials
        )
        self._range = self._max - self._min
        self._offset = self._min / self._range if self._range > 1e-5 else self._min
        astype = self._outcome_type if self._outcome_type else tuple
        if self._issues and Issue.num_outcomes(issues) < max_cache_size:
            outcomes = Issue.enumerate(issues, astype=astype)
        if not outcomes:
            outcomes = Issue.discretize_and_enumerate(
                issues, n_discretization=2, astype=astype, max_n_outcomes=max_cache_size
            )
            n = max_cache_size - len(outcomes)
            if n > 0:
                outcomes += Issue.sample(
                    issues,
                    n,
                    astype=astype,
                    with_replacement=False,
                    fail_if_not_enough=False,
                )
        utils = self.eval_all(outcomes)
        self._ordered_outcomes = sorted(zip(utils, outcomes), key=lambda x: -x[0])
        self._inverse_initialzed = True

    def inverse(
        self,
        u: float,
        eps: Union[float, Tuple[float, float]] = (1e-3, 0.2),
        assume_normalized=True,
        issues: Optional[List["Issue"]] = None,
        outcomes: Optional[List["Outcome"]] = None,
        n_trials: int = 10000,
        return_all_in_range=False,
        max_n_outcomes=10000,
    ) -> Union[Optional["Outcome"], Tuple[List["Outcome"], List[UtilityValue]]]:
        """
        Finds an outcmoe with the given utility value

        Args:
            u: the utility value to find an outcome for.
            eps: An approximation error. If a number, it is mapped to (eps, eps)
            n_trials: the number of time to try (used if random sampling is employed)
            issues: If given the issue space to search (not recommended)
            outcomes: If given the outcomes to search (not recommended)
            assume_normalized: If true, the ufun will be assumed normalized between 0 and 1 (faster)
            return_all_in_range: If given all outcomes in the given range (or samples of them) will be returned up
                                 to `max_n_outcomes`
            max_n_outcomes: Only used if return_all_in_range is given and gives the maximum number of outcomes to return

        Returns:
            - if return_all_in_range:
               - A tuple with all outcomes in the given range (or samples thereof)
               - A tuple of corresponding utility values
            - Otherwise, An outcome with utility between u-eps[0] and u+eps[1] if found

        Remarks:
            - If issues or outcomes are not None, then init_inverse will be called first

        """
        if not isinstance(eps, Iterable):
            mn, mx = u - eps, u + eps
        else:
            mn, mx = u - eps[0], u + eps[1]
        if not assume_normalized:
            if self._range > 1e-5:
                mn = mn / self._range + self._offset
                mx = mx / self._range + self._offset
            else:
                mn = mn + self._offset
                mx = mx + self._offset
        if (not self._inverse_initialzed) or issues or outcomes:
            self.init_inverse(issues, outcomes, n_trials, n_trials)
        # todo use bisection
        if return_all_in_range:
            results = []
            utils = []
            for i, (util, w) in enumerate(self._ordered_outcomes):
                if utils > mx:
                    continue
                if util < mn or len(results) >= max_n_outcomes:
                    break
                results.append(w)
                utils.append(util)
            return results, utils

        for i, (util, w) in enumerate(self._ordered_outcomes):
            if util >= mn:
                continue
            ubefore, wbefore = self._ordered_outcomes[i - 1 if i > 0 else 0]
            if ubefore > mx:
                return None
            return wbefore
        ubefore, wbefore = self._ordered_outcomes[-1]
        if ubefore > mx:
            return None
        return wbefore

    @property
    def is_inverse_initialized(self):
        return self._inverse_initialzed

    def uninialize_inverse(self):
        self._inverse_initialzed = False

    @classmethod
    def random(
        cls, issues, reserved_value=(0.0, 0.5), normalized=True, **kwargs
    ) -> "UtilityFunction":
        """Generates a random ufun of the given type"""

    @classmethod
    def from_str(cls, s) -> "UtilityFunction":
        """Creates an object out of a dict."""
        return deserialize(eval(s))

    def __str__(self):
        return str(serialize(self))


UtilityFunctions = List["UtilityFunction"]
