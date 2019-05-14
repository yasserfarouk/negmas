r"""Models basic types of utility functions.

Utility functions are at the core of negotiation. Agents engage in negotiations with the goal of maximizing some utility
function. In most cases, these utility functions are assumed to be known a-periori and static for the duration of a
single negotiations.

Notes:
    We try to allow for applications that do not necessary have these two assumptions in the following ways:

    * A utility_function *value* (\ `UtilityValue`\ ) can always represent represent a utility_function distribution over all
      possible utility_function values (\ `UtilityDistribution`\ ) or a `KnownUtilityValue` which is a real number.

    * The base class of all utility_function *functions* is
      `UtilityFunction` and is assumed to map outcomes (\ `Outcome` objects) to the aforementioned generic utility *values*
      (\ `UtilityValue` objects).

    * Utility functions can be constructed using any `Callable` which makes it possible to construct them so that
      they change depending on the context or the progression of the negotiation.


"""
import itertools
import pprint
import random
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from functools import reduce
from operator import mul
from typing import (
    MutableMapping,
    Mapping,
    Union,
    Optional,
    Any,
    Sequence,
    Callable,
    Dict,
    List,
    Iterable,
    Tuple,
    Collection,
)
from typing import TYPE_CHECKING

import numpy as np
import pkg_resources

from negmas.common import NamedObject
from negmas.common import AgentMechanismInterface
from negmas.generics import GenericMapping, ienumerate, iget, ivalues
from negmas.helpers import Distribution
from negmas.helpers import snake_case, gmap, ikeys, Floats
from negmas.java import JavaCallerMixin, to_java
from negmas.outcomes import (
    sample_outcomes,
    OutcomeRange,
    Outcome,
    outcome_in_range,
    Issue,
    outcome_is_valid,
    OutcomeType,
    outcome_as_dict,
)

if TYPE_CHECKING:
    from negmas.outcomes import OutcomeRange, Outcome

__all__ = [
    "UtilityDistribution",
    "UtilityValue",
    "UtilityFunction",
    "ConstUFun",
    "LinDiscountedUFun",
    "ExpDiscountedUFun",
    "MappingUtilityFunction",
    "LinearUtilityAggregationFunction",
    "NonLinearUtilityAggregationFunction",
    "HyperRectangleUtilityFunction",
    "NonlinearHyperRectangleUtilityFunction",
    "ComplexWeightedUtilityFunction",
    "ComplexNonlinearUtilityFunction",
    "IPUtilityFunction",
    "pareto_frontier",
    "make_discounted_ufun",
    "normalize",
    "JavaUtilityFunction",
    "RandomUtilityFunction",
]


# Helper Types just used to make type hinting more readable
OutcomeUtilityMapping = Union[
    Callable[[Union["Outcome", int, str, float]], "UtilityValue"],
    Mapping[Union[Sequence, Mapping, int, str, float], "UtilityValue"],
]
OutcomeUtilityMappings = List[OutcomeUtilityMapping]
"""Maps from multi-issue or single-issue outcomes to Negotiator values."""
IssueUtilityFunctionMapping = Union[  # type: ignore
    Callable[["Issue"], "UtilityFunction"],  # type: ignore
    Mapping["Issue", "UtilityFunction"],  # type: ignore
]  # type: ignore
IssueUtilityFunctionMappings = List[IssueUtilityFunctionMapping]

UtilityDistribution = Distribution


class ExactUtilityValue(float):
    """Encapsulates a single offerable_outcomes utility_function value."""


UtilityValue = Union[UtilityDistribution, float]
"""Either a utility_function distribution or an exact offerable_outcomes utility_function value.

`UtilityFunction`s always return a `UtilityValue` which makes it easier to implement algorithms relying  on 
probabilistic modeling of utility_function values."""


class UtilityFunction(ABC, NamedObject):
    """The abstract base class for all utility functions.

    A utility function encapsulates a mapping from outcomes to UtilityValue(s). This is a generalization of standard
    utility functions that are expected to always return a real-value. This generalization is useful for modeling cases
    in which only partial knowledge of the utility function is available.

    Args:

        name (str): Name of the utility function. If None, a random name will be given.
        reserved_value(float): The value to return if the input offer to `apply` is None

    """

    def __init__(
        self,
        name: Optional[str] = None,
        ami: AgentMechanismInterface = None,
        reserved_value: Optional[UtilityValue] = 0.0,
    ) -> None:
        super().__init__(name=name)
        self.reserved_value = reserved_value
        self.ami = ami

    @property
    def is_dynamic(self):
        """Whether the utility function can potentially depend on negotiation state (mechanism information).

        - If this property is `False`,  the ufun can safely be assumed to be static (not dependent on negotiation
          state).
        - If this property is `True`, the ufun may depend on negotiation state but it may also not depend on it.
        """
        return self.ami is None

    @classmethod
    def from_genius(cls, file_name: str, **kwargs):
        """Imports a utility function from a GENIUS XML file.

        Args:

            file_name (str): File name to import from

        Returns:

            A utility function object (depending on the input file)


        Examples:

            >>> from negmas import UtilityFunction
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
        cls, u: "UtilityFunction", issues: List[Issue], file_name: str, **kwargs
    ):
        """Exports a utility function from a GENIUS XML file.

        Args:

            file_name (str): File name to export to
            u: utility function
            issues: The issues being considered as defined in the domain

        Returns:

            None


        Examples:

            >>> from negmas import UtilityFunction
            >>> from negmas import load_genius_domain
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
        """Exports a utility function to a well formatted string


        """
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
        if u.reserved_value is not None and "<reservation value" not in output:
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
        normalize_utility=True,
        max_n_outcomes: int = 1e6,
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
                    max_n_outcomes (int): Maximum number of outcomes allowed (effective only if force_single_issue is True)

                Returns:

                    A utility function object (depending on the input file)


                Examples:

                    >>> u, _ = UtilityFunction.from_xml_str(open(pkg_resources.resource_filename('negmas'
                    ...                                      , resource_name='tests/data/Laptop/Laptop-C-prof1.xml')
                    ...                                      , 'r').read(), force_single_issue=False
                    ... , normalize_utility=True, keep_issue_names=False, keep_value_names=True)
                    >>> assert abs(u(('Dell', '60 Gb', "19'' LCD")) - 0.599329436957658) < 0.1
                    >>> assert abs(u(('HP', '80 Gb', "20'' LCD")) - 0.6342209804130308) < 0.01
                    >>> assert abs(u(('HP', '60 Gb', "19'' LCD")) - 1.0) < 0.0001

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
                    >>> assert abs(u(("Dell+60 Gb+19'' LCD",)) - 0.599329436957658) < 0.1
                    >>> assert abs(u(("HP+80 Gb+20'' LCD",)) - 0.6342209804130308) < 0.01

                    >>> u, _ = UtilityFunction.from_xml_str(open(pkg_resources.resource_filename('negmas'
                    ...                                      , resource_name='tests/data/Laptop/Laptop-C-prof1.xml')
                    ...                                      , 'r').read(), force_single_issue=True
                    ... , keep_issue_names=False, keep_value_names=False, normalize_utility=True)
                    >>> assert abs(u((0,)) - 0.599329436957658) < 0.1

                    >>> u, _ = UtilityFunction.from_xml_str(open(pkg_resources.resource_filename('negmas'
                    ...                                      , resource_name='tests/data/Laptop/Laptop-C-prof1.xml')
                    ...         , 'r').read(), force_single_issue=False, normalize_utility=True)
                    >>> assert abs(u({'Laptop': 'Dell', 'Harddisk': '60 Gb', 'External Monitor': "19'' LCD"}) - 0.599329436957658) < 0.1
                    >>> assert abs(u({'Laptop': 'HP', 'Harddisk': '80 Gb', 'External Monitor': "20'' LCD"}) - 0.6342209804130308) < 0.01
                    >>> assert abs(u({'Laptop': 'HP', 'Harddisk': '60 Gb', 'External Monitor': "19'' LCD"}) - 1.0) < 0.0001

        """
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
            total_util = total_util if max_utility is None else max_utility
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
                        and domain_issues[myname].is_continuous()
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
                            and domain_issues[myname].is_continuous()
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
                            and not domain_issues[myname].is_continuous()
                        ):
                            n_steps = domain_issues[myname].cardinality()
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
                            if mytype == "integer":
                                item_key = int(item_key)
                            issues[issue_key][item_key] = float(
                                item.attrib.get("evaluation", reserved_value)
                            )
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

        # if not keep_issue_names:
        #    issues = [issues[_] for _ in issues.keys()]
        #    real_issues = [real_issues[_] for _ in sorted(real_issues.keys())]
        #    for i, issue in enumerate(issues):
        #        issues[i] = [issue[_] for _ in issue.keys()]

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
            if len(weights) > 0:
                for key, issue in zip(ikeys(issues), ivalues(issues)):
                    try:
                        w = weights[issue_info[key]["index"]]
                    except:
                        w = 1.0
                    for item_key in ikeys(issue):
                        issue[item_key] *= w
            if force_single_issue:
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
                    umax, umin = max(utils), min(utils)
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
                            [item_utility for item_utility in ivalues(items)]
                            for issue_key, items in zip(ikeys(issues), ivalues(issues))
                        ]
                    )
                    utils = list(map(lambda vals: sum(vals), utils))
                    umax, umin = max(utils), min(utils)
                    factor = umax - umin
                    if factor > 1e-8:
                        offset = (umin / len(issues)) / factor
                    else:
                        offset = 0.0
                        factor = 1.0
                    for key, issue in ienumerate(issues):
                        for item_key in ikeys(issue):
                            issues[key][item_key] = (
                                issues[key][item_key] / factor - offset
                            )
                if len(issues) > 1:
                    u = LinearUtilityAggregationFunction(issue_utilities=issues)
                else:
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
                utils = list(map(lambda vals: sum(vals), utils))
                umax, umin = max(utils), min(utils)
                factor = umax - umin
                if factor > 1e-8:
                    offset = (umin / len(real_issues)) / factor
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
        if reserved_value is not None and not ignore_reserved and u is not None:
            u.reserved_value = reserved_value
        if ignore_discount:
            discount_factor = None
        return u, discount_factor

    def __getitem__(self, offer: Outcome) -> Optional[UtilityValue]:
        """Overrides [] operator to call the ufun allowing it to act as a mapping"""
        return self(offer)

    @abstractmethod
    def __call__(self, offer: Outcome) -> Optional[UtilityValue]:
        """Calculate the utility_function value for a given outcome.

        Args:
            offer: The offer to be evaluated.


        Remarks:
            - You cannot return None from overriden apply() functions but raise an exception (ValueError) if it was
              not possible to calculate the UtilityValue.
            - Return A UtilityValue not a float for real-valued utilities for the benefit of inspection code.

        Returns:
            UtilityValue: The utility_function value which may be a distribution. If `None` it means the utility_function value cannot be
            calculated.
        """
        if offer is None:
            return self.reserved_value
        return None

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

        Args:
            cls:
            ufuns:
            issues:
            n_outcomes:
            min_per_dim:
            force_single_issue:

        Returns:

        """
        issues = list(issues)
        outcomes = sample_outcomes(
            issues=issues,
            keep_issue_names=False,
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
            u = [ufun(o) for o in outcomes]
            utils.append(MappingUtilityFunction(mapping=dict(zip(output_outcomes, u))))

        return utils, output_outcomes, output_issues

    def compare(self, o1, o2) -> "UtilityValue":
        """Compares the two outcomes and returns a measure of the difference between their utilities"""
        u1, u2 = self(o1), self(o2)
        if isinstance(u1, float) and isinstance(u2, float):
            return u1 - u2
        raise NotImplementedError(
            "Should calculate the integration [(u1-u2) du1 du2] but not implemented yet."
        )

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
        """Calculate the expected utility_function value.

            Args:
                offer: The offer to be evaluated.

            Returns:
                float: The expected utility_function for utility_priors and just utility_function for real-valued utilities.

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
            conflict_level: How conflicting are the two ufuns to generate. 1.0 means maximum conflict.
            conflict_delta: How variable is the conflict at different outcomes.
            zero_summness: How zero-sum like are the two ufuns.

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
            MappingUtilityFunction(dict(zip(outcomes, u1))),
            MappingUtilityFunction(dict(zip(outcomes, u2))),
        )

    @classmethod
    def generate_random(
        cls, n: int, outcomes: Union[int, List[Outcome]], normalized: bool = True
    ) -> List["UtilityFunction"]:
        """Generates a couple of utility functions

        Args:
            n: number of utility functions to generate
            outcomes: number of outcomes to use
            normalized: if true, the resulting ufuns will be normlized between zero and one.


        """
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
            u1:
            u2:

        Examples:
            - A nonlinear strictly zero sum case
            >>> outcomes = [(_,) for _ in range(10)]
            >>> u1 = MappingUtilityFunction(dict(zip(outcomes, np.random.random(len(outcomes)))))
            >>> u2 = MappingUtilityFunction(dict(zip(outcomes, 1.0 - np.array(list(u1.mapping.values())))))
            >>> print(UtilityFunction.conflict_level(u1=u1, u2=u2, outcomes=outcomes))
            1.0

            - The same ufun
            >>> print(UtilityFunction.conflict_level(u1=u1, u2=u1, outcomes=outcomes))
            0.0

            - A linear strictly zero sum case
            >>> outcomes = [(_,) for _ in range(10)]
            >>> u1 = MappingUtilityFunction(dict(zip(outcomes, np.linspace(0.0, 1.0, len(outcomes), endpoint=True))))
            >>> u2 = MappingUtilityFunction(dict(zip(outcomes, np.linspace(1.0, 0.0, len(outcomes), endpoint=True))))
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
        Finds the conflict level in these two ufuns

        Args:
            u1:
            u2:

        Examples:
            - A nonlinear same ufun case
            >>> outcomes = [(_,) for _ in range(10)]
            >>> u1 = MappingUtilityFunction(dict(zip(outcomes, np.linspace(1.0, 0.0, len(outcomes), endpoint=True))))
            
            - A linear strictly zero sum case
            >>> outcomes = [(_,) for _ in range(10)]
            >>> u1 = MappingUtilityFunction(dict(zip(outcomes, np.linspace(0.0, 1.0, len(outcomes), endpoint=True))))
            >>> u2 = MappingUtilityFunction(dict(zip(outcomes, np.linspace(1.0, 0.0, len(outcomes), endpoint=True))))


        """
        if isinstance(outcomes, int):
            outcomes = [(_,) for _ in range(outcomes)]
        n_outcomes = len(outcomes)
        points = np.array([[u1(o), u2(o)] for o in outcomes])
        order = np.random.permutation(np.array(range(n_outcomes)))
        p1, p2 = points[order, 0], points[order, 1]
        signs = []
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
            signs.append(win)
        signs = np.array(signs)
        if len(signs) == 0:
            return None
        return signs.mean()


UtilityFunctions = List["UtilityFunction"]


class ExpDiscountedUFun(UtilityFunction):
    """A discounted utility function based on some factor of the negotiation

    Args:
        ufun: The utility function that is being discounted
        beta: discount factor
        factor: str -> The name of the AgentMechanismInterface variable based on which discounting operate
        callable -> must receive a mechanism info object and returns a float representing the factor

    """

    def __init__(
        self,
        ufun: UtilityFunction,
        ami: "AgentMechanismInterface",
        beta: Optional[float] = None,
        factor: Union[str, Callable[["AgentMechanismInterface"], float]] = "step",
        name=None,
        reserved_value: Optional[UtilityValue] = 0.0,
        dynamic_reservation=True,
    ):
        super().__init__(name=name, reserved_value=reserved_value, ami=ami)
        self.ufun = ufun
        self.beta = beta
        self.factor = factor
        self.dynamic_reservation = dynamic_reservation

    def __call__(self, offer: Outcome) -> Optional[UtilityValue]:
        if offer is None and not self.dynamic_reservation:
            return self.reserved_value
        u = self.ufun(offer)
        if not self.beta or self.beta == 1.0:
            return u
        if isinstance(self.factor, str):
            factor = getattr(self.ami, self.factor)
        else:
            factor = self.factor(self.ami)
        return (factor ** self.beta) * u

    def xml(self, issues: List[Issue]) -> str:
        output = self.ufun.xml(issues)
        output += "</objective>\n"
        factor = None
        if self.factor is not None:
            factor = str(self.factor)
        if self.beta is not None:
            output += f'<discount_factor value="{self.beta}" '
            if factor is not None and factor != "step":
                output += f' variable="{factor}" '
            output += "/>\n"
        return output

    def __str__(self):
        return f"{self.ufun.type}-cost:{self.beta} based on {self.factor}"

    def __getattr__(self, item):
        return getattr(self.ufun, item)

    @property
    def base_type(self):
        return self.ufun.type

    @property
    def type(self):
        return self.ufun.type + "_exponentially_discounted"


class LinDiscountedUFun(UtilityFunction):
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
        ufun: UtilityFunction,
        ami: "AgentMechanismInterface",
        cost: Optional[float] = None,
        factor: Union[str, Callable[["AgentMechanismInterface"], float]] = "step",
        power: float = 1.0,
        name=None,
        reserved_value: Optional[UtilityValue] = 0.0,
        dynamic_reservation=True,
    ):
        super().__init__(name=name, reserved_value=reserved_value, ami=ami)
        self.ufun = ufun
        self.cost = cost
        self.factor = factor
        self.power = power
        self.dynamic_reservation = dynamic_reservation

    def __call__(self, offer: Outcome) -> Optional[UtilityValue]:
        if offer is None and not self.dynamic_reservation:
            return self.reserved_value
        u = self.ufun(offer)
        if not self.cost or self.cost == 0.0:
            return u
        if isinstance(self.factor, str):
            factor = getattr(self.ami, self.factor)
        else:
            factor = self.factor(self.ami)
        return u - ((factor * self.cost) ** self.power)

    def xml(self, issues: List[Issue]) -> str:
        output = self.ufun.xml(issues)
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

    def __str__(self):
        return f"{self.ufun.type}-cost:{self.cost} raised to {self.power} based on {self.factor}"

    def __getattr__(self, item):
        return getattr(self.ufun, item)

    @property
    def base_type(self):
        return self.ufun.type

    @property
    def type(self):
        return self.ufun.type + "_linearly_discounted"


class ConstUFun(UtilityFunction):
    def __init__(
        self,
        value: float,
        name=None,
        reserved_value: Optional[float] = 0.0,
        ami: AgentMechanismInterface = None,
    ):
        super().__init__(name=name, reserved_value=reserved_value, ami=ami)
        self.value = value

    def __call__(self, offer: Outcome) -> Optional[UtilityValue]:
        if offer is None:
            return self.reserved_value
        return self.value

    def xml(self, issues: List[Issue]) -> str:
        pass

    def __str__(self):
        return str(self.value)


def make_discounted_ufun(
    ufun: "UtilityFunction",
    ami: "AgentMechanismInterface",
    cost_per_round: float = None,
    power_per_round: float = None,
    discount_per_round: float = None,
    cost_per_relative_time: float = None,
    power_per_relative_time: float = None,
    discount_per_relative_time: float = None,
    cost_per_real_time: float = None,
    power_per_real_time: float = None,
    discount_per_real_time: float = None,
    dynamic_reservation: bool = True,
):
    if cost_per_round is not None and cost_per_round > 0.0:
        ufun = LinDiscountedUFun(
            ufun=ufun,
            ami=ami,
            cost=cost_per_round,
            factor="step",
            power=power_per_round,
            dynamic_reservation=dynamic_reservation,
        )
    if cost_per_relative_time is not None and cost_per_relative_time > 0.0:
        ufun = LinDiscountedUFun(
            ufun=ufun,
            ami=ami,
            cost=cost_per_relative_time,
            factor="relative_time",
            power=power_per_relative_time,
            dynamic_reservation=dynamic_reservation,
        )
    if cost_per_real_time is not None and cost_per_real_time > 0.0:
        ufun = LinDiscountedUFun(
            ufun=ufun,
            ami=ami,
            cost=cost_per_real_time,
            factor="real_time",
            power=power_per_real_time,
            dynamic_reservation=dynamic_reservation,
        )
    if discount_per_round is not None and discount_per_round > 0.0:
        ufun = ExpDiscountedUFun(
            ufun=ufun,
            ami=ami,
            beta=discount_per_round,
            factor="step",
            dynamic_reservation=dynamic_reservation,
        )
    if discount_per_relative_time is not None and discount_per_relative_time > 0.0:
        ufun = ExpDiscountedUFun(
            ufun=ufun,
            ami=ami,
            beta=discount_per_relative_time,
            factor="relative_time",
            dynamic_reservation=dynamic_reservation,
        )
    if discount_per_real_time is not None and discount_per_real_time > 0.0:
        ufun = ExpDiscountedUFun(
            ufun=ufun,
            ami=ami,
            beta=discount_per_real_time,
            factor="real_time",
            dynamic_reservation=dynamic_reservation,
        )
    return ufun


class LinearUtilityAggregationFunction(UtilityFunction):
    r"""A linear utility function for multi-issue negotiations.

    Models a linear utility function using predefined weights:\.

    Args:
         issue_utilities: utility functions for individual issues
         weights: weights for combining `issue_utilities`
         name: name of the utility function. If None a random name will be generated.

    Notes:

        The utility value is calculated as:

        .. math::

        u = \sum_{i=0}^{n_{outcomes}-1} {w_i * u_i}


    Examples:

        >>> issues = [Issue((10.0, 20.0), 'price'), Issue(['delivered', 'not delivered'], 'delivery')
        ...           , Issue(5, 'quality')]
        >>> print(list(map(str, issues)))
        ['price: (10.0, 20.0)', "delivery: ['delivered', 'not delivered']", 'quality: 5']
        >>> f = LinearUtilityAggregationFunction({'price': lambda x: 2.0*x
        ...                          , 'delivery': {'delivered': 10, 'not delivered': -10}
        ...                          , 'quality': MappingUtilityFunction(lambda x: x-3)}
        ...         , weights={'price': 1.0, 'delivery': 2.0, 'quality': 4.0})
        >>> float(f({'quality': 2, 'price': 14.0, 'delivery': 'delivered'})
        ...       ) -  (1.0*(2.0*14)+2.0*10+4.0*(2.0-3.0))
        0.0
        >>> f = LinearUtilityAggregationFunction({'price': lambda x: 2.0*x
        ...                          , 'delivery': {'delivered': 10, 'not delivered': -10}}
        ...         , weights={'price': 1.0, 'delivery': 2.0})
        >>> float(f({'quality': 2, 'price': 14.0, 'delivery': 'delivered'})) - (1.0*(2.0*14)+2.0*10)
        0.0

        You can use lists instead of dictionaries for defining outcomes, weights and nonlinearity
        but that is less readable

        >>> f = LinearUtilityAggregationFunction([lambda x: 2.0*x
        ...                          , {'delivered': 10, 'not delivered': -10}
        ...                          , MappingUtilityFunction(lambda x: x-3)]
        ...         , weights=[1.0, 2.0, 4.0])
        >>> float(f((14.0, 'delivered', 2))) - (1.0*(2.0*14)+2.0*10+4.0*(2.0-3.0))
        0.0

    Remarks:
        The mapping need not use all the issues in the output as the last example show.

    """

    def __init__(
        self,
        issue_utilities: Union[
            MutableMapping[Any, GenericMapping], Sequence[GenericMapping]
        ],
        weights: Optional[Union[Mapping[Any, float], Sequence[float]]] = None,
        name: Optional[str] = None,
        reserved_value: Optional[UtilityValue] = 0.0,
        ami: AgentMechanismInterface = None,
    ) -> None:
        super().__init__(name=name, reserved_value=reserved_value, ami=ami)
        self.issue_utilities = issue_utilities
        self.weights = weights
        if self.weights is None:
            self.weights = {}
            for k in ikeys(self.issue_utilities):
                self.weights[k] = 1.0
        for k, v in ienumerate(self.issue_utilities):
            self.issue_utilities[k] = (
                v if isinstance(v, UtilityFunction) else MappingUtilityFunction(v)
            )
        if isinstance(issue_utilities, dict):
            self.issue_indices = dict(
                zip(self.issue_utilities.keys(), range(len(self.issue_utilities)))
            )
        else:
            self.issue_indices = dict(
                zip(range(len(self.issue_utilities)), range(len(self.issue_utilities)))
            )

    def __call__(self, offer: Optional["Outcome"]) -> Optional[UtilityValue]:
        if offer is None:
            return self.reserved_value
        u = ExactUtilityValue(0.0)
        for k in ikeys(self.issue_utilities):
            if isinstance(offer, tuple):
                v = iget(offer, self.issue_indices[k])
            else:
                v = iget(offer, k)
            current_utility = gmap(iget(self.issue_utilities, k), v)
            if current_utility is None:
                return None

            w = iget(self.weights, k)  # type: ignore
            if w is None:
                return None
            try:
                u += w * current_utility
            except FloatingPointError:
                continue
        return u

    def xml(self, issues: List[Issue]) -> str:
        """ Generates an XML string representing the utility function

        Args:
            issues:

        Examples:

            >>> issues = [Issue(values=10, name='i1'), Issue(values=['delivered', 'not delivered'], name='i2')
            ...     , Issue(values=4, name='i3')]
            >>> f = LinearUtilityAggregationFunction([lambda x: 2.0*x
            ...                          , {'delivered': 10, 'not delivered': -10}
            ...                          , MappingUtilityFunction(lambda x: x-3)]
            ...         , weights=[1.0, 2.0, 4.0])
            >>> print(f.xml(issues))
            <issue index="1" etype="discrete" type="discrete" vtype="discrete" name="i1">
                <item index="1" value="0" evaluation="0.0" />
                <item index="2" value="1" evaluation="2.0" />
                <item index="3" value="2" evaluation="4.0" />
                <item index="4" value="3" evaluation="6.0" />
                <item index="5" value="4" evaluation="8.0" />
                <item index="6" value="5" evaluation="10.0" />
                <item index="7" value="6" evaluation="12.0" />
                <item index="8" value="7" evaluation="14.0" />
                <item index="9" value="8" evaluation="16.0" />
                <item index="10" value="9" evaluation="18.0" />
            </issue>
            <issue index="2" etype="discrete" type="discrete" vtype="discrete" name="i2">
                <item index="1" value="delivered" evaluation="10" />
                <item index="2" value="not delivered" evaluation="-10" />
            </issue>
            <issue index="3" etype="discrete" type="discrete" vtype="discrete" name="i3">
                <item index="1" value="0" evaluation="-3" />
                <item index="2" value="1" evaluation="-2" />
                <item index="3" value="2" evaluation="-1" />
                <item index="4" value="3" evaluation="0" />
            </issue>
            <weight index="1" value="1.0">
            </weight>
            <weight index="2" value="2.0">
            </weight>
            <weight index="3" value="4.0">
            </weight>
            <BLANKLINE>
            >>> print(f.xml({i:_ for i, _ in enumerate(issues)}))
            <issue index="1" etype="discrete" type="discrete" vtype="discrete" name="i1">
                <item index="1" value="0" evaluation="0.0" />
                <item index="2" value="1" evaluation="2.0" />
                <item index="3" value="2" evaluation="4.0" />
                <item index="4" value="3" evaluation="6.0" />
                <item index="5" value="4" evaluation="8.0" />
                <item index="6" value="5" evaluation="10.0" />
                <item index="7" value="6" evaluation="12.0" />
                <item index="8" value="7" evaluation="14.0" />
                <item index="9" value="8" evaluation="16.0" />
                <item index="10" value="9" evaluation="18.0" />
            </issue>
            <issue index="2" etype="discrete" type="discrete" vtype="discrete" name="i2">
                <item index="1" value="delivered" evaluation="10" />
                <item index="2" value="not delivered" evaluation="-10" />
            </issue>
            <issue index="3" etype="discrete" type="discrete" vtype="discrete" name="i3">
                <item index="1" value="0" evaluation="-3" />
                <item index="2" value="1" evaluation="-2" />
                <item index="3" value="2" evaluation="-1" />
                <item index="4" value="3" evaluation="0" />
            </issue>
            <weight index="1" value="1.0">
            </weight>
            <weight index="2" value="2.0">
            </weight>
            <weight index="3" value="4.0">
            </weight>
            <BLANKLINE>

        """
        output = ""
        keys = list(ikeys(issues))
        for i, k in enumerate(keys):
            issue_name = iget(issues, k).name
            output += f'<issue index="{i+1}" etype="discrete" type="discrete" vtype="discrete" name="{issue_name}">\n'
            vals = iget(issues, k).all
            for indx, v in enumerate(vals):
                u = gmap(iget(self.issue_utilities, k), v)
                v_ = (
                    v
                    if not (isinstance(v, tuple) or isinstance(v, list))
                    else "-".join([str(_) for _ in v])
                )
                output += (
                    f'    <item index="{indx+1}" value="{v_}" evaluation="{u}" />\n'
                )
            output += "</issue>\n"
        for i, k in enumerate(keys):
            output += (
                f'<weight index="{i+1}" value="{iget(self.weights, k)}">\n</weight>\n'
            )
        return output

    def __str__(self):
        return f"u: {self.issue_utilities}\n w: {self.weights}"


class MappingUtilityFunction(UtilityFunction):
    """Outcome mapping utility function.

    This is the simplest possible utility function and it just maps a set of ``Outcome``s to a set of
    ``UtilityValue``(s). It is only usable with single-issue negotiations. It can be constructed with wither a mapping
    (e.g. a dirct) or a callable function.

    Args:
            mapping: Either a callable or a mapping from ``Outcome`` (dict) to ``UtilityValue``.
            default: value returned for outcomes causing exception (e.g. invalid outcomes).
            name: name of the utility function. If None a random name will be generated.

    Eamples:

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
        ['price: (10.0, 20.0)', "delivery: ['delivered', 'not delivered']", 'quality: 5']
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
        reserved_value: Optional[UtilityValue] = 0.0,
        ami: AgentMechanismInterface = None,
    ) -> None:
        super().__init__(name=name, reserved_value=reserved_value, ami=ami)
        self.mapping = mapping
        self.default = default

    def __call__(self, offer: Optional["Outcome"]) -> Optional[UtilityValue]:
        # noinspection PyBroadException
        if offer is None:
            return self.reserved_value
        try:
            if isinstance(offer, dict) and isinstance(self.mapping, dict):
                m = gmap(self.mapping, tuple(offer.values()))
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


class RandomUtilityFunction(MappingUtilityFunction):
    """A random utility function for a discrete outcome space"""

    def __init__(self, outcomes: List[Outcome]):
        if len(outcomes) < 1:
            raise ValueError("Cannot create a random utility function without outcomes")
        if isinstance(outcomes[0], tuple):
            pass
        else:
            outcomes = [tuple(o.keys()) for o in outcomes]
        super().__init__(mapping=dict(zip(outcomes, np.random.rand(len(outcomes)))))


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
        ['price: (10.0, 20.0)', "delivery: ['delivered', 'not delivered']", 'quality: 5']
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
        reserved_value: Optional[UtilityValue] = 0.0,
        ami: AgentMechanismInterface = None,
    ) -> None:
        super().__init__(name=name, reserved_value=reserved_value, ami=ami)
        self.issue_utilities = issue_utilities
        self.f = f

    def __call__(self, offer: Optional["Outcome"]) -> Optional[UtilityValue]:
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


class HyperRectangleUtilityFunction(UtilityFunction):
    """A utility function defined as a set of hyper-volumes.

    The utility function that is calulated by combining linearly a set of *probably nonlinear* functions applied in
    predefined hyper-volumes of the outcome space.

     Args:
          outcome_ranges: The outcome_ranges for which the `mappings` are defined
          mappings: The *possibly nonlinear* mapppings correponding to the outcome_ranges
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
        reserved_value: Optional[UtilityValue] = 0.0,
        ami: AgentMechanismInterface = None,
    ) -> None:
        super().__init__(name=name, reserved_value=reserved_value, ami=ami)
        self.outcome_ranges = outcome_ranges
        self.mappings = utilities
        self.weights = weights
        self.ignore_issues_not_in_input = ignore_issues_not_in_input
        self.ignore_failing_range_utilities = ignore_failing_range_utilities
        self.adjust_params()

    def adjust_params(self):
        if self.weights is None:
            self.weights = [1.0] * len(self.outcome_ranges)

    def __call__(self, offer: Optional["Outcome"]) -> Optional[UtilityValue]:
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
        reserved_value: Optional[UtilityValue] = 0.0,
        ami: AgentMechanismInterface = None,
    ) -> None:
        super().__init__(name=name, reserved_value=reserved_value, ami=ami)
        self.hypervolumes = hypervolumes
        self.mappings = mappings
        self.f = f

    def __call__(self, offer: Optional["Outcome"]) -> Optional[UtilityValue]:
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


class ComplexWeightedUtilityFunction(UtilityFunction):
    """ A utility function composed of linear aggregation of other utility functions

        Args:
            ufuns: An iterable of utility functions
            weights: Weights used for combination
            name: Utility function name

        """

    def __init__(
        self,
        ufuns: Iterable[UtilityFunction],
        weights: Optional[Iterable[float]] = None,
        name=None,
        reserved_value: Optional[UtilityValue] = 0.0,
        ami: AgentMechanismInterface = None,
    ):
        super().__init__(name=name, reserved_value=reserved_value, ami=ami)
        self.ufuns = list(ufuns)
        if weights is None:
            weights = [1.0] * len(self.ufuns)
        self.weights = list(weights)

    def __call__(self, offer: Outcome) -> Optional[UtilityValue]:
        """Calculate the utility_function value for a given outcome.

        Args:
            offer: The offer to be evaluated.


        Remarks:
            - You cannot return None from overriden apply() functions but raise an exception (ValueError) if it was
              not possible to calculate the UtilityValue.
            - Return A UtilityValue not a float for real-valued utilities for the benefit of inspection code.

        Returns:
            UtilityValue: The utility_function value which may be a distribution. If `None` it means the utility_function value cannot be
            calculated.
        """
        if offer is None:
            return self.reserved_value
        u = ExactUtilityValue(0.0)
        failure = False
        for f, w in zip(self.ufuns, self.weights):
            util = f(offer)
            if util is not None:
                u += w * util
            else:
                failure = True
        return u if not failure else None

    def xml(self, issues: List[Issue]) -> str:
        output = ""
        # @todo implement weights. Here I assume they are always 1.0
        for f, w in zip(self.ufuns, self.weights):
            output += f.xml(issues)
        return output


class ComplexNonlinearUtilityFunction(UtilityFunction):
    """ A utility function composed of nonlinear aggregation of other utility functions

    Args:
        ufuns: An iterable of utility functions
        combination_function: The function used to combine results of ufuns
        name: Utility function name

    """

    def __init__(
        self,
        ufuns: Iterable[UtilityFunction],
        combination_function=Callable[[Iterable[UtilityValue]], UtilityValue],
        name=None,
        reserved_value: Optional[UtilityValue] = 0.0,
        ami: AgentMechanismInterface = None,
    ):
        super().__init__(name=name, reserved_value=reserved_value, ami=ami)
        self.ufuns = list(ufuns)
        self.combination_function = combination_function

    def __call__(self, offer: Outcome) -> Optional[UtilityValue]:
        """Calculate the utility_function value for a given outcome.

        Args:
            offer: The offer to be evaluated.


        Remarks:
            - You cannot return None from overriden apply() functions but raise an exception (ValueError) if it was
              not possible to calculate the UtilityValue.
            - Return A UtilityValue not a float for real-valued utilities for the benefit of inspection code.

        Returns:
            UtilityValue: The utility_function value which may be a distribution. If `None` it means the utility_function value cannot be
            calculated.
        """
        if offer is None:
            return self.reserved_value
        return self.combination_function([f(offer) for f in self.ufuns])

    def xml(self, issues: List[Issue]) -> str:
        raise NotImplementedError(f"Cannot convert {self.__class__.__name__} to xml")


class IPUtilityFunction(UtilityFunction):
    """Independent Probabilistic Utility Function.

    Args:

        outcomes: Iterable of outcomes
        distribs: distributions associated with the outcomes
        name: ufun name

    Examples:

        >>> f = IPUtilityFunction(outcomes=[('o1',), ('o2',)]
        ...         , distributions=[UtilityDistribution(dtype='uniform', loc=0.0, scale=0.5)
        ...         , UtilityDistribution(dtype='uniform', loc=0.1, scale=0.5)])
        >>> f(('o1',))
        U(0.0, 0.5)

        >>> f = IPUtilityFunction(outcomes=[{'cost': 10, 'dist': 20}, {'cost': 10, 'dist': 30}]
        ...         , distributions=[UtilityDistribution(dtype='uniform', loc=0.0, scale=0.5)
        ...         , UtilityDistribution(dtype='uniform', loc=0.1, scale=0.5)])
        >>> f({'cost': 10, 'dist': 30})
        U(0.1, 0.6)


    """

    def __init__(
        self,
        outcomes: Iterable["Outcome"],
        distributions: Iterable["UtilityDistribution"] = None,
        issue_names: Iterable[str] = None,
        name=None,
        reserved_value: Optional[UtilityValue] = 0.0,
        ami: AgentMechanismInterface = None,
    ):
        super().__init__(name=name, reserved_value=reserved_value, ami=ami)
        outcomes, distributions = (
            list(outcomes),
            (list(distributions) if distributions is not None else None),
        )
        if len(outcomes) < 1:
            raise ValueError(
                "IPUtilityFunction cannot be initialized with zero outcomes"
            )
        self.tupelized = False

        self.n_issues = len(outcomes[0])
        if issue_names is None:
            self.issue_names = sorted(ikeys(outcomes[0]))
        else:
            self.issue_names = range(len(outcomes[0]))

        self.issue_keys = dict(zip(range(self.n_issues), self.issue_names))

        if not isinstance(outcomes[0], tuple):
            outcomes = [
                tuple(iget(_, key, None) for key in self.issue_names) for _ in outcomes
            ]
            self.tupelized = True
        if distributions is None:
            distributions = [
                UtilityDistribution(dtype="uniform", loc=0.0, scale=1.0)
                for _ in range(len(outcomes))
            ]
        self.distributions = dict(zip(outcomes, distributions))

    def distribution(self, outcome: "Outcome") -> "UtilityValue":
        """
        Returns the distributon associated with a specific outcome
        Args:
            outcome:

        Returns:

        """
        return self.distributions[self.key(outcome)]

    @classmethod
    def from_ufun(
        cls,
        u: MappingUtilityFunction,
        range: Tuple[float, float] = (0.0, 1.0),
        uncertainty: float = 0.5,
        variability: float = 0.0,
    ) -> "IPUtilityFunction":
        """
        Generates a distribution from which `u` may have been sampled
        Args:
            u:
            range: range of the utility_function values
            uncertainty: uncertainty level

        Examples:

            - No uncertainty
            >>> u = MappingUtilityFunction(mapping=dict(zip([('o1',), ('o2',)], [0.3, 0.7])))
            >>> p = IPUtilityFunction.from_ufun(u, uncertainty=0.0)
            >>> print(p)
            {('o1',): U(0.3, 0.3), ('o2',): U(0.7, 0.7)}

            - Full uncertainty
            >>> u = MappingUtilityFunction(mapping=dict(zip([('o1',), ('o2',)], [0.3, 0.7])))
            >>> p = IPUtilityFunction.from_ufun(u, uncertainty=1.0)
            >>> print(p)
            {('o1',): U(0.0, 1.0), ('o2',): U(0.0, 1.0)}

            - some uncertainty
            >>> u = MappingUtilityFunction(mapping=dict(zip([('o1',), ('o2',)], [0.3, 0.7])))
            >>> p = IPUtilityFunction.from_ufun(u, uncertainty=0.1)
            >>> print([_.scale for _ in p.distributions.values()])
            [0.1, 0.1]
            >>> for k, v in p.distributions.items():
            ...     assert v.loc <= u(k)


        Returns:
            a new IPUtilityFunction
        """
        if isinstance(u.mapping, dict):
            return cls.from_mapping(
                u.mapping, range=range, uncertainty=uncertainty, variability=variability
            )
        return cls.from_mapping(
            dict(zip(ikeys(u.mapping), ivalues(u.mapping))),
            range=range,
            uncertainty=uncertainty,
            variability=variability,
        )

    @classmethod
    def from_mapping(
        cls,
        mapping: Dict["Outcome", float],
        range: Tuple[float, float] = (0.0, 1.0),
        uncertainty: float = 0.5,
        variability: float = 0.0,
    ) -> "IPUtilityFunction":
        """
        Generates a distribution from which `u` may have been sampled
        Args:
            mapping: mapping from outcomes to float values
            range: range of the utility_function values
            uncertainty: uncertainty level

        Examples:

            - No uncertainty
            >>> mapping=dict(zip([('o1',), ('o2',)], [0.3, 0.7]))
            >>> p = IPUtilityFunction.from_mapping(mapping, uncertainty=0.0)
            >>> print(p)
            {('o1',): U(0.3, 0.3), ('o2',): U(0.7, 0.7)}

            - Full uncertainty
            >>> mapping=dict(zip([('o1',), ('o2',)], [0.3, 0.7]))
            >>> p = IPUtilityFunction.from_mapping(mapping, uncertainty=1.0)
            >>> print(p)
            {('o1',): U(0.0, 1.0), ('o2',): U(0.0, 1.0)}

            - some uncertainty
            >>> mapping=dict(zip([('o1',), ('o2',)], [0.3, 0.7]))
            >>> p = IPUtilityFunction.from_mapping(mapping, uncertainty=0.1)
            >>> print([_.scale for _ in p.distributions.values()])
            [0.1, 0.1]
            >>> for k, v in p.distributions.items():
            ...     assert v.loc <= mapping[k]

        Returns:
            a new IPUtilityFunction
        """
        outcomes = list(mapping.keys())
        if isinstance(uncertainty, Iterable):
            uncertainties = uncertainty
        elif variability <= 0.0:
            uncertainties = [uncertainty] * len(outcomes)
        else:
            uncertainties = (
                uncertainty
                + (np.random.rand(len(outcomes)) - 0.5) * variability * uncertainty
            ).tolist()
        return IPUtilityFunction(
            outcomes=outcomes,
            distributions=[
                Distribution.around(value=mapping[o], uncertainty=u, range=range)
                for o, u in zip(outcomes, uncertainties)
            ],
        )

    def __str__(self):
        return pprint.pformat(self.distributions)

    def sample(self) -> MappingUtilityFunction:
        """
        Samples the utility_function distribution to create a mapping utility function


        Examples:
            >>> import random
            >>> f = IPUtilityFunction(outcomes=[('o1',), ('o2',)]
            ...         , distributions=[UtilityDistribution(dtype='uniform', loc=0.0, scale=0.2)
            ...         , UtilityDistribution(dtype='uniform', loc=0.4, scale=0.5)])
            >>> u = f.sample()
            >>> assert u(('o1',)) <= 0.2
            >>> assert 0.4 <= u(('o2',)) <= 0.9

        Returns:

            MappingUtilityFunction
        """
        return MappingUtilityFunction(
            mapping={o: d.sample(1)[0] for o, d in self.distributions.items()}
        )

    def key(self, outcome: "Outcome"):
        """
        Returns the key of the given outcome in self.distributions.

        Args:
            outcome:

        Returns:
            tuple

        Examples:

        >>> f = IPUtilityFunction(outcomes=[('o1',), ('o2',)]
        ...         , distributions=[UtilityDistribution(dtype='uniform', loc=0.0, scale=0.5)
        ...         , UtilityDistribution(dtype='uniform', loc=0.1, scale=0.5)])
        >>> f.key({0:'o1'})
        ('o1',)
        >>> f.key(('o1',))
        ('o1',)
        >>> f.distributions
        {('o1',): U(0.0, 0.5), ('o2',): U(0.1, 0.6)}
        >>> f.distribution(('o1',))
        U(0.0, 0.5)

        >>> f = IPUtilityFunction(outcomes=[{'cost': 10, 'dist': 20}, {'dist': 30, 'cost': 10}]
        ...         , distributions=[UtilityDistribution(dtype='uniform', loc=0.0, scale=0.5)
        ...         , UtilityDistribution(dtype='uniform', loc=0.1, scale=0.5)])
        >>> f.key({'dist': 30, 'cost': 10})
        (10, 30)
        >>> f.key({'cost': 10, 'dist': 30})
        (10, 30)
        >>> f.distributions
        {(10, 20): U(0.0, 0.5), (10, 30): U(0.1, 0.6)}
        >>> f.distribution((10, 20.0))
        U(0.0, 0.5)
        >>> f.distribution({'cost': 10, 'dist': 20})
        U(0.0, 0.5)

        """
        if isinstance(outcome, tuple):
            return outcome
        return tuple((outcome.get(_, None) for _ in self.issue_names))

    def __call__(self, offer: Outcome) -> Optional[UtilityValue]:
        """Calculate the utility_function value for a given outcome.

        Args:
            offer: The offer to be evaluated.


        Remarks:
            - You cannot return None from overriden apply() functions but raise an exception (ValueError) if it was
              not possible to calculate the UtilityValue.
            - Return A UtilityValue not a float for real-valued utilities for the benefit of inspection code.

        Returns:
            UtilityValue: The utility_function value which may be a distribution. If `None` it means the utility_function value cannot be
            calculated.
        """
        if offer is None:
            return self.reserved_value
        if self.tupelized and not isinstance(offer, tuple):
            offer = tuple(ivalues(offer))
        return self.distributions[offer]

    def xml(self, issues: List[Issue]) -> str:
        raise NotImplementedError(f"Cannot convert {self.__class__.__name__} to xml")


def _pareto_frontier(
    points, eps=-1e-18, sort_by_welfare=False
) -> Tuple[List[Tuple[float]], List[int]]:
    """Finds the pareto-frontier of a set of points

    Args:
        points: list of points
        eps: A (usually negative) small number to treat as zero during calculations
        sort_by_welfare: If True, the results are sorted descindingly by total welfare

    Returns:

    """
    points = np.asarray(points)
    n = len(points)
    indices = np.array(range(n))
    for j in range(points.shape[1]):
        order = points[:, 0].argsort()[-1::-1]
        points = points[order]
        indices = indices[order]

    frontier = [(indices[0], points[0, :])]
    for p in range(1, n):
        current = points[p, :]
        for i, (_, f) in enumerate(frontier):
            current_better, current_worse = current > f, current < f
            if np.all(current == f):
                break
            if not np.any(current_better) and np.any(current_worse):
                # current is dominated, break
                break
            if np.any(current_better):
                if not np.any(current_worse):
                    # current dominates f, append it, remove f and scan for anything else dominated by current
                    for j, (_, g) in enumerate(frontier[i + 1 :]):
                        if np.all(current == g):
                            frontier = frontier[:i] + frontier[i + 1 :]
                            break
                        if np.any(current > g) and not np.any(current < g):
                            frontier = frontier[:j] + frontier[j + 1 :]
                    else:
                        frontier[i] = (indices[p], current)
                else:
                    # neither current nor f dominate each other, append current only if it is not
                    # dominated by anything in frontier
                    for j, (_, g) in enumerate(frontier[i + 1 :]):
                        if np.all(current == g) or (
                            np.any(g > current) and not np.any(current > g)
                        ):
                            break
                    else:
                        frontier.append((indices[p], current))
    if sort_by_welfare:
        welfare = [np.sum(_[1]) for _ in frontier]
        indx = sorted(range(len(welfare)), key=lambda x: welfare[x], reverse=True)
        frontier = [frontier[_] for _ in indx]
    return [tuple(_[1]) for _ in frontier], [_[0] for _ in frontier]


def pareto_frontier(
    ufuns: Iterable[UtilityFunction],
    outcomes: Iterable[Outcome] = None,
    issues: Iterable[Issue] = None,
    n_discretization: Optional[int] = 10,
    sort_by_welfare=False,
) -> Tuple[List[Tuple[float]], List[int]]:
    """Finds all pareto-optimal outcomes in the list

    Args:

        ufuns: The utility functions
        outcomes: the outcomes to be checked. If None then all possible outcomes from the issues will be checked
        issues: The set of issues (only used when outcomes is None)
        n_discretization: The number of items to discretize each real-dimension into
        sort_by_welfare: If True, the resutls are sorted descendingly by total welfare

    Returns:
        Two lists of the same length. First list gives the utilities at pareto frontier points and second list gives their indices

    """

    ufuns = list(ufuns)
    if issues:
        issues = list(issues)
    if outcomes:
        outcomes = list(outcomes)

    # calculate all candidate outcomes
    if outcomes is None:
        if issues is None:
            return [], []
        outcomes = itertools.product(
            *[issue.alli(n=n_discretization) for issue in issues]
        )
    points = [[ufun(outcome) for ufun in ufuns] for outcome in outcomes]
    return _pareto_frontier(points, sort_by_welfare=sort_by_welfare)


def normalize(
    ufun: UtilityFunction,
    outcomes: Collection[Outcome],
    rng: Tuple[float, float] = (0.0, 1.0),
    epsilon: float = 1e-6,
    infeasible_cutoff: Optional[float] = -1000.0,
) -> UtilityFunction:
    """Normalizes a utility function to the range [0, 1]

    Args:
        ufun: The utility function to normalize
        outcomes: A collection of outcomes to normalize for
        rng: range to normalize to. Default is [0, 1]
        epsilon: A small number specifying the resolution
        infeasible_cutoff: A value under which any utility is considered infeasible and is not used in normalization

    Returns:
        UtilityFunction: A utility function that is guaranteed to be normalized for the set of given outcomes

    """
    u: List[float]
    u = [ufun(o) for o in outcomes]
    if infeasible_cutoff is not None:
        u = [float(_) for _ in u if _ is not None and float(_) >= infeasible_cutoff]
    else:
        u = [float(_) for _ in u if _ is not None]
    if len(u) == 0:
        return ufun
    mx, mn = max(u), min(u)
    if abs(mx - 1.0) < epsilon and abs(mn) < epsilon:
        return ufun
    if mx == mn:
        if -epsilon <= mn <= 1 + epsilon:
            return ufun
        else:
            r = (
                float(ufun.reserved_value) / mn
                if ufun.reserved_value is not None and mn != 0.0
                else 0.0
            )
            if infeasible_cutoff is not None:
                return ComplexNonlinearUtilityFunction(
                    ufuns=[ufun],
                    combination_function=lambda x: infeasible_cutoff
                    if x[0] is None
                    else x[0]
                    if x[0] < infeasible_cutoff
                    else 0.5 * x[0] / mn,
                )
            else:
                return ComplexWeightedUtilityFunction(
                    ufuns=[ufun],
                    weights=[0.5 / mn],
                    name=ufun.name + "-normalized",
                    reserved_value=r,
                    ami=ufun.ami,
                )
    scale = (rng[1] - rng[0]) / (mx - mn)
    r = scale * (ufun.reserved_value - mn) if ufun.reserved_value else 0.0
    if infeasible_cutoff is not None:
        return ComplexNonlinearUtilityFunction(
            ufuns=[ufun],
            combination_function=lambda x: infeasible_cutoff
            if x[0] is None
            else x[0]
            if x[0] < infeasible_cutoff
            else scale * (x[0] - mn) + rng[0],
        )
    else:
        return ComplexWeightedUtilityFunction(
            ufuns=[ufun, ConstUFun(-mn + rng[0] / scale)],
            weights=[scale, scale],
            name=ufun.name + "-normalized",
            reserved_value=r,
            ami=ufun.ami,
        )


class JavaUtilityFunction(UtilityFunction, JavaCallerMixin):
    """A utility function implemented in Java"""

    def __init__(self, java_object, java_class_name: Optional[str], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_java_bridge(
            java_object=java_object,
            java_class_name=java_class_name,
            auto_load_java=False,
        )
        if java_object is None:
            self._java_object.fromMap(to_java(self))

    def __call__(self, offer: Outcome) -> Optional[UtilityValue]:
        return self._java_object.call(to_java(outcome_as_dict(offer)))

    def xml(self, issues: List[Issue]) -> str:
        return "Java UFun"
