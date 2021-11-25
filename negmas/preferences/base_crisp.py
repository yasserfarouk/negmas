from __future__ import annotations

import itertools
import numbers
import random
import warnings
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from functools import reduce
from math import sqrt
from operator import mul
from typing import (
    Any,
    Callable,
    Collection,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Type,
    Union,
)

import numpy as np

from negmas.generics import ienumerate, ivalues
from negmas.helpers import PATH, ikeys
from negmas.outcomes import (
    Issue,
    Outcome,
    discretize_and_enumerate_issues,
    enumerate_issues,
    num_outcomes,
    outcome_is_valid,
    sample_issues,
)

from .base_probabilistic import ProbUtilityFunction
from .preferences import CardinalPreferences

__all__ = [
    "UtilityFunction",
]


class UtilityFunction(CardinalPreferences, ProbUtilityFunction):
    def normalize(
        self,
        outcomes: list[Outcome] | None = None,
        rng: Tuple[float | None, float | None] = (0.0, 1.0),
        infeasible_cutoff: float = float("-inf"),
        epsilon: float = 1e-6,
    ) -> "UtilityFunction":
        """
        Creates a new utility function that is normalized based on input conditions.

        Args:
            outcomes: A set of outcomes to limit our attention to. If not given,
                      the whole ufun is normalized
            rng: The minimum and maximum value to normalize to. If either is None, it is ignored.
                 This means that passing `(None, 1.0)` will normalize the ufun so that the maximum
                 is `1` but will not guarantee any limit for the minimum and so on.
            infeasible_cutoff: outcomes with utility value less than or equal to this value will not
                               be considered during normalization
            epsilon: A small allowed error in normalization
        """
        from .ops import normalize as gnormalize

        if outcomes is None:
            raise ValueError(
                f"Cannot normalize ufun of type {self.__class__.__name__} without passing `outcomes` to consider"
            )
        return gnormalize(self, outcomes, rng, epsilon, infeasible_cutoff)

    @classmethod
    def from_genius(cls, file_name: PATH, **kwargs) -> Tuple["UtilityFunction", float]:
        """Imports a utility function from a GENIUS XML file.

        Args:

            file_name (str): File name to import from

        Returns:

            A utility function object (depending on the input file)


        Examples:

            >>> from negmas.preferences import UtilityFunction
            >>> import pkg_resources
            >>> from negmas.inout import load_genius_domain
            >>> domain = load_genius_domain(pkg_resources.resource_filename('negmas'
            ...                             , resource_name='tests/data/Laptop/Laptop-C-domain.xml'))
            >>> u, d = UtilityFunction.from_genius(file_name = pkg_resources.resource_filename('negmas'
            ...                                      , resource_name='tests/data/Laptop/Laptop-C-prof1.xml')
            ...                                      , issues=domain.issues)
            >>> u.__class__.__name__
            'LinearUtilityAggregationFunction'
            >>> u.reserved_value
            0.0
            >>> d
            1.0

        Remarks:
            See ``from_xml_str`` for all the parameters

        """
        kwargs["name"] = str(file_name)
        with open(file_name) as f:
            xml_str = f.read()
            return cls.from_xml_str(xml_str=xml_str, **kwargs)

    def to_genius(self, file_name: PATH, issues: List[Issue] = None, **kwargs):
        """Exports a utility function to a GENIUS XML file.

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
            >>> domain = load_genius_domain(domain_file_name=pkg_resources.resource_filename('negmas'
            ...                                             , resource_name='tests/data/Laptop/Laptop-C-domain.xml'))
            >>> u, discount = UtilityFunction.from_genius(file_name=pkg_resources.resource_filename('negmas'
            ...                                             , resource_name='tests/data/Laptop/Laptop-C-prof1.xml')
            ...                                             , issues=domain.issues)
            >>> u.to_genius(discount_factor=discount
            ...     , file_name = pkg_resources.resource_filename('negmas'
            ...                   , resource_name='tests/data/LaptopConv/Laptop-C-prof1.xml')
            ...     , issues=domain.issues)

        Remarks:
            See ``to_xml_str`` for all the parameters

        """
        with open(file_name, "w") as f:
            f.write(self.to_xml_str(issues=issues, **kwargs))

    def to_xml_str(self, issues: List[Issue] = None, discount_factor=None) -> str:
        """Exports a utility function to a well formatted string"""
        if issues is None:
            issues = self.issues

        if issues is not None:
            n_issues = len(issues)
        else:
            n_issues = 0
        output = (
            f'<utility_space type="any" number_of_issues="{n_issues}">\n'
            f'<objective index="1" etype="objective" type="objective" description="" name="any">\n'
        )

        output += self.xml(issues=issues)
        if "</objective>" not in output:
            output += "</objective>\n"
            if discount_factor is not None:
                output += f'<discount_factor value="{discount_factor}" />\n'
        if self.reserved_value != float("-inf") and "<reservation value" not in output:
            output += f'<reservation value="{self.reserved_value}" />\n'
        if "</utility_space>" not in output:
            output += "</utility_space>\n"
        return output

    @classmethod
    def from_xml_str(
        cls,
        xml_str: str,
        issues: Optional[List[Issue]] = None,
        force_numeric=False,
        safe_parsing=True,
        ignore_discount=False,
        ignore_reserved=False,
        name: str = None,
    ) -> Tuple["UtilityFunction", float]:
        """Imports a utility function from a GENIUS XML string.

        Args:

            xml_str (str): The string containing GENIUS style XML utility function definition
            issues (List[Issue] | None): Optional issue space to confirm that the utility function is valid
            force_single_issue (bool): Tries to generate a MappingUtility function with a single issue which is the
            product of all issues in the input
            safe_parsing (bool): Turn on extra checks

        Returns:

            A utility function object (depending on the input file)


        Examples:

            >>> import pkg_resources
            >>> from negmas.inout import load_genius_domain
            >>> domain = load_genius_domain(pkg_resources.resource_filename('negmas'
            ...                             , resource_name='tests/data/Laptop/Laptop-C-domain.xml'))
            >>> u, _ = UtilityFunction.from_xml_str(open(pkg_resources.resource_filename('negmas'
            ...                                      , resource_name='tests/data/Laptop/Laptop-C-prof1.xml')
            ...                                      , 'r').read(), issues=domain.issues)

            >>> u, _ = UtilityFunction.from_xml_str(open(pkg_resources.resource_filename('negmas'
            ...                                      , resource_name='tests/data/Laptop/Laptop-C-prof1.xml')
            ...                                      , 'r').read(), issues=domain.issues)
            >>> assert abs(u(("Dell", "60 Gb", "19'' LCD",)) - 21.987727736172488) < 0.000001
            >>> assert abs(u(("HP", "80 Gb", "20'' LCD",)) - 22.68559475583014) < 0.000001


        """
        from negmas.preferences.complex import ComplexWeightedUtilityFunction
        from negmas.preferences.linear import (
            LinearUtilityAggregationFunction,
            LinearUtilityFunction,
        )
        from negmas.preferences.nonlinear import (
            HyperRectangleUtilityFunction,
            MappingUtilityFunction,
        )

        # keep_issue_names = True
        keep_value_names = True
        force_single_issue = False
        normalize_utility = False
        normalize_max_only = False
        max_n_outcomes: int = 1_000_000

        root = ET.fromstring(xml_str)
        if safe_parsing and root.tag != "utility_space":
            raise ValueError(f"Root tag is {root.tag}: Expected utility_space")

        ordered_issues = []
        issue_order_passed = False
        domain_issues: Optional[Dict[str, Issue]] = None
        if issues is not None:
            if isinstance(issues, list):
                issue_order_passed = True
                ordered_issues = [_ for _ in issues]
                domain_issues = dict(zip([_.name for _ in issues], issues))
            elif isinstance(issues, Issue) and force_single_issue:
                domain_issues = dict(zip([issues.name], [issues]))
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
        found_issues = {}
        real_issues = {}
        linear_issues = {}
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
                        ii = int(r.attrib["index"]) - 1
                        key = issue_keys[ii]
                        ranges[ii] = (
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
                issue_key = myname
                if not issue_order_passed:
                    ordered_issues.append(issue_key)
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
                    found_issues[issue_key] = {}
                    if (
                        domain_issues is not None
                        and domain_issues[myname].is_continuous()
                    ):
                        raise ValueError(
                            f"Got a {mytype} issue but expected a continuous valued issue"
                        )
                    # found_issues[indx]['items'] = {}
                elif mytype in ("integer", "real"):
                    lower, upper = None, None
                    if domain_issues is not None and domain_issues.get(myname, None):
                        lower = domain_issues[myname].min_value
                        upper = domain_issues[myname].max_value
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
                        found_issues[issue_key] = {}
                        if (
                            domain_issues is not None
                            and domain_issues[myname].is_continuous()
                        ):
                            raise ValueError(
                                f"Got a {mytype} issue but expected a continuous valued issue"
                            )
                        # found_issues[indx]['items'] = {}
                        lower, upper = int(lower), int(upper)
                        for i in range(lower, upper + 1):
                            if domain_issues is not None and not outcome_is_valid(
                                (i,), [domain_issues[myname]]
                            ):
                                raise ValueError(
                                    f"Value {i} is not in the domain issue values: "
                                    f"{domain_issues[myname].values}"
                                )
                            found_issues[issue_key][i] = (
                                i if keep_value_names else i - lower
                            )
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
                            found_issues[issue_key] = {}
                            for i in range(lower, upper + 1):
                                found_issues[issue_key][i] = (
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
                            found_issues[issue_key][item_key] = float(val)
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
                                if value_scale is not None:
                                    offset += value_shift * slope
                                    slope *= value_scale * slope
                                fun = lambda x: offset + slope * float(x)
                                linear_issues[issue_key] = dict(
                                    type="linear", slope=slope, offset=offset
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
                                if value_scale is not None:
                                    offset1 += value_shift * slope1
                                    offset2 += value_shift * slope2
                                    slope1 *= value_scale
                                    slope2 *= value_scale
                                fun = (
                                    lambda x: offset1
                                    + slope1 * (value_scale * float(x) + value_shift)
                                    if x < middle
                                    else offset2
                                    + slope2 * (value_scale * float(x) + value_shift)
                                )
                            else:
                                raise ValueError(
                                    f'Unknown ftype {item.attrib["ftype"]}'
                                )
                            if mytype == "real" and value_scale is None:
                                real_issues[issue_key]["fun"] = fun
                            else:
                                for item_key, value in found_issues[issue_key].items():
                                    found_issues[issue_key][item_key] = fun(value)
                                found_values = True
                    if not found_values and issue_key in found_issues.keys():
                        found_issues.pop(issue_key, None)
                else:
                    """Here goes the code for real-valued issues"""

        # if not keep_issue_names:
        #     found_issues = [found_issues[_] for _ in found_issues.keys()]
        #     real_issues = [real_issues[_] for _ in sorted(real_issues.keys())]
        #     for i, issue in enumerate(found_issues):
        #         found_issues[i] = [issue[_] for _ in issue.keys()]

        if safe_parsing and (
            len(weights) > 0
            and len(weights) != len(found_issues) + len(real_issues)
            and len(weights) != len(found_issues)
        ):
            raise ValueError(
                f"Got {len(weights)} weights for {len(found_issues)} issues and {len(real_issues)} real issues"
            )

        if force_single_issue and (
            len(rects) > 0
            or len(real_issues) > 1
            or (len(real_issues) > 0 and len(found_issues) > 0)
        ):
            raise ValueError(
                f"Cannot force single issue with a hyper-volumes based function"
            )

        # add utilities specified not as hyper-rectangles
        u = None
        if len(found_issues) > 0:
            if force_single_issue:
                if len(weights) > 0:
                    for key, issue in zip(ikeys(found_issues), ivalues(found_issues)):
                        try:
                            w = weights[issue_info[key]["index"]]
                        except:
                            w = 1.0
                        for item_key in ikeys(issue):
                            issue[item_key] *= w
                n_outcomes = None
                if max_n_outcomes is not None:
                    n_items = [len(_) for _ in ivalues(found_issues)]
                    n_outcomes = reduce(mul, n_items, 1)
                    if n_outcomes > max_n_outcomes:
                        return None, discount_factor
                if keep_value_names:
                    names = itertools.product(
                        *(
                            [
                                str(item_key).replace("&", "-")
                                for item_key in ikeys(items)
                            ]
                            for issue_key, items in zip(
                                ikeys(found_issues), ivalues(found_issues)
                            )
                        )
                    )
                    names = map(lambda items: ("+".join(items),), names)
                else:
                    if n_outcomes is None:
                        n_items = [len(_) for _ in ivalues(found_issues)]
                        n_outcomes = reduce(mul, n_items, 1)
                    names = [(_,) for _ in range(n_outcomes)]
                utils = itertools.product(
                    *(
                        [item_utility for item_utility in ivalues(items)]
                        for issue_key, items in zip(
                            ikeys(found_issues), ivalues(found_issues)
                        )
                    )
                )
                utils = map(lambda vals: sum(vals), utils)
                if normalize_utility:
                    utils = list(utils)
                    umax, umin = max(utils), (0.0 if normalize_max_only else min(utils))
                    if umax != umin:
                        utils = [(_ - umin) / (umax - umin) for _ in utils]
                u = MappingUtilityFunction(
                    dict(zip(names, utils)), name=name, issues=ordered_issues
                )
            else:
                utils = None
                if normalize_utility:
                    utils = itertools.product(
                        *(
                            [
                                item_utility * weights[issue_info[issue_key]["index"]]
                                for item_utility in ivalues(items)
                            ]
                            for issue_key, items in zip(
                                ikeys(found_issues), ivalues(found_issues)
                            )
                        )
                    )
                    if len(weights) > 0:
                        ws = dict()
                        for key, issue in zip(
                            ikeys(found_issues), ivalues(found_issues)
                        ):
                            try:
                                ws[key] = weights[issue_info[key]["index"]]
                            except:
                                ws[key] = 1.0
                        wsum = sum(weights.values())
                    else:
                        ws = [1.0] * len(found_issues)
                        wsum = len(found_issues)

                    utils = list(map(sum, utils))
                    umax, umin = max(utils), (0.0 if normalize_max_only else min(utils))
                    factor = umax - umin
                    if factor > 1e-8:
                        offset = umin / (wsum * factor)
                    else:
                        offset = 0.0
                        factor = umax if umax > 1e-8 else 1.0
                    for key, issue in ienumerate(found_issues):
                        for item_key in ikeys(issue):
                            found_issues[key][item_key] = (
                                found_issues[key][item_key] / factor - offset
                            )
                if len(found_issues) > 1:
                    ws = dict()
                    if len(weights) > 0:
                        for key, issue in zip(
                            ikeys(found_issues), ivalues(found_issues)
                        ):
                            try:
                                ws[key] = weights[issue_info[key]["index"]]
                            except:
                                ws[key] = 1.0

                    if isinstance(found_issues, list):
                        ws = [ws[i] for i in range(len(found_issues))]

                    if len(found_issues) == len(linear_issues):
                        # todo correct this
                        u = LinearUtilityFunction(
                            weights=ws, issues=ordered_issues, biases=None
                        )
                    else:
                        u = LinearUtilityAggregationFunction(
                            issue_utilities=found_issues,
                            weights=ws,
                            issues=ordered_issues,
                        )
                else:
                    if len(weights) > 0:
                        for key, issue in zip(
                            ikeys(found_issues), ivalues(found_issues)
                        ):
                            try:
                                w = weights[issue_info[key]["index"]]
                            except:
                                w = 1.0
                            for item_key in ikeys(issue):
                                issue[item_key] *= w
                    first_key = list(ikeys(found_issues))[0]
                    if utils is None:
                        utils = ivalues(found_issues[first_key])
                    u = MappingUtilityFunction(
                        dict(
                            zip([(_,) for _ in ikeys(found_issues[first_key])], utils)
                        ),
                        name=name,
                        issues=ordered_issues,
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
                    *(
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
                    )
                )
                if len(weights) > 0:
                    ws = dict()
                    for key, issue in zip(ikeys(found_issues), ivalues(found_issues)):
                        try:
                            ws[key] = weights[issue_info[key]["index"]]
                        except:
                            ws[key] = 1.0
                    wsum = sum(weights.values())
                else:
                    ws = [1.0] * len(found_issues)
                    wsum = len(found_issues)

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
                issue_utilities={
                    _["key"]: _["fun_final"] for _ in real_issues.values()
                },
                issues=ordered_issues,
            )
            if u is None:
                u = u_real
            else:
                u = ComplexWeightedUtilityFunction(
                    ufuns=[u, u_real],
                    weights=[1.0, 1.0],
                    name=name,
                )

        # add hyper rectangles issues
        if len(rects) > 0:
            uhyper = HyperRectangleUtilityFunction(
                outcome_ranges=rects,
                utilities=rect_utils,
                name=name,
            )
            if u is None:
                u = uhyper
            else:
                u = ComplexWeightedUtilityFunction(
                    ufuns=[u, uhyper],
                    weights=[1.0, 1.0],
                    name=name,
                )
        if not ignore_reserved and u is not None:
            u.reserved_value = reserved_value
        if ignore_discount:
            discount_factor = None
        u.name = name
        return u, discount_factor

    def utility_difference(self, o1: Outcome, o2: Outcome) -> float:
        """
        Compares the two outcomes and returns a measure of the difference
        between their utilities.

        Args:
            o1: First outcome
            o2: Second outcome
        """
        u1, u2 = self(o1), self(o2)
        return u1 - u2

    @abstractmethod
    def eval(self, offer: Outcome) -> float:
        ...

    def __call__(self, offer: Outcome | None) -> float:
        """Calculate the utility_function value for a given outcome.

        Args:
            offer: The offer to be evaluated.


        Remarks:

            - It calls the abstract method `eval` after opationally adjusting the
              outcome type.
            - It is preferred to override eval instead of directly overriding this method
            - You cannot return None from overriden eval() functions but raise an exception (ValueError) if it was
              not possible to calculate the UtilityValue.
            - Return a float from your `eval` implementation.
            - Return the reserved value if the offer was None

        Returns:
            UtilityValue: The utility_function value which may be a distribution. If `None` it means the
                          utility_function value cannot be calculated.
        """
        if offer is None:
            return self.reserved_value
        return self.eval(offer)

    def __getitem__(self, offer: Outcome | None) -> float | None:
        """Overrides [] operator to call the ufun allowing it to act as a mapping"""
        return self(offer)

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
        from negmas.preferences.nonlinear import MappingUtilityFunction

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
        from negmas.preferences.nonlinear import MappingUtilityFunction

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
        """Generates N mapping utility functions

        Args:
            n: number of utility functions to generate
            outcomes: number of outcomes to use
            normalized: if true, the resulting ufuns will be normlized between zero and one.


        """
        from negmas.preferences.nonlinear import MappingUtilityFunction

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

    def sample_outcome_with_utility(
        self,
        rng: Tuple[Optional[float], Optional[float]],
        issues: List[Issue] = None,
        outcomes: List[Outcome] = None,
        n_trials: int = 100,
    ) -> Optional[Outcome]:
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
            if issues is None:
                issues = self.issues
            outcomes = sample_issues(
                issues=issues,
                n_outcomes=n_trials,
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
            outcomes = sample_issues(
                issues,
                n_outcomes=max_n_outcomes,
                with_replacement=True,
                fail_if_not_enough=False,
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
        outcomes: Optional[List[Outcome]] = None,
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
        if self._issues and num_outcomes(issues) < max_cache_size:
            outcomes = enumerate_issues(issues)
        if not outcomes:
            outcomes = discretize_and_enumerate_issues(
                issues, n_discretization=2, max_n_outcomes=max_cache_size
            )
            n = max_cache_size - len(outcomes)
            if n > 0:
                outcomes += sample_issues(
                    issues,
                    n,
                    with_replacement=False,
                    fail_if_not_enough=False,
                )
        utils = self.eval_all(outcomes)
        self._ordered_outcomes = sorted(zip(utils, outcomes), key=lambda x: -x[0])
        self.inverse_initialized = True

    def inverse(
        self,
        u: float,
        eps: Union[float, Tuple[float, float]] = (1e-3, 0.2),
        assume_normalized=True,
        issues: Optional[List["Issue"]] = None,
        outcomes: Optional[List[Outcome]] = None,
        n_trials: int = 10000,
        return_all_in_range=False,
        max_n_outcomes=10000,
    ) -> Union[Optional[Outcome], Tuple[List[Outcome], List[UtilityValue]]]:
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
        if (not self.inverse_initialized) or issues or outcomes:
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
        return self.inverse_initialized

    def uninialize_inverse(self):
        self.inverse_initialized = False
