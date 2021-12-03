from __future__ import annotations

import itertools
import math
import random
import xml.etree.ElementTree as ET
from functools import reduce
from operator import mul
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

from negmas.generics import ienumerate, ivalues
from negmas.helpers import PathLike, ikeys
from negmas.outcomes import Issue, Outcome, outcome_is_valid
from negmas.outcomes.outcome_space import CartesianOutcomeSpace

from .preferences import Preferences
from .protocols import (
    HasRange,
    HasReservedValue,
    InverseUFun,
    MultiInverseUFun,
    PartiallyNormalizable,
    PartiallyScalable,
    Randomizable,
    UFun,
    XmlSerializableUFun,
)

__all__ = ["UtilityFunction"]


class UtilityFunction(
    XmlSerializableUFun,
    Randomizable,
    PartiallyNormalizable,
    PartiallyScalable,
    HasRange,
    HasReservedValue,
    UFun,
    Preferences,
):
    def __init__(self, *args, reserved_value: float = float("-inf"), **kwargs):
        super().__init__(*args, **kwargs)
        self.reserved_value = reserved_value

    def scale_min_for(
        self,
        to: float,
        outcome_space: CartesianOutcomeSpace | None,
        outcomes: list[Outcome] | None,
    ) -> "UtilityFunction":
        return self.normalize_for((to, float("inf")), outcome_space, outcomes)

    def scale_max_for(
        self,
        to: float,
        outcome_space: CartesianOutcomeSpace | None,
        outcomes: list[Outcome] | None,
    ) -> "UtilityFunction":
        return self.normalize_for((float("-inf"), to), outcome_space, outcomes)

    def normalize_for(
        self,
        to: Tuple[float, float] = (0.0, 1.0),
        outcome_space: CartesianOutcomeSpace | None = None,
        outcomes: list[Outcome] | None = None,
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
            max_cardinality: Maximum ufun evaluations to conduct
        """
        infeasible_cutoff: float = float("-inf")
        epsilon: float = 1e-6
        max_cardinality: int = 1000
        if not outcome_space:
            outcome_space = self.outcome_space
        if to[0] is None and to[1] is None:
            return self
        max_only = to[0] == float("-inf")
        min_only = to[1] == float("inf")
        # todo check that I normalize the reserved value as well
        from negmas.preferences.complex import (
            ComplexNonlinearUtilityFunction,
            ComplexWeightedUtilityFunction,
        )
        from negmas.preferences.const import ConstUFun

        mn, mx = self.utility_range(
            outcome_space, outcomes, infeasible_cutoff, max_cardinality
        )

        if min_only:
            to = (to[0], mx)
        if max_only:
            to = (mn, to[1])
        if abs(mx - to[1]) < epsilon and abs(mn - to[0]) < epsilon:
            return self
        if mx == mn:
            if -epsilon <= mn <= 1 + epsilon:
                return self
            else:
                if self.reserved_value is None:
                    r = None
                else:
                    r = self.reserved_value / mn if mn != 0.0 else self.reserved_value
                if math.isfinite(infeasible_cutoff):
                    return ComplexNonlinearUtilityFunction(
                        ufuns=[self],
                        combination_function=lambda x: infeasible_cutoff
                        if x[0] is None
                        else x[0]
                        if x[0] < infeasible_cutoff
                        else 0.5 * x[0] / mn,
                        name=self.name,
                    )
                else:
                    return ComplexWeightedUtilityFunction(
                        ufuns=[self],
                        weights=[0.5 / mn],
                        name=self.name,
                        reserved_value=r,
                    )
        if max_only:
            scale = to[1] / mx
        elif min_only:
            scale = to[0] / mn
        else:
            scale = (to[1] - to[0]) / (mx - mn)
        r = scale * (self.reserved_value - mn)
        # if abs(mn - rng[0] / scale) < epsilon:
        #     return self
        if math.isfinite(infeasible_cutoff):
            return ComplexNonlinearUtilityFunction(
                ufuns=[self],
                combination_function=lambda x: infeasible_cutoff
                if x[0] is None
                else x[0]
                if x[0] < infeasible_cutoff
                else scale * (x[0] - mn) + to[0],
                name=self.name,
            )
        return ComplexWeightedUtilityFunction(
            ufuns=[self, ConstUFun(-mn + to[0] / scale)],
            weights=[scale, scale],
            name=self.name,
            reserved_value=r,
        )

    @classmethod
    def from_genius(
        cls, file_name: PathLike, **kwargs
    ) -> Tuple["UtilityFunction" | None, float | None]:
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

    def to_genius(self, file_name: PathLike, issues: List[Issue] = None, **kwargs):
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
            >>> u = UtilityFunction.from_genius(file_name=pkg_resources.resource_filename('negmas'
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
            f.write(self.to_xml_str(outcome_space=outcome_space, **kwargs))

    def to_xml_str(self, issues: List[Issue] = None, discount_factor=None) -> str:
        """Exports a utility function to a well formatted string"""
        if not hasattr(self, "xml"):
            raise ValueError(
                "ufun has no xml() member and cannot be saved to XML string"
            )
        if issues is None:
            issues = self.outcome_space

        if issues is not None:
            n_issues = len(issues)
        else:
            n_issues = 0
        output = (
            f'<utility_space type="any" number_of_issues="{n_issues}">\n'
            f'<objective index="1" etype="objective" type="objective" description="" name="any">\n'
        )

        output += self.xml(issues=issues)  # type: ignore
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
    ) -> Tuple["UtilityFunction" | None, float | None]:
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
        from negmas.preferences.mapping import MappingUtilityFunction
        from negmas.preferences.nonlinear import HyperRectangleUtilityFunction

        # keep_issue_names = True
        keep_value_names = True
        force_single_issue = False
        normalize_utility = False
        normalize_max_only = False
        max_cardinality: int = 1_000_000

        root = ET.fromstring(xml_str)
        if safe_parsing and root.tag != "utility_space":
            raise ValueError(f"Root tag is {root.tag}: Expected utility_space")

        ordered_issues: list[str] = []
        issue_order_passed = False
        domain_issues: Optional[Dict[str, Issue]] = None
        if issues is not None:
            if isinstance(issues, list):
                issue_order_passed = True
                ordered_issues = [_.name for _ in issues]
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
                        # key = issue_keys[ii]
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
                        lower, upper = int(lower), int(upper)  # type: ignore
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
                        lower, upper = float(lower), float(upper)  # type: ignore
                        if (
                            domain_issues is not None
                            and not domain_issues[myname].is_continuous()
                        ):
                            n_steps = domain_issues[myname].cardinality
                            delta = (n_steps - 1) / (upper - lower)
                            value_shift = -lower * delta
                            value_scale = delta
                            lower, upper = 0, n_steps - 1
                            found_issues[issue_key] = {}
                            for i in range(lower, upper + 1):  # type: ignore
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
                            item_name: str = item.attrib.get("value", None)  # type: ignore
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
                                domain_all = list(domain_issues[myname].all)  # type: ignore
                                if len(domain_all) > 0 and isinstance(
                                    domain_all[0], int
                                ):
                                    item_key = int(item_key)
                                if len(domain_all) > 0 and isinstance(
                                    domain_all[0], int
                                ):
                                    item_name: int = int(item_name)  # type: ignore
                                if item_name not in domain_all:
                                    raise ValueError(
                                        f"Value {item_name} is not in the domain issue values: "
                                        f"{domain_issues[myname].values}"
                                    )
                                if len(domain_all) > 0 and isinstance(
                                    domain_all[0], int
                                ):
                                    item_name: int = str(item_name)  # type: ignore
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
                                fun = lambda x: offset + slope * float(x)  # type: ignore
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
                                    + slope1 * (value_scale * float(x) + value_shift)  # type: ignore
                                    if x < middle
                                    else offset2
                                    + slope2 * (value_scale * float(x) + value_shift)  # type: ignore
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
                if max_cardinality is not None:
                    n_items = [len(_) for _ in ivalues(found_issues)]
                    n_outcomes = reduce(mul, n_items, 1)
                    if n_outcomes > max_cardinality:
                        return None, discount_factor
                if keep_value_names:
                    names = itertools.product(
                        *(
                            [
                                str(item_key).replace("&", "-")
                                for item_key in ikeys(items)
                            ]
                            for items in ivalues(found_issues)
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
                        for items in ivalues(found_issues)
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
                    umax, umin = max(utils), (0.0 if normalize_max_only else min(utils))  # type: ignore
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
                            values=found_issues,
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
                        for issue in ivalues(real_issues)
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
                values={_["key"]: _["fun_final"] for _ in real_issues.values()},
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
            The utility of the given outcome
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
    ) -> Tuple["UtilityFunction", "UtilityFunction"]:
        """Generates a couple of utility functions

        Args:

            n_outcomes (int): number of outcomes to use
            conflict_level: How conflicting are the two ufuns to generate.
                            1.0 means maximum conflict.
            conflict_delta: How variable is the conflict at different outcomes.

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
        from negmas.preferences.mapping import MappingUtilityFunction

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
        from negmas.preferences.mapping import MappingUtilityFunction

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
        from negmas.preferences.mapping import MappingUtilityFunction

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


class InverseUtilityFunction(MultiInverseUFun, InverseUFun):
    def __init__(self, ufun: UtilityFunction, max_cache_size: int = 10_000):
        self._ufun = ufun
        self.max_cache_size = max_cache_size
        if ufun.is_stationary():
            self.init()

    def init(self):
        outcome_space = self._ufun.outcome_space
        if outcome_space is None:
            raise ValueError("Cannot find the outcome space.")
        self._worst, self._best = self._ufun.extreme_outcomes()
        self._min, self._max = self._ufun(self._worst), self._ufun(self._best)
        self._range = self._max - self._min
        self._offset = self._min / self._range if self._range > 1e-5 else self._min
        os = outcome_space.to_discrete(levels=2, max_cardinality=self.max_cache_size)
        if os.cardinality <= self.max_cache_size:
            outcomes = list(os.sample(self.max_cache_size, False, False))
        else:
            outcomes = list(os.enumerate())[: self.max_cache_size]
        utils = [self._ufun(_) for _ in outcomes]
        self._ordered_outcomes = sorted(zip(utils, outcomes), key=lambda x: -x[0])

    def some(
        self,
        rng: float | tuple[float, float],
    ) -> list[Outcome]:
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
                                 to `max_cardinality`
            max_cardinality: Only used if return_all_in_range is given and gives the maximum number of outcomes to return

        Returns:
            - if return_all_in_range:
               - A tuple with all outcomes in the given range (or samples thereof)
               - A tuple of corresponding utility values
            - Otherwise, An outcome with utility between u-eps[0] and u+eps[1] if found

        Remarks:
            - If issues or outcomes are not None, then init_inverse will be called first

        """
        if not self._ufun.is_stationary():
            self.init()
        if not isinstance(rng, Iterable):
            rng = (rng, rng)
        mn, mx = rng
        # todo use bisection
        results = []
        for util, w in self._ordered_outcomes:
            if util > mx:
                continue
            if util < mn:
                break
            results.append(w)
        return results

    def worst_in(self, rng: float | tuple[float, float]) -> Outcome | None:
        if not self._ufun.is_stationary():
            self.init()
        if not isinstance(rng, Iterable):
            rng = (rng, rng)
        mn, mx = rng
        for i, (util, _) in enumerate(self._ordered_outcomes):
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

    def best_in(self, rng: float | tuple[float, float]) -> Outcome | None:
        if not self._ufun.is_stationary():
            self.init()
        if not isinstance(rng, Iterable):
            rng = (rng, rng)
        mn, mx = rng
        for util, w in self._ordered_outcomes:
            if util <= mx:
                if util < mn:
                    return None
                return w
        return None

    def one_in(self, rng: float | tuple[float, float]) -> Outcome | None:
        lst = self.some(rng)
        if not lst:
            return None
        return lst[random.randint(0, len(lst) - 1)]
