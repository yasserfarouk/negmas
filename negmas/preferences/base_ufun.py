from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Sequence, TypeVar

from negmas import warnings
from negmas.common import Value
from negmas.helpers import PathLike
from negmas.helpers.prob import Distribution, Real, ScipyDistribution
from negmas.helpers.types import get_full_type_name
from negmas.outcomes import Issue, Outcome, dict2outcome
from negmas.outcomes.common import check_one_at_most, os_or_none
from negmas.outcomes.issue_ops import issues_from_geniusweb_json
from negmas.outcomes.outcome_space import make_os
from negmas.outcomes.protocols import IndependentIssuesOS, OutcomeSpace
from negmas.preferences.value_fun import TableFun
from negmas.serialization import PYTHON_CLASS_IDENTIFIER, deserialize, serialize
from negmas.warnings import warn_if_slow

from .preferences import Preferences
from .protocols import InverseUFun
from .value_fun import make_fun_from_xml

if TYPE_CHECKING:
    from negmas.preferences import (
        ConstUtilityFunction,
        ProbUtilityFunction,
        UtilityFunction,
        WeightedUtilityFunction,
    )

__all__ = [
    "BaseUtilityFunction",
]


MAX_CARDINALITY = 10_000_000_000
T = TypeVar("T", bound="BaseUtilityFunction")


# PartiallyScalable,
# HasRange,
# HasReservedValue,
# StationaryConvertible,
# OrdinalRanking,
# CardinalRanking,
# BasePref,
class BaseUtilityFunction(Preferences, ABC):
    """
    Base class for all utility functions in negmas
    """

    def __init__(self, *args, reserved_value: float = float("-inf"), **kwargs):
        super().__init__(*args, **kwargs)
        self.reserved_value = reserved_value
        self._cached_inverse: InverseUFun | None = None
        self._cached_inverse_type: type[InverseUFun] | None = None

    @abstractmethod
    def eval(self, offer: Outcome) -> Value:
        ...

    def to_stationary(self: T) -> T:
        raise NotImplementedError(
            f"I do not know how to convert a ufun of type {self.type_name} to a stationary ufun."
        )

    def extreme_outcomes(
        self,
        outcome_space: OutcomeSpace | None = None,
        issues: Iterable[Issue] | None = None,
        outcomes: Iterable[Outcome] | None = None,
        max_cardinality=100_000,
    ) -> tuple[Outcome, Outcome]:
        check_one_at_most(outcome_space, issues, outcomes)
        outcome_space = os_or_none(outcome_space, issues, outcomes)
        if not outcome_space:
            outcome_space = self.outcome_space
        if outcome_space and not outcomes:
            outcomes = outcome_space.enumerate_or_sample(
                max_cardinality=max_cardinality
            )
        if not outcomes:
            raise ValueError("Cannot find outcomes to use for finding extremes")
        mn, mx = float("inf"), float("-inf")
        worst, best = None, None
        warn_if_slow(len(list(outcomes)), f"Extreme Outcomes too Slow")
        for o in outcomes:
            u = self(o)
            if u < mn:
                worst, mn = o, u
            if u > mx:
                best, mx = o, u
        if worst is None or best is None:
            raise ValueError(f"Cound not find worst and best outcomes for {self}")
        return worst, best

    def minmax(
        self,
        outcome_space: OutcomeSpace | None = None,
        issues: Sequence[Issue] | None = None,
        outcomes: Sequence[Outcome] | None = None,
        max_cardinality=1000,
        above_reserve=False,
    ) -> tuple[float, float]:
        """Finds the range of the given utility function for the given outcomes

        Args:
            self: The utility function
            issues: List of issues (optional)
            outcomes: A collection of outcomes (optional)
            max_cardinality: the maximum number of outcomes to try sampling (if sampling is used and outcomes are not given)
            above_reserve: If given, the minimum and maximum will be set to reserved value if they were less than it.

        Returns:
            (lowest, highest) utilities in that order

        """
        (worst, best) = self.extreme_outcomes(
            outcome_space, issues, outcomes, max_cardinality
        )
        w, b = self(worst), self(best)
        if isinstance(w, Distribution):
            w = w.min
        if isinstance(b, Distribution):
            b = b.max
        if above_reserve:
            r = self.reserved_value
            if r is None:
                return w, b
            if b < r:
                b, w = r, r
            elif w < r:
                w = r
        return w, b

    @property
    def reserved_distribution(self) -> Distribution:
        return ScipyDistribution(type="uniform", loc=self.reserved_value, scale=0.0)

    def max(self) -> Value:
        _, mx = self.minmax()
        return mx

    def min(self) -> Value:
        mn, _ = self.minmax()
        return mn

    def best(self) -> Outcome:
        _, mx = self.extreme_outcomes()
        return mx

    def worst(self) -> Outcome:
        mn, _ = self.extreme_outcomes()
        return mn

    def eval_normalized(
        self,
        offer: Outcome | None,
        above_reserve: bool = True,
        expected_limits: bool = True,
    ) -> Value:
        """
        Evaluates the ufun normalizing the result between zero and one

        Args:
            offer (Outcome | None): offer
            above_reserve (bool): If True, zero corresponds to the reserved value not the minimum
            expected_limits (bool): If True, the expectation of the utility limits will be used for normalization instead of the maximum range and minimum lowest limit

        Remarks:
            - If the maximum and the minium are equal, finite and above reserve, will return 1.0.
            - If the maximum and the minium are equal, initinte or below reserve, will return 0.0.
            - For probabilistic ufuns, a distribution will still be returned.
            - The minimum and maximum will be evaluated freshly every time. If they are already caached in the ufun, the cache will be used.

        """
        r = self.reserved_value
        u = self.eval(offer) if offer else r
        mn, mx = self.minmax()
        if above_reserve:
            if mx < r:
                mx = mn = float("-inf")
            elif mn < r:
                mn = r
        d = mx - mn
        if isinstance(d, Distribution):
            d = float(d) if expected_limits else d.max
        if isinstance(mn, Distribution):
            mn = float(mn) if expected_limits else mn.min
        if d < 1e-5:
            warnings.warn(
                f"Ufun has equal max and min. The outcome will be normalized to zero if they were finite otherwise 1.0: {mn=}, {mx=}, {r=}, {u=}"
            )
            return 1.0 if math.isfinite(mx) else 0.0
        d = 1 / d
        return (u - mn) * d

    def invert(self, inverter: type[InverseUFun] | None = None) -> InverseUFun:
        """
        Inverts the ufun, initializes it and caches the result.
        """
        from .inv_ufun import PresortingInverseUtilityFunction

        if self._cached_inverse and (
            inverter is None or self._cached_inverse_type == inverter
        ):
            return self._cached_inverse
        if inverter is None:
            inverter = PresortingInverseUtilityFunction
        self._cached_inverse_type = inverter
        self._cached_inverse = inverter(self)
        self._cached_inverse.init()
        return self._cached_inverse

    def is_volatile(self) -> bool:
        return True

    def is_session_dependent(self) -> bool:
        return True

    def is_state_dependent(self) -> bool:
        return True

    def scale_by(
        self: T, scale: float, scale_reserved=True
    ) -> WeightedUtilityFunction | T:
        if scale < 0:
            raise ValueError(f"Cannot scale with a negative multiplier ({scale})")
        from negmas.preferences.complex import WeightedUtilityFunction

        r = (scale * self.reserved_value) if scale_reserved else self.reserved_value
        return WeightedUtilityFunction(
            ufuns=[self],
            weights=[scale],
            name=self.name,
            reserved_value=r,
        )

    def scale_min_for(
        self: T,
        to: float,
        outcome_space: OutcomeSpace | None = None,
        issues: Sequence[Issue] | None = None,
        outcomes: Sequence[Outcome] | None = None,
        rng: tuple[float, float] | None = None,
    ) -> T:
        if rng is None:
            mn, _ = self.minmax(outcome_space, issues, outcomes)
        else:
            mn, _ = rng
        scale = to / mn
        return self.scale_by(scale)

    def scale_min(self: T, to: float, rng: tuple[float, float] | None = None) -> T:
        return self.scale_min_for(to, outcome_space=self.outcome_space, rng=rng)

    def scale_max_for(
        self: T,
        to: float,
        outcome_space: OutcomeSpace | None = None,
        issues: Sequence[Issue] | None = None,
        outcomes: Sequence[Outcome] | None = None,
        rng: tuple[float, float] | None = None,
    ) -> T:
        if rng is None:
            _, mx = self.minmax(outcome_space, issues, outcomes)
        else:
            _, mx = rng
        scale = to / mx
        return self.scale_by(scale)

    def scale_max(self: T, to: float, rng: tuple[float, float] | None = None) -> T:
        return self.scale_max_for(to, outcome_space=self.outcome_space, rng=rng)

    def normalize_for(
        self: T,
        to: tuple[float, float] = (0.0, 1.0),
        outcome_space: OutcomeSpace | None = None,
    ) -> T | ConstUtilityFunction:
        max_cardinality: int = MAX_CARDINALITY
        if not outcome_space:
            outcome_space = self.outcome_space
        if not outcome_space:
            raise ValueError(
                "Cannot find the outcome-space to normalize for. "
                "You must pass outcome_space, issues or outcomes or have the ufun being constructed with one of them"
            )
        mn, mx = self.minmax(outcome_space, max_cardinality=max_cardinality)

        d = float(mx - mn)
        if d < 1e-7:
            from negmas.preferences.crisp.const import ConstUtilityFunction

            return ConstUtilityFunction(
                0.0 if mx < self.reserved_value else 1.0,
                outcome_space=self.outcome_space,
                name=self.name,
                reserved_value=1.0 if mn < self.reserved_value else 0.0,
            )

        scale = float(to[1] - to[0]) / d

        u = self.scale_by(scale, scale_reserved=True)
        return u.shift_by(to[0] - scale * mn, shift_reserved=True)

    def normalize(
        self: T,
        to: tuple[float, float] = (0.0, 1.0),
        normalize_weights: bool = False,
    ) -> T | ConstUtilityFunction:
        _ = normalize_weights
        from negmas.preferences import ConstUtilityFunction

        if not self.outcome_space:
            raise ValueError(f"Cannot normalize a ufun without an outcome-space")
        mn, mx = self.minmax(self.outcome_space, max_cardinality=MAX_CARDINALITY)

        d = float(mx - mn)
        if d < 1e-8:
            return ConstUtilityFunction(
                0.0 if mx < self.reserved_value else 1.0,
                name=self.name,
                reserved_value=1.0 if mn < self.reserved_value else 0.0,
            )

        scale = float(to[1] - to[0]) / d

        # u = self.shift_by(-mn, shift_reserved=True)
        u = self.scale_by(scale, scale_reserved=True)
        return u.shift_by(to[0] - scale * mn, shift_reserved=True)

    def shift_by(
        self: T, offset: float, shift_reserved=True
    ) -> WeightedUtilityFunction | T:
        from negmas.preferences.complex import WeightedUtilityFunction
        from negmas.preferences.crisp.const import ConstUtilityFunction

        r = (self.reserved_value + offset) if shift_reserved else self.reserved_value
        return WeightedUtilityFunction(
            ufuns=[self, ConstUtilityFunction(offset)],
            weights=[1, 1],
            name=self.name,
            reserved_value=r,
        )

    def shift_min_for(
        self: T,
        to: float,
        outcome_space: OutcomeSpace | None = None,
        issues: Sequence[Issue] | None = None,
        outcomes: Sequence[Outcome] | None = None,
        rng: tuple[float, float] | None = None,
    ) -> T:
        if rng is None:
            mn, _ = self.minmax(outcome_space, issues, outcomes)
        else:
            mn, _ = rng
        offset = to - mn
        return self.shift_by(offset)

    def shift_max_for(
        self: T,
        to: float,
        outcome_space: OutcomeSpace | None = None,
        issues: Sequence[Issue] | None = None,
        outcomes: Sequence[Outcome] | None = None,
        rng: tuple[float, float] | None = None,
    ) -> T:
        if rng is None:
            _, mx = self.minmax(outcome_space, issues, outcomes)
        else:
            _, mx = rng
        offset = to - mx
        return self.shift_by(offset)

    def _do_rank(self, vals, descending):
        vals = sorted(vals, key=lambda x: x[1], reverse=descending)
        if not vals:
            return []
        ranks = [([vals[0][0]], vals[0][1])]
        for w, v in vals[1:]:
            if v == ranks[-1][1]:
                ranks[-1][0].append(w)
                continue
            ranks.append(([w], v))
        return ranks

    def argrank_with_weights(
        self, outcomes: Sequence[Outcome | None], descending=True
    ) -> list[tuple[list[Outcome | None], float]]:
        """
        Ranks the given list of outcomes with weights. None stands for the null outcome.

        Returns:

            - A list of tuples each with two values:
                - an list of integers giving the index in the input array (outcomes) of an outcome (at the given utility level)
                - the weight of that outcome
            - The list is sorted by weights descendingly

        """
        vals = zip(range(len(list(outcomes))), (self(_) for _ in outcomes))
        return self._do_rank(vals, descending)

    def argrank(
        self, outcomes: Sequence[Outcome | None], descending=True
    ) -> list[list[Outcome | None]]:
        """
        Ranks the given list of outcomes with weights. None stands for the null outcome.

        Returns:
            A list of lists of integers giving the outcome index in the input. The list is sorted by utlity value

        """
        ranks = self.argrank_with_weights(outcomes, descending)
        return [_[0] for _ in ranks]

    def rank_with_weights(
        self, outcomes: Sequence[Outcome | None], descending=True
    ) -> list[tuple[list[Outcome | None], float]]:
        """
        Ranks the given list of outcomes with weights. None stands for the null outcome.

        Returns:

            - A list of tuples each with two values:
                - an list of integers giving the index in the input array (outcomes) of an outcome (at the given utility level)
                - the weight of that outcome
            - The list is sorted by weights descendingly

        """
        vals = zip(outcomes, (self(_) for _ in outcomes))
        return self._do_rank(vals, descending)

    def rank(
        self, outcomes: Sequence[Outcome | None], descending=True
    ) -> list[list[Outcome | None]]:
        """
        Ranks the given list of outcomes with weights. None stands for the null outcome.

        Returns:
            A list of lists of integers giving the outcome index in the input. The list is sorted by utlity value

        """
        ranks = self.rank_with_weights(outcomes, descending)
        return [_[0] for _ in ranks]

    def eu(self, offer: Outcome | None) -> float:
        """
        calculates the **expected** utility value of the input outcome
        """
        return float(self(offer))

    def to_crisp(self) -> UtilityFunction:
        from negmas.preferences.crisp_ufun import CrispAdapter

        return CrispAdapter(self)

    def to_prob(self) -> ProbUtilityFunction:
        from negmas.preferences.prob_ufun import ProbAdapter

        return ProbAdapter(self)

    def to_dict(self) -> dict[str, Any]:
        d = {PYTHON_CLASS_IDENTIFIER: get_full_type_name(type(self))}
        return dict(
            **d,
            outcome_space=serialize(self.outcome_space),
            reserved_value=self.reserved_value,
            name=self.name,
            id=self.id,
        )

    @classmethod
    def from_dict(cls, d):
        d.pop(PYTHON_CLASS_IDENTIFIER, None)
        d["outcome_space"] = deserialize(d.get("outcome_space", None))
        return cls(**d)

    def sample_outcome_with_utility(
        self,
        rng: tuple[float, float],
        outcome_space: OutcomeSpace | None = None,
        issues: Sequence[Issue] | None = None,
        outcomes: Sequence[Outcome] | None = None,
        n_trials: int = 100,
    ) -> Outcome | None:
        """
        Samples an outcome in the given utiltity range or return None if not possible

        Args:
            rng (Tuple[float, float]): rng
            outcome_space (OutcomeSpace | None): outcome_space
            issues (Sequence[Issue] | None): issues
            outcomes (Sequence[Outcome] | None): outcomes
            n_trials (int): n_trials

        Returns:
            Optional["Outcome"]:
        """
        if rng[0] is None:
            rng = (float("-inf"), rng[1])
        if rng[1] is None:
            rng = (rng[0], float("inf"))
        outcome_space = os_or_none(outcome_space, issues, outcomes)
        if not outcome_space:
            outcome_space = self.outcome_space
        if not outcome_space:
            raise ValueError("No outcome-space is given or defined for the ufun")
        if outcome_space.cardinality < n_trials:
            n_trials = outcome_space.cardinality  # type: ignore I know that it is an int (see the if)
        for o in outcome_space.sample(n_trials, with_replacement=False):
            if o is None:
                continue
            assert (
                o in outcome_space
            ), f"Sampled outcome {o} which is not in the outcome-space {outcome_space}"
            if rng[0] - 1e-6 <= float(self(o)) <= rng[1] + 1e-6:
                return o
        return None

    @classmethod
    def from_xml_str(
        cls,
        xml_str: str,
        issues: Iterable[Issue] | Sequence[Issue],
        safe_parsing=True,
        ignore_discount=False,
        ignore_reserved=False,
        name: str | None = None,
    ) -> tuple[BaseUtilityFunction | None, float | None]:
        """Imports a utility function from a GENIUS XML string.

        Args:

            xml_str (str): The string containing GENIUS style XML utility function definition
            issues (Sequence[Issue] | None): Optional issue space to confirm that the utility function is valid
            product of all issues in the input
            safe_parsing (bool): Turn on extra checks

        Returns:

            A utility function object (depending on the input file)


        Examples:

            >>> from negmas.preferences import UtilityFunction
            >>> import pkg_resources
            >>> from negmas.inout import load_genius_domain
            >>> domain = load_genius_domain(pkg_resources.resource_filename('negmas'
            ...                             , resource_name='tests/data/Laptop/Laptop-C-domain.xml'))
            >>> with open(pkg_resources.resource_filename('negmas'
            ...                                      , resource_name='tests/data/Laptop/Laptop-C-prof1.xml')
            ...                                      , 'r') as ff:
            ...     u, _ = UtilityFunction.from_xml_str(ff.read(), issues=domain.issues)
            >>> with open(pkg_resources.resource_filename('negmas'
            ...                                      , resource_name='tests/data/Laptop/Laptop-C-prof1.xml')
            ...                                      , 'r') as ff:
            ...     u, _ = UtilityFunction.from_xml_str(ff.read(), issues=domain.issues)
            >>> assert abs(u(("Dell", "60 Gb", "19'' LCD",)) - 21.987727736172488) < 0.000001
            >>> assert abs(u(("HP", "80 Gb", "20'' LCD",)) - 22.68559475583014) < 0.000001


        """
        from negmas.preferences.complex import WeightedUtilityFunction
        from negmas.preferences.crisp.linear import (
            AffineUtilityFunction,
            LinearAdditiveUtilityFunction,
        )
        from negmas.preferences.crisp.nonlinear import HyperRectangleUtilityFunction

        root = ET.fromstring(xml_str)
        if safe_parsing and root.tag != "utility_space":
            raise ValueError(f"Root tag is {root.tag}: Expected utility_space")

        issues = list(issues)
        ordered_issues: list[Issue] = []
        domain_issues_dict: dict[str, Issue] | None = None
        ordered_issues = issues
        domain_issues_dict = dict(zip([_.name for _ in issues], issues))
        # issue_indices = dict(zip([_.name for _ in issues], range(len(issues))))
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
            objective = root
        weights = {}
        found_issues = {}
        issue_info = {}
        issue_keys = {}
        rects, rect_utils = [], []
        all_numeric = True
        global_bias = 0

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
            return rects, rect_utils

        for child in objective:
            if child.tag == "weight":
                indx = int(child.attrib["index"]) - 1
                if indx < 0 or indx >= len(issues):
                    global_bias += float(child.attrib["value"])
                    continue
                weights[issues[indx].name] = float(child.attrib["value"])
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
                issue_key = child.attrib["name"]
                if (
                    domain_issues_dict is not None
                    and issue_key not in domain_issues_dict.keys()
                ):
                    raise ValueError(
                        f"Issue {issue_key} is not in the input issue names ({domain_issues_dict.keys()})"
                    )
                issue_info[issue_key] = {"name": issue_key, "index": indx}
                issue_keys[indx] = issue_key
                info = {"type": "discrete", "etype": "discrete", "vtype": "discrete"}
                for a in ("type", "etype", "vtype"):
                    info[a] = child.attrib.get(a, info[a])
                issue_info[issue_key].update(info)
                mytype = info["type"]
                # vtype = info["vtype"]
                if domain_issues_dict is None:
                    raise ValueError(f"unknown domain-issue-dict!!!")

                current_issue = domain_issues_dict[issue_key]

                if mytype == "discrete":
                    found_issues[issue_key] = dict()
                    if current_issue.is_continuous():
                        raise ValueError(
                            f"Got a {mytype} issue but expected a continuous valued issue"
                        )
                elif mytype in ("integer", "real"):
                    lower = current_issue.min_value
                    upper = current_issue.max_value
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
                        if current_issue.is_continuous():
                            raise ValueError(
                                f"Got a {mytype} issue but expected a continuous valued issue"
                            )
                        lower, upper = int(lower), int(upper)  # type: ignore
                    else:
                        lower, upper = float(lower), float(upper)  # type: ignore
                    if lower < current_issue.min_value or upper > current_issue.max_value:  # type: ignore
                        raise ValueError(
                            f"Bounds ({lower}, {upper}) are invalid for issue {issue_key} with bounds: "
                            f"{current_issue.values}"
                        )
                else:
                    raise ValueError(f"Unknown type: {mytype}")
                # now we found ranges for range issues and will find values for all issues
                found_values = False
                for item in child:
                    if item.tag == "item":
                        if mytype != "discrete":
                            raise ValueError(
                                f"cannot specify item utilities for not-discrete type: {mytype}"
                            )
                        all_numeric = False
                        item_indx = int(item.attrib["index"]) - 1
                        item_name = item.attrib.get("value", None)
                        if item_name is None:
                            warnings.warn(
                                f"An item without a value at index {item_indx} for issue {issue_key}",
                                warnings.NegmasIOWarning,
                            )
                            continue
                        # may be I do not need this
                        if current_issue.is_integer():
                            item_name = int(item_name)
                        if current_issue.is_float():
                            item_name = float(item_name)
                        if not current_issue.is_valid(item_name):
                            raise ValueError(
                                f"Value {item_name} is not in the domain issue values: "
                                f"{current_issue.values}"
                            )
                        val = item.attrib.get("evaluation", None)
                        if val is None:
                            raise ValueError(
                                f"Item {item_name} of issue {issue_key} has no evaluation attribute!!"
                            )
                        float(val)
                        found_issues[issue_key][item_name] = float(val)
                        found_values = True
                        issue_info[issue_key]["map_type"] = "dict"
                    elif item.tag == "evaluator":
                        _f, _name = make_fun_from_xml(item)
                        found_issues[issue_key] = _f
                        issue_info[issue_key]["map_type"] = _name
                        found_values = True
                if not found_values and issue_key in found_issues.keys():
                    found_issues.pop(issue_key, None)

        # add utilities specified not as hyper-rectangles
        if not all_numeric and all(_.is_numeric() for _ in issues):
            raise ValueError(
                "Some found issues are not numeric but all input issues are"
            )
        u = None
        if len(found_issues) > 0:
            if all_numeric:
                slopes, biases, ws = [], [], []
                for key in (_.name for _ in issues):
                    if key in found_issues:
                        slopes.append(found_issues[key].slope)
                        biases.append(found_issues[key].bias)
                    else:
                        slopes.append(0.0)
                        biases.append(0.0)
                    ws.append(weights.get(key, 1.0))
                bias = 0.0
                for b, w in zip(biases, ws):
                    bias += b * w
                for i, s in enumerate(slopes):
                    ws[i] *= s

                u = AffineUtilityFunction(
                    weights=ws,
                    outcome_space=make_os(ordered_issues),
                    bias=bias + global_bias,
                )
            else:
                u = LinearAdditiveUtilityFunction(
                    values=found_issues,
                    weights=weights,
                    outcome_space=make_os(ordered_issues),
                    bias=global_bias,
                )

        if len(rects) > 0:
            uhyper = HyperRectangleUtilityFunction(
                outcome_ranges=rects,
                utilities=rect_utils,
                name=name,
                outcome_space=make_os(ordered_issues),
                bias=global_bias,
            )
            if u is None:
                u = uhyper
            else:
                u = WeightedUtilityFunction(
                    ufuns=[u, uhyper],
                    weights=[1.0, 1.0],
                    name=name,
                    outcome_space=make_os(ordered_issues),
                )
        if u is None:
            raise ValueError("No issues found")
        if not ignore_reserved:
            u.reserved_value = reserved_value
        u.name = name
        # if not ignore_discount and discount_factor != 0.0:
        #     from negmas.preferences.discounted import ExpDiscountedUFun
        #     u = ExpDiscountedUFun(ufun=u, discount=discount_factor, name=name)
        if ignore_discount:
            discount_factor = None
        return u, discount_factor

    @classmethod
    def from_genius(
        cls, file_name: PathLike | str, **kwargs
    ) -> tuple[BaseUtilityFunction | None, float | None]:
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
            'LinearAdditiveUtilityFunction'
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

    @classmethod
    def from_geniusweb_json_str(
        cls,
        json_str: str | dict,
        safe_parsing=True,
        issues: Iterable[Issue] | Sequence[Issue] | None = None,
        ignore_discount=False,
        ignore_reserved=False,
        use_reserved_outcome=False,
        name: str | None = None,
    ) -> tuple[BaseUtilityFunction | None, float | None]:
        """Imports a utility function from a GeniusWeb JSON string.

        Args:

            json_str (str): The string containing GENIUS style XML utility function definition
            issues (Sequence[Issue] | None): Optional issue space to confirm that the utility function is valid
            product of all issues in the input
            safe_parsing (bool): Turn on extra checks

        Returns:

            A utility function object (depending on the input file)

        """
        from negmas.preferences.crisp.linear import LinearAdditiveUtilityFunction

        _ = safe_parsing

        if isinstance(json_str, str):
            d = json.loads(json_str)
        else:
            d = json_str
        reserved_outcome, discount_factor, u = None, 1.0, None
        if "LinearAdditiveUtilitySpace" in d.keys():
            udict = d["LinearAdditiveUtilitySpace"]
            domain = (
                make_os(issues_from_geniusweb_json(udict["domain"])[0])
                if "domain" in udict.keys()
                else None
            )
            if domain is None and issues is not None:
                domain = make_os(tuple(issues))
            discount_factor = (
                udict.get("discount_factor", 1.0) if not ignore_discount else 1.0
            )
            reserved_value = udict.get("reserved_value", None)
            uname = udict.get("name", name)
            reserved_dict = udict.get("reservationBid", dict()).get("issuevalues", None)
            if reserved_dict and not ignore_reserved:
                reserved_outcome = dict2outcome(reserved_dict, issues=domain.issues)  # type: ignore
            weights = udict.get("issueWeights", None)
            utils = udict.get("issueUtilities", dict())
            values = dict()
            for iname, idict in utils.items():
                vals = idict.get("discreteutils", dict()).get("valueUtilities", dict())
                values[iname] = TableFun(vals)
            u = LinearAdditiveUtilityFunction(
                values=values,
                weights=weights,
                bias=0.0,
                name=uname,
                reserved_outcome=None,
                reserved_value=None,
                outcome_space=domain,
            )
            if not ignore_reserved:
                if reserved_outcome and not use_reserved_outcome:
                    reserved_value = u(reserved_outcome)
                if use_reserved_outcome:
                    u.reserved_outcome = reserved_outcome  # type: ignore
                u.reserved_value = reserved_value  # type: ignore

        return u, discount_factor

    @classmethod
    def from_geniusweb(
        cls, file_name: PathLike | str, **kwargs
    ) -> tuple[BaseUtilityFunction | None, float | None]:
        """Imports a utility function from a GeniusWeb json file.

        Args:

            file_name (str): File name to import from

        Returns:

            A utility function object (depending on the input file)

        Remarks:
            See ``from_geniusweb_json_str`` for all the parameters

        """
        kwargs["name"] = str(file_name)
        with open(file_name) as f:
            xml_str = f.read()
        return cls.from_geniusweb_json_str(json_str=xml_str, **kwargs)

    def to_xml_str(
        self, issues: Iterable[Issue] | None = None, discount_factor=None
    ) -> str:
        """
        Exports a utility function to a well formatted string
        """
        if not hasattr(self, "xml"):
            raise ValueError(
                f"ufun of type {self.__class__.__name__} has no xml() member and cannot be saved to XML string\nThe ufun params: {self.to_dict()}"
            )
        if issues is None:
            if not isinstance(self.outcome_space, IndependentIssuesOS):
                raise ValueError(
                    f"Cannot convert to xml because the outcome-space of the ufun is not a cartesian outcome space"
                )
            issues = self.outcome_space.issues
            n_issues = 0
        else:
            issues = list(issues)
            n_issues = len(issues)
        output = (
            f'<utility_space type="any" number_of_issues="{n_issues}">\n'
            f'<objective index="1" etype="objective" type="objective" description="" name="any">\n'
        )

        output += self.xml(issues=issues)  # type: ignore
        if "</objective>" not in output:
            output += "</objective>\n"
            if discount_factor is not None:
                output += f'<discount_factor value="{discount_factor}" />\n'
        if (
            self.reserved_value is not None
            and self.reserved_value != float("-inf")
            and "<reservation value" not in output
        ):
            output += f'<reservation value="{self.reserved_value}" />\n'
        if "</utility_space>" not in output:
            output += "</utility_space>\n"
        return output

    def to_genius(
        self, file_name: PathLike | str, issues: Iterable[Issue] | None = None, **kwargs
    ):
        """
        Exports a utility function to a GENIUS XML file.

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
            >>> u, d = UtilityFunction.from_genius(file_name=pkg_resources.resource_filename('negmas'
            ...                                             , resource_name='tests/data/Laptop/Laptop-C-prof1.xml')
            ...                                             , issues=domain.issues)
            >>> u.to_genius(discount_factor=d
            ...     , file_name = pkg_resources.resource_filename('negmas'
            ...                   , resource_name='tests/data/LaptopConv/Laptop-C-prof1.xml')
            ...     , issues=domain.issues)

        Remarks:
            See ``to_xml_str`` for all the parameters

        """
        file_name = Path(file_name).absolute()
        if file_name.suffix == "":
            file_name = file_name.parent / f"{file_name.stem}.xml"
        with open(file_name, "w") as f:
            f.write(self.to_xml_str(issues=issues, **kwargs))

    def difference_prob(
        self, first: Outcome | None, second: Outcome | None
    ) -> Distribution:
        """
        Returns a numeric difference between the utility of the two given outcomes
        """
        f, s = self(first), self(second)
        if not isinstance(f, Distribution):
            f = Real(f)
        if not isinstance(s, Distribution):
            s = Real(s)
        return f - s

    def is_not_worse(self, first: Outcome | None, second: Outcome | None) -> bool:
        return self.difference_prob(first, second) >= 0.0

    def difference(self, first: Outcome | None, second: Outcome | None) -> float:
        """
        Returns a numeric difference between the utility of the two given outcomes
        """
        return float(self(first)) - float(self(second))

    def __call__(self, offer: Outcome | None) -> Value:
        """
        Calculate the utility for a given outcome at the given negotiation state.

        Args:
            offer: The offer to be evaluated.


        Remarks:

            - It calls the abstract method `eval` after opationally adjusting the
              outcome type.
            - It is preferred to override eval instead of directly overriding this method
            - You cannot return None from overriden eval() functions but raise an exception (ValueError) if it was
              not possible to calculate the Value.
            - Return a float from your `eval` implementation.
            - Return the reserved value if the offer was None

        Returns:
            The utility of the given outcome
        """
        if offer is None:
            return self.reserved_value  # type: ignore I know that concrete subclasses will be returning the correct type
        return self.eval(offer)


class _FullyStatic:
    """
    Used internally to indicate that the ufun can **NEVER** change due to anything.
    """

    def is_session_dependent(self) -> bool:
        return False

    def is_volatile(self) -> bool:
        return False

    def is_state_dependent(self) -> bool:
        return False

    def is_stationary(self) -> bool:
        return True


class _ExtremelyDynamic:
    """
    Used internally to indicate that the ufun can change due to anything.
    """

    def is_session_dependent(self) -> bool:
        return True

    def is_volatile(self) -> bool:
        return True

    def is_state_dependent(self) -> bool:
        return True

    def is_stationary(self) -> bool:
        return False
