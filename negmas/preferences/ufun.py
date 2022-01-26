from __future__ import annotations

import random
import warnings
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

from negmas.common import Value
from negmas.helpers import PathLike
from negmas.helpers.numeric import get_one_float
from negmas.helpers.prob import Distribution, ScipyDistribution
from negmas.helpers.types import get_full_type_name
from negmas.outcomes import Issue, Outcome
from negmas.outcomes.common import check_one_at_most, os_or_none
from negmas.outcomes.outcome_space import make_os
from negmas.outcomes.protocols import IndependentIssuesOS, OutcomeSpace
from negmas.serialization import PYTHON_CLASS_IDENTIFIER, deserialize, serialize

from .preferences import Preferences
from .protocols import (
    BasePref,
    CardinalRanking,
    HasRange,
    HasReservedValue,
    InverseUFun,
    MultiInverseUFun,
    OrdinalRanking,
    PartiallyNormalizable,
    PartiallyScalable,
    StationaryConvertible,
    UFunCrisp,
    UFunProb,
)
from .value_fun import MAX_CARINALITY, make_fun_from_xml

if TYPE_CHECKING:
    from negmas.preferences import ComplexWeightedUtilityFunction

__all__ = [
    "BaseUtilityFunction",
    "UtilityFunction",
    "ProbUtilityFunction",
    "StationaryUtilityFunction",
    "PresortingInverseUtilityFunction",
    "SamplingInverseUtilityFunction",
]

import functools


def ignore_unhashable(func):
    uncached = func.__wrapped__
    attributes = functools.WRAPPER_ASSIGNMENTS + ("cache_info", "cache_clear")  # type: ignore (not my code and I do not know what is happening here)

    @functools.wraps(func, assigned=attributes)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except TypeError as error:
            if "unhashable type" in str(error):
                return uncached(*args, **kwargs)
            raise

    wrapper.__uncached__ = uncached
    return wrapper


class BaseUtilityFunction(
    Preferences,
    PartiallyNormalizable,
    PartiallyScalable,
    HasRange,
    HasReservedValue,
    StationaryConvertible,
    OrdinalRanking,
    CardinalRanking,
    BasePref,
    ABC,
):
    def __init__(self, *args, reserved_value: float = float("-inf"), **kwargs):
        super().__init__(*args, **kwargs)
        self.reserved_value = reserved_value

    def is_volatile(self) -> bool:
        return True

    def is_stationary(self) -> bool:
        return False

    def is_state_dependent(self) -> bool:
        return True

    def scale_by(
        self, scale: float, scale_reserved=True
    ) -> "ComplexWeightedUtilityFunction":
        if scale < 0:
            raise ValueError(f"Cannot scale with a negative multiplier ({scale})")
        from negmas.preferences.complex import ComplexWeightedUtilityFunction

        r = scale * self.reserved_value if scale_reserved else self.reserved_value
        return ComplexWeightedUtilityFunction(
            ufuns=[self],
            weights=[scale],
            name=self.name,
            reserved_value=r,
        )

    def shift_by(
        self, offset: float, shift_reserved=True
    ) -> "ComplexWeightedUtilityFunction":
        from negmas.preferences.complex import ComplexWeightedUtilityFunction
        from negmas.preferences.const import ConstUtilityFunction

        r = self.reserved_value + offset if shift_reserved else self.reserved_value
        return ComplexWeightedUtilityFunction(
            ufuns=[self, ConstUtilityFunction(offset)],
            weights=[1, 1],
            name=self.name,
            reserved_value=r,
        )

    def shift_min_for(
        self,
        to: float,
        outcome_space: OutcomeSpace | None = None,
        issues: list[Issue] | None = None,
        outcomes: list[Outcome] | None = None,
        rng: tuple[float, float] | None = None,
    ) -> "UtilityFunction":
        if rng is None:
            mn, _ = self.minmax(outcome_space, issues, outcomes)
        else:
            mn, _ = rng
        offset = to - mn
        return self.shift_by(offset)

    def scale_min_for(
        self,
        to: float,
        outcome_space: OutcomeSpace | None = None,
        issues: list[Issue] | None = None,
        outcomes: list[Outcome] | None = None,
        rng: tuple[float, float] | None = None,
    ) -> "UtilityFunction":
        if rng is None:
            mn, _ = self.minmax(outcome_space, issues, outcomes)
        else:
            mn, _ = rng
        scale = to / mn
        return self.scale_by(scale)

    def shift_max_for(
        self,
        to: float,
        outcome_space: OutcomeSpace | None = None,
        issues: list[Issue] | None = None,
        outcomes: list[Outcome] | None = None,
        rng: tuple[float, float] | None = None,
    ) -> "UtilityFunction":
        if rng is None:
            _, mx = self.minmax(outcome_space, issues, outcomes)
        else:
            _, mx = rng
        offset = to - mx
        return self.shift_by(offset)

    def scale_max_for(
        self,
        to: float,
        outcome_space: OutcomeSpace | None = None,
        issues: list[Issue] | None = None,
        outcomes: list[Outcome] | None = None,
        rng: tuple[float, float] | None = None,
    ) -> "UtilityFunction":
        if rng is None:
            _, mx = self.minmax(outcome_space, issues, outcomes)
        else:
            _, mx = rng
        scale = to / mx
        return self.scale_by(scale)

    def argrank(
        self, outcomes: list[Outcome | None], descending=True
    ) -> list[list[Outcome | None]]:
        """Ranks the given list of outcomes with weights. None stands for the null outcome.

        Returns:
            A list of lists of integers giving the outcome index in the input. The list is sorted by utlity value

        """
        ranks = self.argrank_with_weights(outcomes, descending)
        return [_[0] for _ in ranks]

    def rank(
        self, outcomes: list[Outcome | None], descending=True
    ) -> list[list[Outcome | None]]:
        """Ranks the given list of outcomes with weights. None stands for the null outcome.

        Returns:
            A list of lists of integers giving the outcome index in the input. The list is sorted by utlity value

        """
        ranks = self.rank_with_weights(outcomes, descending)
        return [_[0] for _ in ranks]

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
        self, outcomes: list[Outcome | None], descending=True
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

    def rank_with_weights(
        self, outcomes: list[Outcome | None], descending=True
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

    def eu(self, offer: Outcome | None) -> float:
        """calculates the **expected** utility value of the input outcome"""
        return float(self(offer))

    def __call__(self, offer: Outcome | None) -> Value:
        """Calculate the utility_function value for a given outcome at the given negotiation state.

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
            return self.reserved_value  # type: ignore I know that concrete subclasses will be returning the correct type
        return self.eval(offer)

    @abstractmethod
    def eval(self, offer: Outcome) -> Value:
        ...

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

    def extreme_outcomes(
        self,
        outcome_space: OutcomeSpace | None = None,
        issues: Iterable[Issue] | None = None,
        outcomes: Iterable[Outcome] | None = None,
        max_cardinality=1000,
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
        for o in outcomes:
            u = self(o)
            if u < mn:
                worst, mn = o, u
                continue
            if u > mx:
                best, mx = o, u
        if worst is None or best is None:
            raise ValueError(f"Cound not find worst and best outcomes for {self}")
        return worst, best


class _General:
    def is_session_dependent(self) -> bool:
        return True

    def is_volatile(self) -> bool:
        return True

    def is_state_dependent(self) -> bool:
        return True

    def is_stationary(self) -> bool:
        return False


class UtilityFunction(_General, BaseUtilityFunction, UFunCrisp):
    """Base for all crisp ufuns"""

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

    @abstractmethod
    def eval(self, offer: Outcome) -> float:
        ...

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

            >>> from negmas.preferences import conflict_level
            >>> u1, u2 = UtilityFunction.generate_bilateral(outcomes=10, conflict_level=0.0
            ...                                             , conflict_delta=0.0)
            >>> print(conflict_level(u1, u2, outcomes=10))
            0.0

            >>> u1, u2 = UtilityFunction.generate_bilateral(outcomes=10, conflict_level=1.0
            ...                                             , conflict_delta=0.0)
            >>> print(conflict_level(u1, u2, outcomes=10))
            1.0

            >>> u1, u2 = UtilityFunction.generate_bilateral(outcomes=10, conflict_level=0.5
            ...                                             , conflict_delta=0.0)
            >>> 0.0 < conflict_level(u1, u2, outcomes=10) < 1.0
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
        else:
            outcomes = list(outcomes)
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
        rng: Tuple[float, float],
        outcome_space: OutcomeSpace | None = None,
        issues: List[Issue] | None = None,
        outcomes: List[Outcome] | None = None,
        n_trials: int = 100,
    ) -> Optional["Outcome"]:
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
            if rng[0] - 1e-6 <= self(o) <= rng[1] + 1e-6:
                return o
        return None

    def normalize_for(
        self,
        to: tuple[float, float] = (0.0, 1.0),
        outcome_space: OutcomeSpace | None = None,
        issues: list[Issue] | None = None,
        outcomes: list[Outcome] | None = None,
        minmax: tuple[float, float] | None = None,
    ) -> "UtilityFunction":
        """
        Creates a new utility function that is normalized based on input conditions.

        Args:
            to: The minimum and maximum value to normalize to. If either is None, it is ignored.
                 This means that passing `(None, 1.0)` will normalize the ufun so that the maximum
                 is `1` but will not guarantee any limit for the minimum and so on.
            outcomes: A set of outcomes to limit our attention to. If not given,
                      the whole ufun is normalized
            outcome_space: The outcome-space to focus on when normalizing
            minmax: The current minimum and maximum to use for normalization. Pass if known to avoid
                  calculating them using the outcome-space given or defined for the ufun.
        """
        max_cardinality: int = MAX_CARINALITY
        outcome_space = None
        if minmax is not None:
            mn, mx = minmax
        else:
            check_one_at_most(outcome_space, issues, outcomes)
            outcome_space = os_or_none(outcome_space, issues, outcomes)
            if not outcome_space:
                outcome_space = self.outcome_space
            if not outcome_space:
                raise ValueError(
                    "Cannot find the outcome-space to normalize for. "
                    "You must pass outcome_space, issues or outcomes or have the ufun being constructed with one of them"
                )
            mn, mx = self.minmax(outcome_space, max_cardinality=max_cardinality)

        scale = (to[1] - to[0]) / (mx - mn)

        # u = self.shift_by(-mn, shift_reserved=True)
        u = self.scale_by(scale, scale_reserved=True)
        return u.shift_by(to[0] - scale * mn, shift_reserved=True)

    @classmethod
    def from_genius(
        cls, file_name: PathLike | str, **kwargs
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

    def to_genius(
        self, file_name: PathLike | str, issues: Iterable[Issue] = None, **kwargs
    ):
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
        with open(file_name, "w") as f:
            f.write(self.to_xml_str(issues=issues, **kwargs))

    def to_xml_str(self, issues: Iterable[Issue] = None, discount_factor=None) -> str:
        """Exports a utility function to a well formatted string"""
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
        if self.reserved_value != float("-inf") and "<reservation value" not in output:
            output += f'<reservation value="{self.reserved_value}" />\n'
        if "</utility_space>" not in output:
            output += "</utility_space>\n"
        return output

    @classmethod
    def from_xml_str(
        cls,
        xml_str: str,
        issues: Iterable[Issue],
        safe_parsing=True,
        ignore_discount=False,
        ignore_reserved=False,
        name: str = None,
    ) -> Tuple["UtilityFunction" | None, float | None]:
        """Imports a utility function from a GENIUS XML string.

        Args:

            xml_str (str): The string containing GENIUS style XML utility function definition
            issues (List[Issue] | None): Optional issue space to confirm that the utility function is valid
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
            LinearAdditiveUtilityFunction,
            LinearUtilityFunction,
        )
        from negmas.preferences.nonlinear import HyperRectangleUtilityFunction

        root = ET.fromstring(xml_str)
        if safe_parsing and root.tag != "utility_space":
            raise ValueError(f"Root tag is {root.tag}: Expected utility_space")

        issues = list(issues)
        ordered_issues: list[Issue] = []
        domain_issues_dict: Optional[Dict[str, Issue]] = None
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
                                f"An item without a value at index {item_indx} for issue {issue_key}"
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

                u = LinearUtilityFunction(
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
                u = ComplexWeightedUtilityFunction(
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


class ProbUtilityFunction(_General, BaseUtilityFunction, UFunProb):
    """A probablistic utility function. One that returns a probability distribution when called"""

    def __call__(self, offer: Outcome | None) -> Distribution:
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
            return ScipyDistribution("uniform", loc=self.reserved_value, scale=0.0)
        return self.eval(offer)

    @classmethod
    def generate_bilateral(
        cls,
        outcomes: Union[int, List[Outcome]],
        conflict_level: float = 0.5,
        conflict_delta=0.005,
        scale: float | tuple[float, float] = 0.5,
    ) -> Tuple["ProbUtilityFunction", "ProbUtilityFunction"]:
        """Generates a couple of utility functions

        Args:

            n_outcomes (int): number of outcomes to use
            conflict_level: How conflicting are the two ufuns to generate.
                            1.0 means maximum conflict.
            conflict_delta: How variable is the conflict at different outcomes.

        Examples:

            >>> from negmas.preferences import conflict_level
            >>> u1, u2 = UtilityFunction.generate_bilateral(outcomes=10, conflict_level=0.0
            ...                                             , conflict_delta=0.0)
            >>> print(conflict_level(u1, u2, outcomes=10))
            0.0

            >>> u1, u2 = UtilityFunction.generate_bilateral(outcomes=10, conflict_level=1.0
            ...                                             , conflict_delta=0.0)
            >>> print(conflict_level(u1, u2, outcomes=10))
            1.0

            >>> u1, u2 = UtilityFunction.generate_bilateral(outcomes=10, conflict_level=0.5
            ...                                             , conflict_delta=0.0)
            >>> 0.0 < conflict_level(u1, u2, outcomes=10) < 1.0
            True


        """
        from negmas.preferences.mapping import ProbMappingUtilityFunction

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
            ProbMappingUtilityFunction(
                dict(
                    zip(
                        outcomes,
                        (
                            ScipyDistribution(
                                type="unifomr", loc=_, scale=get_one_float(scale)
                            )
                            for _ in u1
                        ),
                    )
                )
            ),
            ProbMappingUtilityFunction(
                dict(
                    zip(
                        outcomes,
                        (
                            ScipyDistribution(
                                type="unifomr", loc=_, scale=get_one_float(scale)
                            )
                            for _ in u2
                        ),
                    )
                )
            ),
        )

    @classmethod
    def generate_random_bilateral(
        cls, outcomes: Union[int, List[Outcome]], scale: float = 0.5
    ) -> Tuple["ProbUtilityFunction", "ProbUtilityFunction"]:
        """Generates a couple of utility functions

        Args:

            n_outcomes (int): number of outcomes to use
            conflict_level: How conflicting are the two ufuns to generate. 1.0 means maximum conflict.
            conflict_delta: How variable is the conflict at different outcomes.
            zero_summness: How zero-sum like are the two ufuns.


        """
        from negmas.preferences.mapping import ProbMappingUtilityFunction

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
            ProbMappingUtilityFunction(
                dict(
                    zip(
                        outcomes,
                        (
                            ScipyDistribution(
                                type="unifomr", loc=_, scale=get_one_float(scale)
                            )
                            for _ in u1
                        ),
                    )
                )
            ),
            ProbMappingUtilityFunction(
                dict(
                    zip(
                        outcomes,
                        (
                            ScipyDistribution(
                                type="unifomr", loc=_, scale=get_one_float(scale)
                            )
                            for _ in u2
                        ),
                    )
                )
            ),
        )

    @classmethod
    def generate_random(
        cls,
        n: int,
        outcomes: Union[int, List[Outcome]],
        normalized: bool = True,
        scale: float | tuple[float, float] = 0.5,
    ) -> List["ProbUtilityFunction"]:
        """Generates N mapping utility functions

        Args:
            n: number of utility functions to generate
            outcomes: number of outcomes to use
            normalized: if true, the resulting ufuns will be normlized between zero and one.


        """
        from negmas.preferences.mapping import MappingUtilityFunction

        if isinstance(outcomes, int):
            outcomes = [(_,) for _ in range(outcomes)]
        else:
            outcomes = list(outcomes)
        n_outcomes = len(outcomes)
        ufuns = []
        for _ in range(n):
            u1 = np.random.random(n_outcomes)
            if normalized:
                u1 -= u1.min()
                u1 /= u1.max()
            ufuns.append(
                MappingUtilityFunction(
                    dict(
                        zip(
                            outcomes,
                            (
                                ScipyDistribution(
                                    type="unifomr", loc=_, scale=get_one_float(scale)
                                )
                                for _ in u1
                            ),
                        )
                    )
                )
            )
        return ufuns


class StationaryUtilityFunction(UtilityFunction):
    def is_session_dependent(self) -> bool:
        return False

    def is_volatile(self) -> bool:
        return False

    def is_state_dependent(self) -> bool:
        return False

    def is_stationary(self) -> bool:
        return True

    def to_stationary(self):
        return self


class SamplingInverseUtilityFunction(MultiInverseUFun, InverseUFun):
    def __init__(self, ufun: UtilityFunction, max_samples_per_call: int = 10_000):
        self._ufun = ufun
        self.max_samples_per_call = max_samples_per_call

    def init(self):
        pass

    def all(
        self,
        rng: float | tuple[float, float],
    ) -> list[Outcome]:
        """
        Finds all outcomes with in the given utility value range

        Args:
            rng: The range. If a value, outcome utilities must match it exactly

        Remarks:
            - If issues or outcomes are not None, then init_inverse will be called first
            - If the outcome-space is discrete, this method will return all outcomes in the given range

        """
        raise ValueError(
            f"Cannot find all outcomes in a range using a SamplingInverseUtilityFunction. Try a PresortedInverseUtilityFunction"
        )

    def some(
        self,
        rng: float | tuple[float, float],
        n: int | None = None,
    ) -> list[Outcome]:
        """
        Finds some outcomes with the given utility value (if discrete, all)

        Args:
            rng: The range. If a value, outcome utilities must match it exactly
            n: The maximum number of outcomes to return

        Remarks:
            - If issues or outcomes are not None, then init_inverse will be called first
            - If the outcome-space is discrete, this method will return all outcomes in the given range

        """
        if not n:
            n = self.max_samples_per_call
        if not self._ufun.outcome_space:
            return []
        return list(self._ufun.outcome_space.sample(n, False, False))

    def worst_in(self, rng: float | tuple[float, float]) -> Outcome | None:
        some = self.some(rng)
        if not isinstance(rng, Iterable):
            rng = (rng, rng)
        worst_util, worst = float("inf"), None
        for o in some:
            util = self._ufun(o)
            if util < worst_util:
                worst_util, worst = util, o
        return worst

    def best_in(self, rng: float | tuple[float, float]) -> Outcome | None:
        some = self.some(rng)
        if not isinstance(rng, Iterable):
            rng = (rng, rng)
        best_util, best = float("-inf"), None
        for o in some:
            util = self._ufun(o)
            if util < best_util:
                best_util, best = util, o
        return best

    def one_in(self, rng: float | tuple[float, float]) -> Outcome | None:
        if not self._ufun.outcome_space:
            return None
        if not isinstance(rng, Iterable):
            rng = (rng, rng)
        for _ in range(self.max_samples_per_call):
            o = list(self._ufun.outcome_space.sample(1))[0]
            if rng[0] + 1e-7 <= self._ufun(o) <= rng[1] - 1e-7:
                return o
        return None


class PresortingInverseUtilityFunction(MultiInverseUFun, InverseUFun):
    def __init__(
        self, ufun: UtilityFunction, levels: int = 10, max_cache_size: int = 10_000
    ):
        self._ufun = ufun
        self.max_cache_size = max_cache_size
        self.levels = levels
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
        for l in range(self.levels, 0, -1):
            n = outcome_space.cardinality_if_discretized(l)
            if n <= self.max_cache_size:
                break
        else:
            raise ValueError(
                f"Cannot discretize keeping cach size at {self.max_cache_size}"
            )
        os = outcome_space.to_discrete(levels=l, max_cardinality=self.max_cache_size)
        if os.cardinality <= self.max_cache_size:
            outcomes = list(os.sample(self.max_cache_size, False, False))
        else:
            outcomes = list(os.enumerate())[: self.max_cache_size]
        utils = [self._ufun(_) for _ in outcomes]
        self._ordered_outcomes = sorted(zip(utils, outcomes), key=lambda x: -x[0])

    def all(
        self,
        rng: float | tuple[float, float],
    ) -> list[Outcome]:
        """
        Finds all outcomes with in the given utility value range

        Args:
            rng: The range. If a value, outcome utilities must match it exactly

        Remarks:
            - If issues or outcomes are not None, then init_inverse will be called first
            - If the outcome-space is discrete, this method will return all outcomes in the given range

        """
        os_ = self._ufun.outcome_space
        if not os_:
            raise ValueError(f"Unkonwn outcome space. Cannot invert the ufun")

        if os_.is_discrete():
            return self.some(rng)
        raise ValueError(
            f"Cannot find all outcomes in a range for a continous outcome space (there is in general an infinite number of them)"
        )

    def some(
        self,
        rng: float | tuple[float, float],
        n: int | None = None,
    ) -> list[Outcome]:
        """
        Finds some outcomes with the given utility value (if discrete, all)

        Args:
            rng: The range. If a value, outcome utilities must match it exactly
            n: The maximum number of outcomes to return

        Remarks:
            - If issues or outcomes are not None, then init_inverse will be called first
            - If the outcome-space is discrete, this method will return all outcomes in the given range

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
            if n and len(results) >= n:
                return results
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
