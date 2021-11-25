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

from negmas.common import NegotiatorMechanismInterface
from negmas.generics import ienumerate, ivalues
from negmas.helpers import PATH, Distribution, Value, ikeys, snake_case
from negmas.outcomes import (
    Issue,
    Outcome,
    discretize_and_enumerate_issues,
    enumerate_issues,
    num_outcomes,
    outcome_is_valid,
    sample_issues,
    sample_outcomes,
)
from negmas.serialization import deserialize, serialize
from negmas.types import NamedObject

from .preferences import ProbCardinalPreferences

__all__ = [
    "ProbUtilityFunction",
]


class ProbUtilityFunction(ProbCardinalPreferences):
    """The abstract base class for all probabilistic utility functions.

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
        issues: The list of issues for which the ufun is defined (optional)
        ami: The `AgentMechanismInterface` for a mechanism for which the ufun
             is defined (optinoal)
        id: An optional system-wide unique identifier. You should not change
            the default value except in special circumstances like during
            serialization and should always guarantee system-wide uniquness
            if you set this value explicitly

    Remarks:
        - If ami is given, it overrides issues
        - If issues is given, it overrides issue_names
        - One of `eval`, `is_better`, `rank` **MUST** be overriden, otherwise
          calling any of them will lead to an infinite loop which is very hard
          to debug.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._inverse_initialized = False

    def __getitem__(self, offer: Outcome | None) -> Optional[Distribution]:
        """Overrides [] operator to call the ufun allowing it to act as a mapping"""
        return self(offer)

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
            - Return A UtilityValue not a float for real-valued utilities for the benefit of inspection code.
            - Return the reserved value if the offer was None

        Returns:
            UtilityValue: The utility_function value which may be a distribution. If `None` it means the
                          utility_function value cannot be calculated.
        """
        if offer is None:
            return self.reserved_value
        return self.eval(offer)

    @abstractmethod
    def eval(self, offer: Outcome) -> Distribution:
        """
        Calculate the utility value for a given outcome.

        Args:
            offer: The offer to be evaluated.

        Returns:
            UtilitiDistribution: The utility_function value which may be a distribution.
                                 If None` it means the utility_function value cannot
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

    def eval_all(self, outcomes: List[Outcome]) -> Iterable[UtilityValue]:
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
    ) -> Tuple[List["MappingUtilityFunction"], List[Outcome], List["Issue"]]:
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

    @abstractmethod
    def xml(self, issues: List[Issue]) -> str:
        """Converts the function into a well formed XML string preferrably in GENIUS format.

        If the output has with </objective> then discount factor and reserved value should also be included
        If the output has </utility_space> it will not be appended in `to_xml_str`

        """

    def eu(self, offer: Outcome) -> Optional[float]:
        """Calculate the expected utility value.

        Args:
            offer: The offer to be evaluated.

        Returns:
            float: The expected utility value for UFuns that return a distribution and just utility value for real-valued utilities.

        """
        v = self(offer)
        return float(v) if v is not None else None

    @classmethod
    def opposition_level(
        cls,
        ufuns=List["UtilityFunction"],
        max_utils: Union[float, Tuple[float, float]] = 1.0,
        outcomes: Union[int, List[Outcome]] = None,
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
            >>> from negmas.preferences.nonlinear import MappingUtilityFunction
            >>> from negmas.preferences import UtilityFunction
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
            outcomes = enumerate_issues(issues, max_n_outcomes=max_tests)
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
                (1.0 - float(u(outcome)) / max_util) ** 2
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
        outcomes: Union[int, List[Outcome]],
        max_tests: int = 10000,
    ) -> float:
        """
        Finds the conflict level in these two ufuns

        Args:
            u1: first utility function
            u2: second utility function

        Examples:
            - A nonlinear strictly zero sum case
            >>> from negmas.preferences.nonlinear import MappingUtilityFunction
            >>> from negmas.preferences import UtilityFunction
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
        outcomes: Union[int, List[Outcome]],
        max_tests: int = 10000,
    ) -> float:
        """
        Finds the win-win level in these two ufuns

        Args:
            u1: first utility function
            u2: second utility function

        Examples:
            - A nonlinear same ufun case
            >>> from negmas.preferences.nonlinear import MappingUtilityFunction
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

    def utility_difference_prob(self, first: Outcome, second: Outcome) -> Distribution:
        return self(first) - self(second)

    @classmethod
    def random(
        cls, issues, reserved_value=(0.0, 0.5), normalized=True, **kwargs
    ) -> "ProbUtilityFunction":
        """Generates a random ufun of the given type"""

    @classmethod
    def from_str(cls, s) -> "ProbUtilityFunction":
        """Creates an object out of a dict."""
        return deserialize(eval(s))

    def __str__(self):
        return str(serialize(self))
