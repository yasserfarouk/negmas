"""Outcome space implementations for representing negotiation domains."""

from __future__ import annotations

from math import isinf
import numbers
from pathlib import Path
from functools import reduce
import random
from itertools import filterfalse
from operator import mul
from typing import TYPE_CHECKING, Callable, Iterable, Sequence, Union

from attrs import define, field

from negmas.helpers import unique_name
from negmas.helpers.types import get_full_type_name
from negmas.outcomes.outcome_ops import (
    cast_value_types,
    outcome_is_valid,
    outcome_types_are_ok,
)
from negmas.protocols import XmlSerializable
from negmas.serialization import PYTHON_CLASS_IDENTIFIER, deserialize, serialize
from negmas.warnings import NegmasSpeedWarning, warn

from .base_issue import DiscreteIssue, Issue
from .categorical_issue import CategoricalIssue
from .common import Outcome
from .contiguous_issue import ContiguousIssue
from .issue_ops import (
    enumerate_discrete_issues,
    issues_from_outcomes,
    issues_from_xml_str,
    issues_to_xml_str,
    sample_issues,
)
from .protocols import DiscreteOutcomeSpace, OutcomeSpace
from .range_issue import RangeIssue
from .singleton_issue import SingletonIssue

if TYPE_CHECKING:
    from negmas.preferences.protocols import HasReservedOutcome, HasReservedValue

__all__ = [
    "CartesianOutcomeSpace",
    "EnumeratingOutcomeSpace",
    "DiscreteCartesianOutcomeSpace",
    "SingletonOutcomeSpace",
    "make_os",
    "DistanceFun",
    "os_union",
    "os_intersection",
    "os_difference",
]

NLEVELS = 5


DistanceFun = Callable[[Outcome, Outcome, Union[OutcomeSpace, None]], float]
"""A callable that can calculate the distance between two outcomes in an outcome-space"""


def make_os(
    issues: Sequence[Issue] | None = None,
    outcomes: Sequence[Outcome] | None = None,
    name: str | None = None,
    path: Path | None = None,
) -> CartesianOutcomeSpace:
    """
    A factory to create outcome-spaces from lists of `Issue` s or `Outcome` s.

    Remarks:

        - must pass one and exactly one of `issues` and `outcomes`
    """
    if issues and outcomes:
        raise ValueError(
            "Cannot make an outcome space passing both issues and outcomes"
        )
    if not issues and not outcomes:
        raise ValueError(
            "Cannot make an outcome space without passing issues or outcomes"
        )
    if not issues and outcomes:
        issues_ = issues_from_outcomes(outcomes)
    else:
        issues_ = issues
    if issues_ is None:
        raise ValueError(
            "Cannot make an outcome space without passing issues or outcomes"
        )

    issues_ = tuple(issues_)
    if all(_.is_discrete() for _ in issues_):
        return DiscreteCartesianOutcomeSpace(
            issues_, name=name if name else "", path=path
        )
    return CartesianOutcomeSpace(issues_, name=name if name else "", path=path)


@define
class OSWithValidity:
    """OSWithValidity implementation."""

    invalid: set[Outcome] = field(factory=set)
    _baseset: set[Outcome] = field(factory=set)

    def __attrs_post_init__(self):
        """Initializes the base set of valid outcomes after attrs initialization."""
        self.update()

    def update(self):
        """Recomputes the base set by removing invalid outcomes from the enumeration."""
        self._baseset = set(self.enumerate()).difference(self.invalid)


@define
class EnumeratingOutcomeSpace(DiscreteOutcomeSpace, OSWithValidity):
    """An outcome space representing the enumeration of some outcomes. No issues defined"""

    name: str | None = field(eq=False, default=None)
    path: Path | None = field(eq=False, default=None)

    def invalidate(self, outcome: Outcome) -> None:
        """Indicates that the outcome is invalid"""
        self.invalid.add(outcome)
        self.update()

    def validate(self, outcome: Outcome) -> None:
        """Indicates that the outcome is invalid"""
        try:
            self.invalid.remove(outcome)
        except Exception:
            pass
        self.update()

    def is_valid(self, outcome: Outcome) -> bool:
        """Checks if the given outcome is valid for that outcome space"""
        return outcome in self._baseset

    def are_types_ok(self, outcome: Outcome) -> bool:
        """Checks if the type of each value in the outcome is correct for the given issue"""
        return True

    def ensure_correct_types(self, outcome: Outcome) -> Outcome:
        """Returns an outcome that is guaratneed to have correct types or raises an exception"""
        return outcome

    @property
    def cardinality(self) -> int:
        """The space cardinality = the number of outcomes"""
        return len(self._baseset)

    def is_numeric(self) -> bool:
        """Checks whether all values in all outcomes are numeric"""
        samples = random.choices(list(self._baseset), k=int(min(self.cardinality, 10)))
        numeric = [all(isinstance(_, numbers.Number) for _ in s) for s in samples]
        return all(numeric)

    def is_integer(self) -> bool:
        """Checks whether all values in all outcomes are integers"""
        samples = random.choices(list(self._baseset), k=int(min(self.cardinality, 10)))
        numeric = [all(isinstance(_, numbers.Integral) for _ in s) for s in samples]
        return all(numeric)

    def is_float(self) -> bool:
        """Checks whether all values in all outcomes are real"""
        samples = random.choices(list(self._baseset), k=int(min(self.cardinality, 10)))
        numeric = [
            all(
                isinstance(_, numbers.Real) and not isinstance(_, numbers.Integral)
                for _ in s
            )
            for s in samples
        ]
        return all(numeric)

    def to_discrete(
        self, levels: int | float = 5, max_cardinality: int | float = float("inf")
    ) -> DiscreteOutcomeSpace:
        """
        Returns a **stable** finite outcome space. If the outcome-space is already finite. It shoud return itself.

        Args:
            levels: The levels of discretization of any continuous dimension (or subdimension)
            max_cardintlity: The maximum cardinality allowed for the resulting outcomespace (if the original OS was infinite).
                             This limitation is **NOT** applied for outcome spaces that are alredy discretized. See `limit_cardinality()`
                             for a method to limit the cardinality of an already discrete space

        If called again, it should return the same discrete outcome space every time.
        """
        return self

    def random_outcome(self) -> Outcome:
        """Returns a single random outcome."""
        return list(self.sample(1))[0]

    def to_largest_discrete(
        self, levels: int, max_cardinality: int | float = float("inf"), **kwargs
    ) -> DiscreteOutcomeSpace:
        """Returns self since this space is already discrete."""
        return self

    def cardinality_if_discretized(
        self, levels: int, max_cardinality: int | float = float("inf")
    ) -> int:
        """
        Returns the cardinality if discretized the given way.
        """
        return self.cardinality

    def enumerate_or_sample(
        self,
        levels: int | float = float("inf"),
        max_cardinality: int | float = float("inf"),
    ) -> Iterable[Outcome]:
        """Enumerates all outcomes if possible (i.e. discrete space) or returns `max_cardinality` different outcomes otherwise"""
        return self.enumerate()

    def is_discrete(self) -> bool:
        """Checks whether there are no continua components of the space"""
        return True

    def is_finite(self) -> bool:
        """Checks whether the space is finite"""
        return self.is_discrete()

    def __contains__(self, item: Outcome | OutcomeSpace | Issue) -> bool:  # type: ignore
        """Checks if the given item is contained in this outcome space."""
        if isinstance(item, Issue):
            return False
        if isinstance(item, Outcome):
            return item in self._baseset
        if not isinstance(item, OutcomeSpace):
            return False
        if isinf(item.cardinality):
            return False
        return all(x in self for x in item.enumerate_or_sample())

    def enumerate(self) -> Iterable[Outcome]:
        """
        Enumerates the outcome space returning all its outcomes (or up to max_cardinality for infinite ones)
        """
        return self._baseset

    def sample(
        self, n_outcomes: int, with_replacement: bool = False, fail_if_not_enough=False
    ) -> Iterable[Outcome]:
        """Samples up to n_outcomes with or without replacement"""
        if self.cardinality < n_outcomes and not with_replacement:
            return []
        if with_replacement:
            return (random.choice(list(self._baseset)) for _ in range(n_outcomes))
        return random.sample(list(self._baseset), k=n_outcomes)

    def limit_cardinality(
        self,
        max_cardinality: int | float = float("inf"),
        levels: int | float = float("inf"),
    ) -> DiscreteOutcomeSpace:
        """
        Limits the cardinality of the outcome space to the given maximum (or the number of levels for each issue to `levels`)

        Args:
            max_cardinality: The maximum number of outcomes in the resulting space
            levels: The maximum levels allowed per issue (if issues are defined for this outcome space)
        """
        ...

    def to_single_issue(
        self, numeric: bool = False, stringify: bool = True
    ) -> CartesianOutcomeSpace:
        """Converts this space to a single-issue outcome space (not implemented for this class)."""
        ...

    def contains_os(self, x: OutcomeSpace) -> bool:
        """Checks whether an outcome-space is contained in this outcome-space."""
        if not x.is_finite():
            if not self.is_finite():
                raise NotImplementedError(
                    "Cannot check containment between two infinite outcome spaces"
                )
            return False
        return all(outcome in self._baseset for outcome in x.enumerate())  # type: ignore

    def __or__(self, other: OutcomeSpace) -> EnumeratingOutcomeSpace:
        """Returns the union of this outcome space with another (| operator)."""
        return os_union(self, other)

    def __and__(self, other: OutcomeSpace) -> EnumeratingOutcomeSpace:
        """Returns the intersection of this outcome space with another (& operator)."""
        return os_intersection(self, other)

    def __sub__(self, other: OutcomeSpace) -> EnumeratingOutcomeSpace:
        """Returns the difference of this outcome space with another (- operator)."""
        return os_difference(self, other)


@define(frozen=True)
class CartesianOutcomeSpace(XmlSerializable):
    """
    An outcome-space that is generated by the cartesian product of a tuple of `Issue` s.
    """

    issues: tuple[Issue, ...] = field(converter=tuple)
    name: str | None = field(eq=False, default=None)
    path: Path | None = field(eq=False, default=None)

    def __attrs_post_init__(self):
        """Generates a unique name for the outcome space if none was provided."""
        if not self.name:
            object.__setattr__(self, "name", unique_name("os", add_time=False, sep=""))

    def __mul__(self, other: CartesianOutcomeSpace) -> CartesianOutcomeSpace:
        """Returns a new outcome space that is the Cartesian product of this space and another."""
        issues = list(self.issues) + list(other.issues)
        name = f"{self.name}*{other.name}"
        return CartesianOutcomeSpace(tuple(issues), name=name)

    def cartesian_product(self, other: CartesianOutcomeSpace) -> CartesianOutcomeSpace:
        """Returns a new outcome space that is the Cartesian product of this space and another."""
        return self * other

    def contains_issue(self, x: Issue) -> bool:
        """Cheks that the given issue is in the tuple of issues constituting the outcome space (i.e. it is one of its dimensions)"""
        return x in self.issues

    def is_valid(self, outcome: Outcome) -> bool:
        """Checks if the given outcome is valid within this outcome space."""
        return outcome_is_valid(outcome, self.issues)

    def is_discrete(self) -> bool:
        """Checks whether all issues are discrete"""
        return all(_.is_discrete() for _ in self.issues)

    def is_finite(self) -> bool:
        """Checks whether the space is finite"""
        return self.is_discrete()

    def contains_os(self, x: OutcomeSpace) -> bool:
        """Checks whether an outcome-space is contained in this outcome-space"""
        # Handle SingletonOutcomeSpace - check if the single outcome is valid
        if hasattr(x, "outcome") and hasattr(x, "issues"):
            # This is a SingletonOutcomeSpace
            if len(self.issues) != len(x.issues):  # type: ignore
                return False
            return self.is_valid(x.outcome)  # type: ignore
        if isinstance(x, CartesianOutcomeSpace):
            if len(self.issues) != len(x.issues):
                return False
            return all(b in a for a, b in zip(self.issues, x.issues))
        # For EnumeratingOutcomeSpace or other types
        if self.is_finite() and not x.is_finite():
            return False
        if not self.is_finite() and not x.is_finite():
            raise NotImplementedError(
                "Cannot check an infinite outcome space that is not cartesian for inclusion in an infinite cartesian outcome space!!"
            )
        if x.is_finite():
            warn(
                f"Testing inclusion of a finite non-cartesian outcome space in a cartesian outcome space can be slow (will do {x.cardinality} checks)",
                NegmasSpeedWarning,
            )
            return all(self.is_valid(_) for _ in x.enumerate())  # type: ignore
        return False

    def to_dict(self, python_class_identifier=PYTHON_CLASS_IDENTIFIER):
        """Serializes the outcome space to a dictionary representation."""
        d = {python_class_identifier: get_full_type_name(type(self))}
        return dict(
            **d,
            name=self.name,
            issues=serialize(
                self.issues, python_class_identifier=python_class_identifier
            ),
        )

    @classmethod
    def from_dict(cls, d, python_class_identifier=PYTHON_CLASS_IDENTIFIER):
        """Deserializes an outcome space from a dictionary representation."""
        return cls(**deserialize(d, python_class_identifier=python_class_identifier))  # type: ignore

    @property
    def issue_names(self) -> list[str]:
        """Returns an ordered list of issue names"""
        return [_.name for _ in self.issues]

    @property
    def cardinality(self) -> int | float:
        """The space cardinality = the number of outcomes"""
        return reduce(mul, [_.cardinality for _ in self.issues], 1)

    def is_compact(self) -> bool:
        """Checks whether all issues are complete ranges"""
        return all(isinstance(_, RangeIssue) for _ in self.issues)

    def is_all_continuous(self) -> bool:
        """Checks whether all issues are discrete"""
        return all(_.is_continuous() for _ in self.issues)

    def is_not_discrete(self) -> bool:
        """Checks whether all issues are discrete"""
        return any(_.is_continuous() for _ in self.issues)

    def is_numeric(self) -> bool:
        """Checks whether all issues are numeric"""
        return all(_.is_numeric() for _ in self.issues)

    def is_integer(self) -> bool:
        """Checks whether all issues are integer"""
        return all(_.is_integer() for _ in self.issues)

    def is_float(self) -> bool:
        """Checks whether all issues are real"""
        return all(_.is_float() for _ in self.issues)

    def to_discrete(
        self, levels: int | float = 10, max_cardinality: int | float = float("inf")
    ) -> DiscreteCartesianOutcomeSpace:
        """
        Discretizes the outcome space by sampling `levels` values for each continuous issue.

        The result of the discretization is stable in the sense that repeated calls will return the same output.
        """
        if max_cardinality != float("inf"):
            c = reduce(
                mul,
                [_.cardinality if _.is_discrete() else levels for _ in self.issues],
                1,
            )
            if c > max_cardinality:
                raise ValueError(
                    f"Cannot convert OutcomeSpace to a discrete OutcomeSpace with at most {max_cardinality} (at least {c} outcomes are required)"
                )
        issues = tuple(
            issue.to_discrete(
                levels if issue.is_continuous() else None,
                compact=False,
                grid=True,
                endpoints=True,
            )
            for issue in self.issues
        )
        return DiscreteCartesianOutcomeSpace(issues=issues, name=self.name)

    @classmethod
    def from_xml_str(
        cls, xml_str: str, safe_parsing=True, name=None, **kwargs
    ) -> CartesianOutcomeSpace:
        """Parses an outcome space from an XML string representation."""
        issues, _ = issues_from_xml_str(
            xml_str, safe_parsing=safe_parsing, n_discretization=None
        )
        if not issues:
            raise ValueError("Failed to read an issue space from an xml string")
        issues = tuple(issues)
        if all(isinstance(_, DiscreteIssue) for _ in issues):
            return DiscreteCartesianOutcomeSpace(issues, name=name)
        return cls(issues, name=name)

    @staticmethod
    def from_outcomes(
        outcomes: list[Outcome],
        numeric_as_ranges: bool = False,
        issue_names: list[str] | None = None,
        name: str | None = None,
    ) -> DiscreteCartesianOutcomeSpace:
        """Creates a discrete outcome space by inferring issues from a list of outcomes."""
        return DiscreteCartesianOutcomeSpace(
            issues_from_outcomes(outcomes, numeric_as_ranges, issue_names), name=name
        )

    def to_xml_str(self, **kwargs) -> str:
        """Serializes the outcome space to an XML string representation."""
        return issues_to_xml_str(self.issues)

    def are_types_ok(self, outcome: Outcome) -> bool:
        """Checks if the type of each value in the outcome is correct for the given issue"""
        return outcome_types_are_ok(outcome, self.issues)

    def ensure_correct_types(self, outcome: Outcome) -> Outcome:
        """Returns an outcome that is guaratneed to have correct types or raises an exception"""
        return cast_value_types(outcome, self.issues)

    def sample(
        self, n_outcomes: int, with_replacement: bool = True, fail_if_not_enough=True
    ) -> Iterable[Outcome]:
        """Samples outcomes from this space, with or without replacement."""
        return sample_issues(
            self.issues, n_outcomes, with_replacement, fail_if_not_enough
        )

    def random_outcome(self):
        """Generates a single random outcome by sampling one value from each issue."""
        return tuple(_.rand() for _ in self.issues)

    def cardinality_if_discretized(
        self, levels: int, max_cardinality: int | float = float("inf")
    ) -> int:
        """Computes the cardinality that would result from discretizing continuous issues."""
        c = reduce(
            mul, [_.cardinality if _.is_discrete() else levels for _ in self.issues], 1
        )
        return min(c, max_cardinality)

    def to_largest_discrete(
        self, levels: int, max_cardinality: int | float = float("inf"), **kwargs
    ) -> DiscreteCartesianOutcomeSpace:
        """Discretizes to the largest possible space within the given cardinality constraint."""
        for level in range(levels, 0, -1):
            if self.cardinality_if_discretized(levels) < max_cardinality:
                break
        else:
            raise ValueError(
                f"Cannot discretize with levels <= {levels} keeping the cardinality under {max_cardinality} Outocme space cardinality is {self.cardinality}\nOutcome space: {self}"
            )
        return self.to_discrete(level, max_cardinality, **kwargs)

    def enumerate_or_sample_rational(
        self,
        preferences: Iterable[HasReservedValue | HasReservedOutcome],
        levels: int | float = float("inf"),
        max_cardinality: int | float = float("inf"),
        aggregator: Callable[[Iterable[bool]], bool] = any,
    ) -> Iterable[Outcome]:
        """
        Enumerates all outcomes if possible (i.e. discrete space) or returns `max_cardinality` different outcomes otherwise.

        Args:
            preferences: A list of `Preferences` that is used to judge outcomes
            levels: The number of levels to use for discretization if needed
            max_cardinality: The maximum cardinality allowed in case of discretization
            aggregator: A predicate that takes an `Iterable` of booleans representing whether or not an outcome is rational
                        for a given `Preferences` (i.e. better than reservation) and returns a single boolean representing
                        the result for all preferences. Default is any but can be all.
        """
        from negmas.preferences.protocols import HasReservedOutcome, HasReservedValue

        if (
            levels == float("inf")
            and max_cardinality == float("inf")
            and not self.is_discrete()
        ):
            raise ValueError(
                "Cannot enumerate-or-sample an outcome space with infinite outcomes without specifying `levels` and/or `max_cardinality`"
            )
        from negmas.outcomes.outcome_space import DiscreteCartesianOutcomeSpace

        if isinstance(self, DiscreteCartesianOutcomeSpace):
            results = self.enumerate()  # type: ignore We know the outcome space is correct
        else:
            if max_cardinality == float("inf"):
                return self.to_discrete(
                    levels=levels, max_cardinality=max_cardinality
                ).enumerate()
            results = self.sample(
                int(max_cardinality), with_replacement=False, fail_if_not_enough=False
            )

        def is_irrational(x: Outcome):
            """Checks if an outcome is irrational (worse than reservation) for any preference."""

            def irrational(u: HasReservedOutcome | HasReservedValue, x: Outcome):
                """Checks if outcome x is worse than the reservation for preference u."""
                if isinstance(u, HasReservedValue):
                    if u.reserved_value is None:
                        return False
                    return u(x) < u.reserved_value  # type: ignore
                if isinstance(u, HasReservedOutcome) and u.reserved_outcome is not None:
                    return u.is_worse(x, u.reserved_outcome)  # type: ignore
                return False

            return aggregator(irrational(u, x) for u in preferences)

        return filterfalse(lambda x: is_irrational(x), results)

    def enumerate_or_sample(
        self,
        levels: int | float = float("inf"),
        max_cardinality: int | float = float("inf"),
    ) -> Iterable[Outcome]:
        """Enumerates all outcomes if possible (i.e. discrete space) or returns `max_cardinality` different outcomes otherwise"""
        if (
            levels == float("inf")
            and max_cardinality == float("inf")
            and not self.is_discrete()
        ):
            raise ValueError(
                "Cannot enumerate-or-sample an outcome space with infinite outcomes without specifying `levels` and/or `max_cardinality`"
            )
        from negmas.outcomes.outcome_space import DiscreteCartesianOutcomeSpace

        if isinstance(self, DiscreteCartesianOutcomeSpace):
            return self.enumerate()  # type: ignore We know the outcome space is correct
        if max_cardinality == float("inf"):
            return self.to_discrete(
                levels=levels, max_cardinality=max_cardinality
            ).enumerate()
        return self.sample(
            int(max_cardinality), with_replacement=False, fail_if_not_enough=False
        )

    def to_single_issue(
        self,
        numeric=False,
        stringify=True,
        levels: int = NLEVELS,
        max_cardinality: int | float = float("inf"),
    ) -> DiscreteCartesianOutcomeSpace:
        """
        Creates a new outcome space that is a single-issue version of this one discretizing it as needed

        Args:
            numeric: If given, the output issue will be a `ContiguousIssue` otberwise it will be a `CategoricalIssue`
            stringify:  If given, the output issue will have string values. Checked only if `numeric` is `False`
            levels: Number of levels to discretize any continuous issue
            max_cardinality: Maximum allowed number of outcomes in the resulting issue.

        Remarks:
            - Will discretize inifinte outcome spaces
        """
        if isinstance(self, DiscreteCartesianOutcomeSpace) and len(self.issues) == 1:
            return self
        dos = self.to_discrete(levels, max_cardinality)
        return dos.to_single_issue(numeric, stringify)  # type: ignore

    def __contains__(self, item):
        """Checks if the given item (outcome, issue, or outcome space) is contained in this space."""
        if isinstance(item, OutcomeSpace):
            return self.contains_os(item)
        if isinstance(item, Issue):
            return self.contains_issue(item)
        if isinstance(item, Outcome):
            return self.is_valid(item)
        if not isinstance(item, Sequence):
            return False
        if not item:
            return True
        if isinstance(item[0], Issue):
            return len(self.issues) == len(item) and self.contains_os(
                make_os(issues=item)
            )
        if isinstance(item[0], Outcome):
            return len(self.issues) == len(item) and self.contains_os(
                make_os(outcomes=item)
            )
        return False

    def __or__(self, other: OutcomeSpace) -> EnumeratingOutcomeSpace:
        """Returns the union of this outcome space with another (| operator)."""
        return os_union(self, other)

    def __and__(self, other: OutcomeSpace) -> EnumeratingOutcomeSpace:
        """Returns the intersection of this outcome space with another (& operator)."""
        return os_intersection(self, other)

    def __sub__(self, other: OutcomeSpace) -> EnumeratingOutcomeSpace:
        """Returns the difference of this outcome space with another (- operator)."""
        return os_difference(self, other)


@define(frozen=True)
class DiscreteCartesianOutcomeSpace(CartesianOutcomeSpace):
    """
    A discrete outcome-space that is generated by the cartesian product of a tuple of `Issue` s (i.e. with finite number of outcomes).
    """

    def to_largest_discrete(
        self, levels: int, max_cardinality: int | float = float("inf"), **kwargs
    ) -> DiscreteCartesianOutcomeSpace:
        """Returns self since this space is already discrete."""
        return self

    def __attrs_post_init__(self):
        """Validates that all issues in the space are discrete."""
        for issue in self.issues:
            if not issue.is_discrete():
                raise ValueError(
                    f"Issue is not discrete. Cannot be added to a DiscreteOutcomeSpace. You must discretize it first: {issue} "
                )

    @property
    def cardinality(self) -> int:
        """Returns the total number of possible outcomes in this space."""
        return reduce(mul, [_.cardinality for _ in self.issues], 1)

    def cardinality_if_discretized(
        self, levels: int, max_cardinality: int | float = float("inf")
    ) -> int:
        """Returns the cardinality unchanged since this space is already discrete."""
        return self.cardinality

    def enumerate(self) -> Iterable[Outcome]:
        """Iterates over all possible outcomes in this discrete space."""
        return enumerate_discrete_issues(  #  type: ignore I know that all my issues are actually discrete
            self.issues  #  type: ignore I know that all my issues are actually discrete
        )

    def limit_cardinality(
        self,
        max_cardinality: int | float = float("inf"),
        levels: int | float = float("inf"),
    ) -> DiscreteCartesianOutcomeSpace:
        """
        Limits the cardinality of the outcome space to the given maximum (or the number of levels for each issue to `levels`)

        Args:
            max_cardinality: The maximum number of outcomes in the resulting space
            levels: The maximum number of levels for each issue/subissue
        """
        if self.cardinality <= max_cardinality or all(
            _.cardinality < levels for _ in self.issues
        ):
            return self
        new_levels = [_.cardinality for _ in self.issues]  # type: ignore will be corrected the next line
        new_levels = [int(_) if _ < levels else int(levels) for _ in new_levels]
        new_cardinality = reduce(mul, new_levels, 1)

        def _reduce_total_cardinality(new_levels, max_cardinality, new_cardinality):
            sort = reversed(sorted((_, i) for i, _ in enumerate(new_levels)))
            sorted_levels = [_[0] for _ in sort]
            indices = [_[1] for _ in sort]
            needed = new_cardinality - max_cardinality
            current = 0
            n = len(sorted_levels)
            while needed > 0 and current < n:
                nxt = n - 1
                v = sorted_levels[current]
                if v == 1:
                    continue
                for i in range(current + 1, n - 1):
                    if v == sorted_levels[i]:
                        continue
                    nxt = i
                    break
                diff = v - sorted_levels[nxt]
                if not diff:
                    diff = 1
                new_levels[indices[current]] -= 1
                max_cardinality = (max_cardinality // v) * (v - 1)
                sort = reversed(sorted((_, i) for i, _ in enumerate(new_levels)))
                sorted_levels = [_[0] for _ in sort]
                current = 0
                needed = new_cardinality - max_cardinality
            return new_levels

        if new_cardinality > max_cardinality:
            new_levels: list[int] = _reduce_total_cardinality(
                new_levels, max_cardinality, new_cardinality
            )
        issues: list[Issue] = []
        for j, i, issue in zip(
            new_levels, (_.cardinality for _ in self.issues), self.issues
        ):
            issues.append(issue if j >= i else issue.to_discrete(j, compact=True))
        return DiscreteCartesianOutcomeSpace(
            tuple(issues), name=f"{self.name}-{max_cardinality}"
        )

    def is_discrete(self) -> bool:
        """Checks whether there are no continua components of the space"""
        return True

    def to_discrete(
        self, levels: int | float = 10, max_cardinality: int | float = float("inf")
    ) -> DiscreteCartesianOutcomeSpace:
        """Returns self since this space is already discrete."""
        return self

    def to_single_issue(
        self,
        numeric=False,
        stringify=True,
        levels: int = NLEVELS,
        max_cardinality: int | float = float("inf"),
    ) -> DiscreteCartesianOutcomeSpace:
        """
        Creates a new outcome space that is a single-issue version of this one

        Args:
            numeric: If given, the output issue will be a `ContiguousIssue` otherwise it will be a `CategoricalIssue`
            stringify:  If given, the output issue will have string values. Checked only if `numeric` is `False`

        Remarks:
            - maps the agenda and ufuns to work correctly together
            - Only works if the outcome space is finite
        """
        outcomes = list(self.enumerate())
        values = (
            range(len(outcomes))
            if numeric
            else [f"v{_}" for _ in range(len(outcomes))]
            if stringify
            else outcomes
        )
        issue = (
            ContiguousIssue(len(outcomes), name="-".join(self.issue_names))
            if numeric
            else CategoricalIssue(values, name="-".join(self.issue_names))
        )
        return DiscreteCartesianOutcomeSpace(issues=(issue,), name=self.name)

    # def sample(
    #     self,
    #     n_outcomes: int,
    #     with_replacement: bool = False,
    #     fail_if_not_enough=True,
    # ) -> Iterable[Outcome]:
    #     """
    #     Samples up to n_outcomes with or without replacement.
    #     """
    #
    #     return sample_issues(
    #         self.issues, n_outcomes, with_replacement, fail_if_not_enough
    #     )
    #
    #     # outcomes = self.enumerate()
    #     # outcomes = list(outcomes)
    #     # if with_replacement:
    #     #     return random.choices(outcomes, k=n_outcomes)
    #     # if fail_if_not_enough and n_outcomes > self.cardinality:
    #     #     raise ValueError("Cannot sample enough")
    #     # random.shuffle(outcomes)
    #     # return outcomes[:n_outcomes]

    def __iter__(self):
        """Returns an iterator over all outcomes in this discrete space."""
        return self.enumerate().__iter__()

    def __len__(self) -> int:
        """Returns the number of outcomes in this space."""
        return self.cardinality


@define(frozen=True)
class SingletonOutcomeSpace(DiscreteCartesianOutcomeSpace):
    """
    A discrete outcome-space representing a single outcome.

    This is useful for representing a specific outcome as an outcome space,
    e.g., for checking containment or performing set operations.
    """

    issues: tuple[SingletonIssue, ...] = field(converter=tuple)  # type: ignore[assignment]
    name: str | None = field(eq=False, default=None)
    path: Path | None = field(eq=False, default=None)

    @classmethod
    def from_outcome(
        cls,
        outcome: Outcome,
        issue_names: Sequence[str] | None = None,
        name: str | None = None,
    ) -> SingletonOutcomeSpace:
        """
        Creates a SingletonOutcomeSpace from an outcome.

        Args:
            outcome: The outcome tuple to represent
            issue_names: Optional names for the issues. If None, generates names
                         as issue00, issue01, ...
            name: Optional name for the outcome space

        Returns:
            A SingletonOutcomeSpace containing exactly this outcome
        """
        if issue_names is None:
            issue_names = [f"issue{i:02d}" for i in range(len(outcome))]
        if len(issue_names) != len(outcome):
            raise ValueError(
                f"Number of issue names ({len(issue_names)}) must match "
                f"number of values in outcome ({len(outcome)})"
            )
        issues = tuple(
            SingletonIssue(value, name=iname)
            for value, iname in zip(outcome, issue_names)
        )
        return cls(issues=issues, name=name)

    @property
    def outcome(self) -> Outcome:
        """Returns the single outcome in this space."""
        return tuple(issue.value for issue in self.issues)

    @property
    def cardinality(self) -> int:
        """Always returns 1 since this space contains exactly one outcome."""
        return 1

    def enumerate(self) -> Iterable[Outcome]:
        """Returns an iterable containing the single outcome."""
        return [self.outcome]

    def sample(
        self, n_outcomes: int, with_replacement: bool = True, fail_if_not_enough=True
    ) -> Iterable[Outcome]:
        """Returns the single outcome up to n_outcomes times."""
        if n_outcomes > 1 and not with_replacement:
            if fail_if_not_enough:
                raise ValueError(
                    f"Cannot sample {n_outcomes} outcomes without replacement from a singleton space"
                )
            return [self.outcome]
        return [self.outcome for _ in range(n_outcomes)]

    def is_valid(self, outcome: Outcome) -> bool:
        """Checks if the given outcome equals the single outcome in this space."""
        return outcome == self.outcome

    def contains_os(self, x: OutcomeSpace) -> bool:
        """Checks if this singleton space contains another outcome space."""
        if isinstance(x, SingletonOutcomeSpace):
            return self.outcome == x.outcome
        if x.cardinality > 1:
            return False
        if x.cardinality == 0:
            return True
        # For cardinality == 1, check if the single outcome matches
        if x.is_finite():
            outcomes = list(x.enumerate())  # type: ignore
            return len(outcomes) == 1 and outcomes[0] == self.outcome
        return False

    def __repr__(self):
        """Returns a detailed string representation."""
        return f"SingletonOutcomeSpace({self.outcome!r})"

    def __str__(self):
        """Returns a human-readable string."""
        return f"SingletonOS({self.outcome})"


# =============================================================================
# Set Operations for Outcome Spaces
# =============================================================================


def _os_to_outcome_set(os: OutcomeSpace) -> set[Outcome]:
    """Converts a finite outcome space to a set of outcomes."""
    if not os.is_finite():
        raise ValueError(
            f"Cannot convert non-finite outcome space {type(os).__name__} to a set"
        )
    return set(os.enumerate())  # type: ignore


def os_union(
    os1: OutcomeSpace, os2: OutcomeSpace, name: str | None = None
) -> EnumeratingOutcomeSpace:
    """
    Returns the union of two outcome spaces.

    Args:
        os1: First outcome space
        os2: Second outcome space
        name: Optional name for the result

    Returns:
        An EnumeratingOutcomeSpace containing all outcomes in either space
    """
    outcomes1 = _os_to_outcome_set(os1)
    outcomes2 = _os_to_outcome_set(os2)
    result = outcomes1.union(outcomes2)
    return EnumeratingOutcomeSpace(baseset=result, name=name)


def os_intersection(
    os1: OutcomeSpace, os2: OutcomeSpace, name: str | None = None
) -> EnumeratingOutcomeSpace:
    """
    Returns the intersection of two outcome spaces.

    Args:
        os1: First outcome space
        os2: Second outcome space
        name: Optional name for the result

    Returns:
        An EnumeratingOutcomeSpace containing outcomes in both spaces
    """
    outcomes1 = _os_to_outcome_set(os1)
    outcomes2 = _os_to_outcome_set(os2)
    result = outcomes1.intersection(outcomes2)
    return EnumeratingOutcomeSpace(baseset=result, name=name)


def os_difference(
    os1: OutcomeSpace, os2: OutcomeSpace, name: str | None = None
) -> EnumeratingOutcomeSpace:
    """
    Returns the difference of two outcome spaces (os1 - os2).

    Args:
        os1: First outcome space
        os2: Second outcome space
        name: Optional name for the result

    Returns:
        An EnumeratingOutcomeSpace containing outcomes in os1 but not in os2
    """
    outcomes1 = _os_to_outcome_set(os1)
    outcomes2 = _os_to_outcome_set(os2)
    result = outcomes1.difference(outcomes2)
    return EnumeratingOutcomeSpace(baseset=result, name=name)


# def flat_issues(
#     outcome_spaces: tuple[OutcomeSpace, ...],
#     add_index_to_issue_names: bool = False,
#     add_os_to_issue_name: bool = False,
# ) -> tuple[Issue, ...]:
#     """Generates a single outcome-space which is the Cartesian product of input outcome_spaces."""
#
#     from negmas.outcomes import make_issue
#     from negmas.outcomes.optional_issue import OptionalIssue
#
#     def _name(i: int, os_name: str | None, issue_name: str | None) -> str:
#         x = issue_name if issue_name else ""
#         if add_os_to_issue_name and os_name:
#             x = f"{os_name}:{x}"
#         if add_index_to_issue_names:
#             x = f"{x}:{i}"
#         return x
#
#     values, names, nissues = [], [], []
#     for i, os in enumerate(outcome_spaces):
#         if isinstance(os, EnumeratingOutcomeSpace):
#             values.append(list(os.enumerate()))
#             names.append(_name(i, "", os.name))
#             nissues.append(1)
#         elif isinstance(os, CartesianOutcomeSpace):
#             for issue in os.issues:
#                 values.append(issue.values)
#                 names.append(_name(i, os.name, issue.name))
#             nissues.append(len(os.issues))
#         else:
#             raise TypeError(
#                 f"Outcome space of type {type(os)} cannot be combined with other outcome-spaces"
#             )
#     return tuple(OptionalIssue(make_issue(v, n), n) for v, n in zip(values, names))
#
#
# @define(frozen=True)
# class DiscreteCartesianOutcomeSpaceProduct(DiscreteCartesianOutcomeSpace):
#     """
#     A discrete outcome-space that is by multiplying multiple discrete outcome spaces
#     """
#
#     issues: tuple[Issue, ...] = field(init=False)  # type: ignore
#     outcome_spaces: tuple[CartesianOutcomeSpace, ...] = field(init=True, default=None)
#     _sizes: tuple[int, ...] = field(init=False, default=None)
#     _extended_sizes: tuple[int, ...] = field(init=False, default=None)
#     _cardinality: int = field(init=False, default=0)
#
#     def __attrs_post_init__(self):
#         object.__setattr__(self, "issues", flat_issues(self.outcome_spaces))
#         for issue in self.issues:
#             if not issue.is_discrete():
#                 raise ValueError(
#                     f"Issue is not discrete. Cannot be added to a DiscreteOutcomeSpace. You must discretize it first: {issue} "
#                 )
#         object.__setattr__(
#             self, "_sizes", tuple(int(_.cardinality) for _ in self.outcome_spaces)
#         )
#         object.__setattr__(self, "_extended_sizes", tuple(_ + 1 for _ in self._sizes))
#         object.__setattr__(self, "_cardinality", reduce(mul, self._extended_sizes, 1))
#
#     @property
#     def cardinality(self) -> int:
#         return self._cardinality
#
#     def cardinality_if_discretized(
#         self, levels: int, max_cardinality: int | float = float("inf")
#     ) -> int:
#         return self._cardinality
#
#     def enumerate(self) -> Iterable[Outcome]:
#         return enumerate_discrete_issues(  #  type: ignore I know that all my issues are actually discrete
#             self.issues  #  type: ignore I know that all my issues are actually discrete
#         )
#
#     def limit_cardinality(
#         self,
#         max_cardinality: int | float = float("inf"),
#         levels: int | float = float("inf"),
#     ) -> DiscreteCartesianOutcomeSpace:
#         """
#         Limits the cardinality of the outcome space to the given maximum (or the number of levels for each issue to `levels`)
#
#         Args:
#             max_cardinality: The maximum number of outcomes in the resulting space
#             levels: The maximum number of levels for each issue/subissue
#         """
#         if self.cardinality <= max_cardinality or all(
#             _.cardinality < levels for _ in self.issues
#         ):
#             return self
#         new_levels = [_.cardinality for _ in self.issues]  # type: ignore will be corrected the next line
#         new_levels = [int(_) if _ < levels else int(levels) for _ in new_levels]
#         new_cardinality = reduce(mul, new_levels, 1)
#
#         def _reduce_total_cardinality(new_levels, max_cardinality, new_cardinality):
#             sort = reversed(sorted((_, i) for i, _ in enumerate(new_levels)))
#             sorted_levels = [_[0] for _ in sort]
#             indices = [_[1] for _ in sort]
#             needed = new_cardinality - max_cardinality
#             current = 0
#             n = len(sorted_levels)
#             while needed > 0 and current < n:
#                 nxt = n - 1
#                 v = sorted_levels[current]
#                 if v == 1:
#                     continue
#                 for i in range(current + 1, n - 1):
#                     if v == sorted_levels[i]:
#                         continue
#                     nxt = i
#                     break
#                 diff = v - sorted_levels[nxt]
#                 if not diff:
#                     diff = 1
#                 new_levels[indices[current]] -= 1
#                 max_cardinality = (max_cardinality // v) * (v - 1)
#                 sort = reversed(sorted((_, i) for i, _ in enumerate(new_levels)))
#                 sorted_levels = [_[0] for _ in sort]
#                 current = 0
#                 needed = new_cardinality - max_cardinality
#             return new_levels
#
#         if new_cardinality > max_cardinality:
#             new_levels: list[int] = _reduce_total_cardinality(
#                 new_levels, max_cardinality, new_cardinality
#             )
#         issues: list[Issue] = []
#         for j, i, issue in zip(
#             new_levels, (_.cardinality for _ in self.issues), self.issues
#         ):
#             issues.append(issue if j >= i else issue.to_discrete(j, compact=True))
#         return DiscreteCartesianOutcomeSpace(
#             tuple(issues), name=f"{self.name}-{max_cardinality}"
#         )
#
#     def is_discrete(self) -> bool:
#         """Checks whether there are no continua components of the space"""
#         return True
#
#     def to_discrete(
#         self, levels: int | float = 10, max_cardinality: int | float = float("inf")
#     ) -> DiscreteCartesianOutcomeSpace:
#         return self
#
#     def to_single_issue(
#         self,
#         numeric=False,
#         stringify=True,
#         levels: int = NLEVELS,
#         max_cardinality: int | float = float("inf"),
#     ) -> DiscreteCartesianOutcomeSpace:
#         """
#         Creates a new outcome space that is a single-issue version of this one
#
#         Args:
#             numeric: If given, the output issue will be a `ContiguousIssue` otherwise it will be a `CategoricalIssue`
#             stringify:  If given, the output issue will have string values. Checked only if `numeric` is `False`
#
#         Remarks:
#             - maps the agenda and ufuns to work correctly together
#             - Only works if the outcome space is finite
#         """
#         outcomes = list(self.enumerate())
#         values = (
#             range(len(outcomes))
#             if numeric
#             else [f"v{_}" for _ in range(len(outcomes))]
#             if stringify
#             else outcomes
#         )
#         issue = (
#             ContiguousIssue(len(outcomes), name="-".join(self.issue_names))
#             if numeric
#             else CategoricalIssue(values, name="-".join(self.issue_names))
#         )
#         return DiscreteCartesianOutcomeSpace(issues=(issue,), name=self.name)
#
#     # def sample(
#     #     self,
#     #     n_outcomes: int,
#     #     with_replacement: bool = False,
#     #     fail_if_not_enough=True,
#     # ) -> Iterable[Outcome]:
#     #     """
#     #     Samples up to n_outcomes with or without replacement.
#     #     """
#     #
#     #     return sample_issues(
#     #         self.issues, n_outcomes, with_replacement, fail_if_not_enough
#     #     )
#     #
#     #     # outcomes = self.enumerate()
#     #     # outcomes = list(outcomes)
#     #     # if with_replacement:
#     #     #     return random.choices(outcomes, k=n_outcomes)
#     #     # if fail_if_not_enough and n_outcomes > self.cardinality:
#     #     #     raise ValueError("Cannot sample enough")
#     #     # random.shuffle(outcomes)
#     #     # return outcomes[:n_outcomes]
#
#     def __iter__(self):
#         return self.enumerate().__iter__()
#
#     def __len__(self) -> int:
#         return self.cardinality
