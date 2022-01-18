from __future__ import annotations

import pprint
from typing import Dict, Iterable, List, Tuple

import numpy as np

from negmas.generics import iget, ivalues
from negmas.helpers import ScipyDistribution
from negmas.outcomes import Issue, Outcome

from ..helpers.prob import (
    Distribution,
    ScipyDistribution,
    as_distribution,
    uniform_around,
)
from .mapping import MappingUtilityFunction
from .ufun import ProbUtilityFunction, UtilityFunction

__all__ = ["IPUtilityFunction", "ILSUtilityFunction", "UniformUtilityFunction"]


class IPUtilityFunction(ProbUtilityFunction):
    """Independent Probabilistic Utility Function.

    Args:

        outcomes: Iterable of outcomes
        distributions: the distributions. One for each outcome
        name: ufun name

    Examples:

        >>> outcomes = [('o1',), ('o2',)]
        >>> f = IPUtilityFunction(outcomes=outcomes
        ...         , distributions=[Distribution(type='uniform', loc=0.0, scale=0.5)
        ...         , Distribution(type='uniform', loc=0.1, scale=0.5)])
        >>> str(f(('o1',)))
        'U(0.0, 0.5)'

    """

    def __init__(
        self,
        outcomes: Iterable[Outcome],
        distributions: Iterable[Distribution] = None,
        issue_names: Iterable[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        outcomes = list(outcomes)
        distributions = (
            list(distributions)
            if distributions is not None
            else [
                ScipyDistribution(type="uniform", loc=0.0, scale=1.0)
                for _ in range(len(outcomes))
            ]
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
        self.distributions = dict(zip(outcomes, distributions))

    def distribution(self, outcome: "Outcome") -> Distribution:
        """
        Returns the distributon associated with a specific outcome
        Args:
            outcome:

        Returns:

        """
        return self.distributions[self.key(outcome)]

    @classmethod
    def from_preferences(
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
            >>> p = IPUtilityFunction.from_preferences(u, uncertainty=0.0)
            >>> print(p)
            {('o1',): 0.3, ('o2',): 0.7}

            - Full uncertainty
            >>> u = MappingUtilityFunction(mapping=dict(zip([('o1',), ('o2',)], [0.3, 0.7])))
            >>> p = IPUtilityFunction.from_preferences(u, uncertainty=1.0)
            >>> print(p)
            {('o1',): U(0.0, 1.0), ('o2',): U(0.0, 1.0)}

            - some uncertainty
            >>> u = MappingUtilityFunction(mapping=dict(zip([('o1',), ('o2',)], [0.3, 0.7])))
            >>> p = IPUtilityFunction.from_preferences(u, uncertainty=0.1)
            >>> print([_.scale for _ in p.distributions.values()])
            [0.1, 0.1]
            >>> for k, v in p.distributions.items():
            ...     assert v.loc <= u(k)


        Returns:
            a new IPUtilityFunction
        """
        if isinstance(u.mapping, dict):
            return cls.from_mapping(
                u.mapping,
                range=range,
                uncertainty=uncertainty,
                variability=variability,
                reserved_value=u.reserved_value,
            )
        if not u.outcome_space.is_discrete():
            raise ValueError(
                "Cannot be constructed from a ufun with an infinite outcome space"
            )
        outcomes = u.outcome_space.enumerate()  # type: ignore (I know that it is a discrete space)
        d = dict(zip(outcomes, (u(_) for _ in outcomes)))
        return cls.from_mapping(
            d,
            range=range,
            uncertainty=uncertainty,
            variability=variability,
            reserved_value=u.reserved_value,
        )

    @classmethod
    def from_mapping(
        cls,
        mapping: Dict["Outcome", float],
        range: Tuple[float, float] = (0.0, 1.0),
        uncertainty: float = 0.5,
        variability: float = 0.0,
        reserved_value: float = float("-inf"),
    ) -> "IPUtilityFunction":
        """
        Generates a distribution from which `u` may have been sampled
        Args:
            mapping: mapping from outcomes to float values
            range: range of the utility_function values
            uncertainty: uncertainty level
            variability: The variability within the ufun
            reserved_value: The reserved value

        Examples:

            - No uncertainty
            >>> mapping = dict(zip([('o1',), ('o2',)], [0.3, 0.7]))
            >>> p = IPUtilityFunction.from_mapping(mapping, uncertainty=0.0)
            >>> print(p)
            {('o1',): 0.3, ('o2',): 0.7}

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
                uniform_around(value=mapping[o], uncertainty=u, range=range)
                for o, u in zip(outcomes, uncertainties)
            ],
            reserved_value=reserved_value,
        )

    def is_non_stationary(self):
        return True

    def __str__(self):
        return pprint.pformat(self.distributions)

    def sample(self) -> MappingUtilityFunction:
        """
        Samples the utility_function distribution to create a mapping utility function


        Examples:
            >>> import random
            >>> f = IPUtilityFunction(outcomes=[('o1',), ('o2',)]
            ...         , distributions=[Distribution(type='uniform', loc=0.0, scale=0.2)
            ...         , Distribution(type='uniform', loc=0.4, scale=0.5)])
            >>> u = f.sample()
            >>> assert u(('o1',)) <= 0.2
            >>> assert 0.4 <= u(('o2',)) <= 0.9

        Returns:

            MappingUtilityFunction
        """
        return MappingUtilityFunction(
            mapping={o: d.sample(1)[0] for o, d in self.distributions.items()}
        )

    def key(self, outcome: Outcome):
        """
        Returns the key of the given outcome in self.distributions.

        Args:
            outcome:

        Returns:
            tuple

        """
        return outcome

    def eval(self, offer: "Outcome") -> Distribution:
        """Calculate the utility_function value for a given outcome.

        Args:
            offer: The offer to be evaluated.


        Remarks:
            - You cannot return None from overriden apply() functions but raise an exception (ValueError) if it was
              not possible to calculate the Distribution.
            - Return A Distribution not a float for real-valued utilities for the benefit of inspection code.

        """
        if offer is None:
            return as_distribution(self.reserved_value)
        if self.tupelized and not isinstance(offer, tuple):
            offer = tuple(ivalues(offer))
        return self.distributions[offer]

    def xml(self, issues: List[Issue]) -> str:
        raise NotImplementedError(f"Cannot convert {self.__class__.__name__} to xml")


class ILSUtilityFunction(ProbUtilityFunction):
    """
    A utility function which represents the loc and scale deviations as any crisp ufun
    """

    def __init__(
        self, type: str, loc: UtilityFunction, scale: UtilityFunction, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._type = type
        self.loc = loc
        self.scale = scale

    def eval(self, offer: Outcome) -> ScipyDistribution:
        loc, scale = self.loc(offer), self.scale(offer)
        if loc is None or scale is None:
            raise ValueError(
                f"Cannot calculate loc ({loc}) or scale ({scale}) for offer {offer}"
            )
        return ScipyDistribution(self._type, loc=loc, scale=scale)


class UniformUtilityFunction(ILSUtilityFunction):
    """
    A utility function which represents the loc and scale deviations as any crisp ufun
    """

    def __init__(self, loc: UtilityFunction, scale: UtilityFunction, *args, **kwargs):
        super().__init__("uniform", loc, scale, *args, *kwargs)


class GaussialUtilityFunction(ILSUtilityFunction):
    """
    A utility function which represents the mean and std deviations as any crisp ufun
    """

    def __init__(self, loc: UtilityFunction, scale: UtilityFunction, *args, **kwargs):
        super().__init__("norm", loc, scale, *args, *kwargs)
