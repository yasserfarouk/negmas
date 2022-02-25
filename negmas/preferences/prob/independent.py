from __future__ import annotations

import pprint
from typing import TYPE_CHECKING, Iterable

import numpy as np

from negmas.generics import iget, ikeys, ivalues
from negmas.helpers.prob import ScipyDistribution
from negmas.outcomes import Issue, Outcome

from ...helpers.prob import (
    Distribution,
    ScipyDistribution,
    make_distribution,
    uniform_around,
)
from ..crisp.mapping import MappingUtilityFunction
from ..mixins import StationaryMixin
from ..prob_ufun import ProbUtilityFunction
from .mapping import ProbMappingUtilityFunction

if TYPE_CHECKING:
    from ..base import Value

__all__ = ["IPUtilityFunction"]


class IPUtilityFunction(StationaryMixin, ProbUtilityFunction):
    """
    Independent Probabilistic Utility Function.

    Args:

        outcomes: Iterable of outcomes
        distributions: the distributions. One for each outcome
        name: ufun name

    Examples:

        >>> outcomes = [('o1',), ('o2',)]
        >>> f = IPUtilityFunction(outcomes=outcomes
        ...         , distributions=[ScipyDistribution(type='uniform', loc=0.0, scale=0.5)
        ...         , ScipyDistribution(type='uniform', loc=0.1, scale=0.5)])
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

    @classmethod
    def random(
        cls,
        outcome_space,
        reserved_value,
        normalized=True,
        dist_limits=(0.0, 1.0),
        **kwargs,
    ) -> IPUtilityFunction:
        """Generates a random ufun of the given type"""
        raise NotImplementedError("random hyper-rectangle ufuns are not implemented")

    def key(self, outcome: Outcome):
        """
        Returns the key of the given outcome in self.distributions.

        Args:
            outcome:

        Returns:
            tuple

        """
        return outcome

    def distribution(self, outcome: Outcome) -> Distribution:
        """
        Returns the distributon associated with a specific outcome
        Args:
            outcome:

        Returns:

        """
        return self.distributions[self.key(outcome)]

    @classmethod
    def from_mapping(
        cls,
        mapping: dict[Outcome, Value],
        range: tuple[float, float] = (0.0, 1.0),
        uncertainty: float = 0.5,
        variability: float = 0.0,
        reserved_value: float = float("-inf"),
    ) -> IPUtilityFunction:
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
                uniform_around(value=float(mapping[o]), uncertainty=u, range=range)
                for o, u in zip(outcomes, uncertainties)
            ],
            reserved_value=reserved_value,
        )

    @classmethod
    def from_preferences(
        cls,
        u: ProbMappingUtilityFunction,
        range: tuple[float, float] = (0.0, 1.0),
        uncertainty: float = 0.5,
        variability: float = 0.0,
    ) -> IPUtilityFunction:
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
        if not u.outcome_space:
            raise ValueError("Unknown outcome space")
        if not u.outcome_space.is_discrete():
            raise ValueError(
                "Cannot be constructed from a ufun with an infinite outcome space"
            )
        outcomes = u.outcome_space.enumerate()  # type: ignore (I know that it is a discrete space)
        d = dict(zip(outcomes, (u(_) for _ in outcomes)))
        return cls.from_mapping(
            d,  # type: ignore
            range=range,
            uncertainty=uncertainty,
            variability=variability,
            reserved_value=u.reserved_value,
        )

    def is_state_dependent(self):
        return True

    def sample(self) -> MappingUtilityFunction:
        """
        Samples the utility_function distribution to create a mapping utility function


        Examples:
            >>> import random
            >>> f = IPUtilityFunction(outcomes=[('o1',), ('o2',)]
            ...         , distributions=[ScipyDistribution(type='uniform', loc=0.0, scale=0.2)
            ...         , ScipyDistribution(type='uniform', loc=0.4, scale=0.5)])
            >>> u = f.sample()
            >>> assert u(('o1',)) <= 0.2
            >>> assert 0.4 <= u(('o2',)) <= 0.9

        Returns:

            MappingUtilityFunction
        """
        return MappingUtilityFunction(
            mapping={o: d.sample(1)[0] for o, d in self.distributions.items()}
        )

    def eval(self, offer: Outcome) -> Distribution:
        """Calculate the utility_function value for a given outcome.

        Args:
            offer: The offer to be evaluated.


        Remarks:
            - You cannot return None from overriden apply() functions but raise an exception (ValueError) if it was
              not possible to calculate the Distribution.
            - Return A Distribution not a float for real-valued utilities for the benefit of inspection code.

        """
        if offer is None:
            return make_distribution(self.reserved_value)
        if self.tupelized and not isinstance(offer, tuple):
            offer = tuple(ivalues(offer))
        return self.distributions[offer]

    def xml(self, issues: list[Issue]) -> str:
        raise NotImplementedError(f"Cannot convert {self.__class__.__name__} to xml")

    def __str__(self):
        return pprint.pformat(self.distributions)
