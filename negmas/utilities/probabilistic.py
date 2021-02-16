import pprint
from typing import (
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
)

import numpy as np

from negmas.common import AgentMechanismInterface
from negmas.generics import iget, ivalues
from negmas.helpers import Distribution, ikeys
from negmas.outcomes import (
    Issue,
    Outcome,
)
from .base import UtilityFunction, UtilityValue, UtilityDistribution
from .nonlinear import MappingUtilityFunction

__all__ = [
    "IPUtilityFunction",
]


class IPUtilityFunction(UtilityFunction):
    """Independent Probabilistic Utility Function.

    Args:

        outcomes: Iterable of outcomes
        distributions: the distributions. One for each outcome
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
        reserved_value: UtilityValue = float("-inf"),
        ami: AgentMechanismInterface = None,
        outcome_type: Optional[Type] = None,
        id = None,
    ):
        super().__init__(
            name=name, outcome_type=outcome_type, reserved_value=reserved_value, ami=ami, id=id,
        )
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
                u.mapping,
                range=range,
                uncertainty=uncertainty,
                variability=variability,
                reserved_value=u.reserved_value,
            )
        return cls.from_mapping(
            dict(zip(ikeys(u.mapping), ivalues(u.mapping))),
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
            reserved_value=reserved_value,
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

    def eval(self, offer: "Outcome") -> UtilityValue:
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
