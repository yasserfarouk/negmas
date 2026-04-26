"""Module for locscale functionality."""

from __future__ import annotations

from negmas.helpers.prob import ScipyDistribution
from negmas.outcomes import Outcome

from ..crisp_ufun import UtilityFunction
from ..prob_ufun import ProbUtilityFunction

__all__ = ["ILSUtilityFunction", "UniformUtilityFunction"]


class ILSUtilityFunction(ProbUtilityFunction):
    """
    A utility function which represents the loc and scale deviations as any crisp ufun
    """

    def __init__(
        self, type: str, loc: UtilityFunction, scale: UtilityFunction, *args, **kwargs
    ):
        """Initialize the instance.

        Args:
            type: The type of scipy distribution (e.g. 'uniform', 'norm', 'beta').
            loc: Utility function that evaluates to the location parameter of the distribution.
            scale: Utility function that evaluates to the scale parameter of the distribution.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self._type = type
        self.loc = loc
        self.scale = scale

    def eval(self, offer: Outcome) -> ScipyDistribution:
        """Eval.

        Args:
            offer: Offer being considered.

        Returns:
            ScipyDistribution: The result.
        """
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
        """Initializes the instance."""
        super().__init__("uniform", loc, scale, *args, *kwargs)


class GaussianUtilityFunction(ILSUtilityFunction):
    """
    A utility function which represents the mean and std deviations as any crisp ufun
    """

    def __init__(self, loc: UtilityFunction, scale: UtilityFunction, *args, **kwargs):
        """Initializes the instance."""
        super().__init__("norm", loc, scale, *args, *kwargs)


NormalUtilityFunction = GaussianUtilityFunction
"""
An alias for `GaussianUtilityFunction`
"""
