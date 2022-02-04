from __future__ import annotations

from negmas.helpers import ScipyDistribution
from negmas.outcomes import Outcome

from ...helpers.prob import ScipyDistribution
from ..crisp_ufun import UtilityFunction
from ..mixins import StationaryMixin
from ..prob_ufun import ProbUtilityFunction

__all__ = ["ILSUtilityFunction", "UniformUtilityFunction"]


class ILSUtilityFunction(ProbUtilityFunction, StationaryMixin):
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


class UniformUtilityFunction(ILSUtilityFunction, StationaryMixin):
    """
    A utility function which represents the loc and scale deviations as any crisp ufun
    """

    def __init__(self, loc: UtilityFunction, scale: UtilityFunction, *args, **kwargs):
        super().__init__("uniform", loc, scale, *args, *kwargs)


class GaussialUtilityFunction(ILSUtilityFunction, StationaryMixin):
    """
    A utility function which represents the mean and std deviations as any crisp ufun
    """

    def __init__(self, loc: UtilityFunction, scale: UtilityFunction, *args, **kwargs):
        super().__init__("norm", loc, scale, *args, *kwargs)
