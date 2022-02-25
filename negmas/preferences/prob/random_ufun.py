from __future__ import annotations

from random import choice, random
from typing import TYPE_CHECKING

from negmas.common import Distribution
from negmas.helpers.prob import ScipyDistribution

from ..mixins import StationaryMixin
from ..prob_ufun import ProbUtilityFunction

if TYPE_CHECKING:
    from negmas.outcomes import Outcome

__all__ = ["ProbRandomUtilityFunction"]


class ProbRandomUtilityFunction(StationaryMixin, ProbUtilityFunction):
    """A random utility function for a discrete outcome space"""

    def __init__(
        self,
        locs: tuple[float, float] = (0.0, 1.0),
        scales: tuple[float, float] = (0.0, 1.0),
        types: tuple[str, ...] = ("uniform",),
        *args,
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
        )
        self._cache: dict[Outcome | None, tuple[float, float, str]] = dict()
        self._types = [types] if isinstance(types, str) else types
        self._loc_scale = locs[1] - locs[0]
        self._loc_offset = locs[0]
        self._scale_scale = scales[1] - scales[0]
        self._scale_offset = scales[0]

    def eval(self, offer: Outcome | None) -> Distribution:
        v = self._cache.get(offer, None)
        if v is None:
            loc = self._loc_offset + self._loc_scale * random()
            scale = self._scale_offset + self._scale_scale * random()
            typ = choice(self._types)
            self._cache[offer] = (loc, scale, typ)
        else:
            loc, scale, typ = v
        return ScipyDistribution(type=typ, loc=loc, scale=scale)
