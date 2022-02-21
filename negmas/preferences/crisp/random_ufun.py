from __future__ import annotations

from random import random
from typing import TYPE_CHECKING

from ..crisp_ufun import UtilityFunction
from ..mixins import StationaryMixin

if TYPE_CHECKING:
    from negmas.outcomes import Outcome

__all__ = ["RandomUtilityFunction"]


class RandomUtilityFunction(StationaryMixin, UtilityFunction):
    """A random utility function for a discrete outcome space"""

    def __init__(
        self,
        rng: tuple[float, float] = (0.0, 1.0),
        *args,
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
        )
        self._cache: dict[Outcome | None, float] = dict()
        self._scale = rng[1] - rng[0]
        self._offset = rng[0]

    def eval(self, offer: Outcome | None) -> float:
        v = self._cache.get(offer, None)
        if v is None:
            v = self._offset + self._scale * random()
            self._cache[offer] = v
        return v
