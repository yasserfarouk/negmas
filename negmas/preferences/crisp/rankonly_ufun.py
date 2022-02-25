from __future__ import annotations

import math
from itertools import chain
from typing import TYPE_CHECKING

from ..crisp_ufun import UtilityFunction
from ..mixins import StationaryMixin
from .mapping import MappingUtilityFunction

if TYPE_CHECKING:
    from negmas.outcomes import Outcome

    from ..base_ufun import BaseUtilityFunction

__all__ = ["RankOnlyUtilityFunction"]


class RankOnlyUtilityFunction(StationaryMixin, UtilityFunction):
    r"""
    A utility function that keeps trak of outcome order onlyself.

    Remarks:
        - This type of utility function can only be generated for discrete outcome spaces.
        - It can be constructed from any utility function.
        - Given an outcome space of $K$ outcomes, the outcome with maximum utility will have a value $\le K$ when evaluated and the worst outcome will have value $0$.
        - Outcomes with equal utility value will be ordered randomly if `randomize_eqial` is `True`, otherwise outcomes that are `eps` apart will have the same rank.
        - The reserved value will also be mapped to an integer giving its rank

    """

    def eval(self, outcome: Outcome) -> int | None:
        return self._mapping.get(outcome, None)

    def __init__(
        self,
        ufun: BaseUtilityFunction,
        randomize_equal: bool = False,
        eps=1e-7,
        name: str = None,
        id: str = None,
        type_name: str = None,
    ):
        if ufun.outcome_space is None:
            raise ValueError(
                f"Cannot craerte a RankOnly utility function for the given ufun because the outcome space is not konwn"
            )
        if not ufun.outcome_space.is_finite():
            raise ValueError(
                f"Cannot create a RankOnly utility function for the given ufun because the outcome space is not discrete\n{ufun.outcome_space}"
            )
        super().__init__(
            outcome_space=ufun.outcome_space, name=name, id=id, type_name=type_name
        )
        if math.isinf(ufun.reserved_value):
            outcomes = ufun.outcome_space.enumerate_or_sample()
        else:
            outcomes = chain(ufun.outcome_space.enumerate_or_sample(), [None])
        ordered: list[tuple[float, Outcome | None]] = [
            (float(ufun(_)), _) for _ in outcomes
        ]
        ordered = sorted(ordered, key=lambda x: x[0], reverse=True)
        ranked: list[tuple[Outcome | None, int]]
        if randomize_equal:
            n = ufun.outcome_space.cardinality - int(math.isinf(ufun.reserved_value))
            ranked = [(w, int(n - i)) for i, (_, w) in enumerate(ordered)]
        else:
            ranked = []
            last_ufun, current = float("inf"), -1
            for u, o in ordered:
                if abs(last_ufun - u) > eps:
                    current += 1
                last_ufun = u
                ranked.append((o, current))
            for i, (o, r) in enumerate(ranked):
                ranked[i] = (o, current - r)

        self._mapping = dict(ranked)
        if math.isinf(ufun.reserved_value):
            self.reserved_value = ufun.reserved_value
        else:
            self.reserved_value = self.eval(None)  # type: ignore

    def to_mapping_ufun(self) -> MappingUtilityFunction:
        return MappingUtilityFunction(self._mapping, outcome_space=self.outcome_space, reserved_value=self.reserved_value)  # type: ignore
