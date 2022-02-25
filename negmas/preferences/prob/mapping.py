from __future__ import annotations

import random
from typing import Iterable

import numpy as np

from negmas.common import Distribution
from negmas.generics import gmap
from negmas.helpers import get_full_type_name
from negmas.helpers.prob import ScipyDistribution
from negmas.outcomes import Issue, Outcome
from negmas.outcomes.protocols import OutcomeSpace
from negmas.serialization import PYTHON_CLASS_IDENTIFIER, deserialize, serialize

from ..base import OutcomeUtilityMapping
from ..mixins import StationaryMixin
from ..prob_ufun import ProbUtilityFunction

__all__ = ["ProbMappingUtilityFunction"]


class ProbMappingUtilityFunction(StationaryMixin, ProbUtilityFunction):
    """
    Outcome mapping utility function.

    This is the simplest possible utility function and it just maps a set of `Outcome`s to a set of
    `Value`(s). It is only usable with single-issue negotiations. It can be constructed with wither a mapping
    (e.g. a dict) or a callable function.

    Args:
            mapping: Either a callable or a mapping from `Outcome` to `Value`.
            default: value returned for outcomes causing exception (e.g. invalid outcomes).
            name: name of the utility function. If None a random name will be generated.
            reserved_value: The reserved value (utility of not getting an agreement = utility(None) )

    Remarks:
        - If the mapping used failed on the outcome (for example because it is not a valid outcome), then the
        ``default`` value given to the constructor (which defaults to None) will be returned.

    """

    def __init__(
        self,
        mapping: OutcomeUtilityMapping,
        default=None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.mapping = mapping
        self.default = default

    def to_dict(self):
        d = {PYTHON_CLASS_IDENTIFIER: get_full_type_name(type(self))}
        d.update(super().to_dict())
        return dict(
            **d,
            mapping=serialize(self.mapping),
            default=self.default,
        )

    @classmethod
    def from_dict(cls, d):
        d.pop(PYTHON_CLASS_IDENTIFIER, None)
        d["mapping"] = deserialize(d["mapping"])
        return cls(**d)

    def eval(self, offer: Outcome | None) -> Distribution | float | None:
        # noinspection PyBroadException
        if offer is None:
            return self.reserved_value
        try:
            m = gmap(self.mapping, offer)
        except Exception:
            return self.default

        return m

    def xml(self, issues: list[Issue]) -> str:
        raise NotImplementedError(f"Cannot save ProbMappingUtilityFunction to xml")

    @classmethod
    def random(
        cls,
        outcome_space: OutcomeSpace,
        reserved_value=(0.0, 1.0),
        normalized=True,
        max_cardinality: int = 10000,
        type: str = "uniform",
    ):
        # todo: corrrect this for continuous outcome-spaces
        if not isinstance(reserved_value, Iterable):
            reserved_value = (reserved_value, reserved_value)
        os = outcome_space.to_largest_discrete(
            levels=10, max_cardinality=max_cardinality
        )
        mn, rng = 0.0, 1.0
        if not normalized:
            mn = 4 * random.random()
            rng = 4 * random.random()
        locs = np.random.rand(os.cardinality) * rng + mn
        scales = np.random.rand(os.cardinality) * rng + mn
        return cls(
            dict(
                zip(
                    os.enumerate(),
                    (
                        ScipyDistribution(type=type, loc=l, scale=s)
                        for l, s in zip(locs, scales)
                    ),
                )
            ),
            reserved_value=reserved_value[0]  # type: ignore
            + random.random() * (reserved_value[1] - reserved_value[0]),  # type: ignore
        )

    def __str__(self) -> str:
        return f"mapping: {self.mapping}\ndefault: {self.default}"
