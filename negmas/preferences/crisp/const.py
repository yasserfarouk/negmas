from __future__ import annotations

import random

from negmas.helpers import get_full_type_name
from negmas.helpers.numeric import make_range
from negmas.outcomes import Issue, Outcome
from negmas.serialization import PYTHON_CLASS_IDENTIFIER

from ..base import Value
from ..crisp_ufun import UtilityFunction
from ..mixins import StationaryMixin

__all__ = ["ConstUtilityFunction"]


class ConstUtilityFunction(StationaryMixin, UtilityFunction):
    """
    A utility function that returns the same value for all outcomes.

    This type of ufun can be considered a special type of `LinearUtilityFunction` with zero slop
    but it is applicable to any type of issue not only numeric ones.
    """

    def __init__(
        self,
        value: Value,
        *,
        reserved_value: float = float("-inf"),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.reserved_value = reserved_value
        self.value = value

    def to_dict(self):
        d = {PYTHON_CLASS_IDENTIFIER: get_full_type_name(type(self))}
        d.update(super().to_dict())
        return dict(
            value=self.value,
            **d,
        )

    @classmethod
    def from_dict(cls, d):
        d.pop(PYTHON_CLASS_IDENTIFIER, None)
        return cls(**d)

    def eval(self, offer: Outcome) -> Value:
        if offer is None:
            return self.reserved_value
        return self.value

    def xml(self, issues: list[Issue]) -> str:
        from negmas.preferences.crisp.linear import AffineUtilityFunction

        return AffineUtilityFunction(
            [0.0] * len(issues),
            float(self.value),
            issues=issues,
            name=self.name,
            id=self.id,
        ).xml(issues)

    @classmethod
    def random(
        cls,
        issues,
        reserved_value=(0.0, 1.0),
        normalized=True,
        value_range=(0.0, 1.0),
        **kwargs,
    ) -> ConstUtilityFunction:
        """Generates a random ufun of the given type"""
        reserved_value = make_range(reserved_value)
        value_range = make_range(value_range)
        if normalized:
            kwargs["value"] = 1.0
        else:
            kwargs["value"] = (
                random.random() * (value_range[1] - value_range[0]) + value_range[0]
            )
        kwargs["reserved_value"] = (
            random.random() * (reserved_value[1] - reserved_value[0])
            + reserved_value[0]
        )
        return cls(**kwargs)

    def __str__(self):
        return str(self.value)
