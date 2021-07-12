import random
from typing import (
    List,
    Optional,
    Type,
)

import numpy as np

from negmas.common import AgentMechanismInterface
from negmas.outcomes import (
    Issue,
    Outcome,
)
from .base import UtilityFunction, UtilityValue
from negmas.helpers import make_range
from negmas.serialization import PYTHON_CLASS_IDENTIFIER
from negmas.helpers import get_full_type_name

__all__ = [
    "ConstUFun",
]


class ConstUFun(UtilityFunction):
    def __init__(
        self,
        value: float,
        name=None,
        reserved_value: UtilityValue = float("-inf"),
        ami: AgentMechanismInterface = None,
        outcome_type: Optional[Type] = None,
        **kwargs,
    ):
        super().__init__(
            name=name,
            outcome_type=outcome_type,
            reserved_value=reserved_value,
            ami=ami,
            **kwargs,
        )
        self.value = value

    def to_dict(self):
        d = {PYTHON_CLASS_IDENTIFIER: get_full_type_name(type(self))}
        return dict(
            value=self.value,
            name=self.name,
            reserved_value=self.reserved_value,
            **d,
        )

    @classmethod
    def from_dict(cls, d):
        d.pop(PYTHON_CLASS_IDENTIFIER, None)
        return cls(
            value=d.get("value", None),
            name=d.get("name", None),
            reserved_value=d.get("reserved_value", None),
            ami=d.get("ami", None),
            outcome_type=d.get("outcome_type", None),
        )

    def eval(self, offer: "Outcome") -> UtilityValue:
        if offer is None:
            return self.reserved_value
        return self.value

    def xml(self, issues: List[Issue]) -> str:
        pass

    def __str__(self):
        return str(self.value)

    @classmethod
    def random(
        cls,
        issues,
        reserved_value=(0.0, 1.0),
        normalized=True,
        value_range=(0.0, 1.0),
        **kwargs,
    ) -> "ExpDiscountedUFun":
        """Generates a random ufun of the given type"""
        reserved_value = make_range(reserved_value)
        value_range = make_range(value_range)
        kwargs["value"] = (
            random.random() * (value_range[1] - value_range[0]) + value_range[0]
        )
        kwargs["reserved_value"] = (
            random.random() * (reserved_value[1] - reserved_value[0])
            + reserved_value[0]
        )
        return cls(**kwargs)
