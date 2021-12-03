import random
from typing import List

from negmas.helpers import get_full_type_name, make_range
from negmas.outcomes import Issue, Outcome
from negmas.preferences.protocols import IndIssues, StationaryUFun
from negmas.protocols import XmlSerializable
from negmas.serialization import PYTHON_CLASS_IDENTIFIER

from .base import UtilityValue
from .ufun import UtilityFunction

__all__ = ["ConstUFun"]


class ConstUFun(IndIssues, UtilityFunction, XmlSerializable, StationaryUFun):
    def __init__(
        self,
        value: UtilityValue,
        *,
        reserved_value: float = float("-inf"),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.reserved_value = reserved_value
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
        )

    def eval(self, offer: "Outcome") -> UtilityValue:
        if offer is None:
            return self.reserved_value
        return self.value

    def xml(self, issues: List[Issue]) -> str:
        from negmas.preferences.linear import LinearUtilityFunction

        return LinearUtilityFunction(
            [0.0] * len(issues),
            float(self.value),
            issues=self.issues,
            name=self.name,
            id=self.id,
        ).xml(issues)

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
    ) -> "ConstUFun":
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
