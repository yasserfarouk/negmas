from __future__ import annotations

import functools
from typing import Callable

from .negotiator import Negotiator

__all__ = [
    "ControlledNegotiator",
]


class ControlledNegotiator(Negotiator):
    """
    A negotiator that can be used to pass all method calls to a parent (Controller).

    It uses magic dunder methods to implement a general way of passing calls to the parent. This method is slow.

    It is recommended to implement a `ControlledNegotiator` for each mechanism that does this passing explicitly which
    will be much faster.

    For an example, see the implementation of `ControlledSAONegotiator` .

    """

    def __getattribute__(self, item):
        if item in (
            "id",
            "name",
            "on_preferences_changed",
            "has_preferences",
            "preferences",
            "reserved_value",
        ) or item.startswith("_"):
            return super().__getattribute__(item)
        parent = super().__getattribute__("__dict__").get("_Negotiator__parent", None)
        if parent is None:
            return super().__getattribute__(item)
        attr = getattr(parent, item, None)
        if attr is None:
            return super().__getattribute__(item)
        if isinstance(attr, Callable):
            return functools.partial(
                attr,
                negotiator_id=super().__getattribute__("__dict__")[
                    "_NamedObject__uuid"
                ],
            )
        return super().__getattribute__(item)
