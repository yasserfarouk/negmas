"""Negotiators that delegate behavior to a parent controller."""

from __future__ import annotations

import functools
from typing import Callable

from .negotiator import Negotiator

__all__ = ["ControlledNegotiator"]

#: Attribute names that must resolve on the negotiator itself rather than being
#: delegated to the parent controller. A frozenset gives O(1) membership for
#: this very hot ``__getattribute__`` path (called on every attribute access).
_NOT_DELEGATED = frozenset(
    (
        "id",
        "name",
        "on_preferences_changed",
        "has_preferences",
        "preferences",
        "ufun",
        "opponent_ufun",
        "reserved_value",
        "nmi",
        "ami",
        "owner",
        "annotation",
        "private_info",
        "parent",
        "capabilities",
    )
)


class ControlledNegotiator(Negotiator):
    """
    A negotiator that can be used to pass all method calls to a parent (Controller).

    It uses magic dunder methods to implement a general way of passing calls to the parent. This method is slow.

    It is recommended to implement a `ControlledNegotiator` for each mechanism that does this passing explicitly which
    will be much faster.

    For an example, see the implementation of `ControlledSAONegotiator` .

    """

    def __getattribute__(self, item):
        """getattribute  .

        Args:
            item: Item.
        """
        if item.startswith("_") or item in _NOT_DELEGATED:
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
