from __future__ import annotations

from .agent import Agent
from .entity import Entity

__all__ = ["Adapter"]


class Adapter(Agent):
    """
    Represents an adapter agent that makes some included object act as an
    agent in some world.

    Args:

        obj: The object to be adapted.
        include_adapter_type_name: Whether to include the adapter type name. If
                                   None, then it will be included if it does
                                   not start with and underscore.
        include_obj_type_name: Whether to include object type name in this
                               adapter's type name
        type_postfix: A string to add to the end of the type name

    Remarks:

        - Other than keeping an internal copy of the adapted object under
          `obj`, this class is used primarily to provide a way to give
          good type_name and short_type_name properties that combine the name
          of the adapter and the name of the enclosed object nicely.
        - The adapted object must be an `Entity`.
        - The `World` class uses the type names from this adapter whenever
          it needs to get a type-name (either `type_name` or `short_type_name`)
    """

    def __init__(
        self,
        obj,
        include_adapter_type_name: bool | None = None,
        include_obj_type_name=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if include_adapter_type_name is None:
            include_adapter_type_name = not self.__class__.__name__.startswith("_")

        self._obj = obj
        self._include_adapter, self._include_obj = (
            include_adapter_type_name,
            include_obj_type_name,
        )

    @property
    def adapted_object(self) -> Entity:
        return self._obj

    @adapted_object.setter
    def adapted_object(self, x: Entity) -> Entity:
        self._obj = x
        return x

    @property
    def short_type_name(self):
        """Returns a short name of the type of this entity"""
        base = super().short_type_name if self._include_adapter else ""
        obj = self._obj.short_type_name if self._include_obj else ""
        return base + ":" + obj

    @property
    def type_name(self):
        """Returns a short name of the type of this entity"""
        base = super().type_name if self._include_adapter else ""
        obj = self._obj.type_name if self._include_obj else ""
        return base + ":" + obj

    def init(self):
        """Override this method to modify initialization logic"""
        self._obj.init()

    def step(self):
        """Override this method to modify stepping logic"""
        self._obj.step()

    def __getattr__(self, attr):
        return getattr(self._obj, attr)
