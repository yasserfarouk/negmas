from __future__ import annotations

__all__ = ["Entity"]


class Entity:
    """Defines an entity that is a part of the world but does not participate in the simulation"""

    def __init__(self, type_postfix: str = ""):
        self._initialized = False
        self.__type_postfix = type_postfix
        self.__current_step = 0

    def _shorten(self, long_name):
        name = (
            long_name.split(".")[-1]
            .lower()
            .replace("factory_manager", "")
            .replace("manager", "")
        )
        name = (
            name.replace("factory", "")
            .replace("agent", "")
            .replace("miner", "")
            .replace("consumer", "")
        )
        if long_name.startswith("jnegmas"):
            name = f"j-{name}"
        name = name.strip("_")
        return name

    @classmethod
    def _type_name(cls):
        return cls.__module__ + "." + cls.__name__

    @property
    def type_name(self):
        """Returns the name of the type of this entity"""
        return self.__class__._type_name() + self.__type_postfix

    @property
    def short_type_name(self):
        """Returns a short name of the type of this entity"""
        return self._shorten(self.type_name)

    @property
    def type_postfix(self):
        return self.__type_postfix

    def init(self):
        """Override this method to modify initialization logic"""

    def init_(self):
        """Called to initialize the agent **after** the world is initialized. the AWI is accessible at this point."""
        self._initialized = True
        self.__current_step = 0
        self.init()

    def step(self):
        """Override this method to modify stepping logic"""

    def step_(self):
        """Called at every time-step. This function is called directly by the world."""
        if not self._initialized:
            self.init_()
        self.step()
        self.__current_step += 1
