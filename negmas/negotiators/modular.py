from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Iterable

from negmas.common import (
    MechanismState,
    NegotiatorMechanismInterface,
    PreferencesChange,
)

from .components.component import Component
from .negotiator import Negotiator

if TYPE_CHECKING:
    from negmas.preferences import BaseUtilityFunction, Preferences

__all__ = [
    "ModularNegotiator",
]


class ModularNegotiator(Negotiator):
    """
    A generic modular negotiator that can combine multiple negotiation `Component` s.


    This class simply holds a list of components and call them on every event.
    """

    def insert_component(
        self,
        component: Component,
        name: str | None = None,
        index: int = -1,
        override: bool = False,
    ) -> None:
        """
        Adds a component at the given index. If a negative number is given, appends at the end
        """
        if (
            component.negotiator is not None
            and id(component.negotiator) != id(self)
            and not override
        ):
            raise ValueError(
                f"Component {component} already has as parent {component.negotiator.id} that is not the same as this negotiator {self.id}. Cannot add it. To override pass `overrride=True`"
            )
        component.set_negotiator(self)
        if index < 0:
            index = len(self._components)
        self._components.insert(index, component)
        if not name:
            name = str(index)
        self.__component_map[name] = len(self._components)

    def __init__(
        self,
        *args,
        components: Iterable[Component],
        component_names: Iterable[str] | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._components: list[Component] = []
        self.__component_map: dict[str, int] = dict()
        for c, n in zip(
            components, component_names if component_names else itertools.repeat(None)
        ):
            self.insert_component(c, name=n)

    @property
    def components(self) -> tuple[Component, ...]:
        return tuple(self._components)

    def remove_component_at(self, index: int) -> None:
        """
        Removes the component at the givne index.
        """
        self._components = self._components[:index] + self._components[index + 1 :]

    def remove_component(self, name: str) -> None:
        """
        Removes the component with the given name
        """
        self.remove_component_at(self.__component_map[name])

    def on_preferences_changed(self, changes: list[PreferencesChange]):
        """
        Called to inform the component that the ufun has changed and the kinds of change that happened.
        """
        for c in self._components:
            c.on_preferences_changed(changes)

    def join(
        self,
        nmi: NegotiatorMechanismInterface,
        state: MechanismState,
        *,
        preferences: Preferences | None = None,
        ufun: BaseUtilityFunction | None = None,
        role: str = "negotiator",
    ) -> bool:
        if not all(
            _.can_join(nmi, state, preferences=preferences, ufun=ufun, role=role)
            for _ in self._components
        ):
            return False
        joined = super().join(nmi, state, preferences=preferences, ufun=ufun, role=role)
        if not joined:
            return False
        for c in self._components:
            c.after_join(nmi)
        return joined

    def on_negotiation_start(self, state: MechanismState) -> None:
        """
        A call back called at each negotiation start
        """
        for c in self._components:
            c.on_negotiation_start(state)

    def on_round_start(self, state: MechanismState) -> None:
        """
        A call back called at each negotiation round start
        """
        for c in self._components:
            c.on_round_start(state)

    def on_round_end(self, state: MechanismState) -> None:
        """
        A call back called at each negotiation round end
        """
        for c in self._components:
            c.on_round_end(state)

    def on_leave(self, state: MechanismState) -> None:
        """
        A call back called after leaving a negotiation.
        """
        for c in self._components:
            c.on_leave(state)

    def on_negotiation_end(self, state: MechanismState) -> None:
        """
        A call back called at each negotiation end
        """
        for c in self._components:
            c.on_negotiation_end(state)

    def on_mechanism_error(self, state: MechanismState) -> None:
        """
        A call back called whenever an error happens in the mechanism. The error and its explanation are accessible in
        `state`
        """
        for c in self._components:
            c.on_mechanism_error(state)
