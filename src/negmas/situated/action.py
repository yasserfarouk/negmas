"""Action class representing agent actions in a simulated world."""

from __future__ import annotations

from attrs import define

__all__ = ["Action"]


@define
class Action:
    """An action that an `Agent` can execute in a `World` through the `Simulator`."""

    type: str
    """The action type identifier."""
    params: dict
    """Any extra parameters to be passed for the action."""
