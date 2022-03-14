from __future__ import annotations

from attr import define

__all__ = ["Action"]


@define
class Action:
    """An action that an `Agent` can execute in a `World` through the `Simulator`."""

    type: str
    """Action name."""
    params: dict
    """Any extra parameters to be passed for the action."""
