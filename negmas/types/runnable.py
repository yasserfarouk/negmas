"""
Base protocol for all entities that can be stepped/run.
"""
from __future__ import annotations

from typing import Any

from typing_extensions import Protocol, runtime

__all__ = ["Runnable"]


@runtime
class Runnable(Protocol):
    """A protocol defining runnable objects"""

    @property
    def current_step(self) -> int:
        """Returns the current time step index"""
        ...

    def step(self) -> Any:
        pass

    def run(self) -> Any:
        pass
