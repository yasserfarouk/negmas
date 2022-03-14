from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .entity import Entity

if TYPE_CHECKING:
    from .world import World


__all__ = ["StatsMonitor", "WorldMonitor"]


class StatsMonitor(Entity):
    """A monitor object capable of receiving stats of a world"""

    def init(self, stats: dict[str, Any], world_name: str):
        """Called to initialize the monitor before running first step"""

    def step(self, stats: dict[str, Any], world_name: str):
        """Called at the END of every simulation step"""


class WorldMonitor(Entity):
    """A monitor object capable of monitoring a world. It has read/write access to the world"""

    def init(self, world: World):
        """Called to initialize the monitor before running first step"""

    def step(self, world: World):
        """Called at the END of every simulation step"""
