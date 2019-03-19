"""Implements Event management"""
import itertools
import random
from collections import defaultdict
from typing import Any, Dict, Union, Optional, Set

from dataclasses import dataclass

from negmas import NamedObject

__all__ = [
    "Event",
    "EventSource",
    "EventSink",
    "Notification",
    "Notifier",
    "Notifiable"
]


@dataclass
class Event:
    __slots__ = ['type', 'data']
    type: str
    data: Any


class EventSource:
    """An object capable of raising events"""

    def __init__(self):
        super().__init__()
        self.__sinks: Dict[str, list] = defaultdict(list)

    def announce(self, event: Event):
        """Raises an event and informs all event sinks that are registerd for notifications
        on this event type"""

        sinks = self.__sinks.get(event.type, [])
        random.shuffle(sinks)
        for sink in sinks:
            sink.on_event(event=event, sender=self)

    def register_listener(self, event_type: str, listener: "EventSink"):
        self.__sinks[event_type].append(listener)


class EventSink:
    def on_event(self, event: Event, sender: EventSource):
        pass


@dataclass
class Notification:
    __slots__ = ['type', 'data']
    type: str
    data: Any


class Notifier(NamedObject):
    def notify(self, notifiable: 'Notifiable', notification: Notification):
        notifiable.on_notification(notification=notification, notifier=self.id)


class Notifiable:
    def on_notification(self, notification: Notification, notifier: str) -> None:
        pass
