from __future__ import annotations

"""Implements Event management"""
import json
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from negmas import warnings

from .outcomes import Issue
from .serialization import serialize
from .types import NamedObject

__all__ = [
    "Event",
    "EventSource",
    "EventSink",
    "EventLogger",
    "Notification",
    "Notifier",
    "Notifiable",
]


@dataclass
class Event:
    """An event that can be raised and consumed"""

    __slots__ = ["type", "data"]
    type: str
    data: Any


class EventSource:
    """An object capable of raising events"""

    def __init__(self):
        super().__init__()
        self.__sinks: dict[str | None, list[EventSink]] = defaultdict(list)

    def announce(self, event: Event):
        """Raises an event and informs all event sinks that are registered for notifications
        on this event type"""

        sinks = self.__sinks.get(event.type, []) + self.__sinks.get(None, [])
        random.shuffle(sinks)
        for sink in sinks:
            sink.on_event(event=event, sender=self)

    def register_listener(self, event_type: str | None, listener: EventSink):
        """Registers a listener for the given event_type.

        Args:
            event_type: The type to register. If None, the listener will
                        be registered for all types
            listener: The listening agent (must have an `on_event` method
                      that receives an event: `Event` and a sender: `EventSource`)
        """
        self.__sinks[event_type].append(listener)


class EventSink:
    """An object capable of receiving events"""

    def on_event(self, event: Event, sender: EventSource):
        pass


def myvars(x):
    if not x:
        return x
    return {
        k: v
        for k, v in vars(x).items()
        if not k.startswith("_") and not k.endswith("_")
    }


class EventLogger(EventSink):
    """
    Logs events to a file

    Args:
        file_name: Name of the file to save events into
        types: The types of events to save. If None, all events will be saved
    """

    def __init__(self, file_name: str | Path, types: list[str] | None = None):
        file_name = Path(file_name)
        file_name.parent.mkdir(parents=True, exist_ok=True)
        self._file_name = file_name
        self._types = set(types) if types else None
        self._start = time.perf_counter()

    def reset_timer(self):
        self._start = time.perf_counter()

    def on_event(self, event: Event, sender: EventSource):
        if not self._file_name:
            return
        if self._types is not None and event.type not in self._types:
            return

        def _simplify(x):
            if x is None:
                return None
            if isinstance(x, (str, int, float)):
                return x
            if isinstance(x, Issue):
                return dict(name=x.name, values=x.values)
            if isinstance(x, dict):
                return {k: _simplify(v) for k, v in x.items()}
            for y in ("id", "name"):
                if hasattr(x, y):
                    return getattr(x, y)
            if isinstance(x, Iterable):
                return list(_simplify(_) for _ in x)
            return str(x)
            # return _simplify(myvars(x))

        try:
            sid = sender.id if hasattr(sender, "id") else serialize(sender)  # type: ignore
            d = dict(
                sender=sid,
                time=time.perf_counter() - self._start,
                type=event.type,
                data=_simplify(event.data),
            )
            with open(self._file_name, "a") as f:
                f.write(f"{json.dumps(d)},\n")
        except Exception as e:
            warnings.warn(
                f"Failed to log {str(event)}: {str(e)}", warnings.NegmasLoggingWarning
            )


@dataclass
class Notification:
    __slots__ = ["type", "data"]
    type: str
    data: Any


class Notifier(NamedObject):
    """An object that can notify other objects"""

    def notify(self, notifiable: Notifiable, notification: Notification):
        notifiable.on_notification_(notification=notification, notifier=self.id)


class Notifiable:
    """An object that can be notified"""

    def add_handler(
        self, notification_type: str, callback: Callable[[Notification, str], bool]
    ):
        """
        Adds a notification handler to the list of handlers of the given type. These handlers will be called
        in the order in which they are received

        Args:
            notification_type: Notification type as specificed in the type member of the Notification class
            callback: The callback which must receive a Notification object and a string and returns a boolean. If True
                      is returned from one callback, the remaining callbacks will not be called

        Returns:

        """
        if not hasattr(self, "__notification_handlers"):
            self.__notification_handlers: dict[
                str, list[Callable[[Notification, str], bool]]
            ] = defaultdict(list)
        self.__notification_handlers[notification_type].append(callback)

    def handlers(
        self, notification_type: str
    ) -> list[Callable[[Notification, str], bool]]:
        """
        Gets the list of handlers registered for some notification type. This list can be modified in place to change
        the order of handlers for example. It is NOT a copy.
        """
        try:
            return self.__notification_handlers[notification_type]
        except (ValueError, IndexError, AttributeError):
            return []

    def remove_handler(
        self, notification_type: str, callback: Callable[[Notification, str], bool]
    ) -> bool:
        """
        Removes a notification handler from the list of handlers of the given type.

        Args:
            notification_type: Notification type as specificed in the type member of the Notification class
            callback: The callback which must receive a Notification object and a string and returns a boolean. If True
                      is returned from one callback, the remaining callbacks will not be called

        Returns:
            Whether or not the handler was in the list of handlers for this type. In all cases, the handler will not be
            called after this call (either it was not there or it will be removed).

        """
        try:
            self.__notification_handlers[notification_type].remove(callback)
            return True
        except (ValueError, IndexError, AttributeError):
            return False

    def on_notification(self, notification: Notification, notifier: str) -> None:
        """
        Called when a notification is received and is not handled by any registered handler

        Args:
            notification: The notification received
            notifier: The notifier ID

        Remarks:

            - override this method to provide a catch-all notification handling method.
        """

    def on_notification_(self, notification: Notification, notifier: str) -> bool:
        """
        Called when a notification is received. Do NOT directly override this method

        Args:
            notification:
            notifier:

        Returns:

        """
        try:
            for callback in self.__notification_handlers[notification.type]:
                if callback(notification, notifier):
                    break
            return True
        except (IndexError, ValueError, AttributeError):
            self.on_notification(notification, notifier)
            return False
