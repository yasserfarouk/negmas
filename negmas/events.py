"""Implements Event management"""
import itertools
import random
from collections import defaultdict
from typing import Any, Dict, Union, Optional, Set, Callable, List

from dataclasses import dataclass

from negmas import NamedObject

__all__ = [
    "Event",
    "EventSource",
    "EventSink",
    "Notification",
    "Notifier",
    "Notifiable",
]


@dataclass
class Event:
    __slots__ = ["type", "data"]
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
    __slots__ = ["type", "data"]
    type: str
    data: Any


class Notifier(NamedObject):
    def notify(self, notifiable: "Notifiable", notification: Notification):
        notifiable.on_notification_(notification=notification, notifier=self.id)


class Notifiable:
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
            self.__notification_handlers: Dict[
                str, List[Callable[[Notification, str], bool]]
            ] = defaultdict(list)
        self.__notification_handlers[notification_type].append(callback)

    def handlers(
        self, notification_type: str
    ) -> List[Callable[[Notification, str], bool]]:
        """
        Gets the list of handlers registered for some notification type. This list can be modified in place to change
        the order of handlers for example. It is NOT a copy.
        """
        try:
            return self.__notification_handlers[notification_type]
        except (ValueError, IndexError, AttributeError) as e:
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
        except (ValueError, IndexError, AttributeError) as e:
            return False

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
        except (IndexError, ValueError, AttributeError) as e:
            self.on_notification(notification, notifier)
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
        pass
