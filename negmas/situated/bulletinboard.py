from __future__ import annotations

import copy
import re
import uuid
from typing import Any, Callable, Iterable

from negmas.events import Event, EventSource
from negmas.helpers.inout import ConfigReader

__all__ = ["BulletinBoard"]


class BulletinBoard(EventSource, ConfigReader):
    """
    The bulletin-board which carries all public information. It consists of sections each with a dictionary of records.
    """

    # def __getstate__(self):
    #     return self.name, self._data
    #
    # def __setstate__(self, state):
    #     name, self._data = state
    #     super().__init__(name=name)

    def __init__(self):
        """
        Constructor

        Args:
            name: BulletinBoard name
        """
        super().__init__()
        self._data: dict[str, dict[str, Any]] = {}

    def add_section(self, name: str) -> None:
        """
        Adds a section to the bulletin Board

        Args:
            name: Section name

        Returns:

        """
        self._data[name] = {}

    def query(
        self, section: str | list[str] | None, query: Any, query_keys=False
    ) -> dict[str, Any] | None:
        """
        Returns all records in the given section/sections of the bulletin-board that satisfy the query

        Args:
            section: Either a section name, a list of sections or None specifying ALL public sections (see remarks)
            query: The query which is USUALLY a dict with conditions on it when querying values and a RegExp when
            querying keys
            query_keys: Whether the query is to be applied to the keys or values.

        Returns:

            - A dictionary with key:value pairs giving all records that satisfied the given requirements.

        Remarks:

            - A public section is a section with a name that does not start with an underscore
            - If a set of sections is given, and two records in different sections had the same key, only one of them
              will be returned
            - Key queries use regular expressions and match from the beginning using the standard re.match function

        """
        if section is None:
            return self.query(
                section=[_ for _ in self._data.keys() if not _.startswith("_")],
                query=query,
                query_keys=query_keys,
            )

        if isinstance(section, Iterable) and not isinstance(section, str):
            results = [
                self.query(section=_, query=query, query_keys=query_keys)
                for _ in section
            ]
            if len(results) == 0:
                return dict()
            final: dict[str, Any] = {}
            for _ in results:
                final.update(_)
            return final

        sec = self._data.get(section, None)
        if sec is None:
            return {}
        if query is None:
            return copy.deepcopy(sec)
        if query_keys:
            return {k: v for k, v in sec.items() if re.match(str(query), k) is not None}
        return {k: v for k, v in sec.items() if BulletinBoard.satisfies(v, query)}

    @classmethod
    def satisfies(cls, value: Any, query: Any) -> bool:
        method = getattr(value, "satisfies", None)
        if method is not None and isinstance(method, Callable):
            return method(query)
        if isinstance(value, dict) and isinstance(query, dict):
            for k, v in query.items():
                if value.get(k, None) != v:
                    return False
        else:
            raise ValueError(
                f"Cannot check satisfaction of {type(query)} against value {type(value)}"
            )
        return True

    def read(self, section: str, key: str) -> Any:
        """
        Reads the value associated with given key

        Args:
            section: section name
            key: key

        Returns:

            Content of that key in the bulletin-board

        """
        sec = self._data.get(section, None)
        if sec is None:
            return None
        return sec.get(key, None)

    def record(self, section: str, value: Any, key: str | None = None) -> None:
        """
        Records data in the given section of the bulletin-board

        Args:
            section: section name (can contain subsections separated by '/')
            key: The key
            value: The value

        """
        if key is None:
            try:
                skey = str(hash(value))
            except:
                skey = str(uuid.uuid4())
        else:
            skey = key
        self._data[section][skey] = value
        self.announce(
            Event("new_record", data={"section": section, "key": skey, "value": value})
        )

    def remove(
        self,
        section: list[str] | str | None,
        *,
        query: Any | None = None,
        key: str = None,
        query_keys: bool = False,
        value: Any = None,
    ) -> bool:
        """
        Removes a value or a set of values from the bulletin Board

        Args:
            section: The section
            query: the query to use to select what to remove
            key: the key to remove (no need to give a full query)
            query_keys: Whether to apply the query (if given) to keys or values
            value: Value to be removed

        Returns:
            bool: Success of failure
        """
        if section is None:
            return self.remove(
                section=[_ for _ in self._data.keys() if not _.startswith("_")],
                query=query,
                key=key,
                query_keys=query_keys,
            )

        if isinstance(section, Iterable) and not isinstance(section, str):
            return all(
                self.remove(section=_, query=query, key=key, query_keys=query_keys)
                for _ in section
            )

        sec = self._data.get(section, None)
        if sec is None:
            return False
        if value is not None:
            for k, v in sec.items():
                if v == value:
                    key = k
                    break
        if key is not None:
            try:
                self.announce(
                    Event(
                        "will_remove_record",
                        data={"section": sec, "key": key, "value": sec[key]},
                    )
                )
                sec.pop(key, None)
                return True
            except KeyError:
                return False

        if query is None:
            return False

        if query_keys:
            keys = [k for k in sec.keys() if re.match(str(query), k) is not None]
        else:
            keys = [k for k, v in sec.items() if v.satisfies(query)]
        if len(keys) == 0:
            return False
        for k in keys:
            self.announce(
                Event(
                    "will_remove_record",
                    data={"section": sec, "key": k, "value": sec[k]},
                )
            )
            sec.pop(k, None)
        return True

    @property
    def data(self):
        """This property is intended for use only by the world manager. No other agent is allowed to use it"""
        return self._data
