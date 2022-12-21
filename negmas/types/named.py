"""
Implements `NambedObject` for entities that have name/id combinations.


In principle, the user should manage the `name` and use it for printing/logging.
The system manages the `id` making sure it is unique for every entity.
The user **can** override the system assigned `id` but should always make sure that
no entities in the entire system has te same `id`
"""
from __future__ import annotations

import datetime
import uuid
from os import PathLike
from pathlib import Path
from typing import Any, Literal, overload

import dill

from ..helpers import get_full_type_name, shorten, unique_name
from ..helpers.inout import dump, load

__all__ = ["NamedObject"]


class NamedObject:
    """The base class of all named entities.

    All named entities need to call this class's __init__() somewhere during initialization.

    Args:
        name (str): The given name of the entity. Notice that the class will add this to a  base that depends
                    on the child's class name.
        id (str): A unique identifier in the whole system. In principle you should let the system create
                  this identifier by passing None. In special cases like in serialization you may want to
                  set the id directly
        type_name (str): A string to be returned by `type_name` and its short version is returned by `short_type_name`

    """

    def __init__(
        self,
        name: str | None = None,
        *,
        id: str | None = None,
        type_name: str | None = None,
    ) -> None:
        if name is not None:
            name = str(name)
        self.__uuid = (
            (f"{name}-" if name is not None else "") + str(uuid.uuid4())
            if not id
            else id
        )
        if name is None or len(name) == 0:
            name = unique_name("", add_time=False, rand_digits=16)
        self.__name = name
        self.__type_name = type_name
        super().__init__()

    @classmethod
    def create(cls, *args, **kwargs):
        """Creates an object and returns a proxy to it."""
        return cls(*args, **kwargs)

    @classmethod
    def spawn_object(cls, *args, **kwargs):
        return cls(*args, **kwargs)

    #     @classmethod
    #     def spawn_thread(cls, *args, **kwargs):
    #         raise NotImplementedError("Thread objects are not yet supported")
    #
    #     @classmethod
    #     def spawn_process(cls, *args, **kwargs):
    #         raise NotImplementedError("Process objects are not yet supported")
    #
    #     @classmethod
    #     def spawn_remote_tcp(
    #         cls,
    #         remote_tcp_address: str = "localhost",
    #         remote_tcp_port: Optional[int] = None,
    #         *args,
    #         **kwargs,
    #     ):
    #         raise NotImplementedError("TCP objects are not yet supported")
    #
    #     @classmethod
    #     def spawn_remote_http(
    #         cls,
    #         remote_http_address: str = "localhost",
    #         remote_http_port: Optional[int] = None,
    #         *args,
    #         **kwargs,
    #     ):
    #         raise NotImplementedError("HTTP objects are not yet supported")

    @classmethod
    def spawn(
        cls,
        spawn_as="object",
        spawn_params: dict[str, Any] | None = None,
        *args,
        **kwargs,
    ):
        if spawn_as == "object":
            return cls.spawn_object(*args, **kwargs)
        # if spawn_as == "thread":
        #     return cls.spawn_thread(*args, **kwargs)
        # if spawn_as == "gevent" or spawn_as == "green-thread":
        #     return cls.spawn_gevent(*args, **kwargs)
        # if spawn_as == "eventlet":
        #     return cls.spawn_eventlet(*args, **kwargs)
        # if spawn_as == "process":
        #     return cls.spawn_process(*args, **kwargs)
        if spawn_params is None:
            spawn_params = dict()
        # if spawn_as == "tcp":
        #     return cls.spawn_remote_tcp(
        #         spawn_params.get("address", "localhost"),
        #         spawn_params.get("port", None),
        #         *args,
        #         **kwargs,
        #     )
        raise ValueError(f"cannot spawn as {spawn_as}")

    @property
    def name(self):
        """A convenient name of the entity (intended primarily for printing/logging/debugging)."""
        return self.__name

    @name.setter
    def name(self, name):
        self.__name = name

    @property
    def uuid(self):
        """The unique ID of this entity"""
        return self.__uuid

    @uuid.setter
    def uuid(self, uuid):
        self.__uuid = uuid

    @property
    def id(self):
        """The unique ID of this entity"""
        return self.__uuid

    @id.setter
    def id(self, id):
        self.__uuid = id

    def checkpoint(
        self,
        path: PathLike,
        file_name: str | None = None,
        info: dict[str, Any] | None = None,
        exist_ok: bool = False,
        single_checkpoint: bool = True,
        step_attribs: tuple[str, ...] = (
            "current_step",
            "_current_step",
            "_Entity__current_step",
            "_step",
        ),
    ) -> Path:
        """
        Saves a checkpoint of the current object at  the given path.

        Args:

            path: Full path to a directory to store the checkpoint
            file_name: Name of the file to dump into. If not given, a unique name is created
            info: Information to save with the checkpoint (must be json serializable)
            exist_ok: If true, override existing dump
            single_checkpoint: If true, keep a single checkpoint for the last step
            step_attribs: Attributes to represent the time-step of the object. Any of the given attributes will be
                          used in the file name generated if single_checkpoint is False. If single_checkpoint is True, the
                          filename will not contain time-step information

        Returns:
            full path to the file used to save the checkpoint

        """
        if file_name is None:
            base_name = (
                f"{self.__class__.__name__.split('.')[-1].lower()}.{unique_name('', add_time=False, rand_digits=8, sep='-')}"
                f".{self.id.replace('/', '_')}"
            )
        else:
            base_name = file_name
        path = Path(path)
        if path.exists() and path.is_file():
            raise ValueError(f"{str(path)} is a file. It must be a directory")
        path.mkdir(parents=True, exist_ok=True)
        current_step = None
        for attrib in step_attribs:
            try:
                a = getattr(self, attrib)
                if isinstance(a, int):
                    current_step = a
                    break
            except AttributeError:
                pass
        if not single_checkpoint and current_step is not None:
            base_name = f"{current_step:05}.{base_name}"
        full_file_name = path / base_name

        if info is None:
            info = {}
        info.update(
            {
                "type": get_full_type_name(self.__class__),
                "id": self.id,
                "name": self.name,
                "time": datetime.datetime.now().isoformat(),
                "step": current_step,
                "filename": str(full_file_name),
            }
        )

        if (not exist_ok) and full_file_name.exists():
            raise ValueError(
                f"{str(full_file_name)} already exists. Pass exist_ok=True if you want to override it"
            )

        with open(full_file_name, "wb") as f:
            dill.dump(self, f)

        info_file_name = path / (base_name + ".json")
        dump(info, info_file_name)
        return full_file_name

    @overload
    @classmethod
    def from_checkpoint(
        cls, file_name: Path | str, return_info: Literal[False] = False
    ) -> NamedObject:
        ...

    @overload
    @classmethod
    def from_checkpoint(
        cls, file_name: Path | str, return_info: Literal[True] = True
    ) -> tuple[NamedObject, dict[str, Any]]:
        ...

    @classmethod
    def from_checkpoint(
        cls, file_name: Path | str, return_info: bool = False
    ) -> NamedObject | tuple[NamedObject, dict[str, Any]]:
        """
        Creates an object from a saved checkpoint

        Args:
            file_name:
            return_info: If True, tbe information saved when the file was dumped are returned

        Returns:
            Either the object or the object and dump-info as a dict (if return_info was true)

        Remarks:

            - If info is returned, it is guaranteed to have the following members:
                - time: Dump time
                - type: Type of the dumped object
                - id: ID
                - name: name
        """
        file_name = Path(file_name).absolute()
        with open(file_name, "rb") as f:
            obj = dill.load(f)
        if return_info:
            return obj, cls.checkpoint_info(file_name)
        return obj

    @classmethod
    def checkpoint_info(cls, file_name: Path | str) -> dict[str, Any]:
        """
        Returns the information associated with a dump of the object saved in the given file

        Args:
            file_name: Name of the object

        Returns:

        """
        file_name = Path(file_name).absolute()
        return load(file_name.parent / (file_name.name + ".json"))

    @property
    def type_name(self) -> str:
        if self.__type_name:
            return self.__type_name
        return self.__class__.__name__

    @property
    def short_type_name(self) -> str:
        return shorten(self.type_name)
