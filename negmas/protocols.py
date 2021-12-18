from __future__ import annotations

from abc import abstractmethod
from os import PathLike
from typing import Any, Protocol, Type, TypeVar

__all__ = ["XmlSerializable", "DictSerializable"]

X = TypeVar("X", bound="XmlSerializable")


class XmlSerializable(Protocol):
    @classmethod
    def from_genius(cls: Type[X], file_name: PathLike, **kwargs) -> X:
        """Imports a utility function from a GENIUS XML file.

        Args:
            file_name (str): File name to import from

        Returns:
            A utility function object (depending on the input file)
        """
        with open(file_name, "r") as f:
            s = f.read()
        return cls.from_xml_str(s, **kwargs)

    def to_genius(self, file_name: PathLike, **kwargs) -> None:
        """
        Exports a utility function to a GENIUS XML file.

        Args:

            file_name (str): File name to export to

        Returns:

            None

        Remarks:
            See ``to_xml_str`` for all the parameters

        """
        with open(file_name, "w") as f:
            f.write(self.to_xml_str(**kwargs))

    def to_xml_str(self, **kwargs) -> str:
        """Exports a utility function to a well formatted string"""

    @classmethod
    @abstractmethod
    def from_xml_str(cls: Type[X], xml_str: str, **kwargs) -> X:
        """Imports a utility function from a GENIUS XML string.

        Args:

            xml_str (str): The string containing GENIUS style XML utility function definition

        Returns:

            A utility function object (depending on the input file)
        """


D = TypeVar("D", bound="DictSerializable")


class DictSerializable(Protocol):
    def to_dict(self) -> str:
        ...

    @classmethod
    def from_dict(cls: Type[D], v: dict[str, Any]) -> D:
        ...


class HasMinMax(Protocol):
    min_value: Any
    max_value: Any
