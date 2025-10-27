"""Protocol definitions."""

from __future__ import annotations

from abc import abstractmethod
from os import PathLike
from pathlib import Path
from typing import Any, Protocol, TypeVar

__all__ = ["XmlSerializable", "DictSerializable"]

X = TypeVar("X", bound="XmlSerializable")


class XmlSerializable(Protocol):
    @classmethod
    @abstractmethod
    def from_xml_str(cls: type[X], xml_str: str, **kwargs) -> X:
        """Imports a utility function from a GENIUS XML string.

        Args:

            xml_str (str): The string containing GENIUS style XML utility function definition

        Returns:

            A utility function object (depending on the input file)
        """

    @abstractmethod
    def to_xml_str(self, **kwargs) -> str:
        """Exports a utility function to a well formatted string"""

    @classmethod
    def from_genius(cls: type[X], file_name: PathLike, **kwargs) -> X:
        """Imports a utility function from a GENIUS XML file.

        Args:
            file_name (str): File name to import from

        Returns:
            A utility function object (depending on the input file)
        """
        with open(file_name) as f:
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
        file_name = Path(file_name).absolute()
        if file_name.suffix == "":
            file_name = file_name.parent / f"{file_name.stem}.xml"
        with open(file_name, "w") as f:
            f.write(self.to_xml_str(**kwargs))


D = TypeVar("D", bound="DictSerializable")


class DictSerializable(Protocol):
    """DictSerializable implementation."""

    def to_dict(self) -> str:
        """To dict.

        Returns:
            str: The result.
        """
        ...

    @classmethod
    def from_dict(cls: type[D], v: dict[str, Any]) -> D:
        """From dict.

        Args:
            v: V.

        Returns:
            D: The result.
        """
        ...


class HasMinMax(Protocol):
    """HasMinMax implementation."""

    min_value: Any
    max_value: Any
