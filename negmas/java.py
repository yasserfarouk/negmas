from __future__ import annotations

from numbers import Integral

"""
Implements Java interoperability allowing parts of negmas to work smoothly
with their Java counterparts in jnegmas

"""
from typing import Any, Iterable, Optional

from py4j.java_collections import JavaList, JavaMap, JavaSet
from py4j.java_gateway import JavaObject

from .helpers import camel_case, get_class, snake_case

__all__ = [
    "from_java",
    "java_identifier",
    "from_java",
    "to_dict",
    "PYTHON_CLASS_IDENTIFIER",
]

DEFAULT_JNEGMAS_PATH = "external/jnegmas-1.0-SNAPSHOT-all.jar"
PYTHON_CLASS_IDENTIFIER = "__python_class__"


def java_identifier(s: str):
    if s != PYTHON_CLASS_IDENTIFIER:
        return camel_case(s)
    return s


def python_identifier(k: str) -> str:
    """
    Converts a key to snake case keeping dunder keys alone

    Args:

        k: string to be converted

    Returns:

        snake-case string
    """
    return k if k.startswith("__") else snake_case(k)


def from_java(
    d: Any, deep=True, remove_type_field=True, fallback_class_name: Optional[str] = None
):
    """Decodes a dict coming from java recovering all objects in the way

    Args:

        d: The value to be decoded. If it is not a dict, it is returned as it is.
        deep: If true, decode recursively
        remove_type_field: If true the field called `PYTHON_CLASS_IDENTIFIER` will be removed if found.
        fallback_class_name: If given, it is used as the fall-back  type if ``PYTHON_CLASS_IDENTIFIER` is not in the dict.

    Remarks:

        - If the object is not a dict or if it has no `PYTHON_CLASS_IDENTIFIER` field and no `fallback_class_name` is
          given, the input `d` is returned as it is. It will not even be copied.

    See Also:
        `PYTHON_CLASS_IDENTIFIER`



    """

    def good_field(k: str):
        return (
            not k.startswith("python_")
            and not k.startswith("java")
            and not (k != PYTHON_CLASS_IDENTIFIER and k.startswith("_"))
        )

    if d is None or isinstance(d, int) or isinstance(d, float) or isinstance(d, str):
        return d
    if isinstance(d, JavaList):
        return [_ for _ in d]
    if isinstance(d, JavaSet):
        return {_ for _ in d}
    if isinstance(d, JavaMap):
        if deep:
            d = {python_identifier(k): from_java(v) for k, v in d.items()}
        else:
            d = {python_identifier(k): v for k, v in d.items()}
        deep = False
    if isinstance(d, dict):
        if remove_type_field:
            python_class_name = d.pop(PYTHON_CLASS_IDENTIFIER, fallback_class_name)
        else:
            python_class_name = d.get(PYTHON_CLASS_IDENTIFIER, fallback_class_name)
        if python_class_name is not None:
            python_class_name = py_class_name(python_class_name)
            # if python_class_name.endswith("Issue"):
            #     python_class = get_class("negmas.outcomes.Issue")
            # else:
            python_class = get_class(python_class_name)
            # we resolve sub-objects first from the dict if deep is specified before calling from_java on the class
            if deep:
                d = {
                    python_identifier(k): from_java(v)
                    for k, v in d.items()
                    if good_field(k)
                }
            # from_java needs to do a shallow conversion from a dict as deep conversion is taken care of already.
            if hasattr(python_class, "from_java"):
                return python_class.from_java(
                    {python_identifier(k): v for k, v in d.items()}, python_class_name
                )
            if deep:
                d = {
                    python_identifier(k): from_java(v)
                    for k, v in d.items()
                    if good_field(k)
                }
            else:
                d = {python_identifier(k): v for k, v in d.items() if good_field(k)}
            return python_class(**d)
        return d
    raise (ValueError(str(d)))


def to_dict(value, deep=True, add_type_field=True, camel=True):
    """Encodes the given value as nothing more complex than simple dict of either dicts, lists or builtin numeric
    or string values
    Args:
        value: Any object
        deep: Whether we should go deep in the encoding or do a shallow encoding
        add_type_field: Whether to add a type field. If True, A field named `PYTHON_CLASS_IDENTIFIER` will be added
        giving the type of `value`
        camel: Convert to camel_case if True
    Remarks:
        - All iterables are converted to lists when `deep` is true.
        - If the `value` object has a `to_dict` member, it will be called to do the conversion, otherwise its `__dict__`
          or `__slots__` member will be used.
    See Also:
          `from_java`, `PYTHON_CLASS_IDENTIFIER`
    """

    def _j(s: str) -> str:
        if camel:
            return java_identifier(s)
        return str(s)

    def good_field(k: str):
        return (
            not k.startswith("python_")
            and not k.startswith("java")
            and not (k != PYTHON_CLASS_IDENTIFIER and k.startswith("_"))
        )

    if isinstance(value, dict):
        if not deep:
            return {_j(k): v for k, v in value.items()}
        return {
            _j(k): to_dict(v, add_type_field=add_type_field, camel=camel)
            for k, v in value.items()
            if good_field(k)
        }
    if isinstance(value, Iterable) and not deep:
        return value
    if isinstance(value, Iterable) and not isinstance(value, str):
        return [to_dict(_, add_type_field=add_type_field, camel=camel) for _ in value]
    if hasattr(value, "to_dict"):
        converted = value.to_dict()
        if isinstance(converted, dict):
            if add_type_field and (PYTHON_CLASS_IDENTIFIER not in converted.keys()):
                converted[PYTHON_CLASS_IDENTIFIER] = (
                    value.__class__.__module__ + "." + value.__class__.__name__
                )
            return {_j(k): v for k, v in converted.items()}
        else:
            return converted
    if hasattr(value, "__dict__"):
        if deep:
            d = {
                _j(k): to_dict(v, add_type_field=add_type_field, camel=camel)
                for k, v in value.__dict__.items()
                if good_field(k)
            }
        else:
            d = {_j(k): v for k, v in value.__dict__.items() if good_field(k)}
        if add_type_field:
            d[PYTHON_CLASS_IDENTIFIER] = (
                value.__class__.__module__ + "." + value.__class__.__name__
            )

        # ugly ugly ugly ugly
        if "_NamedObject__uuid" in value.__dict__:
            d["id"] = value.__dict__["_NamedObject__uuid"]
        if "_NamedObject__name" in value.__dict__:
            d["name"] = value.__dict__["_NamedObject__name"]

        return d
    if hasattr(value, "__slots__"):
        if deep:
            d = dict(
                zip(
                    (_j(k) for k in value.__slots__),
                    (
                        to_dict(
                            getattr(value, _),
                            add_type_field=add_type_field,
                            camel=camel,
                        )
                        for _ in value.__slots__
                    ),
                )
            )
        else:
            d = dict(
                zip(
                    (_j(k) for k in value.__slots__),
                    (getattr(value, _) for _ in value.__slots__),
                )
            )
        if add_type_field:
            d[PYTHON_CLASS_IDENTIFIER] = (
                value.__class__.__module__ + "." + value.__class__.__name__
            )
        return d
    # a builtin
    if isinstance(value, Integral):
        return int(value)
    return value


def py_class_name(python_class_name: str) -> str:
    """
    Converts a class name that we got from Java to the corresponding class name in python

    Args:

        python_class_name: The class name we got from JNEgMAS

    Returns:

        The class name in negmas.

    """
    lst = python_class_name.split(".")
    if lst[-1].startswith("Python"):
        lst[-1] = lst[-1][len("Python") :]
    python_class_name = ".".join(lst)
    return python_class_name
