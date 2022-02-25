#!/usr/bin/env python
"""
Datatypes that do not directly relate to negotiation.
"""
from __future__ import annotations

import functools
import importlib
import json
from enum import Enum
from os import PathLike
from types import FunctionType, LambdaType
from typing import Any, Callable, Type

import stringcase

__all__ = [
    "PathLike",
    "TYPE_START",
    "ReturnCause",
    "get_class",
    "import_by_name",
    "get_full_type_name",
    "instantiate",
    "is_jsonable",
    "is_lambda_function",
    "is_partial_function",
    "is_lambda_or_partial_function",
    "is_type",
    "is_not_lambda_nor_partial_function",
]

TYPE_START = "__TYPE__:"


class ReturnCause(Enum):
    TIMEOUT = 0
    SUCCESS = 1
    FAILURE = 2


def get_full_type_name(t: type[Any] | Callable | str) -> str:
    """
    Gets the ful typename of a type. You *should not* pass an instance to this function but it may just work.

    An exception is that if the input is of type `str` or if it is None, it will be returned as it is
    """
    if t is None or isinstance(t, str):
        return t
    if not hasattr(t, "__module__") and not hasattr(t, "__name__"):
        t = type(t)
    return t.__module__ + "." + t.__name__


def import_by_name(full_name: str) -> Any:
    """Imports something form a module using its full name"""
    if not isinstance(full_name, str):
        return full_name
    modules: list[str] = []
    parts = full_name.split(".")
    modules = parts[:-1]
    module_name = ".".join(modules)
    item_name = parts[-1]
    if len(modules) < 1:
        raise ValueError(
            f"Cannot get the object {item_name} in module {module_name}  (modules {modules})"
        )
    module = importlib.import_module(module_name)
    return getattr(module, item_name)


def get_class(
    class_name: str | type,
    module_name: str = None,
    scope: dict = None,
    allow_nonstandard_names=False,
) -> type:
    """Imports and creates a class object for the given class name"""
    if not isinstance(class_name, str):
        return class_name

    # remove explicit type annotation in the string. Used when serializing
    while class_name.startswith(TYPE_START):
        class_name = class_name[len(TYPE_START) :]

    modules: list[str] = []

    if module_name is not None:
        modules = module_name.split(".")
    modules += class_name.split(".")
    if len(modules) < 1:
        raise ValueError(
            f"Cannot get the class {class_name} in module {module_name}  (modules {modules})"
        )
    if not class_name.startswith("builtins") or allow_nonstandard_names:
        class_name = stringcase.pascalcase(modules[-1])
    else:
        class_name = modules[-1]
    if len(modules) < 2:
        return eval(class_name, scope)
    module_name = ".".join(modules[:-1])
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def instantiate(
    class_name: str | type, module_name: str = None, scope: dict = None, **kwargs
) -> Any:
    """Imports and instantiates an object of a class"""
    return get_class(class_name, module_name)(**kwargs)


def is_lambda_function(obj):
    """Checks if the given object is a lambda function"""
    return isinstance(obj, LambdaType) and obj.__name__ == "<lambda>"


def is_partial_function(obj):
    """Checks if the given object is a lambda function"""
    return isinstance(obj, functools.partial)


def is_lambda_or_partial_function(obj):
    """Checks if the given object is a lambda function or a partial function"""
    return is_lambda_function(obj) or is_partial_function(obj)


def is_type(obj):
    """Checks if the given object is a type converted to string"""
    return isinstance(obj, Type)


def is_not_type(obj):
    """Checks if the given object is not a type converted to string"""
    return not is_type(obj)


def is_not_lambda_nor_partial_function(obj):
    """Checks if the given object is not a lambda function"""
    return isinstance(obj, FunctionType) and (
        obj.__name__ != "<lambda>" and not isinstance(obj, functools.partial)
    )


def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except:
        return False


# class Proxy:
#     """A general proxy class."""
#
#     def __init__(self, obj):
#         self._obj = obj
#
#     def __getattr__(self, item):
#         return getattr(self._obj, item)

# class LazyInitializable:
#     """
#     Not used
#
#     Supports a set_params function that can be used for lazy initialization
#     """
#
#     def __init__(self) -> None:
#         super().__init__()
#
#     def set_params(self, **kwargs) -> None:
#         """Sets the attributes of the object.
#
#         This function can be used to set the attributes of any object to the
#         same values used in its construction which allows for lazy
#         initialization.
#
#         Args:
#             **kwargs: The parameters usually passed to the constructor as a dict
#
#         Example:
#
#             >>> class A(LazyInitializable):
#             ...     def __init__(self, a=None, b=None) -> None:
#             ...         super().__init__()
#             ...         self.a = a
#             ...         self.b = b
#
#             Now you can do the following::
#
#             >>> a = A()
#             >>> a.set_params(a=3, b=2)
#
#             which will be equivalent to:
#
#             >>> b = A(a=3, b=2)
#
#         Remarks:
#             - See ``adjust_params()`` for an example in which the constuctor needs to do more processing than just
#               assinging its inputs to instance members.
#
#         """
#         for k, v in kwargs.items():
#             setattr(self, k, v)
#         self.adjust_params()
#
#     def adjust_params(self) -> None:
#         """
#         Adjust the internal attributes following ``set_attributes()`` or construction using ``__init__()``.
#
#         This function needs to be implemented only if the constructor needs to
#         do some processing on the inputs other than assigning it to instance
#         attributes. In such case, move these adjustments to this function and
#         call it in the constructor.
#
#         Examples:
#
#             >>> class A(object):
#             ...     def __init__(self, a=None, b=None):
#             ...         self.a = a
#             ...         self.b = b if b is not None else []
#
#             should now be defined as follows:
#
#             >>> class A(LazyInitializable):
#             ...     def __init__(self, a, b):
#             ...         super().__init__()
#             ...         self.a = a
#             ...         self.b = b
#             ...         self.adjust_params()
#             ...
#             ...     def adjust_params(self):
#             ...         if self.b is None: self.b = []
#
#         Remarks:
#             - Remember to call `super().__init__()` first in your constructor and to call your `adjust_params()` by
#               the end of the constructor.
#             - The constructor should ONLY copy the parameters it receives to internal variables and then calls
#               `adjust_params()` if any more processing is needed. This makes it possible to use `set_params()` with
#               this object.
#             - You should **never** call `adjust_params()` directly anywhere.
#         """
