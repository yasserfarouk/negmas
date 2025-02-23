"""
Implements serialization to and from strings and secondary storage.

"""

from __future__ import annotations
import importlib
import types
from pathlib import Path
from typing import Any, Callable, Iterable
import sys

import cloudpickle
import numpy as np
from pandas import json_normalize

from negmas import warnings

from .helpers import (
    TYPE_START,
    get_class,
    get_full_type_name,
    is_jsonable,
    is_lambda_or_partial_function,
    is_not_lambda_nor_partial_function,
)
from .helpers.inout import dump, load

__all__ = [
    "serialize",
    "deserialize",
    "dump",
    "load",
    "to_flat_dict",
    "PYTHON_CLASS_IDENTIFIER",
]

PYTHON_CLASS_IDENTIFIER = "__python_class__"
PATH_START = "__PATH__:"
LAMBDA_START = b"__LAMBDAOBJ__:"
FUNCTION_START = b"__FUNCTION_START__:"
# JSON_START = b"__JSON_START__:"
CLOUDPICKLE_START = b"__CLOUDPICKLE_START__:"
SPECIAL_FIELDS = ("_NamedObject__uuid", "_NamedObject__name")
SPECIAL_FIELDS_SHORT_NAMES = ("id", "name")


def serialize(
    value,
    deep=True,
    add_type_field=True,
    keep_private=False,
    ignore_methods=True,
    ignore_lambda=False,
    shorten_type_field=False,
    objmem=None,
    python_class_identifier=PYTHON_CLASS_IDENTIFIER,
):
    """
    Encodes the given value as nothing more complex than simple dict
    of either dicts, lists or builtin numeric or string values. The resulting
    dictionary will be json serializable

    Args:

        value: Any object
        deep: Whether we should go deep in the encoding or do a shallow encoding
        add_type_field: Whether to add a type field. If True, A field named `PYTHON_CLASS_IDENTIFIER` will be added
        giving the type of `value`
        keep_private: Keeps fields starting with "_"
        shorten_type_field: IF given, the type will be shortened to class name only if it starts with "negmas."

    Remarks:

        - All iterables are converted to lists when `deep` is true.
        - If the `value` object has a `to_dict` member, it will be called to
          do the conversion, otherwise its `__dict__` or `__slots__` member
          will be used.

    See Also:
          `deserialize`, `PYTHON_CLASS_IDENTIFIER`

    """

    def add_to_mem(x, objmem):
        if not objmem:
            objmem = {id(x)}
        else:
            objmem.add(id(x))
        return objmem

    def good_field(k: str, v, objmem):
        if not isinstance(k, str):
            return True
        if objmem and id(v) in objmem:
            return False
        if ignore_lambda and is_lambda_or_partial_function(v):
            return False
        if ignore_methods and is_not_lambda_nor_partial_function(v):
            return False
        if not isinstance(k, str):
            return False
        return keep_private or not (k != python_class_identifier and k.startswith("_"))

    def adjust_dict(d):
        if not isinstance(d, dict):
            return d
        for a, b in zip(SPECIAL_FIELDS, SPECIAL_FIELDS_SHORT_NAMES):
            if a in d.keys():
                if b in d.keys() and d[b] != d[a]:
                    warnings.warn(
                        f"Field {a} and {b} already exist and are not equal.",
                        warnings.NegmasSarializationWarning,
                    )
                d[b] = d[a]
                del d[b]
        return d

    if value is None:
        return None

    def get_type_field(value):
        t = value.__class__.__name__
        if shorten_type_field and t.startswith("negmas."):
            return t
        return value.__class__.__module__ + "." + t

    if isinstance(value, dict):
        if not deep:
            return adjust_dict({k: v for k, v in value.items()})
        return adjust_dict(
            {
                k: serialize(
                    v,
                    deep=deep,
                    add_type_field=add_type_field,
                    objmem=objmem,
                    python_class_identifier=python_class_identifier,
                )
                for k, v in value.items()
                if good_field(k, v, objmem)
            }
        )
    if isinstance(value, Iterable) and not deep:
        # add_to_mem(value)
        return value
    if isinstance(value, type):
        return TYPE_START + get_full_type_name(value)
    if isinstance(value, Path):
        return PATH_START + str(value)
    # if isinstance(value, np.ndarray):
    #     return value.tolist()
    if isinstance(value, (list, tuple)) and not isinstance(value, str):
        objmem = add_to_mem(value, objmem)
        try:
            return adjust_dict(
                type(value)(
                    serialize(
                        _,
                        deep=deep,
                        add_type_field=add_type_field,
                        objmem=objmem,
                        python_class_identifier=python_class_identifier,
                    )
                    for _ in value
                )
            )
        except Exception:
            return adjust_dict(
                [
                    serialize(
                        _,
                        deep=deep,
                        add_type_field=add_type_field,
                        objmem=objmem,
                        python_class_identifier=python_class_identifier,
                    )
                    for _ in value
                ]
            )

    def convertwith(value, method, pass_identifier=False):
        if hasattr(value, method) and isinstance(
            getattr(value, method), types.MethodType
        ):
            if pass_identifier:
                converted = getattr(value, method)(
                    python_class_identifier=python_class_identifier
                )  # type: ignore
            else:
                converted = getattr(value, method)()  # type: ignore
            if isinstance(converted, dict):
                if add_type_field and (python_class_identifier not in converted.keys()):
                    converted[python_class_identifier] = get_type_field(value)
                return adjust_dict({k: v for k, v in converted.items()})
            else:
                return adjust_dict(converted)

    for method in ("to_dict", "asdict", "dict"):
        try:
            converted = convertwith(value, method, pass_identifier=True)
        except Exception:
            converted = convertwith(value, method, pass_identifier=False)
        if converted is not None:
            return converted
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        if (
            value.startswith(FUNCTION_START)
            or value.startswith(LAMBDA_START)
            or value.startswith(CLOUDPICKLE_START)
            # or value.startswith(JSON_START)
        ):
            warnings.warn(
                f"{value} starts with a reserved part!! Will just keep it as"
                f" it is. May be you are serializing an already serialized object",
                warnings.NegmasSarializationWarning,
            )
        return value

    if is_lambda_or_partial_function(value):
        return LAMBDA_START + cloudpickle.dumps(value)

    if is_not_lambda_nor_partial_function(value):
        return FUNCTION_START + cloudpickle.dumps(value)

    if hasattr(value, "__dict__"):
        if deep:
            objmem = add_to_mem(value, objmem)
            d = {
                k: serialize(
                    v,
                    deep=deep,
                    add_type_field=add_type_field,
                    objmem=objmem,
                    python_class_identifier=python_class_identifier,
                )
                for k, v in value.__dict__.items()
                if good_field(k, v, objmem)
            }
        else:
            d = {k: v for k, v in value.__dict__.items() if good_field(k, v, objmem)}
        if add_type_field:
            d[python_class_identifier] = get_type_field(value)
        return adjust_dict(d)

    if hasattr(value, "__slots__"):
        if deep:
            objmem = add_to_mem(value, objmem)
            d = dict(
                zip(
                    (k for k in value.__slots__),  # type: ignore
                    (
                        serialize(
                            getattr(value, _),
                            deep=deep,
                            add_type_field=add_type_field,
                            objmem=objmem,
                            python_class_identifier=python_class_identifier,
                        )
                        for _ in value.__slots__  # type: ignore
                    ),
                )
            )
        else:
            d = dict(
                zip(
                    (k for k in value.__slots__),  # type: ignore
                    (getattr(value, _) for _ in value.__slots__),  # type: ignore
                )
            )
        if add_type_field:
            d[python_class_identifier] = get_type_field(value)
        return adjust_dict(d)
    if isinstance(value, np.int64):  # type: ignore
        return int(value)
    # a builtin
    if is_jsonable(value):
        return value
    try:
        vv = CLOUDPICKLE_START + cloudpickle.dumps(value)
        return vv
    except Exception:
        pass
    warnings.warn(
        f"{value} of type {type(value)} is not serializable",
        warnings.NegmasSarializationWarning,
    )
    return value


def to_flat_dict(
    value,
    deep=True,
    add_type_field=False,
    shorten_type_field=False,
    python_class_identifier=PYTHON_CLASS_IDENTIFIER,
) -> dict[str, Any]:
    """
    Encodes the given value as a flat dictionary

    Args:
        value: The value to be converted to a flat dictionary
        deep: Converting all sub-objects
        add_type_field: If true, a special field for the object type will be added
        shorten_type_field: If true, the type field will be shortened to just class name if it is defined in NegMAS

    Returns:

    """
    d = serialize(
        value,
        add_type_field=add_type_field,
        shorten_type_field=shorten_type_field,
        deep=deep,
        python_class_identifier=python_class_identifier,
    )
    if d is None:
        return {}
    if not isinstance(d, dict):
        raise ValueError(
            f"value is of type {type(value)} cannot be converted to a flat dict"
        )
    for k, v in d.items():
        if isinstance(v, list) or isinstance(v, tuple):
            d[k] = str(v)
    return json_normalize(d, errors="ignore", sep="_").to_dict(orient="records")[0]


def deserialize(
    d: Any,
    deep=True,
    remove_type_field=True,
    keep_private=False,
    fallback_class_name: str | None = None,
    base_module: str = "",
    deep_ignore: bool = True,
    ignored_keys: tuple[str, ...] = tuple(),
    python_class_identifier=PYTHON_CLASS_IDENTIFIER,
    type_marker: str = TYPE_START,
    path_marker: str = PATH_START,
    lambda_marker: bytes = LAMBDA_START,
    function_marker: bytes = FUNCTION_START,
    cloudpickle_marker: bytes = CLOUDPICKLE_START,
    extra_paths: tuple[str | Path, ...] = tuple(),
    extra_modules: tuple[str, ...] = tuple(),
    type_name_adapter: Callable[[str], str] | None = None,
    path_adapter: Callable[[str], str] | None = None,
):
    """Decodes a dict/object coming from `serialize`

    Args:

        d: The value to be decoded. If it is not a dict, it is returned as it is.
        deep: If true, decode recursively
        remove_type_field: If true the field called `PYTHON_CLASS_IDENTIFIER` will be removed if found.
        keep_private: If given, private fields (starting with _) will be kept
        fallback_class_name: If given, it is used as the fall-back  type if ``PYTHON_CLASS_IDENTIFIER` is not in the dict.
        ignored_keys: Keys to ignore
        deep_ignore: if given, ignored keys are ignored in all components recusrively when deep is specified

    Remarks:

        - If the object is not a dict or if it has no `PYTHON_CLASS_IDENTIFIER` field and no `fallback_class_name` is
          given, the input `d` is returned as it is. It will not even be copied.

    See Also:
        `serialize`, `PYTHON_CLASS_IDENTIFIER`



    """

    def do_nothing(x):
        return x

    params_ = dict(
        deep=deep,
        remove_type_field=remove_type_field,
        keep_private=keep_private,
        fallback_class_name=fallback_class_name,
        base_module=base_module,
        deep_ignore=deep_ignore if deep_ignore else False,
        ignored_keys=ignored_keys if deep_ignore else tuple(),
        python_class_identifier=python_class_identifier,
        type_marker=type_marker,
        path_marker=path_marker,
        lambda_marker=lambda_marker,
        function_marker=function_marker,
        cloudpickle_marker=cloudpickle_marker,
        extra_paths=extra_paths,
        extra_modules=extra_modules,
        type_name_adapter=type_name_adapter,
        path_adapter=path_adapter,
    )

    if type_name_adapter is None:
        type_name_adapter = do_nothing
    if path_adapter is None:
        path_adapter = do_nothing
    for p in extra_paths:
        sys.path.append(str(p))
    for module in extra_modules:
        importlib.import_module(module)

    def good_field(k: str):
        if k in ignored_keys:
            return False
        if not isinstance(k, str):
            return True
        return keep_private or not (k != python_class_identifier and k.startswith("_"))

    if d is None or isinstance(d, int) or isinstance(d, float) or isinstance(d, str):
        return d
    if isinstance(d, dict):
        if remove_type_field:
            python_class_name = d.pop(python_class_identifier, fallback_class_name)
        else:
            python_class_name = d.get(python_class_identifier, fallback_class_name)
        if python_class_name is not None and python_class_name != "functools.partial":
            try:
                python_class = get_class(type_name_adapter(python_class_name))
            except Exception as e:
                if base_module:
                    python_class = get_class(
                        type_name_adapter(f"{base_module}.{python_class_name}")
                    )
                else:
                    raise e
            # we resolve sub-objects first from the dict if deep is specified before calling deserialize on the class
            if deep:
                d = {
                    k: deserialize(v, **params_)  # type: ignore
                    for k, v in d.items()
                    if good_field(k)
                }
            # deserialize needs to do a shallow conversion from a dict as deep conversion is taken care of already.
            #
            if hasattr(python_class, "from_dict"):
                try:
                    return python_class.from_dict(
                        {k: v for k, v in d.items() if k not in ignored_keys},
                        python_class_identifier=python_class_identifier,
                    )  # type: ignore
                except Exception:
                    return python_class.from_dict(
                        {k: v for k, v in d.items() if k not in ignored_keys}
                    )
            if deep:
                d = {
                    k: deserialize(v, **params_)  # type: ignore
                    for k, v in d.items()
                    if good_field(k)
                }
            else:
                d = {k: v for k, v in d.items() if good_field(k)}
            return python_class(**d)
        if not deep:
            return d
        return {k: deserialize(v, **params_) for k, v in d.items() if good_field(k)}  # type: ignore
    if not deep:
        return d
    if isinstance(d, str):
        if d.startswith(type_marker):
            return get_class(type_name_adapter(d[len(type_marker) :]))
        elif d.startswith(path_marker):
            return Path(path_adapter(d[len(path_marker) :]))
        return d
    if isinstance(d, bytes):
        if d.startswith(lambda_marker):
            return cloudpickle.loads(d[len(lambda_marker) :])
        if d.startswith(function_marker):
            return cloudpickle.loads(d[len(function_marker) :])
        if d.startswith(cloudpickle_marker):
            return cloudpickle.loads(d[len(cloudpickle_marker) :])
        # if d.startswith(JSON_START):
        #     return json.loads(d[JSON_START:])
        return d
    if isinstance(d, tuple) or isinstance(d, list):
        return type(d)(
            deserialize(_, **params_)  # type: ignore
            for _ in d
        )
    return d
