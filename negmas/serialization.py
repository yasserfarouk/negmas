"""
Implements serialization to and from strings and secondary storage.

"""
import warnings
import json
from typing import Any, Dict, Iterable, Optional
import cloudpickle

import numpy as np
from pandas import json_normalize

from .helpers import (
    get_class,
    is_lambda_function,
    is_non_lambda_function,
    is_jsonable,
    dump,
    load,
)

__all__ = [
    "serialize",
    "deserialize",
    "dump",
    "load",
    "to_flat_dict",
    "PYTHON_CLASS_IDENTIFIER",
]

PYTHON_CLASS_IDENTIFIER = "__python_class__"
LAMBDA_START = b"__LAMBDAOBJ__:"
FUNCTION_START = b"__FUNCTION_START__:"
# JSON_START = b"__JSON_START__:"
CLOUDPICKLE_START = b"__CLOUDPICKLE_START__:"
SPECIAL_FIELDS = ("_NamedObject__uuid", "_NamedObject__name")
SPECIAL_FIELDS_SHORT_NAMES = ("id", "name")


def to_flat_dict(value, deep=True, add_type_field=False) -> Dict[str, Any]:
    """
    Encodes the given value as a flat dictionary

    Args:
        value: The value to be converted to a flat dictionary
        deep: Converting all sub-objects
        add_type_field: If true, a special field for the object type will be added

    Returns:

    """
    d = serialize(value, add_type_field=add_type_field, deep=deep)
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


def serialize(
    value,
    deep=True,
    add_type_field=True,
    keep_private=False,
    ignore_methods=True,
    ignore_lambda=False,
    objmem=None,
):
    """Encodes the given value as nothing more complex than simple dict
    of either dicts, lists or builtin numeric or string values. The resulting
    dictionary will be json serializable

    Args:

        value: Any object
        deep: Whether we should go deep in the encoding or do a shallow encoding
        add_type_field: Whether to add a type field. If True, A field named `PYTHON_CLASS_IDENTIFIER` will be added
        giving the type of `value`
        keep_private: Keeps fields starting with "_"

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
        if objmem and id(v) in objmem:
            return False
        if ignore_methods and is_non_lambda_function(v):
            return False
        if ignore_lambda and is_lambda_function(v):
            return False
        if not isinstance(k, str):
            return False
        return keep_private or not (k != PYTHON_CLASS_IDENTIFIER and k.startswith("_"))

    def adjust_dict(d):
        if not isinstance(d, dict):
            return d
        for a, b in zip(SPECIAL_FIELDS, SPECIAL_FIELDS_SHORT_NAMES):
            if a in d.keys():
                if b in d.keys() and d[b] != d[a]:
                    warnings.warn(f"Field {a} and {b} already exist and are not equal.")
                d[b] = d[a]
                del d[b]
        return d

    if value is None:
        return None

    if isinstance(value, dict):
        if not deep:
            return adjust_dict({k: v for k, v in value.items()})
        return adjust_dict(
            {
                k: serialize(v, deep=deep, add_type_field=add_type_field, objmem=objmem)
                for k, v in value.items()
                if good_field(k, v, objmem)
            }
        )
    if isinstance(value, Iterable) and not deep:
        # add_to_mem(value)
        return value
    # if isinstance(value, np.ndarray):
    #     return value.tolist()
    if isinstance(value, (list, tuple)) and not isinstance(value, str):
        objmem = add_to_mem(value, objmem)
        return adjust_dict(
            type(value)(
                (
                    serialize(
                        _, deep=deep, add_type_field=add_type_field, objmem=objmem
                    )
                    for _ in value
                )
            )
        )
    if hasattr(value, "to_dict"):
        converted = value.to_dict()
        if isinstance(converted, dict):
            if add_type_field and (PYTHON_CLASS_IDENTIFIER not in converted.keys()):
                converted[PYTHON_CLASS_IDENTIFIER] = (
                    value.__class__.__module__ + "." + value.__class__.__name__
                )
            return adjust_dict({k: v for k, v in converted.items()})
        else:
            return adjust_dict(converted)
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
                f" it is. May be you are serializing an already serialized object"
            )
        return value

    if is_lambda_function(value):
        return LAMBDA_START + cloudpickle.dumps(value)

    if is_non_lambda_function(value):
        return FUNCTION_START + cloudpickle.dumps(value)

    if hasattr(value, "__dict__"):
        if deep:
            objmem = add_to_mem(value, objmem)
            d = {
                k: serialize(v, deep=deep, add_type_field=add_type_field, objmem=objmem)
                for k, v in value.__dict__.items()
                if good_field(k, v, objmem)
            }
        else:
            d = {k: v for k, v in value.__dict__.items() if good_field(k, v, objmem)}
        if add_type_field:
            d[PYTHON_CLASS_IDENTIFIER] = (
                value.__class__.__module__ + "." + value.__class__.__name__
            )
        return adjust_dict(d)

    if hasattr(value, "__slots__"):
        if deep:
            objmem = add_to_mem(value, objmem)
            d = dict(
                zip(
                    (k for k in value.__slots__),
                    (
                        serialize(
                            getattr(value, _),
                            deep=deep,
                            add_type_field=add_type_field,
                            objmem=objmem,
                        )
                        for _ in value.__slots__
                    ),
                )
            )
        else:
            d = dict(
                zip(
                    (k for k in value.__slots__),
                    (getattr(value, _) for _ in value.__slots__),
                )
            )
        if add_type_field:
            d[PYTHON_CLASS_IDENTIFIER] = (
                value.__class__.__module__ + "." + value.__class__.__name__
            )
        return adjust_dict(d)
    if isinstance(value, np.int64):
        return int(value)
    # a builtin
    if is_jsonable(value):
        return value
    try:
        vv = CLOUDPICKLE_START + cloudpickle.dumps(value)
        return vv
    except:
        pass
    warnings.warn(f"{value} of type {type(value)} is not serializable")
    return value


def deserialize(
    d: Any,
    deep=True,
    remove_type_field=True,
    keep_private=False,
    fallback_class_name: Optional[str] = None,
):
    """Decodes a dict/object coming from `serialize`

    Args:

        d: The value to be decoded. If it is not a dict, it is returned as it is.
        deep: If true, decode recursively
        remove_type_field: If true the field called `PYTHON_CLASS_IDENTIFIER` will be removed if found.
        keep_private: If given, private fields (starting with _) will be kept
        fallback_class_name: If given, it is used as the fall-back  type if ``PYTHON_CLASS_IDENTIFIER` is not in the dict.

    Remarks:

        - If the object is not a dict or if it has no `PYTHON_CLASS_IDENTIFIER` field and no `fallback_class_name` is
          given, the input `d` is returned as it is. It will not even be copied.

    See Also:
        `serialize`, `PYTHON_CLASS_IDENTIFIER`



    """

    def good_field(k: str):
        return keep_private or not (k != PYTHON_CLASS_IDENTIFIER and k.startswith("_"))

    if d is None or isinstance(d, int) or isinstance(d, float) or isinstance(d, str):
        return d
    if isinstance(d, dict):
        if remove_type_field:
            python_class_name = d.pop(PYTHON_CLASS_IDENTIFIER, fallback_class_name)
        else:
            python_class_name = d.get(PYTHON_CLASS_IDENTIFIER, fallback_class_name)
        if python_class_name is not None:
            python_class_name = python_class_name
            if python_class_name.endswith("Issue"):
                python_class = get_class("negmas.outcomes.Issue")
            else:
                python_class = get_class(python_class_name)
            # we resolve sub-objects first from the dict if deep is specified before calling deserialize on the class
            if deep:
                d = {k: deserialize(v, deep=deep) for k, v in d.items() if good_field(k)}
            # deserialize needs to do a shallow conversion from a dict as deep conversion is taken care of already.
            if hasattr(python_class, "from_dict"):
                return python_class.from_dict({k: v for k, v in d.items()})
            if deep:
                d = {k: deserialize(v) for k, v in d.items() if good_field(k)}
            else:
                d = {k: v for k, v in d.items() if good_field(k)}
            return python_class(**d)
        if not deep:
            return d
        return {k: deserialize(v, deep=deep) for k, v in d.items() if good_field(k)}
    if not deep:
        return d
    if isinstance(d, str):
        return d
    if isinstance(d, bytes):
        if d.startswith(LAMBDA_START):
            return cloudpickle.loads(d[len(LAMBDA_START):])
        if d.startswith(FUNCTION_START):
            return cloudpickle.loads(d[len(FUNCTION_START):])
        if d.startswith(CLOUDPICKLE_START):
            return cloudpickle.loads(d[len(CLOUDPICKLE_START):])
        # if d.startswith(JSON_START):
        #     return json.loads(d[JSON_START:])
        return d
    if isinstance(d, Iterable):
        return type(d)(deserialize(_) for _ in d)
    return d
