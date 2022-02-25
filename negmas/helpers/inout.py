#!/usr/bin/env python
"""A set of utilities that can be used by agents developed for the platform.

This set of utlities can be extended but must be backward compatible for at
least two versions
"""
from __future__ import annotations

import base64
import json
import os
import pathlib
from os import PathLike
from pathlib import Path
from typing import Any, Iterable

import dill as pickle
import inflect
import numpy as np
import pandas as pd
import stringcase
import yaml

from negmas import warnings
from negmas.config import NEGMAS_CONFIG

from .types import TYPE_START, get_class, get_full_type_name, is_jsonable

__all__ = [
    "is_nonzero_file",
    "ConfigReader",
    "DEFAULT_DUMP_EXTENSION",
    "dump",
    "load",
    "add_records",
    "TYPE_START",
]
# conveniently named classes
BYTES_START = "__BYTES__:"
PATH_START = "__PATH__:"
"""Maps from a single issue to a Negotiator function."""

DEFAULT_DUMP_EXTENSION = NEGMAS_CONFIG.get("default_dump_extension", "json")


def is_nonzero_file(fpath: PathLike) -> bool:
    """Whether or not the path is for an existing nonzero file.

    Args:
        fpath: path to the file to test. It accepts both str and pathlib.Path

    """
    return os.path.isfile(fpath) and os.path.getsize(fpath) > 0


_inflect_engine = inflect.engine()


class ConfigReader:
    @classmethod
    def _split_key(cls, key: str) -> tuple[str, str | None]:
        """Splits the key into a key name and a class name

        Remarks:

            - Note that if the given key has multiple colons the first two will be parsed as key name: class name and
              the rest will be ignored. This can be used to add comments

        """
        keys = key.split(":")
        if len(keys) == 1:
            return keys[0], None
        else:
            return keys[0], keys[1]

    @classmethod
    def _parse_children_config(cls, children, scope):
        """Parses children in the given scope"""
        remaining_children = {}
        myconfig = {}
        setters = []
        for key, v in children.items():
            k, class_name = cls._split_key(key)
            if isinstance(v, dict):
                if class_name is None:
                    class_name = stringcase.pascalcase(k)
                the_class = get_class(class_name=class_name, scope=scope)
                obj, obj_children = the_class.from_config(
                    config=v,
                    ignore_children=False,
                    try_parsing_children=True,
                    scope=scope,
                )
                if obj_children is not None and len(obj_children) > 0:
                    remaining_children[k] = obj_children
                setter_name = "set_" + k
                if hasattr(cls, setter_name):
                    setters.append((setter_name, obj))
                else:
                    myconfig[k] = obj
            elif isinstance(v, Iterable) and not isinstance(v, str):
                singular = _inflect_engine.singular_noun(k)
                if singular is False:
                    singular = k
                if class_name is None:
                    class_name = stringcase.pascalcase(singular)
                setter_name = "set_" + k
                objs = []
                for current in list(v):
                    the_class = get_class(class_name=class_name, scope=scope)
                    obj = the_class.from_config(
                        config=current,
                        ignore_children=True,
                        try_parsing_children=True,
                        scope=scope,
                    )
                    objs.append(obj)
                if hasattr(cls, setter_name):
                    setters.append((setter_name, objs))
                else:
                    myconfig[k] = objs
            else:
                # not a dictionary and not an iterable.
                remaining_children[k] = v

        return myconfig, remaining_children, setters

    @classmethod
    def read_config(cls, config: str | dict, section: str = None) -> dict[str, Any]:
        """
        Reads the configuration from a file or a dict and prepares it for parsing

        Args:
            config: Either a file name or a dictionary
            section: A section in the file or a key in the dictionary to use for loading params

        Returns:

            A dict ready to be parsed by from_config

        Remarks:


        """
        if isinstance(config, str):
            # If config is a string, assume it is a file and read it from the appropriate location
            def exists(nm):
                return os.path.exists(nm) and not os.path.isdir(nm)

            if not exists(config):
                name = pathlib.Path("./") / pathlib.Path(config)
                if exists(name):
                    config = str(name.absolute())
                else:
                    name = (pathlib.Path("./.negmas") / config).absolute()
                    if exists(name):
                        config = str(name)
                    else:
                        name = (
                            pathlib.Path(os.path.expanduser("~/.negmas")) / config
                        ).absolute()
                        if exists(name):
                            config = str(name)
                        else:
                            raise ValueError(f"Cannot find config in {config}.")
            with open(config) as f:
                if config.endswith(".json"):
                    config = json.load(f)
                elif config.endswith(".cfg"):
                    config = eval(f.read())
                elif config.endswith(".yaml") or config.endswith(".yml"):
                    config = yaml.safe_load(f)
                else:
                    raise ValueError(f"Cannot parse {config}")

        if section is not None:
            config = config[section]  # type: ignore

        return config  # type: ignore

    @classmethod
    def from_config(
        cls,
        config: str | dict,
        section: str = None,
        ignore_children: bool = True,
        try_parsing_children: bool = True,
        scope=None,
    ):
        """
        Creates an object of this class given the configuration info

        Args:
            config: Either a file name or a dictionary
            section: A section in the file or a key in the dictionary to use for loading params
            ignore_children: If true then children will be ignored and there will be a single return
            try_parsing_children: If true the children will first be parsed as `ConfigReader` classes if they are not
            simple types (e.g. int, str, float, Iterable[int|str|float]
            scope: The scope at which to evaluate any child classes. This MUST be passed as scope=globals() if you are
            having any children that are to be parsed.

        Returns:

            An object of cls if ignore_children is True or a tuple with an object of cls and a dictionary with children
            that were not parsed.

        Remarks:

            - This function will return an object of its class after passing the key-value pairs found in the config to
              the init function.

            - Requiring passing scope=globals() to this function is to get around the fact that in python eval() will be
              called with a globals dictionary based on the module in which the function is defined not called. This means
              that in general when eval() is called to create the children, it will not have access to the class
              definitions of these children (except if they happen to be imported in this file). To avoid this problem
              causing an undefined_name exception, the caller must pass her globals() as the scope.

        """
        config = cls.read_config(config=config, section=section)

        if config is None:
            if ignore_children:
                return None
            else:
                return None, {}

        # now we have a dict called config which has our configuration

        myconfig = {}  # parts of the config that can directly be parsed
        children = {}  # parts of the config that need further parsing
        setters = (
            []
        )  # the setters are those configs that have a set_ function for them.

        def _is_simple(x):
            """Tests whether the input can directly be parsed"""
            return (
                x is None
                or isinstance(x, int)
                or isinstance(x, str)
                or isinstance(x, float)
                or (
                    isinstance(x, Iterable)
                    and not isinstance(x, dict)
                    and all(_is_simple(_) for _ in list(x))
                )
            )

        def _set_simple_config(key, v) -> dict[str, Any] | None:
            """Sets a simple value v for key taken into accout its class and the class we are constructing"""
            key_name, class_name = cls._split_key(key)
            _setter = "set_" + key_name
            params = {}
            if hasattr(cls, _setter):
                setters.append((_setter, v))
                return None
            params[key_name] = (
                v
                if class_name is None
                else get_class(class_name=class_name, scope=scope)(v)
            )
            return params

        # read the configs key by key and try to parse anything that is simple enough to parse

        for k, v in config.items():  # type: ignore
            if isinstance(v, dict):
                children[k] = v
            elif isinstance(v, Iterable) and not isinstance(v, str):
                lst = list(v)
                if all(_is_simple(_) for _ in lst):
                    # that is a simple value of the form k:class_name = v. We construct class_name (if it exists) with v
                    # notice that we need to remove class_name when setting the key in myconfig
                    val = _set_simple_config(k, v)
                    if val is not None:
                        myconfig.update(val)
                else:
                    children[k] = v  # type: ignore
            else:
                # that is a simple value of the form k:class_name = v. We construct class_name (if it exists) with v
                val = _set_simple_config(k, v)
                if val is not None:
                    myconfig.update(val)

        # now myconfig has all simply parsed parts and children has all non-parsed parts

        if len(children) > 0 and try_parsing_children:
            if scope is None:
                ValueError(
                    f"scope is None but that is not allowed. You must pass scope=globals() or scope=locals() to "
                    f"from_config. If your classes are defined in the global scope pass globals() and if they "
                    f"are defined in local scope then pass locals(). You can only pass scope=None if you are "
                    f"sure that all of the constructor parameters of the class you are creating are simple "
                    f"values like ints floats and strings."
                )
            parsed_conf, remaining_children, setters = cls._parse_children_config(
                children=children, scope=scope
            )
            myconfig.update(parsed_conf)
            children = remaining_children

        main_object = cls(**myconfig)  # type: ignore

        if try_parsing_children:
            # we will only have setters if we have children
            for setter, value in setters:
                getattr(main_object, setter)(value)

        if ignore_children:
            return main_object
        return main_object, children


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, bytes):
            encoded = base64.b64encode(
                obj
            )  # b'ZGF0YSB0byBiZSBlbmNvZGVk' (notice the "b")
            return BYTES_START + encoded.decode("ascii")  #
        elif isinstance(obj, Path):
            return PATH_START + str(obj)
        elif not is_jsonable(obj):
            # it may be a type. Always convert types to full names when saving to json
            try:
                obj = TYPE_START + get_full_type_name(obj)
            except:
                return obj
            return obj
        else:
            return super().default(obj)


class NpDecorder(json.JSONDecoder):
    def default(self, obj):
        if isinstance(obj, str):
            if obj.startswith(BYTES_START):
                return base64.b64decode(
                    obj[BYTES_START:]
                )  # b'ZGF0YSB0byBiZSBlbmNvZGVk' (notice the "b")
            elif obj.startswith(TYPE_START):
                return get_class(
                    obj[TYPE_START:]
                )  # b'ZGF0YSB0byBiZSBlbmNvZGVk' (notice the "b")
            elif obj.startswith(PATH_START):
                return Path(
                    obj[PATH_START:]
                )  # b'ZGF0YSB0byBiZSBlbmNvZGVk' (notice the "b")
        return super().default(obj)  #  type: ignore


def dump(
    d: Any,
    file_name: str | os.PathLike | pathlib.Path,
    sort_keys=True,
    compact=False,
) -> None:
    """
    Saves an object depending on the extension of the file given. If the filename given has no extension,
    `DEFAULT_DUMP_EXTENSION` will be used

    Args:
        d: Object to save
        file_name: file name
        sort_keys: If true, the keys will be sorted before saving
        compact: If given, a compact representation will be tried

    Remarks:

        - Supported formats are json, yaml
        - If None is given, the file will be created but will be empty
        - Numpy arrays will be converted to lists before being dumped

    """
    file_name = pathlib.Path(file_name).expanduser().absolute()
    if file_name.suffix == "":
        file_name = pathlib.Path(str(file_name) + "." + DEFAULT_DUMP_EXTENSION)

    if d is None:
        with open(file_name, "w") as f:
            pass
    if file_name.suffix == ".json":
        with open(file_name, "w") as f:
            json.dump(
                d,
                f,
                sort_keys=sort_keys,
                indent=2 if not compact else None,
                cls=NpEncoder,
            )
    elif file_name.suffix == ".yaml":
        with open(file_name, "w") as f:
            yaml.safe_dump(d, f)
    elif file_name.suffix == ".pickle":
        with open(file_name, "wb") as f:
            pickle.dump(d, f)
    elif file_name.suffix == ".csv":
        if not isinstance(d, pd.DataFrame):
            try:
                d = pd.DataFrame(d)
            except Exception as e:
                raise ValueError(f"Failed to convert to a dataframe: {str(e)}")
        d.to_csv(file_name)
    else:
        raise ValueError(f"Unknown extension {file_name.suffix} for {file_name}")


def load(file_name: str | os.PathLike | pathlib.Path) -> Any:
    """
    Loads an object depending on the extension of the file given. If the filename given has no extension,
    `DEFAULT_DUMP_EXTENSION` will be used

    Args:
        file_name: file name

    Remarks:

        - Supported formats are json, yaml
        - If None is given, the file will be created but will be empty

    """
    file_name = pathlib.Path(file_name).expanduser().absolute()
    if file_name.suffix == "":
        file_name = pathlib.Path(str(file_name) + "." + DEFAULT_DUMP_EXTENSION)
    d = {}
    if not file_name.exists() or os.stat(file_name).st_size < 2:
        return d

    if file_name.suffix == ".json":
        with open(file_name) as f:
            d = json.load(f)
    elif file_name.suffix == ".yaml":
        with open(file_name) as f:
            yaml.safe_load(f)
    elif file_name.suffix == ".pickle":
        with open(file_name, "rb") as f:
            d = pickle.load(f)
    elif file_name.suffix == ".csv":
        d = pd.read_csv(file_name).to_dict()  # type: ignore
    else:
        raise ValueError(f"Unknown extension {file_name.suffix} for {file_name}")
    return d


def add_records(
    file_name: str | os.PathLike,
    data: Any,
    col_names: list[str] | None = None,
    raise_exceptions=False,
) -> None:
    """
    Adds records to a csv file

    Args:

        file_name: file name
        data: data to use for creating the record
        col_names: Names in the data.
        raise_exceptions: If given, exceptions  are raised on failure

    Returns:

        None

    Remarks:

        - If col_names are not given, the function will try to normalize the input data if it
          was a dict or a list of dicts

    """
    if col_names is None and (
        isinstance(data, dict)
        or (isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict))
    ):
        data = pd.json_normalize(data)
    else:
        data = pd.DataFrame(data=data, columns=col_names)
    if len(data) < 1:
        return
    file_name = pathlib.Path(file_name)
    file_name.parent.mkdir(parents=True, exist_ok=True)
    new_file = True
    mode = "a"
    # if file_name.exists():
    if is_nonzero_file(file_name):
        new_file = False
        with open(file_name) as f:
            header = f.readline().strip().strip("\n")
        cols = header.split(",")
        for col in cols:
            if len(col) > 0 and col not in data.columns:
                data[col] = None
        if {_ for _ in data.columns} == set(cols):
            data = data.loc[:, cols]
        else:
            try:
                old_data = pd.read_csv(file_name, index_col=None)
                data = pd.concat((old_data, data), axis=0, ignore_index=True)  # type: ignore
            except Exception as e:
                if raise_exceptions:
                    raise e
                warnings.warn(
                    f"Failed to read data from file {str(file_name)} will override it\n{e}",
                    warnings.NegmasIOWarning,
                )

            mode = "w"
            new_file = True

    data.to_csv(str(file_name), index=False, index_label="", mode=mode, header=new_file)
