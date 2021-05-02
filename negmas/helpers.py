#!/usr/bin/env python
"""A set of utilities that can be used by agents developed for the platform.

This set of utlities can be extended but must be backward compatible for at
least two versions
"""
from pathlib import Path
import warnings
import base64
from types import LambdaType, FunctionType
import atexit
import copy
import datetime
import importlib
import itertools
import json
import logging
import math
import os
import pathlib
import random
import re
import socket
import string
import sys
import traceback
from collections import defaultdict
import concurrent
from concurrent.futures import TimeoutError
from concurrent.futures.thread import ThreadPoolExecutor
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)

import colorlog
import dill as pickle
import inflect
import numpy as np
import pandas as pd
import scipy.stats as stats
import stringcase
import yaml

from negmas.config import NEGMAS_CONFIG
from negmas.generics import GenericMapping, IterableMapping, gmap, ikeys

if TYPE_CHECKING:
    pass

__all__ = [
    "get_one_float",
    "get_one_int",
    "make_range",
    "TimeoutError",
    "TimeoutCaller",
    "PATH",
    "shortest_unique_names",
    "create_loggers",
    # 'MultiIssueUtilityFunctionMapping',
    "ReturnCause",
    "Distribution",  # A probability distribution
    "snake_case",
    "camel_case",
    "unique_name",
    "is_nonzero_file",
    "pretty_string",
    "ConfigReader",
    "get_class",
    "import_by_name",
    "get_full_type_name",
    "instantiate",
    "humanize_time",
    "gmap",
    "ikeys",
    "Floats",
    "DEFAULT_DUMP_EXTENSION",
    "dump",
    "add_records",
    "load",
    "exception2str",
    "is_jsonable",
    "is_lambda_function",
    "is_non_lambda_function",
]
# conveniently named classes
TYPE_START = "__TYPE__:"
BYTES_START = "__BYTES__:"
PATH_START = "__PATH__:"
"""Maps from a single issue to a Negotiator function."""
# MultiIssueUtilityFunctionMapping = Union[
#    Callable[['Issues'], 'UtilityFunction'], Mapping['Issues', 'UtilityFunction']]  # type: ignore
# """Maps between multiple issues and a Negotiator function."""
ParamList = List[Union[int, str]]
GenericMappings = List[GenericMapping]
IterableMappings = List[IterableMapping]
# MultiIssueUtilityFunctionMappings = List[MultiIssueUtilityFunctionMapping]
ParamLists = Iterable[ParamList]
Floats = List[float]

COMMON_LOG_FILE_NAME = "./logs/{}_{}.txt".format(
    "log", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
)

MODULE_LOG_FILE_NAME: Dict[str, str] = dict()

LOGS_BASE_DIR = "./logs"

DEFAULT_DUMP_EXTENSION = NEGMAS_CONFIG.get("default_dump_extension", "json")

PATH = Union[pathlib.Path, str]


def get_one_int(i: Union[int, Tuple[int, int]]):
    if isinstance(i, int):
        return i
    return random.randint(*i)


def get_one_float(rng: Union[float, Tuple[float, float]]):
    if isinstance(rng, float):
        return rng
    return random.random() * (rng[1] - rng[0]) + rng[0]


def make_range(x: Union[Any, Tuple[Any, Any]]) -> Tuple[Any, Any]:
    if isinstance(x, Iterable):
        return x
    return (x, x)


class ReturnCause(Enum):
    TIMEOUT = 0
    SUCCESS = 1
    FAILURE = 2


def shortest_unique_names(
    strs: List[str], sep=".", max_compression=False, guarantee_unique=False
):
    """
    Finds the shortest unique strings starting from the end of each input
    string based on the separator.

    The final strings will only be unique if the inputs are unique.

    Args:
        strs: A list of strings
        sep: The separator used to separate fields in each string
        max_compression: If True, each string will be further compressed
                         by taking the shortest prefix that keeps the
                         strings unique (if they were originally unique)
        guarantee_unique: If given, random characters will be postfixed on
                         strings to guarantee uniquness

    Example:
        given ["a.b.cat", "d.e.f", "a.d.cat"] it will generate ["b.c", "f", "d.cat"]
        if max_compression was false and will generate ["b", "f", "d"] if it was
        True
    """
    if len(strs) < 2:
        return strs
    if guarantee_unique and len(set(strs)) != len(strs):
        chars = string.digits + string.ascii_letters
        for i in range(len(strs) - 1):
            others = set(strs[:i] + strs[i + 1 :])
            while strs[i] in others:
                for a in chars:
                    if strs[i] + a not in others:
                        strs[i] = strs[i] + a
                        break
                else:
                    strs[i] = strs[i] + unique_name("", False, 1, "")

    lsts = [_.split(sep) for _ in strs]
    names = [_[-1] for _ in lsts]
    if len(names) != len(set(names)):
        locs = defaultdict(list)
        for i, s in enumerate(names):
            locs[s].append(i)
        mapping = {"": ""}
        for s, l in locs.items():
            if len(s) < 1:
                continue
            if len(l) == 1:
                mapping[strs[l[0]]] = s
                continue
            strs_new = [sep.join(lsts[_][:-1]) for _ in l]
            prefixes = shortest_unique_names(
                strs_new, sep, max_compression, guarantee_unique
            )
            for loc, prefix in zip(l, prefixes):
                x = sep.join([prefix, s])
                if x.startswith(sep):
                    x = x[len(sep) :]
                mapping[strs[loc]] = x
        strs = [mapping[_] for _ in strs]
    else:
        strs = names
    if not max_compression:
        return strs
    for i, s in enumerate(strs):
        for j in range(1, len(s)):
            for k in itertools.chain(range(i), range(i + 1, len(strs))):
                if strs[k][:j] == s[:j]:
                    break
            else:
                strs[i] = s[:j]
                break
    return strs


def create_loggers(
    file_name: Optional[str] = None,
    module_name: Optional[str] = None,
    screen_level: Optional[int] = logging.WARNING,
    file_level: Optional[int] = logging.DEBUG,
    format_str: str = "%(asctime)s - %(levelname)s - %(message)s",
    colored: bool = True,
    app_wide_log_file: bool = True,
    module_wide_log_file: bool = False,
) -> logging.Logger:
    """
    Create a set of loggers to report feedback.

    The logger created can log to both a file and the screen at the  same time
    with adjustable level for each of them. The default is to log everything to
    the file and to log WARNING at least to the screen

    Args:
        module_wide_log_file:
        app_wide_log_file:
        file_name: The file to export_to the logs to. If None only the screen
                    is used for logging. If empty, a time-stamp is used
        module_name: The module name to use. If not given the file name
                    without .py is used
        screen_level: level of the screen logger
        file_level: level of the file logger
        format_str: the format of logged items
        colored: whether or not to try using colored logs

    Returns:
        logging.Logger: The logger

    """
    if module_name is None:
        module_name = __file__.split("/")[-1][:-3]
    # create logger if it does not already exist
    logger = None
    if module_wide_log_file or app_wide_log_file:
        logger = logging.getLogger(module_name)
        if len(logger.handlers) > 0:
            return logger
        logger.setLevel(logging.DEBUG)
    else:
        logger = logging.getLogger()
    # create formatter
    file_formatter = logging.Formatter(format_str)
    if colored and "colorlog" in sys.modules and os.isatty(2) and screen_level:
        date_format = "%Y-%m-%d %H:%M:%S"
        cformat = "%(log_color)s" + format_str
        screen_formatter = colorlog.ColoredFormatter(
            cformat,
            date_format,
            log_colors={
                "DEBUG": "magenta",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            },
        )
    else:
        screen_formatter = logging.Formatter(format_str)
    if screen_level is not None and (module_wide_log_file or app_wide_log_file):
        # create console handler and set level to logdebug
        screen_logger = logging.StreamHandler()
        screen_logger.setLevel(screen_level)
        # add formatter to ch
        screen_logger.setFormatter(screen_formatter)
        # add ch to logger
        logger.addHandler(screen_logger)
    if file_name is not None and file_level is not None:
        file_name = str(file_name)
        if logger is None:
            logger = logging.getLogger(file_name)
            logger.setLevel(file_level)
        if len(file_name) == 0:
            if app_wide_log_file:
                file_name = COMMON_LOG_FILE_NAME
            elif module_wide_log_file and module_name in MODULE_LOG_FILE_NAME.keys():
                file_name = MODULE_LOG_FILE_NAME[module_name]
            else:
                file_name = "{}/{}_{}.txt".format(
                    LOGS_BASE_DIR,
                    module_name,
                    datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
                )
                MODULE_LOG_FILE_NAME[module_name] = file_name

            os.makedirs(f"{LOGS_BASE_DIR}", exist_ok=True)
        os.makedirs(os.path.dirname(file_name), exist_ok=True)  # type: ignore
        file_logger = logging.FileHandler(file_name)
        file_logger.setLevel(file_level)
        file_logger.setFormatter(file_formatter)
        logger.addHandler(file_logger)
    return logger


def snake_case(s: str) -> str:
    """ Converts a string from CamelCase to snake_case

    Example:

        >>> print(snake_case('ThisIsATest'))
        this_is_a_test



    Args:
        s: input string

    Returns:
        str: converted string
    """
    return (
        re.sub("(((?<=[a-z])[A-Z])|([A-Z](?![A-Z]|$)))", "_\\1", s).lower().strip("_")
    )


def camel_case(
    s: str, capitalize_first: bool = False, lower_first: bool = False
) -> str:
    """ Converts a string from snake_case to CamelCase

    Example:

        >>> print(camel_case('this_is_a_test'))
        thisIsATest
        >>> print(camel_case('this_is_a_test', capitalize_first=True))
        ThisIsATest
        >>> print(camel_case('This_is_a_test', lower_first=True))
        thisIsATest
        >>> print(camel_case('This_is_a_test'))
        ThisIsATest

    Args:
        s: input string
        capitalize_first: if true, the first character will be capitalized
        lower_first: If true, the first character will be lowered

    Returns:
        str: converted string
    """
    if len(s) < 1:
        return s
    parts = s.split("_")
    if capitalize_first:
        parts = [_.capitalize() for _ in parts]
    elif lower_first:
        parts = [parts[0].lower()] + [_.capitalize() for _ in parts[1:]]
    else:
        parts = [parts[0]] + [_.capitalize() for _ in parts[1:]]

    return "".join(parts)


def unique_name(
    base: Union[pathlib.Path, str],
    add_time=True,
    add_host=False,
    rand_digits=8,
    sep="/",
) -> str:
    """Return a unique name.

    Can be used to return a unique directory name on the givn base.

    Args:
        base: str (str): base path/string
        add_time (bool, optional): Defaults to True. Add current time
        rand_digits (int, optional): Defaults to 8. The number of random
            characters to add to the name

    Examples:

        >>> a = unique_name('')
        >>> len(a) == 8 + 1 + 6 + 8 + 6
        True

    Returns:
        str: The unique name.

    """
    _time, rand_part = "", ""
    host_part = socket.gethostname() if add_host else ""
    if rand_digits > 0:
        rand_part = "".join(
            random.choices(string.digits + string.ascii_letters, k=rand_digits)
        )
    if add_time:
        _time = datetime.datetime.now().strftime("%Y%m%dH%H%M%S%f")
    sub = _time + host_part + rand_part
    if len(sub) == 0:
        return base
    if len(base) == 0:
        return sub
    return f"{str(base)}{sep}{sub}"


def is_nonzero_file(fpath: str) -> bool:
    """Whether or not the path is for an existing nonzero file.

    Args:
        fpath: path to the file to test. It accepts both str and pathlib.Path

    """
    return os.path.isfile(fpath) and os.path.getsize(fpath) > 0


def pretty_string(src: Any, tab_size=2, compact=False) -> str:
    """Recursively print nested elements.

        Args:
            src (Any): The source to be converted to a printable string
            tab_size (int): Tab size in spaces
            compact (bool): If true the output is  converted into a single line

        Returns:
            str: The pretty version of the input

        Remarks:
            - This function assumes that the patterns `` "`` and ``":`` do not appear anywhere in the input.
              If they appear, the space, : will be removed.
        """
    s = _pretty_string(src, dpth=0, current_key="", tab_size=tab_size)
    if compact:
        return s.replace("\n", "")

    else:
        return s.replace(' "', " ").replace('":', ":")


def _pretty_string(src, dpth=0, current_key="", tab_size=2) -> str:
    """Recursively print nested elements.

    Args:
        dpth (int): Current depth
        current_key (str): Current key being printed
        tab_size: Tab size in spaces

    Returns:
        str: The pretty version of the input
    """

    def tabs(n):
        return " " * n * tab_size  # or 2 or 8 or...

    output = ""
    if isinstance(src, dict):
        output += tabs(dpth) + "{\n"
        for key, value in src.items():
            output += _pretty_string(value, dpth + 1, key) + "\n"
        output += tabs(dpth) + "}"
    elif isinstance(src, list) or isinstance(src, tuple):
        output += tabs(dpth) + "[\n"
        for litem in src:
            output += _pretty_string(litem, dpth + 1) + "\n"
        output += tabs(dpth) + "]"
    else:
        if len(current_key) > 0:
            output += tabs(dpth) + '"%s":%s' % (current_key, src)
        else:
            output += tabs(dpth) + "%s" % src
    return output


class LazyInitializable(object):
    """Base Negotiator for all agents

    Supports a set_params function that can be used for lazy initialization
    """

    def __init__(self) -> None:
        super().__init__()

    def set_params(self, **kwargs) -> None:
        """Sets the attributes of the object.

        This function can be used to set the attributes of any object to the
        same values used in its construction which allows for lazy
        initialization.

        Args:
            **kwargs: The parameters usually passed to the constructor as a dict

        Example:

            >>> class A(LazyInitializable):
            ...     def __init__(self, a=None, b=None) -> None:
            ...         super().__init__()
            ...         self.a = a
            ...         self.b = b

            Now you can do the following::

            >>> a = A()
            >>> a.set_params(a=3, b=2)

            which will be equivalent to:

            >>> b = A(a=3, b=2)

        Remarks:
            - See ``adjust_params()`` for an example in which the constuctor needs to do more processing than just
              assinging its inputs to instance members.

        """
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.adjust_params()

    def adjust_params(self) -> None:
        """Adjust the internal attributes following ``set_attributes()`` or construction using ``__init__()``.

        This function needs to be implemented only if the constructor needs to
        do some processing on the inputs other than assigning it to instance
        attributes. In such case, move these adjustments to this function and
        call it in the constructor.

        Examples:

            >>> class A(object):
            ...     def __init__(self, a=None, b=None):
            ...         self.a = a
            ...         self.b = b if b is not None else []

            should now be defined as follows:

            >>> class A(LazyInitializable):
            ...     def __init__(self, a, b):
            ...         super().__init__()
            ...         self.a = a
            ...         self.b = b
            ...         self.adjust_params()
            ...
            ...     def adjust_params(self):
            ...         if self.b is None: self.b = []

        Remarks:
            - Remember to call `super().__init__()` first in your constructor and to call your `adjust_params()` by
              the end of the constructor.
            - The constructor should ONLY copy the parameters it receives to internal variables and then calls
              `adjust_params()` if any more processing is needed. This makes it possible to use `set_params()` with
              this object.
            - You should **never** call `adjust_params()` directly anywhere.
        """


class Distribution(object):
    """Any distribution from scipy.stats with overloading of addition and multiplication.

    Args:
            dtype (str): Data type of the distribution as a string. It must be one defined in `scipy.stats`
            loc (float): The location of the distribution (corresponds to mean in Gaussian)
            scale (float): The _scale of the distribution (corresponds to standard deviation in Gaussian)
            multipliers: An iterable of other distributon to *multiply* with this one
            adders: An iterable of other utility_priors to *add* to this one
            **kwargs:

    Examples:

        >>> d2 = Distribution('uniform')
        >>> print(d2.mean())
        0.5

        >>> try:
        ...     d = Distribution('something crazy')
        ... except ValueError as e:
        ...     print(str(e))
        Unknown distribution something crazy

    """

    def __init__(self, dtype: str, **kwargs) -> None:
        super().__init__()
        dist = getattr(stats, dtype.lower(), None)
        if dist is None:
            raise ValueError(f"Unknown distribution {dtype}")
        if "loc" not in kwargs.keys():
            kwargs["loc"] = 0.0
        if "scale" not in kwargs.keys():
            kwargs["scale"] = 1.0

        self.dist = dist(**kwargs)
        self.dtype = dtype
        self.__cached = None

    @classmethod
    def around(
        cls,
        value: float = 0.5,
        range: Tuple[float, float] = (0.0, 1.0),
        uncertainty: float = 0.5,
    ) -> "Distribution":
        """
        Generates a uniform distribution around the input value in the given range with given uncertainty

        Args:
            value: The value to generate the distribution around
            range: The range of possible values
            uncertainty: The uncertainty level required. 0.0 means no uncertainty and 1.0 means full uncertainty

        Returns:
            Distribution A uniform distribution around `value` with uncertainty (scale) `uncertainty`
        """
        if uncertainty >= 1.0:
            return cls(dtype="uniform", loc=range[0], scale=range[1])
        if uncertainty <= 0.0:
            return cls(dtype="uniform", loc=value, scale=0.0)
        scale = uncertainty * (range[1] - range[0])
        loc = max(range[0], (random.random() - 1.0) * scale + value)
        if loc + scale > range[1]:
            loc -= loc + scale - range[1]
        return cls(dtype="uniform", loc=loc, scale=scale)

    def mean(self) -> float:
        if self.dtype != "uniform":
            raise NotImplementedError(
                "Only uniform distributions are supported for now"
            )
        if self.scale < 1e-6:
            return self.loc
        mymean = self.dist.mean()
        return float(mymean)

    def __float__(self):
        return float(self.mean())

    def __and__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            return float(other)
        if self.dtype == "uniform":
            beg = max(self.loc, other.loc)
            end = min(self.scale + self.loc, other.loc + other.scale)
            return Distribution(self.dtype, loc=beg, scale=end - beg)
        raise NotImplementedError()

    def __or__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            return float(other)
        # if self.dtype == "uniform":
        #     raise NotImplementedError(
        #         "Current implementation assumes an overlap otherwise a mixture must be returned"
        #     )
        # beg = min(self.loc, other.loc)
        # end = max(self.scale + self.loc, other.loc + other.scale)
        # return Distribution(self.dtype, loc=beg, scale=end - beg)
        raise NotImplementedError()

    def prob(self, val: float) -> float:
        """Returns the probability for the given value
        """
        return self.dist.prob(val)

    def sample(self, size: int = 1) -> np.ndarray:
        return self.dist.rvs(size=size)

    @property
    def loc(self):
        return self.dist.kwds.get("loc", 0.0)

    @property
    def scale(self):
        return self.dist.kwds.get("scale", 0.0)

    def min(self):
        return self.loc - self.scale

    def max(self):
        return self.loc + self.scale

    def __str__(self):
        if self.dtype == "uniform":
            return f"U({self.loc}, {self.loc+self.scale})"
        return f"{self.dtype}(loc:{self.loc}, scale:{self.scale})"

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        return result

    __repr__ = __str__

    def __eq__(self, other):
        return float(self) == other

    def __ne__(self, other):
        return float(self) == other

    def __lt__(self, other):
        return float(self) == other

    def __le__(self, other):
        return float(self) == other

    def __gt__(self, other):
        return float(self) == other

    def __ge__(self, other):
        return float(self) == other

    def __sub__(self, other):
        return float(self) - other

    def __add__(self, other):
        return float(self) + other

    def __radd__(self, other):
        return float(self) + other

    def __mul__(self, other):
        return float(self) * float(other)

    def __rmul__(self, other):
        return float(other) * float(self)

    def __divmod__(self, other):
        return float(self).__divmod__(other)


_inflect_engine = inflect.engine()


class ConfigReader:
    @classmethod
    def _parse_children_config(cls, children, scope):
        """Parses children in the given scope"""
        remaining_children = {}
        myconfig = {}
        setters = []
        for key, v in children.items():
            k, class_name = cls._split_key(key)
            if isinstance(v, Dict):
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
    def _split_key(cls, key: str) -> Tuple[str, Optional[str]]:
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
    def read_config(
        cls, config: Union[str, dict], section: str = None
    ) -> Dict[str, Any]:
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
            with open(config, "r") as f:
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
        config: Union[str, dict],
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

        def _set_simple_config(key, v) -> Optional[Dict[str, Any]]:
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
            if isinstance(v, Dict):
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


class Proxy:
    """A general proxy class."""

    def __init__(self, obj):
        self._obj = obj

    def __getattr__(self, item):
        return getattr(self._obj, item)


def get_full_type_name(t: Union[Type[Any], Callable, str]) -> str:
    """Gets the ful typename of a type. You *should not* pass an instance to this function but it may just work.

    An exception is that if the input is of type `str` or if it is None, it will be returned as it is"""
    if t is None or isinstance(t, str):
        return t
    if not hasattr(t, "__module__") and not hasattr(t, "__name__"):
        t = type(t)
    return t.__module__ + "." + t.__name__


def import_by_name(full_name: str) -> Any:
    """Imports something form a module using its full name"""
    if not isinstance(full_name, str):
        return full_name
    modules: List[str] = []
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
    class_name: Union[str, Type], module_name: str = None, scope: dict = None, allow_nonstandard_names=False,
) -> Type:
    """Imports and creates a class object for the given class name"""
    if not isinstance(class_name, str):
        return class_name
    modules: List[str] = []
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
    class_name: Union[str, Type], module_name: str = None, scope: dict = None, **kwargs
) -> Any:
    """Imports and instantiates an object of a class"""
    return get_class(class_name, module_name)(**kwargs)


def humanize_time(secs, align=False, always_show_all_units=False):
    """
    Prints time that is given as seconds in human readable form. Useful only for times >=1sec.

    :param secs: float: number of seconds
    :param align: bool, optional: whether to align outputs so that they all take the same size (not implemented)
    :param always_show_all_units: bool, optional: Whether to always show days, hours, and minutes even when they
                                are zeros. default False
    :return: str: formated string with the humanized form
    """
    units = [("d", 86400), ("h", 3600), ("m", 60), ("s", 1)]
    parts = []
    for unit, mul in units:
        if secs / mul >= 1 or mul == 1 or always_show_all_units:
            if mul > 1:
                n = int(math.floor(secs / mul))
                secs -= n * mul
            else:
                n = secs if secs != int(secs) else int(secs)
            if align:
                parts.append("%2d%s%s" % (n, unit, ""))
            else:
                parts.append("%2d%s%s" % (n, unit, ""))
    return ":".join(parts)

def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except:
        return False

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, bytes):
            encoded =  base64.b64encode(obj)  # b'ZGF0YSB0byBiZSBlbmNvZGVk' (notice the "b")
            return BYTES_START + encoded.decode('ascii')            #
        elif isinstance(obj, Path):
            return PATH_START + str(obj)
        elif not is_jsonable(obj):
            # it may be a type. Always convert types to full names when saving to json
            try:
                obj  = TYPE_START + get_full_type_name(obj)
            except:
                return obj
            return obj
        else:
            return super().default(obj)

class NpDecorder(json.JSONDecoder):
    def default(self, obj):
        if isinstance(obj, str):
            if obj.startswith(BYTES_START):
                return base64.b64decode(obj[BYTES_START:])  # b'ZGF0YSB0byBiZSBlbmNvZGVk' (notice the "b")
            elif obj.startswith(TYPE_START):
                return get_class(obj[TYPE_START:])  # b'ZGF0YSB0byBiZSBlbmNvZGVk' (notice the "b")
            elif obj.startswith(PATH_START):
                return Path(obj[PATH_START:])  # b'ZGF0YSB0byBiZSBlbmNvZGVk' (notice the "b")
        return super().default(obj)

def dump(
    d: Any, file_name: Union[str, os.PathLike, pathlib.Path], sort_keys=True, compact=False
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
            json.dump(d, f, sort_keys=sort_keys, indent=2 if not compact else None, cls=NpEncoder)
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


def load(file_name: Union[str, os.PathLike, pathlib.Path]) -> Any:
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
        with open(file_name, "r") as f:
            d = json.load(f)
    elif file_name.suffix == ".yaml":
        with open(file_name, "r") as f:
            yaml.safe_load(f)
    elif file_name.suffix == ".pickle":
        with open(file_name, "rb") as f:
            d = pickle.load(f)
    elif file_name.suffix == ".csv":
        d = pd.read_csv(file_name).to_dict()
    else:
        raise ValueError(f"Unknown extension {file_name.suffix} for {file_name}")
    return d


def is_lambda_function(obj):
    """Checks if the given object is a lambda function"""
    return isinstance(obj, LambdaType) and obj.__name__ == "<lambda>"

def is_non_lambda_function(obj):
    """Checks if the given object is a lambda function"""
    return isinstance(obj, FunctionType) and obj.__name__ != "<lambda>"

def add_records(
    file_name: Union[str, os.PathLike], data: Any, col_names: Optional[List[str]] = None
) -> None:
    """
    Adds records to a csv file

    Args:

        file_name: file name
        data: data to use for creating the record
        col_names: Names in the data.

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
    if file_name.exists():
        new_file = False
        with open(file_name, "r") as f:
            header = f.readline().strip().strip("\n")
        cols = header.split(",")
        for col in cols:
            if len(col) > 0 and col not in data.columns:
                data[col] = None
        if set([_ for _ in data.columns]) == set(cols):
            data = data.loc[:, cols]
        else:
            try:
                old_data = pd.read_csv(file_name, index_col=None)
                data = pd.concat((old_data, data), axis=0, ignore_index=True)
            except Exception:
                warnings.warn(
                    f"Failed to read data from file {str(file_name)} will override it"
                )

            mode = "w"
            new_file = True

    data.to_csv(str(file_name), index=False, index_label="", mode=mode, header=new_file)


def exception2str(limit=None, chain=True) -> str:
    return traceback.format_exc(limit=limit, chain=chain)


class TimeoutCaller:
    pool = None

    @classmethod
    def run(cls, to_run, timeout: float):
        pool = cls.get_pool()
        future = pool.submit(to_run)
        try:
            result = future.result(timeout)
            return result
        except TimeoutError as s:
            future.cancel()
            raise s

    @classmethod
    def get_pool(cls):
        if cls.pool is None:
            cls.pool = ThreadPoolExecutor()
        return cls.pool

    @classmethod
    def cleanup(cls):
        if cls.pool is not None:
            try:
                cls.pool.shutdown(wait=False)
                for thread in cls.pool._threads:
                    del concurrent.futures.thread._threads_queues[thread]
            except:
                warnings.warn(
                    "NegMAS have finished processing but there are some "
                    "threads still hanging there!! If your program does "
                    "not die by itself. Please press Ctrl-c to kill it"
                )


atexit.register(TimeoutCaller.cleanup)
