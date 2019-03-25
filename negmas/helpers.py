#!/usr/bin/env python
"""A set of utilities that can be used by agents developed for the platform.

This set of utlities can be extended but must be backward compatible for at
least two versions
"""
import datetime
import importlib
import json
import logging
import math
import pathlib
import string
import sys
from typing import List, Optional, Iterable, Union, Callable, Mapping, Any, Sequence, Tuple, Dict, Type
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

import colorlog
import numpy as np
import os
import random
import re
import scipy.stats as stats
import inflect
import stringcase
from enum import Enum
import yaml
from negmas.generics import *
import copy

__all__ = [
    'create_loggers',
    # 'MultiIssueUtilityFunctionMapping',
    'ReturnCause',
    # The cause for returning from a sync call
    # 'LazyInitializable',    # A class that allows setting initialization parameters later
    # via a set_params() call
    'LoggerMixin',
    # Adds the ability to log to screen and file
    # The base for all named _entities in the system
    'Distribution',  # A probability distribution
    'snake_case',
    'camel_case',
    'unique_name',
    'is_nonzero_file',
    'pretty_string',
    'ConfigReader',
    'get_class',
    'import_by_name',
    'get_full_type_name',
    'instantiate',
    'humanize_time'
]
# conveniently named classes

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

COMMON_LOG_FILE_NAME = './logs/{}_{}.txt'.format('log', datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))

MODULE_LOG_FILE_NAME: Dict[str, str] = dict()

LOGS_BASE_DIR = './logs'


class ReturnCause(Enum):
    TIMEOUT = 0
    SUCCESS = 1
    FAILURE = 2


def create_loggers(
    file_name: Optional[str] = None,
    module_name: Optional[str] = None,
    screen_level: Optional[int] = logging.WARNING,
    file_level: Optional[int] = logging.DEBUG,
    format_str: str = '%(asctime)s - %(levelname)s - %(message)s',
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
        module_name = __file__.split('/')[-1][:-3]
    # create logger if it does not already exist
    logger = logging.getLogger(module_name)
    if len(logger.handlers) > 0:
        return logger
    logger.setLevel(logging.DEBUG)
    # create formatter
    if colored and 'colorlog' in sys.modules and os.isatty(2):
        date_format = '%Y-%m-%d %H:%M:%S'
        cformat = '%(log_color)s' + format_str
        formatter = colorlog.ColoredFormatter(
            cformat,
            date_format,
            log_colors={
                'DEBUG': 'reset',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'bold_red',
            },
        )
    else:
        formatter = logging.Formatter(format_str)
    if screen_level is not None:
        # create console handler and set level to logdebug
        screen_logger = logging.StreamHandler()
        screen_logger.setLevel(screen_level)
        # add formatter to ch
        screen_logger.setFormatter(formatter)
        # add ch to logger
        logger.addHandler(screen_logger)
    if file_name is not None and file_level is not None:
        file_name = str(file_name)
        if len(file_name) == 0:
            if app_wide_log_file:
                file_name = COMMON_LOG_FILE_NAME
            elif module_wide_log_file and module_name in MODULE_LOG_FILE_NAME.keys():
                file_name = MODULE_LOG_FILE_NAME[module_name]
            else:
                file_name = '{}/{}_{}.txt'.format(LOGS_BASE_DIR,
                    module_name, datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
                )
                MODULE_LOG_FILE_NAME[module_name] = file_name

            os.makedirs(f'{LOGS_BASE_DIR}', exist_ok=True)
        os.makedirs(os.path.dirname(file_name), exist_ok=True) # type: ignore
        file_logger = logging.FileHandler(file_name)
        file_logger.setLevel(file_level)
        file_logger.setFormatter(formatter)
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
    return re.sub(
        '(((?<=[a-z])[A-Z])|([A-Z](?![A-Z]|$)))', '_\\1', s
    ).lower().strip(
        '_'
    )


def camel_case(s: str, capitalize_first: bool = False, lower_first: bool = False) -> str:
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
    parts = s.split('_')
    if capitalize_first:
        parts = [_.capitalize() for _ in parts]
    elif lower_first:
        parts = [parts[0].lower()] +  [_.capitalize() for _ in parts[1:]]
    else:
        parts = [parts[0]] + [_.capitalize() for _ in parts[1:]]

    return ''.join(parts)


def unique_name(base: str, add_time=True, rand_digits=8) -> str:
    """Return a unique name.

    Can be used to return a unique directory name on the givn base.

    Args:
        base: str (str): base path/string
        add_time (bool, optional): Defaults to True. Add current time
        rand_digits (int, optional): Defaults to 8. The number of random
            characters to add to the name

    Examples:

        >>> a = unique_name('')
        >>> len(a) == 8 + 1 + 6 + 8
        True

    Returns:
        str: The unique name.

    """
    if rand_digits > 0:
        characters = string.ascii_letters + string.digits
        password = "".join(
            random.choice(characters) for _ in range(rand_digits)
        )
    else:
        password = ''
    if add_time:
        _time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    else:
        _time = ''
    sub = _time + password
    return os.path.join(base, sub) if len(sub) > 0 else base


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
    s = _pretty_string(src, dpth=0, current_key='', tab_size=tab_size)
    if compact:
        return s.replace('\n', '')

    else:
        return s.replace(' "', ' ').replace('":', ':')


def _pretty_string(src, dpth=0, current_key='', tab_size=2) -> str:
    """Recursively print nested elements.

    Args:
        dpth (int): Current depth
        current_key (str): Current key being printed
        tab_size: Tab size in spaces

    Returns:
        str: The pretty version of the input
    """

    def tabs(n):
        return ' ' * n * tab_size  # or 2 or 8 or...

    output = ''
    if isinstance(src, dict):
        output += tabs(dpth) + '{\n'
        for key, value in src.items():
            output += _pretty_string(value, dpth + 1, key) + '\n'
        output += tabs(dpth) + '}'
    elif isinstance(src, list) or isinstance(src, tuple):
        output += tabs(dpth) + '[\n'
        for litem in src:
            output += _pretty_string(litem, dpth + 1) + '\n'
        output += tabs(dpth) + ']'
    else:
        if len(current_key) > 0:
            output += (tabs(dpth) + '"%s":%s' % (current_key, src))
        else:
            output += (tabs(dpth) + '%s' % src)
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
        pass


class LoggerMixin(object):
    """Parent of all agents that can log to the console/file.

    Args:
        file_name (Optional[str]): File name to use for logging

    Examples:
        Create a temporary file to test with

        >>> from tempfile import mkstemp
        >>> _, f_name = mkstemp()
        >>> l = LoggerMixin(f_name)
        >>> l.loginfo('test info')
        >>> l.logerror('test error')
        >>> l.logwarning('test warning')
        >>> l.logdebug('test debug')
        >>> with open(f_name, 'r') as f:
        ...     lines = f.readlines()
        >>> ('INFO - test info' in lines[0]) if len(lines) > 0 else True
        True
        >>> ('ERROR - test error' in lines[1]) if len(lines) > 1 else True
        True
        >>> ('WARNING - test warning' in lines[2])  if len(lines) > 2 else True
        True
        >>> ('DEBUG - test debug' in lines[3])  if len(lines) > 3 else True
        True

    """

    def __init__(self, file_name: Optional[str] = None, screen_log: bool=False) -> None:
        """Constructor

        Constructs a logger agent

        Args:
            file_name (str, optional): Defaults to None. File used for

        """
        self.log_file_name = file_name
        if screen_log:
            self.logger = create_loggers(self.log_file_name)
        else:
            self.logger = create_loggers(self.log_file_name, screen_level=None)

    def loginfo(self, s: str) -> None:
        """logs info-level information

        Args:
            s (str): The string to log

        """
        self.logger.info(s)

    def logdebug(self, s) -> None:
        """logs debug-level information

        Args:
            s (str): The string to log

        """
        self.logger.debug(s)

    def logwarning(self, s) -> None:
        """logs warning-level information

        Args:
            s (str): The string to log

        """
        self.logger.warning(s)

    def logerror(self, s) -> None:
        """logs error-level information

        Args:
            s (str): The string to log

        """
        self.logger.error(s)


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

    def __init__(
        self,
        dtype: str,
        **kwargs,
    ) -> None:
        super().__init__()
        dist = getattr(stats, dtype.lower(), None)
        if dist is None:
            raise ValueError(f'Unknown distribution {dtype}')
        if 'loc' not in kwargs.keys():
            kwargs['loc'] = 0.0
        if 'scale' not in kwargs.keys():
            kwargs['scale'] = 1.0

        self.dist = dist(**kwargs)
        self.dtype = dtype
        self.__cached = None

    @classmethod
    def around(cls, value: float = 0.5, range: Tuple[float, float] = (0.0, 1.0),
               uncertainty: float = 0.5) -> 'Distribution':
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
            return cls(dtype='uniform', loc=range[0], scale=range[1])
        if uncertainty <= 0.0:
            return cls(dtype='uniform', loc=value, scale=0.0)
        scale = uncertainty * (range[1] - range[0])
        loc = max(range[0], (random.random() - 1.0) * scale + value)
        if loc + scale > range[1]:
            loc -= (loc + scale - range[1])
        return cls(dtype='uniform', loc=loc, scale=scale)

    def mean(self) -> float:
        if self.dtype != 'uniform':
            raise NotImplementedError('Only uniform distributions are supported for now')
        if self.scale < 1e-6:
            return self.loc
        mymean = self.dist.mean()
        return float(mymean)

    def __float__(self):
        return float(self.mean())

    def __and__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            return float(other)
        if self.dtype == 'uniform':
            beg = max(self.loc, other.loc)
            end = min(self.scale + self.loc, other.loc + other.scale)
            return Distribution(self.dtype
                                , loc=beg
                                , scale=end - beg
                                )
        raise NotImplementedError()

    def __or__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            return float(other)
        if self.dtype == 'uniform':
            raise NotImplementedError('Current implementation assumes an overlap otherwise a mixture must be returned')
            beg = min(self.loc, other.loc)
            end = max(self.scale + self.loc, other.loc + other.scale)
            return Distribution(self.dtype
                                , loc=beg
                                , scale=end - beg
                                )
        raise NotImplementedError()

    def prob(self, val: float) -> float:
        """Returns the probability for the given value
        """
        return self.dist.prob(val)

    def sample(self, size: int = 1) -> np.ndarray:
        return self.dist.rvs(size=size)

    @property
    def loc(self):
        return self.dist.kwds.get('loc', 0.0)

    @property
    def scale(self):
        return self.dist.kwds.get('scale', 0.0)

    def __str__(self):
        if self.dtype == 'uniform':
            return f'U({self.loc}, {self.loc+self.scale})'
        return f'{self.dtype}(loc:{self.loc}, scale:{self.scale})'

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
                obj, obj_children = the_class.from_config(config=v, ignore_children=False, try_parsing_children=True
                                                                        , scope=scope)
                if obj_children is not None and len(obj_children) > 0:
                    remaining_children[k] = obj_children
                setter_name = 'set_' + k
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
                setter_name = 'set_' + k
                objs = []
                for current in list(v):
                    the_class = get_class(class_name=class_name, scope=scope)
                    obj = the_class.from_config(config=current, ignore_children=True
                                                              , try_parsing_children=True, scope=scope)
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
        keys = key.split(':')
        if len(keys) == 1:
            return keys[0], None
        else:
            return keys[0], keys[1]

    @classmethod
    def read_config(cls, config: Union[str, dict], section: str = None) -> Dict[str, Any]:
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
                name = pathlib.Path('./') / pathlib.Path(config)
                if exists(name):
                    config = str(name.absolute())
                else:
                    name = (pathlib.Path('./.negmas') / config).absolute()
                    if exists(name):
                        config = str(name)
                    else:
                        name = (pathlib.Path(os.path.expanduser('~/.negmas')) / config).absolute()
                        if exists(name):
                            config = str(name)
                        else:
                            raise ValueError(f'Cannot find config in {config}.')
            with open(config, 'r') as f:
                if config.endswith('.json'):
                    config = json.load(f)
                elif config.endswith('.cfg'):
                    config = eval(f.read())
                elif config.endswith('.yaml') or config.endswith('.yml'):
                    config = yaml.safe_load(f)
                else:
                    raise ValueError(f'Cannot parse {config}')

        if section is not None:
            config = config[section]  # type: ignore

        return config # type: ignore

    @classmethod
    def from_config(cls
                    , config: Union[str, dict]
                    , section: str = None
                    , ignore_children: bool = True
                    , try_parsing_children: bool = True
                    , scope=None):
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

        myconfig = {}   # parts of the config that can directly be parsed
        children = {}   # parts of the config that need further parsing
        setters = []  # the setters are those configs that have a set_ function for them.

        def _is_simple(x):
            """Tests whether the input can directly be parsed"""
            return x is None or isinstance(x, int) or isinstance(x, str) or isinstance(x, float) or \
                   (isinstance(x, Iterable) and not isinstance(x, dict) and all(_is_simple(_) for _ in list(x)))

        def _set_simple_config(key, v) -> Optional[Dict[str, Any]]:
            """Sets a simple value v for key taken into accout its class and the class we are constructing"""
            key_name, class_name = cls._split_key(key)
            _setter = 'set_' + key_name
            params = {}
            if hasattr(cls, _setter):
                setters.append((_setter, v))
                return None
            params[key_name] = v if class_name is None else get_class(class_name=class_name, scope=scope)(v)
            return params

        # read the configs key by key and try to parse anything that is simple enough to parse

        for k, v in config.items(): # type: ignore
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
                    children[k] = v # type: ignore
            else:
                # that is a simple value of the form k:class_name = v. We construct class_name (if it exists) with v
                val = _set_simple_config(k, v)
                if val is not None:
                    myconfig.update(val)

        # now myconfig has all simply parsed parts and children has all non-parsed parts

        if len(children) > 0 and try_parsing_children:
            if scope is None:
                ValueError(f'scope is None but that is not allowed. You must pass scope=globals() or scope=locals() to '
                           f'from_config. If your classes are defined in the global scope pass globals() and if they '
                           f'are defined in local scope then pass locals(). You can only pass scope=None if you are '
                           f'sure that all of the constructor parameters of the class you are creating are simple '
                           f'values like ints floats and strings.')
            parsed_conf, remaining_children, setters = cls._parse_children_config(children=children
                                                                                  , scope=scope)
            myconfig.update(parsed_conf)
            children = remaining_children

        main_object = cls(**myconfig) # type: ignore

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


def get_full_type_name(t: Union[Type[Any], Callable]) -> str:
    """Gets the ful typename of a type. You *should not* pass an instance to this function but it may just work."""
    if not hasattr(t, '__module__') and not hasattr(t, '__name__'):
        t = type(t)
    return t.__module__ + '.' + t.__name__


def import_by_name(full_name: str) -> Any:
    """Imports something form a module using its full name"""
    if not isinstance(full_name, str):
        return full_name
    modules: List[str] = []
    parts = full_name.split('.')
    modules = parts[:-1]
    module_name = '.'.join(modules)
    item_name = parts[:-1]
    if len(modules) < 1:
        raise ValueError(f'Cannot get the object {item_name} in module {module_name}  (modules {modules})')
    module = importlib.import_module(module_name)
    return getattr(module, item_name)


def get_class(class_name: Union[str, Type], module_name: str = None, scope: dict = None) -> Type:
    """Imports and creates a class object for the given class name"""
    if not isinstance(class_name, str):
        return class_name
    modules: List[str] = []
    if module_name is not None:
        modules = module_name.split('.')
    modules += class_name.split('.')
    if len(modules) < 1:
        raise ValueError(f'Cannot get the class {class_name} in module {module_name}  (modules {modules})')
    class_name = stringcase.pascalcase(modules[-1])
    if len(modules) < 2:
        return eval(class_name, scope)
    module_name = '.'.join(modules[:-1])
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def instantiate(class_name: Union[str, Type], module_name: str = None, scope: dict = None, **kwargs) -> Any:
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
