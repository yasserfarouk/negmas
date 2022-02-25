#!/usr/bin/env python
"""
A set of utilities to handle strings
"""
from __future__ import annotations

import datetime
import itertools
import math
import random
import re
import socket
import string
import traceback
from collections import defaultdict

__all__ = [
    "shortest_unique_names",
    "snake_case",
    "camel_case",
    "unique_name",
    "pretty_string",
    "exception2str",
    "humanize_time",
    "shorten",
]

COMMON_NAME_PARTS = (
    "Mechanism",
    "Negotiator",
    "Agent",
    "Controller",
    "Acceptance",
    "Component",
    "Model",
    "Strategy",
    "Offering",
    "Entity",
)
"""Default parts of names removed by `shorten` """


def shorten(name: str, length: int = 4, common_parts=COMMON_NAME_PARTS) -> str:
    """
    Returns a short version of the name.

    Remarks:
        - Removes common parts of names in negmas like Negotiator, Agent, etc
        - Keeps Capital letters up to the given length
        - Adds some of the lowercase letters to fit the length
        - If the input is shorter than the length, it is returned as it is
    """
    for p in common_parts:
        if len(name) <= len(p):
            continue
        name = name.replace(p, "")
    if len(name) <= length:
        return name
    caps = [_ for _ in name if _.isupper()]
    if len(caps) >= length:
        return "".join(caps[:length])
    needed = length - len(caps)
    caps = []
    for c in name:
        if len(caps) >= length:
            break
        if c.isupper():
            caps.append(c)
            continue
        if needed < 1:
            continue
        needed -= 1
        caps.append(c)
    return "".join(caps[:length])


def unique_name(
    base,
    add_time=True,
    add_host=False,
    rand_digits=8,
    sep="/",
) -> str:
    """Return a unique name.

    Can be used to return a unique directory name on the givn base.

    Args:
        base: (any): base path/string (it is converted to string whatever it is)
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
    base = str(base)
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


def shortest_unique_names(
    strs: list[str], sep=".", max_compression=False, guarantee_unique=False
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


def snake_case(s: str) -> str:
    """Converts a string from CamelCase to snake_case

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
    """Converts a string from snake_case to CamelCase

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
            output += tabs(dpth) + f'"{current_key}":{src}'
        else:
            output += tabs(dpth) + "%s" % src
    return output


def pretty_string(src, tab_size=2, compact=False) -> str:
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


def exception2str(limit=None, chain=True) -> str:
    return traceback.format_exc(limit=limit, chain=chain)


def humanize_time(
    secs,
    align=False,
    always_show_all_units=False,
    show_us=False,
    show_ms=False,
    always_show_from="",
):
    """
    Prints time that is given as seconds in human readable form. Useful only for times >=1sec.

    :param secs: float: number of seconds
    :param align: bool, optional: whether to align outputs so that they all take the same size (not implemented)
    :param always_show_all_units: bool, optional: Whether to always show days, hours, and minutes even when they
                                are zeros. default False
    :param always_show_from: One of d,h,m,s,ms,u (day, hour, minute, second, milli-sec, micro-sec) to always show
                             as well as everything shorter than it (i.e passing 'm' shows minutes, seconds, ... etc)
    :param show_us: bool, if given microseconds and milliseconds will be shown
    :param show_ms: bool, if given milliseconds will be shown
    :return: str: formated string with the humanized form
    """
    if show_us:
        secs *= 1_000_000
        units = [
            ("d", 86400_000_000),
            ("h", 3600_000_000),
            ("m", 60_000_000),
            ("s", 1_000_000),
            ("ms", 1000),
            ("u", 1),
        ]
    elif show_ms:
        secs *= 1_000
        units = [
            ("d", 86400_000),
            ("h", 3600_000),
            ("m", 60_000),
            ("s", 1_000),
            ("ms", 1),
        ]
    else:
        units = [("d", 86400), ("h", 3600), ("m", 60), ("s", 1)]
    parts = []
    for unit, mul in units:
        if unit == always_show_from:
            always_show_all_units = True
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
    return "".join(parts)
