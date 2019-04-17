r"""A set of generic classes and corresponding functions to iterate and index them.


The main goal of having this module is to allow for flexible modeling of collections with *most* python builtin
collections (e.g. `list`, `tuple`, `generator`) or even `Callable`\ s without awkward `isinstance` calls in
other modules of the library.

"""
from typing import Iterable, Union, Mapping, Sequence, Callable, Any, Tuple, Dict

__all__ = [
    "IterableMapping",  # A mapping combining dicts, lists, tuples, and generators
    "GenericMapping",
    # An iterable mapping or a callable (e.g. function/functor)
    # GenericMapping functions (apply to GMapping and IterableMapping)
    "gmap",  # An index on an IterableMapping or a call on a Callable
    "gget",
    # Like `gmap` but can return a default value
    # IterableMapping functions (apply only to IterableMappings)
    "ienumerate",  # enumerates a mapping as key-value pairs
    "iitems",  # enumerates a mapping as key-value pairs
    "ivalues",  # enumerates all values in a mapping (no keys)
    "ikeys",  # enumerates all keys in a mapping (no values)
    "iget",  # Similar to `gget` but applies only to IterableMappings
]
# Generic classes that work with a variety of python mappings like collections or callables
IterableMapping = Union[Mapping, Sequence]
"""Something that can be iterated upon with Key-value pairs (e.g. list, dict, tuple)."""
GenericMapping = Union[Callable[[Any], Any], IterableMapping]
"""Something that can be indexed using [] or called using ()"""


def gmap(group: GenericMapping, param: Any) -> Any:
    """Calls or indexes the group by the param

    Args:
        group: Either a Callable or a Mapping
        param: The parameters to use for mapping

    Examples:

        >>> gmap([1, 23, 44], 1)
        23
        >>> gmap({'a': 3, 'b': 5, 'c': 4}, 'c')
        4
        >>> gmap(lambda x: 3*x, 20)
        60

    Returns:

    """
    if hasattr(group, "__call__"):
        return group(param)  # type: ignore

    return group[param]  # type: ignore


def gget(x: GenericMapping, _key: Any, default=None) -> Any:
    """Get an item from an IterableMapping

    Args:
        x: the generic mapping
        _key: key (must be immutable)
        default: default value if no value attached with the key is found

    Examples:

        Example with a list

        >>> [gget([10, 20, 30], _) for _ in (0, 2,1, -1, 4)]
        [10, 30, 20, 30, None]

        Example with a dictionary

        >>> [gget({'a':10, 'b':20, 'c':30}, _) for _ in ('a', 'c','b', -1, 'd')]
        [10, 30, 20, None, None]

        Example with a tuple

        >>> [gget((10, 20, 30), _) for _ in (0, 2,1, -1, 4)]
        [10, 30, 20, 30, None]

        Example with a generator

        >>> [gget(range(10, 40, 10), _) for _ in (0, 2,1, -1, 4)]
        [10, 30, 20, 30, None]

    Returns:

    """
    if hasattr(x, "apply"):
        # noinspection PyBroadException
        try:
            return x(_key)  # type: ignore

        except Exception:
            return default

    else:
        return iget(x, _key, default)  # type: ignore


def ienumerate(x: IterableMapping) -> Iterable[Tuple[Any, Any]]:
    """Enumerates a GenericMapping.

    Args:
        x (IterableMapping): A generic mapping (see `GenericMapping`)

    Examples:

        Example with a list

        >>> for k, cutoff_utility in ienumerate([10, 20, 30]): print(k, cutoff_utility, end='-')
        0 10-1 20-2 30-

        Example with a dictionary

        >>> for k, cutoff_utility in ienumerate({'a': 10, 'b': 20, 'c': 30}): print(k, cutoff_utility, end='-')
        a 10-b 20-c 30-

        Example with a tuple

        >>> for k, cutoff_utility in ienumerate((10, 20, 30)): print(k, cutoff_utility, end='-')
        0 10-1 20-2 30-

        Example with a generator

        >>> for k, cutoff_utility in ienumerate(range(10, 40, 10)): print(k, cutoff_utility, end='-')
        0 10-1 20-2 30-

    Returns:
        a generator/iterator with tuples of key-value pairs.

    """
    if isinstance(x, Dict):
        return x.items()

    else:
        return enumerate(x)


iitems = ienumerate


def ivalues(x: IterableMapping) -> Iterable[Any]:
    """Returns all keys of the iterable.

    Args:
        x (IterableMapping): A generic mapping (see `GenericMapping`)

    Examples:

        Example with a list

        >>> for k in ivalues([10, 20, 30]): print(k, end='-')
        10-20-30-

        Example with a dictionary

        >>> for k in ivalues({'a': 10, 'b': 20, 'c': 30}): print(k, end='-')
        10-20-30-

        Example with a tuple

        >>> for k in ivalues((10, 20, 30)): print(k, end='-')
        10-20-30-

        Example with a generator

        >>> for k in ivalues(range(10, 40, 10)): print(k, end='-')
        10-20-30-

    Returns:
        a generator/iterator with tuples of key-value pairs.

    """
    if isinstance(x, Dict):
        return list(x.values())

    else:
        return x


def ikeys(x: IterableMapping) -> Iterable[Any]:
    """Returns all keys of the iterable.

    Args:
        x (IterableMapping): A generic mapping (see `GenericMapping`)

    Examples:

        Example with a list

        >>> for k in ikeys([10, 20, 30]): print(k, end='-')
        0-1-2-

        Example with a dictionary

        >>> for k in ikeys({'a': 10, 'b': 20, 'c': 30}): print(k, end='-')
        a-b-c-

        Example with a tuple

        >>> for k in ikeys((10, 20, 30)): print(k, end='-')
        0-1-2-

        Example with a generator

        >>> for k in ikeys(range(10, 40, 10)): print(k, end='-')
        0-1-2-

    Returns:
        a generator/iterator with tuples of key-value pairs.

    """
    if isinstance(x, Dict):
        return list(x.keys())

    else:
        return range(len(x))  # type: ignore


def iget(x: IterableMapping, _key: Any, default=None) -> Any:
    """Get an item from an IterableMapping

    Args:
        x: the generic mapping
        _key: key (must be immutable)
        default: default value if no value attached with the key is found

    Examples:

        Example with a list

        >>> [iget([10, 20, 30], _) for _ in (0, 2,1, -1, 4)]
        [10, 30, 20, 30, None]

        Example with a dictionary

        >>> [iget({'a':10, 'b':20, 'c':30}, _) for _ in ('a', 'c','b', -1, 'd')]
        [10, 30, 20, None, None]

        Example with a tuple

        >>> [iget((10, 20, 30), _) for _ in (0, 2,1, -1, 4)]
        [10, 30, 20, 30, None]

        Example with a generator

        >>> [iget(range(10, 40, 10), _) for _ in (0, 2,1, -1, 4)]
        [10, 30, 20, 30, None]

    Returns:

    """
    if isinstance(x, Dict):
        return x.get(_key, default)

    try:
        return x[_key]  # type: ignore
    except IndexError:
        return default
