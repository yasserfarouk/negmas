"""
Common datastructures used in the outcomes module.
"""
from dataclasses import dataclass, fields
from enum import Enum
from typing import Any, Collection, Dict, List, Mapping, Tuple, Union

__all__ = [
    "Outcome",
    "OutcomeType",
    "OutcomeRange",
    "ResponseType",
    "is_outcome",
]


class ResponseType(Enum):
    """Possible answers to offers during negotiation."""

    ACCEPT_OFFER = 0
    REJECT_OFFER = 1
    END_NEGOTIATION = 2
    NO_RESPONSE = 3
    WAIT = 4


@dataclass
class OutcomeType:
    """A helper class allowing for definition of types that behave as outcomes (either in the form of dict or tuple).

    This class is intended to be used when a simple tuple or dict is not enough for describing an outcome (e.g. to use
    editor features like auto-completion of members). You simply define your class as a dataclass and add your fields to
    it then inherit from OutcomeType. As we do nothing in the __init__ function, that is compatible with python
    dataclasses.


    Examples:

        >>> from negmas.outcomes import OutcomeType, Issue
        >>> @dataclass
        ... class MyOutcome(OutcomeType):
        ...     price: float = 0.0
        ...     quantity: int = 0

        You can use MyOutcome as an outcome directly or convert it to a tuple/dict for other functions

        >>> outcome = MyOutcome(price=2.0, quantity=3)
        >>> outcome.price
        2.0
        >>> outcome['price']
        2.0
        >>> outcome.astuple()
        (2.0, 3)
        >>> outcome.asdict()
        {'price': 2.0, 'quantity': 3}

        You can also use outputs from issues to initialize your class

        >>> issues = [Issue(name='price', values=(0.0, 3.0)), Issue(name='quantity', values=10)]
        >>> sample = Issue.sample(issues=issues, n_outcomes=1)[0]

        >>> outcome = MyOutcome(**sample)
        >>> outcome.price == outcome['price']
        True


    """

    def __getitem__(self, item):
        """Makes the outcome type behave like a dict"""

        return self.__dict__[item]

    def keys(self) -> Tuple[str, ...]:
        return tuple(_.name for _ in fields(self))

    def values(self) -> Tuple[str, ...]:
        return tuple(self.__dict__[_.name] for _ in fields(self))

    def astuple(self):
        """Converts the outcome to a tuple where the order of items is the same as they are defined as fields"""

        return tuple(self.__dict__[_.name] for _ in fields(self))

    def asdict(self):
        """Converts the outcome to a dict containing all fields"""

        return {_.name: self.__dict__[_.name] for _ in fields(self)}

    def get(self, name, default: Any = None):
        """Acts like dict.get"""
        try:
            return getattr(self, name, default)
        except:
            return default


Value = Union[int, float, str]
"""Possible value types for a single issue"""

Outcome = Union[
    OutcomeType,
    Tuple[Value],
    Dict[str, Value],
]
"""An outcome is either a tuple of values or a dict with name/value pairs."""

OutcomeRange = Mapping[
    Union[int, str],
    Union[
        int,
        float,
        str,
        List[int],
        List[float],
        List[str],
        Tuple[int, int],
        Tuple[float, float],
        List[Tuple[Union[int, float], Union[int, float]]],
    ],
]
"""Represents a range of outcomes."""

Outcomes = Collection[Outcome]
OutcomeRanges = Collection[OutcomeRange]


def is_outcome(x: Any) -> bool:
    """Checks if x is acceptable as an outcome type"""

    return isinstance(x, dict) or isinstance(x, tuple) or isinstance(x, OutcomeType)
