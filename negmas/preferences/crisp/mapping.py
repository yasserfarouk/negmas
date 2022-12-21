from __future__ import annotations

import random
from typing import Callable, Iterable

import numpy as np

from negmas.common import Distribution
from negmas.generics import gmap
from negmas.helpers import get_full_type_name
from negmas.outcomes import Issue, Outcome
from negmas.outcomes.base_issue import DiscreteIssue
from negmas.outcomes.common import os_or_none
from negmas.outcomes.protocols import OutcomeSpace
from negmas.serialization import PYTHON_CLASS_IDENTIFIER, deserialize, serialize

from ..base import OutcomeUtilityMapping
from ..crisp_ufun import UtilityFunction
from ..mixins import StationaryMixin

__all__ = ["MappingUtilityFunction"]


class MappingUtilityFunction(StationaryMixin, UtilityFunction):
    """
    Outcome mapping utility function.

    This is the simplest possible utility function and it just maps a set of `Outcome`s to a set of
    `Value`(s). It is only usable with single-issue negotiations. It can be constructed with wither a mapping
    (e.g. a dict) or a callable function.

    Args:
            mapping: Either a callable or a mapping from `Outcome` to `Value`.
            default: value returned for outcomes causing exception (e.g. invalid outcomes).
            name: name of the utility function. If None a random name will be generated.
            reserved_value: The reserved value (utility of not getting an agreement = utility(None) )

    Examples:

        Single issue outcome case:

        >>> from negmas.outcomes import make_issue
        >>> issue =make_issue(values=['to be', 'not to be'], name='THE problem')
        >>> print(str(issue))
        THE problem: ['to be', 'not to be']
        >>> f = MappingUtilityFunction({'to be':10.0, 'not to be':0.0})
        >>> print(list(map(f, ['to be', 'not to be'])))
        [10.0, 0.0]
        >>> f = MappingUtilityFunction(mapping={'to be':-10.0, 'not to be':10.0})
        >>> print(list(map(f, ['to be', 'not to be'])))
        [-10.0, 10.0]
        >>> f = MappingUtilityFunction(lambda x: float(len(x)))
        >>> print(list(map(f, ['to be', 'not to be'])))
        [5.0, 9.0]

        Multi issue case:

        >>> issues = [make_issue((10.0, 20.0), 'price'), make_issue(['delivered', 'not delivered'], 'delivery')
        ...           , make_issue(5, 'quality')]
        >>> print(list(map(str, issues)))
        ['price: (10.0, 20.0)', "delivery: ['delivered', 'not delivered']", 'quality: (0, 4)']
        >>> f = MappingUtilityFunction(lambda x: x['price'] if x['delivery'] == 'delivered' else -1.0)
        >>> g = MappingUtilityFunction(lambda x: x['price'] if x['delivery'] == 'delivered' else -1.0
        ...     , default=-1000 )
        >>> f({'price': 16.0}) is None
        True
        >>> g({'price': 16.0})
        -1000
        >>> f({'price': 16.0, 'delivery':  'delivered'})
        16.0
        >>> f({'price': 16.0, 'delivery':  'not delivered'})
        -1.0

    Remarks:
        - If the mapping used failed on the outcome (for example because it is not a valid outcome), then the
        ``default`` value given to the constructor (which defaults to None) will be returned.

    """

    def __init__(
        self,
        mapping: OutcomeUtilityMapping,
        default=None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if self.outcome_space is None and isinstance(mapping, dict):
            self.outcome_space = os_or_none(
                None, None, list((_,) for _ in mapping.keys())
            )
        self.mapping = mapping
        self.default = default

    def to_dict(self):
        d = {PYTHON_CLASS_IDENTIFIER: get_full_type_name(type(self))}
        d.update(super().to_dict())
        return dict(
            **d,
            mapping=serialize(self.mapping),
            default=self.default,
        )

    @classmethod
    def from_dict(cls, d):
        d.pop(PYTHON_CLASS_IDENTIFIER, None)
        d["mapping"] = deserialize(d["mapping"])
        return cls(**d)

    def eval(self, offer: Outcome | None) -> Distribution | float | None:
        # noinspection PyBroadException
        if offer is None:
            return self.reserved_value
        try:
            m = gmap(self.mapping, offer)
        except Exception:
            return self.default

        return m

    def xml(self, issues: list[Issue]) -> str:
        """

        Examples:

            >>> from negmas.outcomes import make_issue
            >>> issue =make_issue(values=['to be', 'not to be'], name='THE problem')
            >>> print(str(issue))
            THE problem: ['to be', 'not to be']
            >>> f = MappingUtilityFunction({'to be':10.0, 'not to be':0.0})
            >>> print(list(map(f, ['to be', 'not to be'])))
            [10.0, 0.0]
            >>> print(f.xml([issue]))
            <issue index="1" etype="discrete" type="discrete" vtype="discrete" name="THE problem">
                <item index="1" value="to be"  cost="0"  evaluation="10.0" description="to be">
                </item>
                <item index="2" value="not to be"  cost="0"  evaluation="0.0" description="not to be">
                </item>
            </issue>
            <weight index="1" value="1.0">
            </weight>
            <BLANKLINE>
        """
        if len(issues) > 1:
            raise ValueError(
                "Cannot call xml() on a mapping utility function with more than one issue"
            )
        issue = issues[0]  # type: ignore We will raise an exception if the type is not discrete anyway
        if issue.is_continuous():
            raise ValueError(
                "Cannot call xml() on a mapping utility function with a continuous issue"
            )
        issue: DiscreteIssue
        output = f'<issue index="1" etype="discrete" type="discrete" vtype="discrete" name="{issue.name}">\n'
        if isinstance(self.mapping, Callable):
            for i, k in enumerate(issue.all):
                output += (
                    f'    <item index="{i+1}" value="{k}"  cost="0"  evaluation="{self(k)}" description="{k}">\n'
                    f"    </item>\n"
                )
        else:
            for i, (k, v) in enumerate(self.mapping.items()):
                output += (
                    f'    <item index="{i+1}" value="{k}"  cost="0"  evaluation="{v}" description="{k}">\n'
                    f"    </item>\n"
                )
        output += "</issue>\n"
        output += '<weight index="1" value="1.0">\n</weight>\n'
        return output

    @classmethod
    def random(
        cls,
        outcome_space: OutcomeSpace,
        reserved_value=(0.0, 1.0),
        normalized=True,
        max_cardinality: int = 10000,
    ):
        # todo: corrrect this for continuous outcome-spaces
        if not isinstance(reserved_value, Iterable):
            reserved_value = (reserved_value, reserved_value)
        os = outcome_space.to_largest_discrete(
            levels=10, max_cardinality=max_cardinality
        )
        mn, rng = 0.0, 1.0
        if not normalized:
            mn = 4 * random.random()
            rng = 4 * random.random()
        return cls(
            dict(
                zip(
                    os.enumerate(),
                    np.random.rand(os.cardinality) * rng + mn,
                )
            ),
            reserved_value=reserved_value[0]  # type: ignore
            + random.random() * (reserved_value[1] - reserved_value[0]),  # type: ignore
            outcome_space=os,
        )

    def __str__(self) -> str:
        return f"mapping: {self.mapping}\ndefault: {self.default}"
