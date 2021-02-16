"""
Defines basic concepts related to outcomes

Outcomes in this package are always assumed to be multi-issue outcomes where  single-issue outcomes can be implemented
as the special case with a single issue.

- Both Continuous and discrete issues are supported. All issue will have names. If none is given, a random name will be
  used. It is HIGHLY recommended to always name your issues.
- Outcomes are dictionaries with issue names as keys and issue values as values.

Examples:

  Different ways to create issues:

  >>> issues = [Issue((0.5, 2.0), 'price'), Issue(['2018.10.'+ str(_) for _ in range(1, 4)], 'date')
  ...           , Issue(20, 'count')]
  >>> for _ in issues: print(_)
  price: (0.5, 2.0)
  date: ['2018.10.1', '2018.10.2', '2018.10.3']
  count: (0, 19)

  Outcome example compatible with the given set of issues:

  >>> a = {'price': 1.2, 'date': '2018.10.04', 'count': 4}

"""

import sys
import warnings
import copy
import itertools
import numbers
import random
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, fields
from enum import Enum
from functools import reduce
from operator import mul
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Optional,
    Collection,
    Dict,
    Generator,
    Iterable,
    List,
    Mapping,
    Sequence,
    Tuple,
    Type,
    Union,
)

import numpy as np

from .common import NamedObject
from .generics import ienumerate, iget, ikeys, ivalues
from .helpers import unique_name, PATH
from .java import PYTHON_CLASS_IDENTIFIER

if TYPE_CHECKING:
    from negmas import Mechanism

LARGE_NUMBER = 100
__all__ = [
    "Outcome",
    "OutcomeType",
    "OutcomeRange",
    "ResponseType",
    "Issue",
    "Issues",
    "outcome_is_valid",
    "outcome_is_complete",
    "outcome_range_is_valid",
    "outcome_range_is_complete",
    "outcome_in_range",
    "enumerate_outcomes",
    "sample_outcomes",
    "outcome_as_dict",
    "outcome_as_tuple",
    "outcome_as",
    "is_outcome",
    "num_outcomes",
    "outcome_for",
]


class ResponseType(Enum):
    """Possible answers to offers during negotiation."""

    ACCEPT_OFFER = 0
    REJECT_OFFER = 1
    END_NEGOTIATION = 2
    NO_RESPONSE = 3
    WAIT = 4


class Issue(NamedObject):
    """Encodes an Issue.

    Args:
            values: Possible values for the issue
            name: Name of the issue. If not given, a random name will be generated
            min_value: Minimum value ( used only if the issue values parameter was a callable )
            max_value: Maximum value ( used only if the issue values parameter was a callable )
            value_type: If a type is given, all values will be forced to this type

    Examples:

        >>> print(Issue(['to be', 'not to be'], name='THE problem'))
        THE problem: ['to be', 'not to be']
        >>> print(Issue(3, name='Cars'))
        Cars: (0, 2)
        >>> print(Issue((0.0, 1.0), name='required accuracy'))
        required accuracy: (0.0, 1.0)
        >>> a = Issue((0.0, 1.0), name='required accuracy')
        >>> a.is_uncountable()
        True
        >>> a.is_countable()
        False
        >>> a.is_continuous()
        True
        >>> a = Issue(lambda: random.randint(10), min_value=0, max_value=9)
        >>> a.is_uncountable()
        True
        >>> a.is_countable()
        False
        >>> a.is_continuous()
        False

    Remarks:

        - Issues can be initialized by either an iterable of strings, an integer or a tuple of two values with
          the following meanings:

          - ``list of anything`` : This is an issue that can any value within the given set of values
            (strings, ints, floats, etc)
          - ``int`` : This is an issue that takes any value from 0 to the given value -1 (int)
          - Tuple[ ``float`` , ``float`` ] : This is an issue that can take any real value between the given limits (min, max)
          - Tuple[ ``int`` , ``int`` ] : This is an issue that can take any integer value between the given limits (min, max)
          - ``Callable`` : The callable should take no parameters and should act as a generator of issue values. This
             type of issue is always assumed to be neither countable nor continuous and are called uncountable. For
             example, you can use this type to make an issue that generates all integers from 0 to infinity.
        - If a list is given, min, max must be callable on it.
    """

    def __init__(
        self,
        values: Union[
            List[Any],
            int,
            Tuple[float, float],
            Tuple[int, int],
            List[int],
            Callable[[], Any],
        ],
        name: Optional[str] = None,
        min_value: Any = None,
        max_value: Any = None,
        value_type: Optional[Type] = None,
        id=None,
    ) -> None:
        super().__init__(name, id=id)

        if (
            value_type is not None
            and value_type is not numbers.Integral
            and isinstance(values, numbers.Integral)
        ):
            raise ValueError(
                "Cannot force a type that is not int while passing a singular integer as values"
            )
        # if isinstance(values, int) and values <= LARGE_NUMBER:
        #    values = list(range(values))
        self._value_type: Optional[Type] = value_type
        self._n_values: Union[int, float] = float("inf")
        self._is_generator: bool = False
        self._is_range: bool = False
        self._is_int_range: bool = False
        self._is_float_range: bool = False
        self._is_integer_valued: bool = False
        self._is_real_valued: bool = False

        if isinstance(values, numbers.Integral):
            values = (0, int(values) - 1)

        if isinstance(values, tuple):
            if len(values) != 2:
                raise ValueError(
                    f"Passing {values} is illegal. Issues with ranges need 2-values tuples"
                )

            if value_type is not None:
                values = (value_type(values[0]), value_type(values[1]))
            isint0, isint1 = (
                isinstance(values[0], numbers.Integral),
                isinstance(values[1], numbers.Integral),
            )
            isreal0, isreal1 = (
                isinstance(values[0], numbers.Real)
                and not isinstance(values[0], numbers.Integral),
                isinstance(values[1], numbers.Real)
                and not isinstance(values[0], numbers.Integral),
            )

            if (
                (isint0 and not isint1)
                or (isint1 and not isint0)
                or (isreal0 and not isreal1)
                or (isreal1 and not isreal0)
                or (not (isint0 or isreal0) or not (isint1 or isreal1))
            ):
                raise ValueError(
                    f"Confusing types: Received types ({type(values[0])}, {type(values[1])})"
                )

            values = (values[0], values[1])
            self._is_int_range = isinstance(values[0], numbers.Integral)
            self._is_float_range = (
                isinstance(values[0], numbers.Real) and not self._is_int_range
            )
            self._is_range = True
            self._value_type = type(values[0])
            self._is_integer_valued = isinstance(values[0], numbers.Integral)
            self._is_real_valued = not self._is_integer_valued and isinstance(
                values[0], numbers.Real
            )

            if self._is_int_range:
                self._n_values = values[1] - values[0] + 1

        elif isinstance(values, Callable):
            self._is_generator = isinstance(values, Callable)
        else:
            values = list(values)

            if len(values) < 1:
                raise ValueError("values must include at least one item")

            if value_type is not None:
                values = [value_type(_) for _ in values]
            self._value_type = type(values[0])

            if value_type is None and not all(
                isinstance(_, self._value_type) for _ in values
            ):
                raise ValueError(f"Not all values are of the same type: {values}")
            self._n_values = len(values)
            self._is_integer_valued = isinstance(values[0], numbers.Integral)
            self._is_real_valued = not self._is_integer_valued and isinstance(
                values[0], numbers.Real
            )
        self._values = values

        if isinstance(self._values, tuple):
            self.min_value, self.max_value = self._values
        elif isinstance(self._values, list):
            self.min_value, self.max_value = min(self._values), max(self._values)
        else:
            self.min_value, self.max_value = min_value, max_value

    @property
    def value_type(self):
        return self._value_type

    @property
    def values(self):
        return self._values

    @classmethod
    def from_outcomes(
        cls, outcomes: List["Outcome"], numeric_as_ranges: bool = False
    ) -> List["Issue"]:
        """
        Create a set of issues given some outcomes

        Args:

            outcomes: A list of outcomes
            numeric_as_ranges: If True, all numeric issues generated will have ranges that are defined by the minimum
                               and maximum values of that issue in the given outcomes instead of a list of the values
                               that appeared in them.

        Returns:

            a list of issues that include the given outcomes.

        Remarks:

            - The outcome space spanned by the generated issues can in principle contain many more possible outcomes
              than the ones given

        """

        def convert_type(v, old, values):
            if isinstance(v, numbers.Integral) and not isinstance(
                old, numbers.Integral
            ):
                return float(v)

            if not isinstance(v, numbers.Integral) and isinstance(
                old, numbers.Integral
            ):
                for i, _ in enumerate(values):
                    values[i] = float(_)

                return v

            if isinstance(v, str) and (isinstance(old, numbers.Number)):
                raise ValueError("a string after a number")

            if isinstance(old, str) and (isinstance(v, numbers.Number)):
                raise ValueError("a number after a string")

            return v

        names = None
        n_issues = None
        values = defaultdict(list)

        for i, o in enumerate(outcomes):
            o = outcome_as_dict(o)

            if n_issues is not None and len(o) != n_issues:
                raise ValueError(
                    f"Outcome {o} at {i} has {len(o)} issues but an earlier outcome had {n_issues} issues"
                )
            n_issues = len(o)

            if names is not None and not all(a == b for a, b in zip(names, o.keys())):
                raise ValueError(
                    f"Outcome {o} at {i} has issues {list(o.keys())} but an earlier outcome had issues {names}"
                )
            names = list(o.keys())

            for k, v in o.items():
                if len(values[k]) > 0:
                    try:
                        v = convert_type(v, values[k][-1], values[k])
                    except ValueError as e:
                        raise ValueError(
                            f"Outcome {o} at {i} has value {v} for issue {k} which is incompatible with an earlier "
                            f"value {values[k][-1]} ({str(e)})"
                        )
                values[k].append(v)

        for k, vals in values.items():
            values[k] = sorted(list(set(vals)))

        if numeric_as_ranges:
            return [
                Issue(values=(v[0], v[-1]), name=n)
                if len(v) > 0 and (isinstance(v[0], numbers.Number))
                else Issue(values=v, name=n)
                for n, v in values.items()
            ]
        else:
            return [Issue(values=v, name=n) for n, v in values.items()]

    @classmethod
    def to_xml_str(cls, issues: List["Issue"], enumerate_integer: bool = True) -> str:
        """Converts the list of issues into a well-formed xml string

        Examples:

            >>> issues = [Issue(values=10, name='i1'), Issue(values=['a', 'b', 'c'], name='i2'),
            ... Issue(values=(2.5, 3.5), name='i3')]
            >>> s = Issue.to_xml_str(issues)
            >>> print(s.strip())
            <negotiation_template>
            <utility_space number_of_issues="3">
            <objective description="" etype="objective" index="0" name="root" type="objective">
                <issue etype="discrete" index="1" name="i1" type="discrete" vtype="discrete">
                    <item index="1" value="0" cost="0" description="0">
                    </item>
                    <item index="2" value="1" cost="0" description="1">
                    </item>
                    <item index="3" value="2" cost="0" description="2">
                    </item>
                    <item index="4" value="3" cost="0" description="3">
                    </item>
                    <item index="5" value="4" cost="0" description="4">
                    </item>
                    <item index="6" value="5" cost="0" description="5">
                    </item>
                    <item index="7" value="6" cost="0" description="6">
                    </item>
                    <item index="8" value="7" cost="0" description="7">
                    </item>
                    <item index="9" value="8" cost="0" description="8">
                    </item>
                    <item index="10" value="9" cost="0" description="9">
                    </item>
                </issue>
                <issue etype="discrete" index="2" name="i2" type="discrete" vtype="discrete">
                    <item index="1" value="a" cost="0" description="a">
                    </item>
                    <item index="2" value="b" cost="0" description="b">
                    </item>
                    <item index="3" value="c" cost="0" description="c">
                    </item>
                </issue>
                <issue etype="real" index="3" name="i3" type="real" vtype="real">
                    <range lowerbound="2.5" upperbound="3.5"></range>
                </issue>
            </objective>
            </utility_space>
            </negotiation_template>

            >>> issues2, _ = Issue.from_xml_str(s)
            >>> print([_.__class__.__name__ for _ in issues2])
            ['Issue', 'Issue', 'Issue']
            >>> print(len(issues2))
            3
            >>> print([str(_) for _ in issues2])
            ["i1: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']", "i2: ['a', 'b', 'c']", 'i3: (2.5, 3.5)']
            >>> print([_.values for _ in issues2])
            [['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], ['a', 'b', 'c'], (2.5, 3.5)]


        """
        output = (
            f'<negotiation_template>\n<utility_space number_of_issues="{len(issues)}">\n'
            f'<objective description="" etype="objective" index="0" name="root" type="objective">\n'
        )

        for indx, issue in enumerate(issues):
            if issue._values is None or issue._is_generator:
                raise ValueError(
                    f"Cannot convert issue {issue.name} to xml because it has no defined values (or uses a "
                    f"generator)."
                )

            if issue._is_int_range:
                if enumerate_integer:
                    output += f'    <issue etype="discrete" index="{indx + 1}" name="{issue.name}" type="discrete" vtype="discrete">\n'

                    for i, v in enumerate(
                        range(issue._values[0], issue._values[1] + 1)
                    ):
                        output += f'        <item index="{i + 1}" value="{v}" cost="0" description="{v}">\n        </item>\n'
                    output += "    </issue>\n"
                else:
                    output += (
                        f'    <issue etype="integer" index="{indx + 1}" name="{issue.name}" type="integer" vtype="integer"'
                        f' lowerbound="{issue._values[0]}" upperbound="{issue._values[1]}" />\n'
                    )
            elif issue._is_float_range:
                output += (
                    f'    <issue etype="real" index="{indx + 1}" name="{issue.name}" type="real" vtype="real">\n'
                    f'        <range lowerbound="{issue._values[0]}" upperbound="{issue._values[1]}"></range>\n    </issue>\n'
                )
            else:
                output += f'    <issue etype="discrete" index="{indx + 1}" name="{issue.name}" type="discrete" vtype="discrete">\n'

                for i, v in enumerate(issue._values):
                    output += f'        <item index="{i + 1}" value="{v}" cost="0" description="{v}">\n        </item>\n'
                output += "    </issue>\n"
        output += f"</objective>\n</utility_space>\n</negotiation_template>"

        return output

    @classmethod
    def to_genius(
        cls, issues: Iterable["Issue"], file_name: PATH, enumerate_integer: bool = True
    ) -> None:
        """Exports a the domain issues to a GENIUS XML file.

        Args:

            issues: The issues to be exported
            file_name (str): File name to export to

        Returns:

            A List[Issue] or Dict[Issue]


        Examples:

            >>> import pkg_resources
            >>> issues, _ = Issue.from_genius(file_name = pkg_resources.resource_filename('negmas'
            ...                                      , resource_name='tests/data/Laptop/Laptop-C-domain.xml'))
            >>> Issue.to_genius(issues=issues, file_name = pkg_resources.resource_filename('negmas'
            ...                                    , resource_name='tests/data/LaptopConv/Laptop-C-domain.xml'))
            >>> issues2, _ = Issue.from_genius(file_name = pkg_resources.resource_filename('negmas'
            ...                                    , resource_name='tests/data/LaptopConv/Laptop-C-domain.xml'))
            >>> print('\\n'.join([' '.join(list(issue.all)) for issue in issues]))
            Dell Macintosh HP
            60 Gb 80 Gb 120 Gb
            19'' LCD 20'' LCD 23'' LCD
            >>> print('\\n'.join([' '.join(list(issue.all)) for issue in issues2]))
            Dell Macintosh HP
            60 Gb 80 Gb 120 Gb
            19'' LCD 20'' LCD 23'' LCD

            - Forcing Single outcome

            >>> issues, _ = Issue.from_genius(file_name = pkg_resources.resource_filename('negmas'
            ...                                      , resource_name='tests/data/Laptop/Laptop-C-domain.xml')
            ...     , force_single_issue=True, keep_value_names=False, keep_issue_names=False)
            >>> print(list(issues[0].all))
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
            >>> Issue.to_genius(issues=issues, enumerate_integer=True
            ...     , file_name = pkg_resources.resource_filename('negmas'
            ...                                   , resource_name='tests/data/LaptopConv/Laptop-C-domain.xml'))
            >>> issues3, _ = Issue.from_genius(file_name=pkg_resources.resource_filename('negmas'
            ...                                    , resource_name='tests/data/LaptopConv/Laptop-C-domain.xml'))
            >>> print([list(issue.all) for issue in issues3])
            [['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26']]

        Remarks:
            See ``from_xml_str`` for all the parameters

        """
        with open(file_name, "w") as f:
            f.write(
                cls.to_xml_str(issues=list(issues), enumerate_integer=enumerate_integer)
            )

    @classmethod
    def from_xml_str(
        cls,
        xml_str: str,
        force_single_issue=False,
        force_numeric=False,
        keep_value_names=True,
        keep_issue_names=True,
        safe_parsing=True,
        n_discretization: Optional[int] = None,
        max_n_outcomes: int = 1_000_000,
    ) -> Tuple[Optional[List["Issue"]], Optional[List[str]]]:
        """Exports a list/dict of issues from a GENIUS XML file.

        Args:

            xml_str (str): The string containing GENIUS style XML domain issue definitions
            force_single_issue (bool): Tries to generate a MappingUtility function with a single issue which is the
            product of all issues in the input
            keep_value_names (bool): Keep names of values
            keep_issue_names (bool): Keep names of issues
            safe_parsing (bool): Turn on extra checks
            n_discretization (Optional[int]): If not None, real valued issues are discretized with the given
            number of values
            max_n_outcomes (int): Maximum number of outcomes allowed (effective only if force_single_issue is True)

        Returns:

            - List[Issue] The issues (note that issue names will be stored in the name attribute of each issue if keep_issue_names)
            - List[dict] A list of agent information dicts each contains 'agent', 'class', 'utility_file_name'

        Examples:

            >>> import pkg_resources
            >>> domain_file_name = pkg_resources.resource_filename('negmas'
            ...                                      , resource_name='tests/data/Laptop/Laptop-C-domain.xml')
            >>> issues, _ = Issue.from_xml_str(open(domain_file_name, 'r').read()
            ... , force_single_issue=True, keep_value_names=False, keep_issue_names=False)
            >>> issue = issues[0]
            >>> print(issue.cardinality)
            27
            >>> print(list(issue.all))
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]

            >>> issues, _ = Issue.from_xml_str(open(domain_file_name, 'r').read()
            ... , force_single_issue=True, keep_value_names=False, keep_issue_names=True)
            >>> print(issues[0].name)
            Laptop-Harddisk-External Monitor
            >>> print(len(issues))
            1
            >>> print(list(issues[0].all))
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]

            >>> issues, _ = Issue.from_xml_str(open(domain_file_name, 'r').read()
            ... , force_single_issue=True, keep_value_names=True, keep_issue_names=False)
            >>> issue = issues[0]
            >>> print(issue.cardinality)
            27
            >>> print('\\n'.join(list(issue.all)[:5]))
            Dell+60 Gb+19'' LCD
            Dell+60 Gb+20'' LCD
            Dell+60 Gb+23'' LCD
            Dell+80 Gb+19'' LCD
            Dell+80 Gb+20'' LCD

            >>> issues, _ = Issue.from_xml_str(open(domain_file_name, 'r').read()
            ... , force_single_issue=False, keep_issue_names=False, keep_value_names=True)
            >>> type(issues)
            <class 'list'>
            >>> str(issues[0])
            "0: ['Dell', 'Macintosh', 'HP']"
            >>> print([_.cardinality for _ in issues])
            [3, 3, 3]
            >>> print('\\n'.join([' '.join(list(issue.all)) for issue in issues]))
            Dell Macintosh HP
            60 Gb 80 Gb 120 Gb
            19'' LCD 20'' LCD 23'' LCD

            >>> issues, _ = Issue.from_xml_str(open(domain_file_name, 'r').read()
            ... , force_single_issue=False, keep_issue_names=True, keep_value_names=True)
            >>> len(issues)
            3
            >>> str(issues[0])
            "Laptop: ['Dell', 'Macintosh', 'HP']"
            >>> print([_.cardinality for _ in issues])
            [3, 3, 3]
            >>> print('\\n'.join([' '.join(list(issue.all)) for issue in issues]))
            Dell Macintosh HP
            60 Gb 80 Gb 120 Gb
            19'' LCD 20'' LCD 23'' LCD


            >>> issues, _ = Issue.from_xml_str(open(domain_file_name, 'r').read()
            ... , force_single_issue=False, keep_issue_names=False, keep_value_names=False)
            >>> len(issues)
            3
            >>> type(issues)
            <class 'list'>
            >>> str(issues[0]).split(': ')[-1]
            '(0, 2)'
            >>> print([_.cardinality for _ in issues])
            [3, 3, 3]

            >>> domain_file_name = pkg_resources.resource_filename('negmas'
            ...                              , resource_name='tests/data/fuzzyagent/single_issue_domain.xml')
            >>> issues, _ = Issue.from_xml_str(open(domain_file_name, 'r').read()
            ... , force_single_issue=False, keep_issue_names=False, keep_value_names=False)
            >>> len(issues)
            1
            >>> type(issues)
            <class 'list'>
            >>> str(issues[0]).split(': ')[-1]
            '(10.0, 40.0)'
            >>> print([_.cardinality for _ in issues])
            [inf]
        """
        root = ET.fromstring(xml_str)

        if safe_parsing and root.tag != "negotiation_template":
            raise ValueError(f"Root tag is {root.tag}: negotiation_template")

        utility_space = None
        agents = []

        for child in root:
            if child.tag == "utility_space":
                utility_space = child

                for _ in utility_space:
                    if _.tag == "objective":
                        utility_space = _

                        break
            elif child.tag == "agent":
                agents.append(child.attrib)

        if utility_space is None:
            if safe_parsing:
                raise ValueError(f"No objective child was found in the root")
            utility_space = root
        issues_dict: Dict[Union[int, str], Any] = {}
        issue_info = {}
        all_discrete = True

        for child in utility_space:
            if child.tag == "issue":
                indx = int(child.attrib["index"]) - 1
                myname = str(child.attrib["name"])
                issue_key: Union[int, str] = myname if keep_issue_names else indx
                issue_info[issue_key] = {"name": myname, "index": indx}
                info = {"type": "discrete", "etype": "discrete", "vtype": "discrete"}

                for a in ("type", "etype", "vtype"):
                    info[a] = child.attrib.get(a, info[a])
                mytype = info["type"]

                if mytype == "discrete":
                    issues_dict[issue_key] = []

                    for item in child:
                        if item.tag == "item":
                            item_indx = int(item.attrib["index"]) - 1
                            item_name = item.attrib.get("value", None)
                            item_key = (
                                item_name
                                if keep_value_names
                                and item_name is not None
                                and not force_numeric
                                else item_indx
                            )

                            if (
                                item_key not in issues_dict[issue_key]
                            ):  # ignore repeated items
                                issues_dict[issue_key].append(item_key)

                    if not keep_value_names:
                        issues_dict[issue_key] = len(issues_dict[issue_key])
                elif mytype in ("integer", "real"):
                    lower_, upper_ = (
                        child.attrib.get("lowerbound", None),
                        child.attrib.get("upperbound", None),
                    )

                    for rng_child in child:
                        if rng_child.tag == "range":
                            lower_, upper_ = (
                                rng_child.attrib.get("lowerbound", lower_),
                                rng_child.attrib.get("upperbound", upper_),
                            )
                    if lower_ is None:
                        if upper_ is not None and float(upper_) < 0:
                            lower_ = str(
                                -(sys.maxsize // 2)
                                if mytype == "integer"
                                else float("-inf")
                            )
                        else:
                            lower_ = "0"
                    if upper_ is None:
                        upper_ = str(
                            (sys.maxsize // 2) if mytype == "integer" else float("-inf")
                        )
                    if mytype == "integer":
                        lower, upper = int(lower_), int(upper_)

                        if keep_value_names:
                            if (upper + 1 - lower) > 1_000_000:
                                warnings.warn(
                                    f"Issue {issue_key} has bounds ({lower}, {upper}) which means "
                                    f"{upper + 1 - lower} values. Consider NOT using keep_value_names"
                                    f" to reduce memory consumption"
                                )
                            issues_dict[issue_key] = list(range(lower, upper + 1))
                        else:
                            issues_dict[issue_key] = lower, upper
                    else:
                        lower, upper = float(lower_), float(upper_)
                        if n_discretization is None:
                            all_discrete = False
                            issues_dict[issue_key] = lower, upper
                        else:
                            issues_dict[issue_key] = n_discretization
                else:
                    # I should add the real-valued issues_dict code here
                    raise ValueError(f"Unknown type: {mytype}")
            else:
                raise ValueError(f"Unknown child for objective: {child.tag}")

        for key, value in zip(ikeys(issues_dict), ivalues(issues_dict)):
            issues_dict[key] = Issue(
                values=value,
                name=issue_info[key]["name"]
                if keep_issue_names
                else str(issue_info[key]["index"]),
            )
        issues = list(issues_dict.values())

        if force_single_issue:
            issue_name_ = (
                "-".join([_.name for _ in issues]) if keep_issue_names else "0"
            )

            if all_discrete or n_discretization is not None:
                n_outcomes = None

                if max_n_outcomes is not None:
                    n_items = [len(list(_.alli(n=n_discretization))) for _ in issues]
                    n_outcomes = reduce(mul, n_items, 1)

                    if n_outcomes > max_n_outcomes:
                        return None, None

                if keep_value_names:
                    if len(issues) > 1:
                        all_values = itertools.product(
                            *[
                                [str(_) for _ in issue.alli(n=n_discretization)]
                                for issue_key, issue in zip(range(len(issues)), issues)
                            ]
                        )
                        all_values = list(
                            map(lambda items: "+".join(items), all_values)
                        )
                    else:
                        all_values = [
                            str(_) for _ in issues[0].alli(n=n_discretization)
                        ]
                    issues = [Issue(values=all_values, name=issue_name_)]
                else:
                    if n_outcomes is None:
                        n_items = [_.cardinality for _ in issues]
                        n_outcomes = reduce(mul, n_items, 1)
                    issues = [Issue(values=n_outcomes, name=issue_name_)]
            else:
                return None, None

        return issues, agents

    @classmethod
    def from_genius(
        cls,
        file_name: PATH,
        force_single_issue=False,
        force_numeric=False,
        keep_value_names=True,
        keep_issue_names=True,
        safe_parsing=True,
        n_discretization: Optional[int] = None,
        max_n_outcomes: int = 1_000_000,
    ) -> Tuple[Optional[List["Issue"]], Optional[List[str]]]:
        """Imports a the domain issues from a GENIUS XML file.

        Args:

            max_n_outcomes: Maximum number of outcomes to use
            n_discretization: Number of discretization levels per issue
            safe_parsing: Add more checks to parsing
            keep_issue_names: Use dictionaries instead of tuples to represent outcomes
            keep_value_names: keep value names in case of strings
            force_numeric: Force the issue values to be numeric
            force_single_issue: Combine all issues into a single issue
            file_name (str): File name to import from

        Returns:

            A two optional lists:

            - List[Issue] containing the issues
            - List[str] containing agent names (that are sometimes stored in the genius domain)


        Examples:

            >>> import pkg_resources
            >>> issues, _ = Issue.from_genius(file_name = pkg_resources.resource_filename('negmas'
            ...                                      , resource_name='tests/data/Laptop/Laptop-C-domain.xml'))
            >>> print([_.name for _ in issues])
            ['Laptop', 'Harddisk', 'External Monitor']

        Remarks:
            See ``from_xml_str`` for all the parameters

        """
        with open(file_name, "r", encoding="utf-8") as f:
            xml_str = f.read()

            return cls.from_xml_str(
                xml_str=xml_str,
                force_single_issue=force_single_issue,
                keep_value_names=keep_value_names,
                keep_issue_names=keep_issue_names,
                safe_parsing=safe_parsing,
                n_discretization=n_discretization,
                max_n_outcomes=max_n_outcomes,
                force_numeric=force_numeric,
            )

    @staticmethod
    def num_outcomes(issues: Iterable["Issue"]) -> Union[int, float]:
        """Returns the total number of outcomes in a set of issues. `-1` indicates infinity"""
        n = 1

        for issue in issues:
            n *= issue.cardinality

        return n

    @staticmethod
    def generate(
        issues: Sequence[
            Union[int, List[str], Tuple[int, int], Callable, Tuple[float, float]]
        ],
        counts: Optional[Sequence[int]] = None,
        names: Optional[Sequence[str]] = None,
    ) -> List["Issue"]:
        """Generates a set of issues with given parameters. Each is optionally repeated

        Args:

            issues: The parameters of the issues
            counts: The number of times to repeat each of the `issues`
            names: The names to assign to the issues. If None, then string representations of integers
                   starting from zero will be used.

        Returns:
            List['Issue']: The list of issues with given conditions

        """
        one_each = counts is None
        int_names = names is None
        result = []
        nxt = 0

        for i, issue in enumerate(issues):
            count = 1 if one_each else counts[i]  # type: ignore

            for _ in range(count):
                name = str(nxt) if int_names else names[i]  # type: ignore
                # if count > 1:
                #    name = name + f' {j}'
                nxt += 1
                result.append(Issue(values=issue, name=name))

        return result

    @property
    def type(self) -> str:
        """The type of the issue.

        Returns:
            str: either 'continuous', 'uncountable' or 'discrete'

        Remarks:

            - IF values is set to None or to a callable, the issue is treated as continuous

        """

        if self._is_float_range:
            return "continuous"

        return "uncountable" if self._is_generator else "discrete"

    def is_numeric(self) -> bool:
        return self._is_integer_valued or self._is_real_valued

    def is_integer(self) -> bool:
        return self._is_integer_valued

    def is_float(self) -> bool:
        return self._is_real_valued

    def is_continuous(self) -> bool:
        """Test whether the issue is a continuous issue

        Returns:
            bool: uncountable (including continuous) or not

        """

        return self._is_float_range

    def is_uncountable(self) -> bool:
        """Test whether the issue has uncountable possible outcomes

        Returns:
            bool: uncountable (including continuous) or not

        """

        return self._is_float_range or self._is_generator

    def is_countable(self) -> bool:
        """Test whether the issue is a discrete issue

        Returns:
            bool: countable or not

        """

        return not self._is_float_range and not self._is_generator

    is_discrete = is_countable

    @property
    def all(self) -> Generator:
        """A generator that generates all possible values.

        Remarks:
            - This function returns a generator for the case when the number of values is very large.
            - If you need a list then use something like:

            >>> from negmas.outcomes import Issue
            >>> list(Issue(5).all)
            [0, 1, 2, 3, 4]

        """

        if self.is_uncountable():
            raise ValueError(
                "Cannot return all possibilities of a continuous/uncountable issue"
            )

        if self._is_int_range:
            yield from range(self._values[0], self._values[1] + 1)
        else:
            yield from self._values  # type: ignore

    def alli(self, n: Optional[int] = 10) -> Generator:
        """A generator that generates all possible values or samples n values for real Issues.

        Remarks:
            - This function returns a generator for the case when the number of values is very large.
            - If you need a list then use something like:

            >>> from negmas.outcomes import Issue
            >>> list(Issue(5).all)
            [0, 1, 2, 3, 4]

        """

        if self._is_float_range:
            if n is None:
                raise ValueError("Real valued issue with no discretization value")
            yield from np.linspace(
                self._values[0], self._values[1], num=n, endpoint=True
            ).tolist()
        elif self._is_generator:
            if n is None:
                raise ValueError("Real valued issue with no discretization value")
            yield from (self._values() for _ in range(n))
        elif self._is_int_range:
            yield from range(self._values[0], self._values[1] + 1)
        else:
            yield from self._values  # type: ignore

    @property
    def cardinality(self) -> Union[int, float]:
        """The number of possible outcomes for the issue. Returns infinity for continuous and uncountable spaces"""

        return self._n_values

    def rand(self) -> Union[int, float, str]:
        """Picks a random valid value."""

        if self._is_float_range:
            return (
                random.random() * (self._values[1] - self._values[0]) + self._values[0]
            )  # type: ignore

        if self._is_int_range:
            return random.randint(*self._values)

        if self._is_generator:
            return self._values()

        return random.choice(self._values)  # type: ignore

    def rand_outcomes(
        self, n: int, with_replacement=False, fail_if_not_enough=False
    ) -> Iterable["Outcome"]:
        """Picks a random valid value."""

        if self._is_int_range:
            if n > self._values[1] and not with_replacement:
                if fail_if_not_enough:
                    raise ValueError(
                        f"Cannot sample {n} outcomes out of {self._values} without replacement"
                    )
                else:
                    return [_ for _ in range(self._values[0], self._values[1] + 1)]

            if with_replacement:
                return np.random.randint(
                    low=self._values[0], high=self._values[1] + 1, size=n
                ).tolist()
            else:
                return random.shuffle(
                    [_ for _ in range(self._values[0], self._values[1] + 1)]
                )[:n]

        if self._is_float_range:
            if with_replacement:
                return (
                    np.random.rand(n) * (self._values[1] - self._values[0])
                    + self._values[0]
                ).tolist()
            else:
                return np.linspace(
                    self._values[0], self._values[1], num=n, endpoint=True
                ).tolist()

        if self._is_generator:
            if not with_replacement:
                raise ValueError(
                    f"values is specified as a callables for this issue. Cannot sample from it without "
                    f"replacement"
                )

            return [self._values() for _ in range(n)]

        if n > len(self._values) and not with_replacement:
            if fail_if_not_enough:
                raise ValueError(
                    f"Cannot sample {n} outcomes out of {self._values} without replacement"
                )
            else:
                return self._values

        return np.random.choice(
            np.asarray(self._values, dtype=self._value_type),
            size=n,
            replace=with_replacement,
        ).tolist()

    rand_valid = rand

    def rand_invalid(self) -> Union[int, float, str]:
        """Pick a random *invalid* value"""

        if self._is_int_range:
            return random.randint(self.max_value + 1, 2 * self.max_value)

        if self._is_float_range:
            return (
                random.random() * (self.max_value - self.min_value)
                + self.max_value * 1.1
            )

        if self._is_generator:
            raise ValueError(
                f"Cannot generate invalid outcomes because values is given as a callable"
            )

        if self._is_real_valued:
            return random.random() * self.max_value + self.max_value * 1.1

        if self._is_integer_valued:
            return random.randint(self.max_value + 1, self.max_value * 2)

        return unique_name("") + str(random.choice(self._values)) + unique_name("")

    @classmethod
    def enumerate(
        cls,
        issues: Collection["Issue"],
        max_n_outcomes: int = None,
        astype: Type = dict,
    ) -> List["Outcome"]:
        """
        Enumerates the outcomes of a list of issues.

        Args:
            issues: The list of issues.
            max_n_outcomes: The maximum number of outcomes to return
            astype: The type to use for the returned outcomes. Can be `typle`
                    `dict`, or any `OutcomeType`.
        Returns:
            List of outcomes of the given type.
        """
        n = num_outcomes(issues)

        if n is None or n == float("inf"):
            return cls.sample(
                issues=issues,
                n_outcomes=max_n_outcomes,
                astype=astype,
                fail_if_not_enough=False,
                with_replacement=False,
            )

        if max_n_outcomes is not None and n > max_n_outcomes:
            values = sample_outcomes(
                issues=issues, n_outcomes=max_n_outcomes, astype=tuple
            )
        else:
            values = enumerate_outcomes(issues, astype=tuple)
        if issubclass(astype, tuple):
            return values
        issue_names = [_.name for _ in issues]
        if issubclass(astype, dict):
            return [outcome_as_dict(value, issue_names) for value in values]
        return [astype(**outcome_as_dict(value, issue_names)) for value in values]

    @classmethod
    def discretize_and_enumerate(
        cls,
        issues: Collection["Issue"],
        n_discretization: int = 10,
        astype: Type = dict,
        max_n_outcomes: int = None,
    ) -> List["Outcome"]:
        """
        Enumerates the outcomes of a list of issues.

        Args:
            issues: The list of issues.
            max_n_outcomes: The maximum number of outcomes to return
            astype: The type to use for the returned outcomes. Can be `typle`
                    `dict`, or any `OutcomeType`.
        Returns:
            List of outcomes of the given type.
        """
        issues = [
            _
            if _.is_countable()
            else Issue(values=_.alli(n_discretization), name=_.name)
            for _ in issues
        ]
        return cls.enumerate(issues, max_n_outcomes=max_n_outcomes, astype=astype)

    @classmethod
    def sample(
        cls,
        issues: Collection["Issue"],
        n_outcomes: int,
        astype: Type = dict,
        with_replacement: bool = True,
        fail_if_not_enough=True,
    ) -> List["Outcome"]:
        """
        Samples some outcomes from the issue space defined by the list of issues

        Args:

            issues: List of issues to sample from
            n_outcomes: The number of outcomes required
            astype: The type of the outcomes. It can be `tuple`, `dict` or any `OutcomeType`
            with_replacement: Whether sampling is with replacement (allowing repetition)
            fail_if_not_enough: IF given then an exception is raised if not enough outcomes are available

        Returns:

            a list of outcomes

        Examples:

            >>> from negmas.outcomes import Issue, OutcomeType
            >>> issues = [Issue(name='price', values=(0.0, 3.0)), Issue(name='quantity', values=10)]

            Sampling outcomes as tuples

            >>> samples = Issue.sample(issues=issues, n_outcomes=10, astype=tuple)
            >>> len(samples) == 10
            True
            >>> type(samples[0]) == tuple
            True

            Sampling outcomes as dicts

            >>> samples = Issue.sample(issues=issues, n_outcomes=10, astype=dict)
            >>> len(samples) == 10
            True
            >>> type(samples[0]) == dict
            True
            >>> list(samples[0].keys())
            ['price', 'quantity']

            >>> @dataclass
            ... class MyOutcome(OutcomeType):
            ...     price: float = 0.0
            ...     quantity: int = 0


            Sampling outcomes as an arbitrary class

            >>> samples = Issue.sample(issues=issues, n_outcomes=10, astype=MyOutcome)
            >>> len(samples) == 10
            True
            >>> type(samples[0]) == MyOutcome
            True
            >>> list(samples[0].keys())
            ['price', 'quantity']


        """
        if astype is None:
            astype = dict
        n_total = num_outcomes(issues)

        if (
            n_total is not None
            and n_total != float("inf")
            and n_outcomes is not None
            and n_total < n_outcomes
            and fail_if_not_enough
            and not with_replacement
        ):
            raise ValueError(
                f"Cannot sample {n_outcomes} from a total of possible {n_total} outcomes"
            )

        if n_total is not None and n_total != float("inf") and n_outcomes is None:
            values = enumerate_outcomes(issues=issues, astype=tuple)
        elif n_total is None and n_outcomes is None:
            raise ValueError(
                f"Cannot sample unknown number of outcomes from continuous outcome spaces"
            )
        else:
            samples = []

            for issue in issues:
                samples.append(
                    issue.rand_outcomes(
                        n=n_outcomes, with_replacement=True, fail_if_not_enough=True
                    )
                )
            values = []

            for i in range(n_outcomes):
                values.append([s[i] for s in samples])

            if not with_replacement and not any(i.is_uncountable() for i in issues):
                tmp_values = []

                for value in values:
                    tmp_values.append(tuple(value))
                tmp_values = set(tmp_values)
                remaining = n_outcomes - len(tmp_values)
                n_max_trials, i = 10, 0

                while remaining < 0 and i < n_max_trials:
                    tmp_values = tmp_values.union(
                        set(
                            cls.sample(
                                issues=issues,
                                n_outcomes=remaining,
                                astype=tuple,
                                with_replacement=True,
                                fail_if_not_enough=False,
                            )
                        )
                    )
                    i += 1
                values = list(tmp_values)

        if issubclass(astype, tuple):
            return [tuple(_) for _ in values]
        issue_names = [_.name for _ in issues]
        if issubclass(astype, dict):
            return [outcome_as_dict(value, issue_names) for value in values]
        return [astype(**outcome_as_dict(value, issue_names)) for value in values]

    @property
    def outcome_range(self) -> "OutcomeRange":
        """An outcome range that represents the full space of the issues"""
        outcome_range = {}

        if self._is_range:
            outcome_range[self.name] = self._values
        elif self._is_generator:
            outcome_range[self.name] = None
        else:
            outcome_range[self.name] = self.all

        return outcome_range

    def __str__(self):
        return f"{self.name}: {self._values}"

    __repr__ = __str__

    @classmethod
    def combine(
        cls,
        issues: Iterable["Issue"],
        issue_name="combined",
        keep_issue_names=True,
        keep_value_names=True,
        issue_sep="_",
        value_sep="-",
    ) -> Optional["Issue"]:
        """
        Combines multiple issues into a single issue

        Args:
            issues: The issues to be combined
            issue_name: used only if `keep_issue_names` is False
            keep_issue_names: If true, the final issue name will be a
                              concatenation of issue names separated by `issue_sep`
                              otherwise `issue_name` will be used.
            keep_value_names: If true, the values for the generated issue
                              will be a concatenation of values from earlier
                              issues separated by `value_sep`.
            issue_sep: Separator for the issue name (used only if `keep_issue_names`)
            value_sep: Separator for the issue name (used only if `keep_value_names`)

        Remarks:

            - Only works if the issues have finite cardinality

        """
        n_outcomes = cls.num_outcomes(issues)
        if n_outcomes == float("inf"):
            return None
        if keep_issue_names:
            issue_name = issue_sep.join([_.name for _ in issues])
        if keep_value_names:
            values = [
                value_sep.join([str(_) for _ in outcome])
                for outcomes in cls.enumerate(issues, astype=tuple)
            ]
        else:
            values = n_outcomes
        return Issue(name=issue_name, values=values)

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return self._values == other.values and self.name == other.name

    def __copy__(self):
        return Issue(name=self.name, values=self._values)

    def __deepcopy__(self, memodict={}):
        if isinstance(self._values, list):
            return Issue(name=self.name, values=[_ for _ in self._values])

        return Issue(name=self.name, values=self._values)

    @classmethod
    def from_java(cls, d: Dict[str, Any], class_name: str) -> "Issue":
        if class_name.endswith("ListIssue"):
            return Issue(name=d.get("name", None), values=d["values"])

        if class_name.endswith("RangeIssue"):
            return Issue(name=d.get("name", None), values=(d["min"], d["max"]))
        raise ValueError(
            f"Unknown issue type: {class_name} with dict {d} received from Java"
        )

    def to_java(self):
        if self._values is None:
            return None

        if self._is_int_range:
            return {
                "name": self.name,
                "min": int(self._values[0]),
                "max": int(self._values[0]),
                PYTHON_CLASS_IDENTIFIER: "negmas.outcomes.IntRangeIssue",
            }

        if self._is_float_range:
            return {
                "name": self.name,
                "min": float(self._values[0]),
                "max": float(self._values[0]),
                PYTHON_CLASS_IDENTIFIER: "negmas.outcomes.DoubleRangeIssue",
            }

        if self._is_generator:
            raise ValueError("Cannot convert issues created by a callable to JAVA")

        if self._is_integer_valued:
            return {
                "name": self.name,
                "values": [int(_) for _ in self._values],
                PYTHON_CLASS_IDENTIFIER: "negmas.outcomes.IntListIssue",
            }

        if self._is_real_valued:
            return {
                "name": self.name,
                "values": [float(_) for _ in self._values],
                PYTHON_CLASS_IDENTIFIER: "negmas.outcomes.DoubleListIssue",
            }

        return {
            "name": self.name,
            "values": [str(_) for _ in self._values],
            PYTHON_CLASS_IDENTIFIER: "negmas.outcomes.StringListIssue",
        }


class Issues(object):
    """Encodes a set of Issues.

    Args:

        name-value pairs



    Remarks:

        - Issues can be initialized by either an iterable of strings, an integer or a tuple of two real values with
          the following meanings:
          - ``iterable of strings``: This is an issue that can any value within the given set of values (strings)
          - ``int``: This is an issue that takes any value from 0 to the given value -1 (int)
          - ``float``: This is an issue that can take any real value between the given limits (min, max)

    """

    def __init__(self, **kwargs) -> None:
        self.issues = [Issue(name=k, values=v) for k, v in kwargs.items()]

    @classmethod
    def from_issue_collection(cls, issues: Iterable[Issue]):
        out = Issues()
        out.issues = issues

        return out

    @classmethod
    def from_single_issue(cls, issue: Issue):
        return Issues.from_issue_collection([issue])

    @property
    def num_outcomes(self) -> Union[int, float]:
        """Returns the total number of outcomes in a set of issues. Infinity is returned for uncountable or continuous
        outcomes """
        n = 1

        for issue in self.issues:
            n *= issue._n_values

        return n

    @property
    def types(self) -> List[str]:
        """The type of the issue.

        Returns:
            str: either 'continuous' or 'discrete'

        """
        types_ = []

        for issue in self.issues:
            types_.append(issue.type)

        return types_

    def is_infinite(self) -> bool:
        """Test whether any issue is continuous (infinite outcome space)

        Returns:
            bool: continuous or not

        """

        return any(_.startswith("c") for _ in self.types)

    def is_finite(self) -> bool:
        """Test whether all issues are discrete (finite outcome space)

        Returns:
            bool: discrete or not

        """

        return all(_.startswith("d") for _ in self.types)

    @property
    def all(self) -> Generator:
        """A generator that generates all possible values.

        Remarks:
            - This function returns a generator for the case when the number of values is very large.
            - If you need a list then use something like:


        """

        if self.is_infinite():
            raise ValueError("Cannot return all possibilities of a continuous issues")

        yield from itertools.product(_.all for _ in self.issues)

    cardinality = num_outcomes
    """The number of possible outcomes for the set of issues. """

    def rand(self) -> Dict[str, Union[int, float, str]]:
        """Picks a random valid value."""

        return {_.name: _.rand() for _ in self.issues}

    rand_valid = rand

    def rand_invalid(self) -> Dict[str, Union[int, float, str]]:
        """Pick a random *invalid* value"""

        return {_.name: _.rand_invalid() for _ in self.issues}

    @property
    def outcome_range(self) -> "OutcomeRange":
        """An outcome range that represents the full space of the issues"""
        outcome_range = {}

        for issue in self.issues:
            if issue.is_continuous():
                outcome_range[issue.name] = issue.values
            elif issue.is_uncountable():
                outcome_range[issue.name] = None
            else:
                outcome_range[issue.name] = issue.all

        return outcome_range

    def __str__(self):
        return "\n".join([str(_) for _ in self.issues])


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

    def keys(self) -> List[str]:
        return tuple(_.name for _ in fields(self))

    def values(self) -> List[str]:
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


Outcome = Union[
    OutcomeType,
    Tuple[Union[int, float, str, list]],
    Dict[Union[int, str], Union[int, float, str, list]],
]
"""An outcome is either a tuple of values or a dict with name/value pairs."""

Outcomes = List["Outcome"]
OutcomeRanges = List["OutcomeRange"]

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


def num_outcomes(issues: Collection[Issue]) -> Optional[int]:
    """
    Returns the total number of outcomes in a set of issues.
    `-1` indicates infinity
    """
    return Issue.num_outcomes(issues)


def is_outcome(x: Any) -> bool:
    """Checks if x is acceptable as an outcome type"""

    return isinstance(x, dict) or isinstance(x, tuple) or isinstance(x, OutcomeType)


def enumerate_outcomes(
    issues: Iterable[Issue], keep_issue_names=None, astype=dict
) -> Optional[Union[List["Outcome"], Dict[str, "Outcome"]]]:
    """Enumerates all outcomes of this set of issues if possible

    Args:
        issues: A list of issues
        keep_issue_names: DEPRECTED. use `astype` instead
        astype: The type to use for returning outcomes. Can be tuple, dict or any `OutcomeType`

    Returns:
        list of outcomes
    """
    if keep_issue_names is not None:
        warnings.warn(
            "keep_issue_names is depricated. Use outcome_type instead.\n"
            "keep_issue_names=True <--> outcome_type=dict\n"
            "keep_issue_names=False <--> outcome_type=tuple\n",
            DeprecationWarning,
        )
        astype = dict if keep_issue_names else tuple
    try:
        outcomes = list(tuple(_) for _ in itertools.product(*[_.all for _ in issues]))
    except:
        return None

    if issubclass(astype, dict):
        issue_names = [_.name for _ in issues]
        outcomes = [outcome_as_dict(_, issue_names) for _ in outcomes]
    elif not issubclass(astype, tuple):
        issue_names = [_.name for _ in issues]
        outcomes = [astype(**outcome_as_dict(_, issue_names)) for _ in outcomes]

    return outcomes


def sample_outcomes(
    issues: Iterable[Issue],
    n_outcomes: Optional[int] = None,
    keep_issue_names=None,
    astype=dict,
    min_per_dim=5,
    expansion_policy=None,
) -> Optional[List[Optional["Outcome"]]]:
    """Discretizes the issue space and returns either a predefined number of outcomes or uniform samples

    Args:
        issues: The issues describing the issue space to be discretized
        n_outcomes: If None then exactly `min_per_dim` bins will be used for every continuous dimension and all outcomes
        will be returned
        keep_issue_names: DEPRICATED. Use `astype` instead
        min_per_dim: Max levels of discretization per dimension
        expansion_policy: None or 'repeat' or 'null' or 'no'. If repeat, then some of the outcomes will be repeated
        if None or 'no' then no expansion will happen if the total number of outcomes is less than
        n_outcomes: If 'null' then expansion will be with None values
        astype: The type used for returning outcomes. Can be tuple, dict or any `OutcomeType`

    Returns:
        List of outcomes

    Examples:

        enumberate the whole space

        >>> issues = [Issue(values=(0.0, 1.0), name='Price'), Issue(values=['a', 'b'], name='Name')]
        >>> sample_outcomes(issues=issues)
        [{'Price': 0.0, 'Name': 'a'}, {'Price': 0.0, 'Name': 'b'}, {'Price': 0.25, 'Name': 'a'}, {'Price': 0.25, 'Name': 'b'}, {'Price': 0.5, 'Name': 'a'}, {'Price': 0.5, 'Name': 'b'}, {'Price': 0.75, 'Name': 'a'}, {'Price': 0.75, 'Name': 'b'}, {'Price': 1.0, 'Name': 'a'}, {'Price': 1.0, 'Name': 'b'}]

        enumerate with sampling for very large space (we have 10 outcomes in the discretized space)

        >>> issues = [Issue(values=(0, 1), name='Price', value_type=float), Issue(values=['a', 'b'], name='Name')]
        >>> issues[0].is_continuous()
        True
        >>> sampled=sample_outcomes(issues=issues, n_outcomes=5)
        >>> len(sampled)
        5
        >>> len(set(tuple(_.values()) for _ in sampled))
        5

        >>> issues = [Issue(values=(0, 1), name='Price'), Issue(values=['a', 'b'], name='Name')]
        >>> issues[0].is_continuous()
        False
        >>> sampled=sample_outcomes(issues=issues, n_outcomes=5)
        >>> len(sampled)
        4
        >>> len(set(tuple(_.values()) for _ in sampled))
        4

    """
    if keep_issue_names is not None:
        warnings.warn(
            "keep_issue_names is depricated. Use outcome_type instead.\n"
            "keep_issue_names=True <--> outcome_type=dict\n"
            "keep_issue_names=False <--> outcome_type=tuple\n",
            DeprecationWarning,
        )
        astype = dict if keep_issue_names else tuple
    issues = [copy.deepcopy(_) for _ in issues]
    continuous = []
    uncountable = []
    indx = []
    uindx = []
    discrete = []
    n_disc = 0

    for i, issue in enumerate(issues):
        if issue.is_continuous():
            continuous.append(issue)
            indx.append(i)
        elif issue.is_uncountable():
            uncountable.append(issue)
            uindx.append(i)
        else:
            discrete.append(issue)
            n_disc += issue.cardinality

    if len(continuous) > 0:
        if n_outcomes is not None:
            n_per_issue = max(min_per_dim, (n_outcomes - n_disc) / len(continuous))
        else:
            n_per_issue = min_per_dim

        for i, issue in enumerate(continuous):
            issues[indx[i]] = Issue(
                name=issue.name,
                values=list(
                    np.linspace(
                        issue.min_value, issue.max_value, num=n_per_issue, endpoint=True
                    ).tolist()
                ),
            )

    if len(uncountable) > 0:
        if n_outcomes is not None:
            n_per_issue = max(min_per_dim, (n_outcomes - n_disc) / len(uncountable))
        else:
            n_per_issue = min_per_dim

        for i, issue in enumerate(uncountable):
            issues[uindx[i]] = Issue(
                name=issue.name, values=[issue.values() for _ in range(n_per_issue)]
            )

    cardinality = 1

    for issue in issues:
        cardinality *= issue.cardinality

    if cardinality == n_outcomes or n_outcomes is None:
        return list(enumerate_outcomes(issues, astype=astype))

    if cardinality < n_outcomes:
        outcomes = list(enumerate_outcomes(issues, astype=astype))

        if expansion_policy == "no" or expansion_policy is None:
            return outcomes
        elif expansion_policy == "null":
            return outcomes + [None] * (n_outcomes - cardinality)
        elif expansion_policy == "repeat":
            n_reps = n_outcomes // cardinality
            n_rem = n_outcomes % cardinality

            if n_reps > 1:
                for _ in n_reps:
                    outcomes += outcomes

            if n_rem > 0:
                outcomes += outcomes[:n_rem]

            return outcomes
    n_per_issue = 1 + int(n_outcomes ** (1 / len(issues)))
    vals = []
    n_found = 1
    for issue in issues:
        if n_per_issue < 2:
            if random.random() < 0.5:
                vals.append((issue.min_value,))
            else:
                vals.append((issue.max_value,))
            continue
        vals.append(issue.alli(n=n_per_issue))
        n_found *= n_per_issue
        if n_found > n_outcomes:
            n_per_issue = 1
    outcomes = itertools.product(*vals)
    if issubclass(astype, tuple):
        return list(outcomes)[:n_outcomes]
    names = [i.name for i in issues]
    outcomes = list(dict(zip(names, o)) for o in outcomes)[:n_outcomes]
    if issubclass(astype, dict):
        return outcomes
    return [astype(*o) for o in outcomes]


def _is_single(x):
    """Checks whether a value is a single value which is defined as either a string or not an Iterable."""

    return isinstance(x, str) or isinstance(x, numbers.Number)


def outcome_is_valid(outcome: "Outcome", issues: Collection[Issue]) -> bool:
    """Test validity of an outcome given a set of issues.

    Examples:
        >>> issues = [Issue((0.5, 2.0), 'price'), Issue(['2018.10.'+ str(_) for _ in range(1, 4)], 'date')\
                , Issue(20, 'count')]
        >>> for _ in issues: print(_)
        price: (0.5, 2.0)
        date: ['2018.10.1', '2018.10.2', '2018.10.3']
        count: (0, 19)
        >>> print([outcome_is_valid({'price':3.0}, issues), outcome_is_valid({'date': '2018.10.4'}, issues)\
            , outcome_is_valid({'count': 21}, issues)])
        [False, False, False]
        >>> valid_incomplete = {'price': 1.9}
        >>> print(outcome_is_valid(valid_incomplete, issues))
        True
        >>> print(outcome_is_complete(valid_incomplete, issues))
        False
        >>> valid_incomplete.update({'date': '2018.10.2', 'count': 5})
        >>> print(outcome_is_complete(valid_incomplete, issues))
        True

    Args:
        outcome: outcome tested which can contain values for a partial set of issue values
        issues: issues

    Returns:
        Union[bool, Tuple[bool, str]]: If return_problem is True then a second return value contains a string with
                                      reason of failure
    """
    outcome = outcome_as_dict(outcome, [_.name for _ in issues])

    for issue in issues:
        for key in ikeys(outcome):
            if str(issue.name) == str(key):
                break

        else:
            continue

        value = iget(outcome, key)

        if issue._is_range and (
            isinstance(value, str) or not issue.min_value <= value <= issue.max_value
        ):
            return False

        if isinstance(issue._values, list) and value not in issue._values:
            return False

    return True


def outcome_is_complete(outcome: "Outcome", issues: Collection[Issue]) -> bool:
    """Tests that the outcome is valid and complete.

    Examples:
        >>> issues = [Issue((0.5, 2.0), 'price'), Issue(['2018.10.'+ str(_) for _ in range(1, 4)], 'date')\
                , Issue(20, 'count')]
        >>> for _ in issues: print(_)
        price: (0.5, 2.0)
        date: ['2018.10.1', '2018.10.2', '2018.10.3']
        count: (0, 19)
        >>> print([outcome_is_complete({'price':3.0}, issues), outcome_is_complete({'date': '2018.10.4'}, issues)\
            , outcome_is_complete({'count': 21}, issues)])
        [False, False, False]
        >>> valid_incomplete = {'price': 1.9}
        >>> print(outcome_is_complete(valid_incomplete, issues))
        False
        >>> valid_incomplete.update({'date': '2018.10.2', 'count': 5})
        >>> print(outcome_is_complete(valid_incomplete, issues))
        True
        >>> invalid = {'price': 2000, 'date': '2018.10.2', 'count': 5}
        >>> print(outcome_is_complete(invalid, issues))
        False
        >>> invalid = {'unknown': 2000, 'date': '2018.10.2', 'count': 5}
        >>> print(outcome_is_complete(invalid, issues))
        False

     Args:
        outcome: outcome tested which much contain valid values all issues if it is to be considered complete.
        issues: issues

    Returns:
        Union[bool, Tuple[bool, str]]: If return_problem is True then a second return value contains a string with
                                      reason of failure

    """
    outcome = outcome_as_dict(outcome, [_.name for _ in issues])

    if len(outcome) != len(issues):
        return False

    valid = outcome_is_valid(outcome, issues)

    if not valid:
        return False

    outcome_keys = [str(k) for k in ikeys(outcome)]

    for issue in issues:
        if str(issue.name) not in outcome_keys:
            return False

    return True


def outcome_range_is_valid(
    outcome_range: OutcomeRange, issues: Optional[Collection[Issue]] = None
) -> Union[bool, Tuple[bool, str]]:
    """Tests whether the outcome range is valid for the set of issues.

    Args:
        outcome_range:
        issues:

    Example:
        >>> try:
        ...     outcome_range_is_valid({'price': (0, 10)})
        ... except NotImplementedError:
        ...     print('Not implemented')
        Not implemented

    Returns:

    """
    # TODO implement this function
    raise NotImplementedError()


def outcome_range_is_complete(
    outcome_range: OutcomeRange, issues: Optional[Collection[Issue]] = None
) -> Union[bool, Tuple[bool, str]]:
    """Tests whether the outcome range is valid and complete for the set of issues

    Args:
        outcome_range:
        issues:

    Example:
        >>> try:
        ...     outcome_range_is_complete({'price': (0, 10)})
        ... except NotImplementedError:
        ...     print('Not implemented')
        Not implemented

    Returns:

    """
    # TODO implement this function
    raise NotImplementedError()


#################################
# Outcome space implementation  #
#################################
def outcome_in_range(
    outcome: "Outcome",
    outcome_range: OutcomeRange,
    *,
    strict=False,
    fail_incomplete=False,
) -> bool:
    """
    Tests that the outcome is contained within the given range of outcomes.

    An outcome range defines a value or a range of values for each issue.

    Args:

        outcome: "Outcome" being tested
        outcome_range: "Outcome" range being tested against
        strict: Whether to enforce that all issues in the outcome must be mentioned in the outcome_range
        fail_incomplete: If True then outcomes that do not sepcify a value for all keys in the outcome_range
        will be considered not falling within it. If False then these outcomes will be considered falling
        within the range given that the values for the issues mentioned in the outcome satisfy the range
        constraints.

    Examples:

        >>> outcome_range = {'price': (0.0, 2.0), 'distance': [0.3, 0.4], 'type': ['a', 'b'], 'area': 3}
        >>> outcome_range_2 = {'price': [(0.0, 1.0), (1.5, 2.0)], 'area': [(3, 4), (7, 9)]}
        >>> outcome_in_range({'price':3.0}, outcome_range)
        False
        >>> outcome_in_range({'date': '2018.10.4'}, outcome_range)
        True
        >>> outcome_in_range({'date': '2018.10.4'}, outcome_range, strict=True)
        False
        >>> outcome_in_range({'area': 3}, outcome_range, fail_incomplete=True)
        False
        >>> outcome_in_range({'area': 3}, outcome_range)
        True
        >>> outcome_in_range({'type': 'c'}, outcome_range)
        False
        >>> outcome_in_range({'type': 'a'}, outcome_range)
        True
        >>> outcome_in_range({'date': '2018.10.4'}, outcome_range_2)
        True
        >>> outcome_in_range({'area': 3.1}, outcome_range_2)
        True
        >>> outcome_in_range({'area': 3}, outcome_range_2)
        False
        >>> outcome_in_range({'area': 5}, outcome_range_2)
        False
        >>> outcome_in_range({'price': 0.4}, outcome_range_2)
        True
        >>> outcome_in_range({'price': 0.4}, outcome_range_2, fail_incomplete=True)
        False
        >>> outcome_in_range({'price': 1.2}, outcome_range_2)
        False
        >>> outcome_in_range({'price': 0.4, 'area': 3.9}, outcome_range_2)
        True
        >>> outcome_in_range({'price': 0.4, 'area': 10}, outcome_range_2)
        False
        >>> outcome_in_range({'price': 1.2, 'area': 10}, outcome_range_2)
        False
        >>> outcome_in_range({'price': 1.2, 'area': 4}, outcome_range_2)
        False
        >>> outcome_in_range({'type': 'a'}, outcome_range_2)
        True
        >>> outcome_in_range({'type': 'a'}, outcome_range_2, strict=True)
        False
        >>> outcome_range = {'price': 10}
        >>> outcome_in_range({'price': 10}, outcome_range)
        True
        >>> outcome_in_range({'price': 11}, outcome_range)
        False

    Returns:

        bool: Success or failure

    Remarks:
        Outcome ranges specify regions in an outcome space. They can have any of the following conditions:

        - A key/issue not mentioned in the outcome range does not add any constraints meaning that **All**
          values are acceptable except if strict == True. If strict == True then *NO* value will be accepted for issues
          not in the outcome_range.
        - A key/issue with the value None in the outcome range means **All** values on this issue are acceptable.
          This is the same as having this key/issue removed from the outcome space
        - A key/issue withe the value [] (empty list) accepts *NO* outcomes
        - A key/issue with  a single value means that it is the only one acceptable
        - A key/issue with a single 2-items tuple (min, max) means that any value within that range is acceptable.
        - A key/issue with a list of values means an output is acceptable if it falls within the condition specified
          by any of the values in the list (list == union). Each such value can be a single value, a 2-items
          tuple or another list. Notice that lists of lists can always be combined into a single list of values

    """

    if (
        fail_incomplete
        and len(set(ikeys(outcome_range)).difference(ikeys(outcome))) > 0
    ):
        return False

    for key, value in ienumerate(outcome):
        if key not in ikeys(outcome_range):
            if strict:
                return False

            continue

        values = iget(outcome_range, key, None)

        if values is None:
            return False

        if _is_single(values) and value != values:
            return False

        if isinstance(values, tuple) and not values[0] < value < values[1]:
            return False

        if isinstance(values, list):
            for constraint in values:
                if _is_single(constraint):
                    if value == constraint:
                        break

                elif isinstance(constraint, list):
                    if value in constraint:
                        break

                elif isinstance(constraint, tuple):
                    if constraint[0] < value < constraint[1]:
                        break

            else:
                return False

            continue

    return True


def outcome_as_dict(outcome: "Outcome", issue_names: List[str] = None):
    """Converts the outcome to a dict no matter what was its type"""

    if outcome is None:
        return None

    if isinstance(outcome, np.ndarray):
        outcome = tuple(outcome.tolist())

    if isinstance(outcome, dict):
        return outcome

    if isinstance(outcome, OutcomeType):
        return outcome.asdict()

    if issue_names is not None:
        return dict(zip(issue_names, outcome))
    warnings.warn(f"Outcome {outcome} is converted to a dict without issue names!!")
    return dict(zip((str(_) for _ in range(len(outcome))), outcome))


def outcome_as_tuple(outcome: "Outcome"):
    """Converts the outcome to a tuple no matter what was its type"""

    if outcome is None:
        return None

    if isinstance(outcome, tuple):
        return outcome

    if isinstance(outcome, dict):
        return tuple(list(outcome.values()))

    if isinstance(outcome, OutcomeType):
        return outcome.astuple()

    if isinstance(outcome, np.ndarray):
        return tuple(outcome.tolist())

    if isinstance(outcome, Iterable):
        return tuple(outcome)

    raise ValueError(f"Unknown type for outcome {type(outcome)}")


def outcome_as(outcome: "Outcome", astype: Type, issue_names: List[str] = None):
    """Converts the outcome to tuple, dict or any `OutcomeType`.

    Args:
         outcome: The outcome to adjust type
         astype: The type to return. None returns the outcome as it is
         issue_names: Only needed if `astype` is not tuple

    """
    if astype is None:
        return outcome
    if issubclass(astype, tuple):
        return outcome_as_tuple(outcome)
    if issubclass(astype, dict):
        return outcome_as_dict(outcome, issue_names)
    return astype(**outcome_as_dict(outcome, issue_names))


def outcome_for(outcome: "Outcome", ami: "Mechanism") -> Optional["Outcome"]:
    """Converts the outcome the type specified by the mechanism

    Args:
         outcome: The outcome to adjust type
         ami: The Agent Mechanism Interface

    """
    astype = ami.outcome_type
    issue_names = None if issubclass(astype, tuple) else [_.name for _ in ami.issues]
    return outcome_as(outcome, astype, issue_names)
