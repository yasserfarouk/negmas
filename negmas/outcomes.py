"""Defines basic concept related to outcomes

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
  count: 20

  Outcome example compatible with the given set of issues:

  >>> a = {'price': 1.2, 'date': '2018.10.04', 'count': 4}

"""

import itertools
import math
import random
import xml.etree.ElementTree as ET
from enum import Enum
from functools import reduce
from operator import mul
from typing import Optional, Collection, List, Generator, Iterable, Sequence, Union, Type
from typing import Tuple, Mapping, Dict, Any

import numpy as np
import pkg_resources
from dataclasses import dataclass, fields


from negmas import NamedObject, NamedObject
from negmas.generics import *
from negmas.helpers import unique_name

LARGE_NUMBER = 100
__all__ = [
    'Outcome',
    'OutcomeType',
    'OutcomeRange',
    'ResponseType',
    'Issue',
    'Issues',
    'outcome_is_valid',
    'outcome_is_complete',
    'outcome_range_is_valid',
    'outcome_range_is_complete',
    'outcome_in_range',
    'enumerate_outcomes',
    'sample_outcomes',
    'outcome_as_dict',
    'outcome_as_tuple',
    'num_outcomes',
]


class ResponseType(Enum):
    """Possible answers to offers during negotiation."""
    ACCEPT_OFFER = 1
    NO_RESPONSE = 0
    REJECT_OFFER = -2
    END_NEGOTIATION = -3


class Issue(NamedObject):
    """Encodes an Issue.

    Args:
            values: Possible values for the issue
            name: Name of the issue. If not given, a random name will be generated

    Examples:
        >>> print(Issue(['to be', 'not to be'], name='THE problem'))
        THE problem: ['to be', 'not to be']
        >>> print(Issue(3, name='Cars'))
        Cars: 3
        >>> print(Issue((0.0, 1.0), name='required accuracy'))
        required accuracy: (0.0, 1.0)
        >>> a = Issue((0.0, 1.0), name='required accuracy')
        >>> a.is_continuous()
        True
        >>> a.is_discrete()
        False

    Remarks:
        - Issues can be initialized by either an iterable of strings, an integer or a tuple of two real values with
          the following meanings:
          - ``iterable of strings``: This is an issue that can any value within the given set of values (strings)
          - ``int``: This is an issue that takes any value from 0 to the given value -1 (int)
          - ``float``: This is an issue that can take any real value between the given limits (min, max)

    """

    def __init__(
        self,
        values: Union[List[str], int, Tuple[float, float]],
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name)
        # if isinstance(values, int) and values <= LARGE_NUMBER:
        #    values = list(range(values))
        if isinstance(values, tuple):
            values = (float(values[0]), float(values[1]))
        self.values = values

    @classmethod
    def to_xml_str(cls, issues: List['Issue']
                   , enumerate_integer: bool = True) -> str:
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
        output = f'<negotiation_template>\n<utility_space number_of_issues="{len(issues)}">\n' \
                 f'<objective description="" etype="objective" index="0" name="root" type="objective">\n'
        for indx, issue in enumerate(issues):
            if isinstance(issue.values, int):
                if enumerate_integer:
                    output += f'    <issue etype="discrete" index="{indx+1}" name="{issue.name}" type="discrete" vtype="discrete">\n'
                    for i, v in enumerate(range(issue.values)):
                        output += f'        <item index="{i+1}" value="{v}" cost="0" description="{v}">\n        </item>\n'
                    output += '    </issue>\n'
                else:
                    output += f'    <issue etype="integer" index="{indx+1}" name="{issue.name}" type="integer" vtype="integer"' \
                              f' lowerbound="0" upperbound="{issue.values-1}" />\n'
            elif isinstance(issue.values, tuple):
                output += f'    <issue etype="real" index="{indx+1}" name="{issue.name}" type="real" vtype="real">\n' \
                          f'        <range lowerbound="{issue.values[0]}" upperbound="{issue.values[1]}"></range>\n    </issue>\n'
            else:
                output += f'    <issue etype="discrete" index="{indx+1}" name="{issue.name}" type="discrete" vtype="discrete">\n'
                for i, v in enumerate(issue.values):
                    output += f'        <item index="{i+1}" value="{v}" cost="0" description="{v}">\n        </item>\n'
                output += '    </issue>\n'
        output += f'</objective>\n</utility_space>\n</negotiation_template>'
        return output

    @classmethod
    def to_genius(cls, issues: Iterable['Issue'], file_name: str, enumerate_integer: bool = True) -> None:
        """Exports a the domain issues to a GENIUS XML file.

                Args:

                    issues: The issues to be exported
                    file_name (str): File name to export to

                Returns:

                    A List[Issue] or Dict[Issue]


                Examples:

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
        with open(file_name, 'w') as f:
            f.write(cls.to_xml_str(issues=list(issues), enumerate_integer=enumerate_integer))

    @classmethod
    def from_xml_str(cls, xml_str: str
                     , force_single_issue=False, keep_value_names=True, keep_issue_names=True, safe_parsing=True
                     , n_discretization: Optional[int] = None
                     , max_n_outcomes: int = 1e6):
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

                    >>> domain_file_name = pkg_resources.resource_filename('negmas'
                    ...                                      , resource_name='tests/data/Laptop/Laptop-C-domain.xml')
                    >>> issues, _ = Issue.from_xml_str(open(domain_file_name, 'r').read()
                    ... , force_single_issue=True, keep_value_names=False, keep_issue_names=False)
                    >>> issue = issues[0]
                    >>> print(issue.cardinality())
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
                    >>> print(issue.cardinality())
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
                    >>> print([_.cardinality() for _ in issues])
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
                    >>> print([_.cardinality() for _ in issues])
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
                    '3'
                    >>> print([_.cardinality() for _ in issues])
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
                    >>> print([_.cardinality() for _ in issues])
                    [-1]
        """
        root = ET.fromstring(xml_str)
        if safe_parsing and root.tag != 'negotiation_template':
            raise ValueError(f'Root tag is {root.tag}: negotiation_template')

        utility_space = None
        agents = []
        for child in root:
            if child.tag == 'utility_space':
                utility_space = child
                for _ in utility_space:
                    if _.tag == 'objective':
                        utility_space = _
                        break
            elif child.tag == 'agent':
                agents.append(child.attrib)

        if utility_space is None:
            if safe_parsing:
                raise ValueError(f'No objective child was found in the root')
            utility_space = root
        weights = {}
        issues = {}
        issue_info = {}
        all_discrete = True
        for child in utility_space:
            if child.tag == 'issue':
                indx = int(child.attrib['index']) - 1
                myname = child.attrib['name']
                issue_key = myname if keep_issue_names else indx
                issue_info[issue_key] = {'name': myname, 'index': indx}
                info = {'type': 'discrete', 'etype': 'discrete', 'vtype': 'discrete'}
                for a in ('type', 'etype', 'vtype'):
                    info[a] = child.attrib.get(a, info[a])
                mytype = info['type']
                if mytype == 'discrete':
                    issues[issue_key] = []
                    for item in child:
                        if item.tag == 'item':
                            item_indx = int(item.attrib['index']) - 1
                            item_name = item.attrib.get('value', None)
                            item_key = item_name if keep_value_names and item_name is not None else item_indx
                            if item_key not in issues[issue_key]:  # ignore repeated items
                                issues[issue_key].append(item_key)
                    if not keep_value_names:
                        issues[issue_key] = len(issues[issue_key])
                elif mytype in ('integer', 'real'):
                    lower, upper = child.attrib.get('lowerbound', None), child.attrib.get('upperbound', None)
                    for rng_child in child:
                        if rng_child.tag == 'range':
                            lower, upper = rng_child.attrib.get('lowerbound', lower), rng_child.attrib.get('upperbound',
                                                                                                           upper)
                    if mytype == 'integer':
                        lower, upper = int(lower), int(upper)
                        if keep_value_names:
                            issues[issue_key] = list(range(lower, upper + 1))
                        else:
                            issues[issue_key] = upper - lower + 1
                    else:
                        if n_discretization is None:
                            all_discrete = False
                            issues[issue_key] = (float(lower), float(upper))
                        else:
                            issues[issue_key] = n_discretization
                else:
                    # I should add the real-valued issues code here
                    raise ValueError(f'Unknown type: {mytype}')
            else:
                raise ValueError(f'Unknown child for objective: {child.tag}')

        for key, value in zip(ikeys(issues), ivalues(issues)):
            issues[key] = Issue(values=value,
                                name=issue_info[key]['name'] if keep_issue_names else str(issue_info[key]['index']))
        issues = list(issues.values())
        if force_single_issue:
            issue_name_ = '-'.join([_.name for _ in issues]) if keep_issue_names else '0'
            if all_discrete or n_discretization is not None:
                n_outcomes = None
                if max_n_outcomes is not None:
                    n_items = [len(list(_.alli(n=n_discretization))) for _ in issues]
                    n_outcomes = reduce(mul, n_items, 1)
                    if n_outcomes > max_n_outcomes:
                        return None, None
                if keep_value_names:
                    if len(issues) > 1:
                        all_values = itertools.product(*[[str(_) for _ in issue.alli(n=n_discretization)]
                                                         for issue_key, issue in zip(range(len(issues)), issues)])
                        all_values = list(map(lambda items: '+'.join(items), all_values))
                    else:
                        all_values = [str(_) for _ in issues[0].alli(n=n_discretization)]
                    issues = [Issue(values=all_values, name=issue_name_)]
                else:
                    if n_outcomes is None:
                        n_items = [_.cardinality() for _ in issues]
                        n_outcomes = reduce(mul, n_items, 1)
                    issues = [Issue(values=n_outcomes, name=issue_name_)]
            else:
                return None, None
        return issues, agents

    @classmethod
    def from_genius(cls, file_name: str, force_single_issue=False, keep_value_names=True, keep_issue_names=True,
                    safe_parsing=True
                    , n_discretization: Optional[int] = None
                    , max_n_outcomes: int = 1e6):
        """Imports a the domain issues from a GENIUS XML file.

                Args:

                    file_name (str): File name to import from

                Returns:

                    A List[Issue] or Dict[Issue]


                Examples:

                    >>> issues, _ = Issue.from_genius(file_name = pkg_resources.resource_filename('negmas'
                    ...                                      , resource_name='tests/data/Laptop/Laptop-C-domain.xml'))
                    >>> print([_.name for _ in issues])
                    ['Laptop', 'Harddisk', 'External Monitor']

                Remarks:
                    See ``from_xml_str`` for all the parameters

                """
        with open(file_name, 'r', encoding='utf-8') as f:
            xml_str = f.read()
            return cls.from_xml_str(xml_str=xml_str, force_single_issue=force_single_issue
                                    , keep_value_names=keep_value_names, keep_issue_names=keep_issue_names
                                    , safe_parsing=safe_parsing, n_discretization=n_discretization
                                    , max_n_outcomes=max_n_outcomes)

    @staticmethod
    def n_outcomes(issues: Iterable['Issue']) -> int:
        """Returns the total number of outcomes in a set of issues. `-1` indicates infinity"""
        n = 1
        for issue in issues:
            n *= issue.cardinality()
            if n < 0:
                return -1

        return n

    @staticmethod
    def generate(
        issues: Sequence[Union[int, List[str], Tuple[float, float]]],
        counts: Optional[Sequence[int]] = None,
        names: Optional[Sequence[str]] = None,
    ) -> List['Issue']:
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
            for j in range(count):
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
            str: either 'continuous' or 'discrete'

        """
        if isinstance(self.values, tuple) and isinstance(
            self.values[0], float
        ):
            return 'continuous'

        elif isinstance(self.values, int) or isinstance(self.values, list):
            return 'discrete'

        raise ValueError(
            'Unknown type. Note that a Tuple[int, int] is not allowed as a range.'
        )

    def is_continuous(self) -> bool:
        """Test whether the issue is a continuous issue

        Returns:
            bool: continuous or not

        """
        return self.type.startswith('c')

    def is_discrete(self) -> bool:
        """Test whether the issue is a discrete issue

        Returns:
            bool: discrete or not

        """
        return self.type.startswith('d')

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
        if self.is_continuous():
            raise ValueError(
                'Cannot return all possibilities of a continuous issue'
            )

        if isinstance(self.values, int):
            yield from range(self.values)

        else:
            yield from self.values  # type: ignore

    def alli(self, n: Optional[int] = 10) -> Generator:
        """A generator that generates all possible values or samples n values for real Issues.

        Remarks:
            - This function returns a generator for the case when the number of values is very large.
            - If you need a list then use something like:

            >>> from negmas.outcomes import Issue
            >>> list(Issue(5).all)
            [0, 1, 2, 3, 4]

        """
        if self.is_continuous():
            if n is None:
                raise ValueError('Real valued issue with no discretization value')
            yield from np.linspace(self.values[0], self.values[1], num=n, endpoint=True).tolist()

        if isinstance(self.values, int):
            yield from range(self.values)
        else:
            yield from self.values  # type: ignore

    def cardinality(self) -> int:
        """The number of possible outcomes for the issue. A negative number means infinite"""
        if isinstance(self.values, int):
            return self.values

        elif self.is_continuous():
            return -1

        return len(self.values)  # type: ignore

    def rand(self) -> Union[int, float, str]:
        """Picks a random valid value."""
        if isinstance(self.values, int):
            return random.randint(0, self.values - 1)

        elif self.is_continuous():
            return random.random() * (
                self.values[1] - self.values[0]
            ) + self.values[
                       0
                   ]  # type: ignore

        return random.choice(self.values)  # type: ignore

    def rand_outcomes(self, n: int, with_replacement=False, fail_if_not_enough=False) -> Iterable["Outcome"]:
        """Picks a random valid value."""
        if isinstance(self.values, int):
            if n > self.values and not with_replacement:
                if fail_if_not_enough:
                    raise ValueError(f'Cannot sample {n} outcomes out of {self.values} without replacement')
                else:
                    return [_ for _ in range(self.values)]
            if with_replacement:
                return np.random.randint(low=0, high=self.values, size=n).tolist()
            else:
                return random.shuffle([_ for _ in range(self.values)])[:n]
        elif self.is_continuous():
            if with_replacement:
                return (np.random.rand(n) * (self.values[1] - self.values[0]) + self.values[0]).tolist()
            else:
                return np.linspace(self.values[0], self.values[1], num=n, endpoint=True).tolist()

        if n > len(self.values) and not with_replacement:
            if fail_if_not_enough:
                raise ValueError(f'Cannot sample {n} outcomes out of {self.values} without replacement')
            else:
                return self.values
        return np.random.choice(np.asarray(self.values, dtype=type(self.values[0])), size=n
                                , replace=with_replacement).tolist()

    rand_valid = rand

    def rand_invalid(self) -> Union[int, float, str]:
        """Pick a random *invalid* value"""
        if isinstance(self.values, int):
            return random.randint(self.values + 1, 2 * self.values)

        elif self.is_continuous():
            return random.random() * (self.values[1]) + self.values[
                1
            ] + 1e-3  # type: ignore

        pick = unique_name('') + str(random.choice(self.values)) + unique_name(
            ''
        )  # type: ignore
        return pick

    @classmethod
    def enumerate(cls, issues: Collection['Issue'], max_n_outcomes: int=None, astype: Type = dict) -> List["Outcome"]:
        n = num_outcomes(issues)
        if n is None:
            return cls.sample(issues=issues, n_outcomes=max_n_outcomes, astype=astype)
        values = enumerate_outcomes(issues, keep_issue_names=False)
        if n > max_n_outcomes:
            values = random.sample(values, max_n_outcomes)
        outcomes = []
        for value in values:
            if astype == tuple:
                outcomes.append(tuple(value))
            elif astype == dict:
                outcomes.append(dict(zip((i.name for i in issues), value)))
            else:
                outcomes.append(astype(**dict(zip((i.name for i in issues), value))))
        return outcomes

    @classmethod
    def sample(cls, issues: Collection['Issue'], n_outcomes: int, astype: Type = dict
               , with_replacement: bool=True, fail_if_not_enough=True) -> List["Outcome"]:
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

            >>> from negmas import Issue, OutcomeType
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
        n_total = num_outcomes(issues)
        if n_total is not None and n_outcomes is not None and n_total < n_outcomes \
            and fail_if_not_enough and not with_replacement:
            raise ValueError(f'Cannot sample {n_outcomes} from a total of possible {n_total} outcomes')
        if n_total is not None and n_outcomes is None:
            values = enumerate_outcomes(issues=issues, keep_issue_names=False)
        elif n_total is None and n_outcomes is None:
            raise ValueError(f'Cannot sample unknown number of outcomes from continuous outcome spaces')
        else:
            samples = []
            for issue in issues:
                samples.append(issue.rand_outcomes(n=n_outcomes, with_replacement=True, fail_if_not_enough=True))
            values = []
            for i in range(n_outcomes):
                values.append([s[i] for s in samples])
            if not with_replacement:
                tmp_values = []
                for value in values:
                    tmp_values.append(tuple(value))
                tmp_values = set(tmp_values)
                remaining = n_outcomes - len(tmp_values)
                n_max_trials, i = 10, 0
                while remaining < 0 and i < n_max_trials:
                    tmp_values = tmp_values.union(set(cls.sample(issues=issues, n_outcomes=remaining, astype=tuple
                                                         , with_replacement=True, fail_if_not_enough=False)))
                    i += 1
                values = list(tmp_values)

        outcomes = []
        for value in values:
            if astype == tuple:
                outcomes.append(tuple(value))
            elif astype == dict:
                outcomes.append(dict(zip((i.name for i in issues), value)))
            else:
                outcomes.append(astype(**dict(zip((i.name for i in issues), value))))
        return outcomes

    @property
    def outcome_range(self) -> 'OutcomeRange':
        """An outcome range that represents the full space of the issues"""
        outcome_range = {}
        if self.is_continuous():
            outcome_range[self.name] = self.values
        else:
            outcome_range[self.name] = self.all
        return outcome_range

    def __str__(self):
        return f'{self.name}: {self.values}'

    __repr__ = __str__

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return self.values == other.values and self.name == other.name

    def __copy__(self):
        return Issue(name=self.name, values=self.values)

    def __deepcopy__(self, memodict={}):
        if isinstance(self.values, list):
            return Issue(name=self.name, values=[_ for _ in self.values])
        return Issue(name=self.name, values=self.values)

    class Java:
        implements = ['jnegmas.Issue']


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
    def from_issue_collection(cls, issues: Iterable[Issue], name: str = None):
        out = Issues(name=name)
        out.issues = issues
        return out

    @classmethod
    def from_single_issue(cls, issue: Issue, name: str = None):
        return Issues.from_issue_collection([issue], name=name)

    def n_outcomes(self) -> int:
        """Returns the total number of outcomes in a set of issues. `-1` indicates infinity"""
        n = 1
        for issue in self.issues:
            n *= issue.cardinality()
            if n < 0:
                return -1

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
        return any(_.startswith('c') for _ in self.types)

    def is_finite(self) -> bool:
        """Test whether all issues are discrete (finite outcome space)

        Returns:
            bool: discrete or not

        """
        return all(_.startswith('d') for _ in self.types)

    @property
    def all(self) -> Generator:
        """A generator that generates all possible values.

        Remarks:
            - This function returns a generator for the case when the number of values is very large.
            - If you need a list then use something like:


        """
        if self.is_infinite():
            raise ValueError(
                'Cannot return all possibilities of a continuous issues'
            )

        yield from itertools.product(_.all for _ in self.issues)

    def cardinality(self) -> int:
        """The number of possible outcomes for the issue. A negative number means infinite"""
        return self.n_outcomes()

    def rand(self) -> Dict[str, Union[int, float, str]]:
        """Picks a random valid value."""
        return {_.name: _.rand() for _ in self.issues}

    rand_valid = rand

    def rand_invalid(self) -> Dict[str, Union[int, float, str]]:
        """Pick a random *invalid* value"""
        return {_.name: _.rand_invalid() for _ in self.issues}

    @property
    def outcome_range(self) -> 'OutcomeRange':
        """An outcome range that represents the full space of the issues"""
        outcome_range = {}
        for issue in self.issues:
            if issue.is_continuous():
                outcome_range[issue.name] = issue.values
            else:
                outcome_range[issue.name] = issue.all
        return outcome_range

    def __str__(self):
        return '\n'.join([str(_) for _ in self.issues])


@dataclass
class OutcomeType:
    """A helper class allowing for definition of types that behave as outcomes (either in the form of dict or tuple).

    This class is intended to be used when a simple tuple or dict is not enough for describing an outcome (e.g. to use
    editor features like auto-completion of members). You simply define your class as a dataclass and add your fields to
    it then inherit from OutcomeType. As we do nothing in the __init__ function, that is compatible with python
    dataclasses.


    Examples:

        >>> from negmas import OutcomeType, Issue
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


Outcome = Union[OutcomeType, Tuple[Union[int, float, str, list]], Dict[Union[int, str], Union[int, float, str, list]]]
"""An outcome is either a tuple of values or a dict with name/value pairs."""

Outcomes = List['Outcome']
OutcomeRanges = List['OutcomeRange']


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


def num_outcomes(issues: Collection[Issue]) -> Optional[int]:
    n = 1
    for issue in issues:
        c = issue.cardinality()
        if c < 0:
            return None
        n *= c
    return n


def enumerate_outcomes(issues: Iterable[Issue], keep_issue_names=True) \
    -> Optional[Union[List['Outcome'], Dict[str, 'Outcome']]]:
    """Enumerates all outcomes of this set of issues if possible

    Args:
        issues:
        keep_issue_names:

    Returns:
        List
    """
    try:
        outcomes = list(itertools.product(*[_.all for _ in issues]))
    except:
        return None
    if keep_issue_names:
        issue_names = [_.name for _ in issues]
        for i, outcome in enumerate(outcomes):
            outcomes[i] = dict(zip(issue_names, outcome))
    return outcomes


def sample_outcomes(issues: Iterable[Issue], n_outcomes: Optional[int] = None, keep_issue_names=True
                    , min_per_dim=5, expansion_policy=None) \
    -> Optional[List[Optional['Outcome']]]:
    """Discretizes the issue space and returns either a predefined number of outcomes or uniform samples

    Args:
        issues: The issues describing the issue space to be discretized
        n_outcomes: If None then exactly `min_per_dim` bins will be used for every continuous dimension and all outcomes
        will be returned
        keep_issue_names:
        min_per_dim:
        expansion_policy: None or 'repeat' or 'null' or 'no'. If repeat, then some of the outcomes will be repeated
        if None or 'no' then no expansion will happen if the total number of outcomes is less than
        n_outcomes. If 'null' then expansion will be with None values

    Returns:
        List


    Examples:

        enumberate the whole space
        >>> issues = [Issue(values=(0, 1), name='Price'), Issue(values=['a', 'b'], name='Name')]
        >>> sample_outcomes(issues=issues)
        [{'Price': 0.0, 'Name': 'a'}, {'Price': 0.0, 'Name': 'b'}, {'Price': 0.25, 'Name': 'a'}, {'Price': 0.25, 'Name': 'b'}, {'Price': 0.5, 'Name': 'a'}, {'Price': 0.5, 'Name': 'b'}, {'Price': 0.75, 'Name': 'a'}, {'Price': 0.75, 'Name': 'b'}, {'Price': 1.0, 'Name': 'a'}, {'Price': 1.0, 'Name': 'b'}]

        enumerate with sampling for very large space (we have 10 outcomes in the discretized space)
        >>> issues = [Issue(values=(0, 1), name='Price'), Issue(values=['a', 'b'], name='Name')]
        >>> sampled=sample_outcomes(issues=issues, n_outcomes=5)
        >>> len(sampled)
        5
        >>> len(set(tuple(_.values()) for _ in sampled))
        5


    """
    issues = [_ for _ in issues]
    continuous = []
    indx = []
    discrete = []
    n_disc = 0
    for i, issue in enumerate(issues):
        if issue.is_continuous():
            continuous.append(issue)
            indx.append(i)
        else:
            discrete.append(issue)
            n_disc += issue.cardinality()
    if len(continuous) > 0:
        if n_outcomes is not None:
            n_per_issue = max(min_per_dim, (n_outcomes - n_disc) / len(continuous))
        else:
            n_per_issue = min_per_dim
        for i, issue in enumerate(continuous):
            issues[indx[i]] = Issue(name=issue.name, values=list(np.linspace(issue.values[0], issue.values[1],
                                                                        num=n_per_issue, endpoint=True).tolist()))

    cardinality = 1
    for issue in issues:
        cardinality *= issue.cardinality()

    if cardinality == n_outcomes or n_outcomes is None:
        return list(enumerate_outcomes(issues, keep_issue_names=keep_issue_names))

    if cardinality < n_outcomes:
        outcomes = list(enumerate_outcomes(issues, keep_issue_names=keep_issue_names))
        if expansion_policy == 'no' or expansion_policy is None:
            return outcomes
        elif expansion_policy == 'null':
            return outcomes + [None] * (n_outcomes - cardinality)
        elif expansion_policy == 'repeat':
            n_reps = n_outcomes // cardinality
            n_rem = n_outcomes % cardinality
            if n_reps > 1:
                for _ in n_reps:
                    outcomes += outcomes
            if n_rem > 0:
                outcomes += outcomes[:n_rem]
            return outcomes

    return list(random.sample(enumerate_outcomes(issues, keep_issue_names=keep_issue_names), n_outcomes))


def _is_single(x):
    """Checks whether a value is a single value which is defined as either a string or not an Iterable."""
    return isinstance(x, str) or isinstance(x, int) or isinstance(x, float)


def outcome_is_valid(outcome: Outcome, issues: Collection[Issue]) -> bool:
    """Test validity of an outcome given a set of issues.

    Examples:
        >>> issues = [Issue((0.5, 2.0), 'price'), Issue(['2018.10.'+ str(_) for _ in range(1, 4)], 'date')\
                , Issue(20, 'count')]
        >>> for _ in issues: print(_)
        price: (0.5, 2.0)
        date: ['2018.10.1', '2018.10.2', '2018.10.3']
        count: 20
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
    for issue in issues:
        for key in ikeys(outcome):
            if issue.name == str(key):
                break

        else:
            continue

        value = iget(outcome, key)
        if isinstance(issue.values, int) and (
            isinstance(value, str) or not 0 <= value < issue.values
        ):
            return False

        elif isinstance(issue.values, tuple) and (
            isinstance(value, str)
            or not issue.values[0] < value < issue.values[1]
        ):
            return False

        elif isinstance(issue.values, list) and value not in issue.values:
            return False

    return True


def outcome_is_complete(outcome: Outcome, issues: Collection[Issue]) -> bool:
    """Tests that the outcome is valid and complete.

    Examples:
        >>> issues = [Issue((0.5, 2.0), 'price'), Issue(['2018.10.'+ str(_) for _ in range(1, 4)], 'date')\
                , Issue(20, 'count')]
        >>> for _ in issues: print(_)
        price: (0.5, 2.0)
        date: ['2018.10.1', '2018.10.2', '2018.10.3']
        count: 20
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
    if len(outcome) != len(issues):
        return False

    valid = outcome_is_valid(outcome, issues)
    if not valid:
        return False

    for issue in issues:
        if issue.name not in outcome.keys():
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
    outcome: Outcome,
    outcome_range: OutcomeRange,
    *,
    strict=False,
    fail_incomplete=False,
) -> bool:
    """Tests that the outcome is contained within the given range of outcomes.

        An outcome range defines a value or a range of values for each issue.

        Args:
            outcome: Outcome being tested
            outcome_range: Outcome range being tested against
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

         Args:
            outcome: The outcome to check
            outcome_range: The outcome range

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
    if fail_incomplete and len(
        set(ikeys(outcome_range)).difference(ikeys(outcome))
    ) > 0:
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

#
# def _make_iterable(
#     v: Union[list, tuple, int, str, float]
# ) -> Union[list, tuple]:
#     """Ensures the output is an Iterable if it was a single value.
#
#     >>> [_make_iterable(1), _make_iterable('1'), _make_iterable(1.0), _make_iterable([1])]
#     [{1}, {'1'}, {1.0}, {1}]
#
#     """
#     if _is_single(v):
#         return {v}
#
#     return set(v)
#
#
# def _intersect_tuples(v1: tuple, v2: tuple) -> tuple:
#     """Finds intersection between two tuples.
#
#     Examples:
#
#         >>> print(_intersect_tuples((0.0, 1.0), (1.1, 2.0)))
#         (0.0, 0.0)
#         >>> _intersect_tuples((0.0, 1.0), (1.0, 2.0))
#         (1.0, 1.0)
#         >>> _intersect_tuples((0.0, 1.0), (0.5, 2.0))
#         (0.5, 1.0)
#
#     """
#     if not (v1[0] <= v2[0] <= v1[1] or v2[0] <= v1[0] <= v2[1]):
#         return (v1[0], v1[0])
#
#     return max((v1[0], v2[0])), min((v1[1], v2[1]))
#
#
# def _union_tuples(v1: tuple, v2: tuple) -> Union[tuple, List[tuple]]:
#     """The union of two tuples.
#
#     Examples:
#
#         >>> _union_tuples((0.0, 1.0), (0.0, 1.0))
#         (0.0, 1.0)
#         >>> _union_tuples((0.0, 1.0), (0.5, 1.0))
#         (0.0, 1.0)
#         >>> _union_tuples((0.0, 1.0), (0.5, 2.0))
#         (0.0, 2.0)
#         >>> _union_tuples((0.0, 1.0), (1.0, 2.0))
#         [(0.0, 1.0), (1.0, 2.0)]
#         >>> _union_tuples((0.0, 1.0), (2.0, 3.0))
#         [(0.0, 1.0), (2.0, 3.0)]
#
#     """
#     if v1[0] == v2[0] and v1[1] == v2[1]:
#         return v1
#
#     if not (v1[0] < v2[0] < v1[1] or v2[0] < v1[0] < v2[1]):
#         return [v1, v2]
#
#     return min((v1[0], v2[0])), max((v1[1], v2[1]))
#
#
# def _subtract_tuples(v1: tuple, v2: tuple) -> Union[tuple, List[tuple]]:
#     """The difference between first and second tuple (v1 - v2)
#
#     Examples:
#
#         >>> _subtract_tuples((0.0, 1.0), (-1.0, 0.0))
#         (0.0, 1.0)
#         >>> _subtract_tuples((0.0, 1.0), (-1.0, 0.5))
#         (0.5, 1.0)
#         >>> _subtract_tuples((0.0, 1.0), (0.2, 0.5))
#         [(0.0, 0.2), (0.5, 1.0)]
#         >>> _subtract_tuples((0.0, 1.0), (0.5, 1.5))
#         (0.0, 0.5)
#         >>> _subtract_tuples((0.0, 1.0), (1.5, 2.0))
#         (0.0, 1.0)
#
#     """
#     v = _intersect_tuples(v1, v2)
#     if v[0] == v[1]:
#         return v1
#
#     if v[0] <= v1[0] and v[1] <= v1[1]:
#         return (v[1], v1[1])
#
#     if v[0] > v1[0] and v[1] < v1[1]:
#         return [(v1[0], v[0]), (v[1], v1[1])]
#
#     return (v1[0], v[0])
#
#
# def _apply_on_lists(
#     v1: Union[set, list], v2: Union[set, list], op: str
# ) -> set:
#     """Applies a set operation on lists.
#
#     Examples:
#
#         >>> _apply_on_lists([1, 2, 3], [2, 4 ,5], 'intersection')
#         {2}
#         >>> _apply_on_lists([1, 2, 3], [2, 4 ,5], 'union')
#         {1, 2, 3, 4, 5}
#         >>> _apply_on_lists([1, 2, 3], [2, 4 ,5], 'subtract')
#         {1, 3}
#
#     """
#     v1, v2 = set(v1), set(v2)
#     if op.startswith('i'):
#         return v1.intersection(v2)
#
#     elif op.startswith('u'):
#         return v1.union(v2)
#
#     elif op.startswith('s'):
#         return v1.difference(v2)
#     raise ValueError()
#
#
# def _intersect_list_tuple(lst: list, tpl: tuple) -> list:
#     """Intersection between a list and tuple.
#
#     Examples:
#
#         >>> _intersect_list_tuple([1, 2, 3], (2.5, 4.0))
#         [3]
#         >>> _intersect_list_tuple([1, 2, 3], (6.0, 7.0))
#         []
#         >>> _intersect_list_tuple([1, 2, 3], (0.0, 3.0))
#         [1, 2, 3]
#         >>> _intersect_list_tuple([], (0.0, 3.0))
#         []
#
#     """
#     result = []
#     for _ in lst:
#         if tpl[0] <= _ <= tpl[1]:
#             result.append(_)
#     return result
#
#
# def _union_list_tuple(
#     lst: list, tpl: tuple
# ) -> Union[tuple, List[Union[tuple, list]]]:
#     """Finds the union between a list and a tuple.
#
#     Examples:
#
#         >>> _union_list_tuple([1, 2, 3], (2.5, 4.0))
#         [(2.5, 4.0), [1, 2, 3]]
#         >>> _union_list_tuple([1, 2, 3], (6.0, 7.0))
#         [(6.0, 7.0), [1, 2, 3]]
#         >>> _union_list_tuple([1, 2, 3], (0.0, 3.0))
#         (0.0, 3.0)
#         >>> _union_list_tuple([], (0.0, 3.0))
#         (0.0, 3.0)
#
#     """
#     for _ in lst:
#         if not (tpl[0] <= _ <= tpl[1]):
#             return [tpl, lst]
#
#     return tpl


# def range_intersection(r1: OutcomeRange, r2: OutcomeRange) -> OutcomeRange:
#     """A new outcome range at the intersection of given outcome ranges
#
#     Examples:
#         >>> r1 = {'price': (0.0, 1.0), 'cost':[1, 2, 3, 4, 5]
#         ...        , 'delivery': ['yes', 'no'], 'honor':'yes'}
#         >>> r2 = {'price': (0.5, 2.0), 'cost':[4, 5, 6]
#         ...        , 'delivery': ['yes'], 'honor':'yes'}
#         >>> r3 = {'price': (0.5, 2.0), 'cost':[4, 5, 6]
#         ...        , 'delivery': ['yes'], 'honor':'no'}
#         >>> r4 = {'price': (0.5, 2.0), 'cost':[4, 5, 6]
#         ...        , 'delivery': 'yes', 'honor':'yes'}
#         >>> r5 = {'price': [1.0, 3.0], 'cost':[4, 5, 6]
#         ...        , 'delivery': 'yes', 'honor':None}
#         >>> r6 = {'price': [1.0, 3.0], 'cost':[4, 5, 6]
#         ...        , 'delivery': 'yes', 'honor':[]}
#         >>> r7 = {'price': 9.0, 'cost':[4, 5, 6]
#         ...        , 'delivery': 'yes', 'honor':None}
#         >>> r8 = {'price': 0.5, 'cost':[9]
#         ...        , 'delivery': 'no', 'honor':None}
#         >>> r9 = {'price': 0.5}
#         >>> range_intersection(r1, r2)
#         {'price': (0.5, 1.0), 'cost': [4, 5], 'delivery': 'yes', 'honor': 'yes'}
#         >>> range_intersection(r1, r3)
#         {'price': (0.5, 1.0), 'cost': [4, 5], 'delivery': 'yes', 'honor': []}
#         >>> range_intersection(r1, r4)
#         {'price': (0.5, 1.0), 'cost': [4, 5], 'delivery': 'yes', 'honor': 'yes'}
#         >>> range_intersection(r1, r5)
#         {'price': 1.0, 'cost': [4, 5], 'delivery': 'yes', 'honor': 'yes'}
#         >>> range_intersection(r1, r6)
#         {'price': 1.0, 'cost': [4, 5], 'delivery': 'yes', 'honor': []}
#         >>> range_intersection(r1, r7)
#         {'price': [], 'cost': [4, 5], 'delivery': 'yes', 'honor': 'yes'}
#         >>> range_intersection(r1, r8)
#         {'price': 0.5, 'cost': [], 'delivery': 'no', 'honor': 'yes'}
#         >>> range_intersection(r1, r9)
#         {'price': 0.5, 'cost': [1, 2, 3, 4, 5], 'delivery': ['yes', 'no'], 'honor': 'yes'}
#
#     """
#     r = {}
#     r1 = range_simplify(r1, compress=False)
#     r2 = range_simplify(r2, compress=False)
#     for k, v1 in r1.items():
#         if k not in r2.keys():
#             r[k] = copy.copy(v1)
#             continue
#         v2 = r2[k]
#         if v1 is None:
#             r[k] = copy.copy(v2)
#             continue
#         if v2 is None:
#             r[k] = copy.copy(v1)
#             continue
#         v1 = copy.copy(_make_iterable(v1))
#         v2 = copy.copy(_make_iterable(v2))
#         if isinstance(v1, tuple) and isinstance(v2, tuple):
#             r[k] = _intersect_tuples(v1, v2)
#         elif isinstance(v1, tuple) and not isinstance(v2, tuple):
#             r[k] = _intersect_list_tuple(lst=v2, tpl=v1)
#         elif not isinstance(v1, tuple) and isinstance(v2, tuple):
#             r[k] = _intersect_list_tuple(lst=v1, tpl=v2)
#         else:
#             r[k] = set(v1).intersection(v2)
#         if isinstance(r[k], list) and len(r[k]) == 1:
#             r[k] = r[k][0]
#     return r
#
#
# def range_simplify(r: OutcomeRange, compress: bool=False) -> OutcomeRange:
#     """Simplifies an outcome range.
#
#     Args:
#
#         r: The outcome range
#         compress: If true the dimensions that are not constrained are removed.
#
#     Examples:
#
#         >>> range_simplify({'a': [5], 'b': [4, 1, 3, 2], 'c': (1.0, 1.0)})
#         {'a': 5, 'b': [4, 1, 3, 2], 'c': 1.0}
#
#         >>> range_simplify({'a': [5], 'b': [4, 1, 3, 2], 'c': [(0.0, 2.0), (1.5, 3.0)
#         ...                              , 4, 2, [1, 2, 3], [1], (0.5, 0.5)]})
#         {'a': 5, 'b': [1, 2, 3, 4], 'c': [(0.0, 3.0), [1, 2, 3, 4]]
#
#
#     """
#
#     def _simplify_value(cutoff_utility):
#         """Puts the value in the most compressed form. """
#
#         if _is_single(cutoff_utility):
#             return cutoff_utility
#         if isinstance(cutoff_utility, tuple) and cutoff_utility[0] == cutoff_utility[1]:
#             return cutoff_utility[0]
#         if isinstance(cutoff_utility, list) and len(cutoff_utility) == 1:
#             return cutoff_utility
#         if isinstance(cutoff_utility, list):
#             combined_list = []
#             tuples = []
#             for item in cutoff_utility:
#                 if _is_single(item):
#                     combined_list.append(item)
#                 elif isinstance(item, tuple):
#                     tuples.append(item)
#                 elif isinstance(item, list):
#                     combined_list += item
#                 else:
#                     raise ValueError('not a single, tuple or list!!! cannot understand')
#             if len(tuples) > 1:
#                 results = _union_tuples(tuples[0], tuples[1])
#                 for i in range(2, len(tuples)):
#                     results = _union_tuples(results, tuples[i])
#                 tuples = results
#             if isinstance(tuples, tuple):
#                 tuples = [tuples]
#             cutoff_utility = tuples + sorted(list(set(combined_list)))
#             if len(cutoff_utility) == 1:
#                 cutoff_utility = cutoff_utility[0]
#         return cutoff_utility
#
#     keys = sorted(r.keys())
#     return {k: _simplify_value(r[k]) for k in keys}
#
#
# def range_equal(r1: OutcomeRange, r2: OutcomeRange) -> OutcomeRange:
#     """Tests if the two outcomes are equal
#
#     Examples:
#         >>> range_equal({'a': 5, 'b': [1, 2, 3, 4], 'c': (1.0, 2.0)},
#         ...             {'a': [5], 'b': [4, 1, 3, 2], 'c': (1.0, 2.0)})
#
#     """
#
# def range_union(r1: OutcomeRange, r2: OutcomeRange) -> OutcomeRange:
#     """A new outcome range covering the union of input outcome ranges.
#
#     Examples:
#         >>> r1 = {'price': (0.0, 1.0), 'cost':[1, 2, 3, 4, 5]
#         ...        , 'delivery': ['yes', 'no'], 'honor':'yes'}
#         >>> r2 = {'price': (0.5, 2.0), 'cost':[4, 5, 6]
#         ...        , 'delivery': ['yes'], 'honor':'yes'}
#         >>> r3 = {'price': (0.5, 2.0), 'cost':[4, 5, 6]
#         ...        , 'delivery': ['yes'], 'honor':'no'}
#         >>> r4 = {'price': (0.5, 2.0), 'cost':[4, 5, 6]
#         ...        , 'delivery': 'yes', 'honor':'yes'}
#         >>> r5 = {'price': [1.0, 3.0], 'cost':[4, 5, 6]
#         ...        , 'delivery': 'yes', 'honor':None}
#         >>> r6 = {'price': [1.0, 3.0], 'cost':[4, 5, 6]
#         ...        , 'delivery': 'yes', 'honor':[]}
#         >>> r7 = {'price': 9.0, 'cost':[4, 5, 6]
#         ...        , 'delivery': 'yes', 'honor':None}
#         >>> r8 = {'price': 0.5, 'cost':[9]
#         ...        , 'delivery': 'no', 'honor':None}
#         >>> r9 = {'price': 0.5}
#         >>> range_union(r1, r2)
#         {'honor': 'yes', 'cost': [1, 2, 3, 4, 5, 6], 'price': (0.0, 2.0), 'delivery': ['yes', 'no']}
#         >>> range_union(r1, r3)
#         {'honor': ['yes', 'no'], 'cost': [1, 2, 3, 4, 5, 6], 'price': (0.0, 2.0), 'delivery': ['yes', 'no']}
#         >>> range_union(r1, r4)
#         {'honor': 'yes', 'cost': [1, 2, 3, 4, 5, 6], 'price': (0.0, 2.0), 'delivery': ['yes', 'no']}
#         >>> range_union(r1, r5)
#         {'honor': None, 'cost': [1, 2, 3, 4, 5, 6], 'price': [(0.0, 1.0), [3]], 'delivery': ['yes', 'no']}
#         >>> range_union(r1, r6)
#         {'honor': 'yes', 'cost': [1, 2, 3, 4, 5, 6], 'price': [(0.0, 1.0), [3]], 'delivery': ['yes', 'no']}
#         >>> range_union(r1, r7)
#         {'honor': None, 'cost': [1, 2, 3, 4, 5, 9], 'price': [(0.0, 1.0), [9.0]], 'delivery': ['yes', 'no']}
#         >>> range_union(r1, r8)
#         {'honor': None, 'cost': [1, 2, 3, 4, 5, 6], 'price': (0.0, 1.0), 'delivery': ['yes', 'no']}
#         >>> range_union(r1, r9)
#         {'honor': None, 'cost': None, 'price': (0.0, 1.0), 'delivery': None}
#
#     """
#     r = {}
#     all_k = list(set(r1.keys()).union(r2.keys()))
#     for k in all_k:
#         if k not in r1.keys():
#             r[k] = None
#             continue
#         if k not in r2.keys():
#             r[k] = None
#             continue
#         v1, v2 = r1[k], r2[k]
#         if v1 is None or v2 is None:
#             r[k] = None
#             continue
#         v1 = _make_iterable(v1)
#         v2 = _make_iterable(v2)
#         if isinstance(v1, tuple) and isinstance(v2, tuple):
#             r[k] = _union_tuples(v1, v2)
#         elif isinstance(v1, tuple) and not isinstance(v2, tuple):
#             r[k] = _union_list_tuple(lst=v2, tpl=v1)
#         elif not isinstance(v1, tuple) and isinstance(v2, tuple):
#             r[k] = _union_list_tuple(lst=v1, tpl=v2)
#         else:
#             r[k] = _apply_on_lists(v1, v2, 'union')
#     return r
#
# def range_subtract(r1: OutcomeRange, r2: OutcomeRange, parent1: OutcomeRange=None, parent2: OutcomeRange=None) -> OutcomeRange:
#     r = {}
#
#     for k, v1 in r1.items():
#         if k not in r2.keys():
#             r[k] = []
#             continue
#         v2 = r2[k]
#         if v1 is None:
#             if parent1 is None:
#                 raise RuntimeError('No parent is defined, cannot invert')
#             r[k] = range_invert({k: v2}, {k: parent1.get(k, None)})
#             continue
#         v1 = _make_iterable(v1)
#         v2 = _make_iterable(v2)
#         if isinstance(v1, tuple) and isinstance(v2, tuple):
#             r[k] = _subtract_tuples(v1, v2)
#         elif isinstance(v1, tuple) and not isinstance(v2, tuple):
#             r[k] = _subtract_list_tuple(lst=v2, tpl=v1)
#         elif not isinstance(v1, tuple) and isinstance(v2, tuple):
#             r[k] = _subtract_list_tuple(lst=v1, tpl=v2)
#         else:
#             r[k] = _apply_on_lists(v1, v2, 'subtract')
#     return r
#
#
# def range_invert(r1: OutcomeRange, issues=Iterable[Issue], resolution=1e-6) -> OutcomeRange:
#     r = {}
#
#     for k, v1 in r1.items():
#         if v1 is None:
#             r[k] = []
#             continue
#     # TODO complete this
#
#     return r
#
#
# def _dim_cardinality(x: Any, recursing=False) -> Union[int, float]:
#     if isinstance(x, str):
#         return 1
#     if isinstance(x, int) and not recursing:
#         return x
#     if isinstance(x, tuple):
#         if x[0] != x[1]:
#             return math.inf
#         else:
#             return 1
#     if isinstance(x, Iterable):
#         return sum([_dim_cardinality(_, recursing=True) for _ in x])
#     return 1
#
#
# def _all_values(x: Any, recursing=False):
#     if isinstance(x, str):
#         yield x
#         raise StopIteration
#     if isinstance(x, int) and not recursing:
#         yield from range(x)
#         raise StopIteration
#     if isinstance(x, tuple):
#         raise RuntimeError('Cannot iterate over ranges (tuples)')
#     if isinstance(x, Iterable):
#         yield from x
#     raise StopIteration
SingleValue = Union[int, float, str]
EPSILON = 1e-9


# class SingleIssueSpace(NamedObject):
#     """Encodes the space of values for a single issue.
#
#     Examples:
#
#         You can define a complex issue space combining ranges and sets of values:
#         >>> everything = SingleIssueSpace(ranges= (20.0, 30.0), count = 10, name='everything')
#         >>> print(everything)
#         everything: (20.0, 30.0) and {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
#
#         You can create child issue spaces that belong to it as needed
#         >>> i1 = SingleIssueSpace(ranges= (20.0, 25.0), values = [1, 2, 3, 4, 5], name='i1', parent=everything)
#         >>> i2 = SingleIssueSpace(ranges= (20.0, 25.0), name='i2', parent=everything)
#         >>> i3 = SingleIssueSpace(values = [1, 2, 3, 4, 5], name='i3', parent=everything)
#         >>> i4 = SingleIssueSpace(values = [6, 7, 8, 9], ranges = (25.0, 29.0), name='i3', parent=everything)
#
#         You can make complex spaces by combining them arbitrary using arithmetic operators
#         >>> print(everything - ((i2 + i3) ))
#         (everything-(i2+i3)): (25.0, 30.0) and {0, 6, 7, 8, 9}
#         >>> print(-(i2 + i4))
#         -(i2+i3): (29.0, 30.0) and {0, 1, 2, 3, 4, 5}
#
#         You can also use logical operators
#         >>> print(everything % ((i2 | i3) ))
#         (everything-(i2+i3)): (25.0, 30.0) and {0, 6, 7, 8, 9}
#         >>> print(~(i2 | i4))
#         -(i2+i3): (29.0, 30.0) and {0, 1, 2, 3, 4, 5}
#
#
#         See the documentation of arithmetic operators to find more examples
#
#     """
#     __slots__ = ['ranges', 'values', 'parent', 'everything']
#
#     def __init__(
#         self,
#         ranges: Union[Tuple[Any, Any], Iterable[Tuple[Any, Any]]] = None,
#         values: Iterable[SingleValue] = None,
#         count: int = None,
#         parent: 'SingleIssueSpace' = None,
#         everything=False,
#         name: str = None,
#     ):
#         super().__init__(name=name)
#         self.parent = parent
#         self.everything = everything
#         if everything:
#             self.ranges = self.values = None
#         else:
#             if ranges is None:
#                 ranges = []
#             if values is None:
#                 values = []
#             if isinstance(ranges, tuple):
#                 ranges = [ranges]
#             self.ranges = list(ranges)
#             self.values = set(values)
#             if count is not None:
#                 self.values = self.values.union(range(count))
#             self.simplify()
#
#     @property
#     def n_outcomes(self) -> Union[int, float]:
#         """Returns the total number of outcomes in a set of issues. `math.inf` indicates infinity
#
#         Examples:
#
#             >>> from negmas import SingleIssueSpace
#             >>> SingleIssueSpace(ranges=(0.0, 1.0), values=[2.0, 3.0, 4.0]).n_outcomes
#             inf
#             >>> SingleIssueSpace(values=[2.0, 3.0, 4.0]).n_outcomes
#             3
#             >>> SingleIssueSpace(count=10).n_outcomes
#             10
#             >>> SingleIssueSpace(count=10, values=[20, 30, 40]).n_outcomes
#             13
#             >>> SingleIssueSpace(count=10, values=[20, 30, 40, 40, 40]).n_outcomes
#             13
#             >>> SingleIssueSpace(count=10, values=[1, 2, 4]).n_outcomes
#             10
#
#
#         """
#         if self.everything:
#             return self.parent.n_outcomes if self.parent is not None else math.inf
#
#         if len(self.ranges) > 0:
#             return math.inf
#
#         return len(self.values)
#
#     @property
#     def cardinalities(self) -> Dict[str, Union[int, float]]:
#         """The cardinality of that issue space with its name. Useful when creating complex OutcomeSpaces
#
#         Examples:
#
#             >>> from negmas import SingleIssueSpace
#             >>> SingleIssueSpace(name='issue', ranges=(0.0, 1.0), values=[2.0, 3.0, 4.0]).cardinalities
#             {'issue': inf}
#             >>> SingleIssueSpace(name='issue', values=[2.0, 3.0, 4.0]).cardinalities
#             {'issue': 3}
#             >>> SingleIssueSpace(name='issue', count=10).cardinalities
#             {'issue': 10}
#             >>> SingleIssueSpace(name='issue', count=10, values=[20, 30, 40]).cardinalities
#             {'issue': 13}
#             >>> SingleIssueSpace(name='issue', count=10, values=[20, 30, 40, 40, 40]).cardinalities
#             {'issue': 13}
#             >>> SingleIssueSpace(name='issue', count=10, values=[1, 2, 4]).cardinalities
#             {'issue': 10}
#
#         """
#         return {self.name: self.cardinality}
#
#     @property
#     def type(self) -> str:
#         """The type of the issue.
#
#         Returns:
#             str: either 'continuous' or 'discrete'
#
#         Examples:
#
#             >>> from negmas import SingleIssueSpace
#             >>> SingleIssueSpace(name='issue', ranges=(0.0, 1.0), values=[2.0, 3.0, 4.0]).type
#             'continuous'
#             >>> SingleIssueSpace(name='issue', values=[2.0, 3.0, 4.0]).type
#             'discrete'
#             >>> SingleIssueSpace(name='issue', count=10).type
#             'discrete'
#             >>> SingleIssueSpace(name='issue', count=10, values=[20, 30, 40]).type
#             'discrete'
#
#         """
#         if self.everything:
#             return self.parent.type if self.parent is not None else (
#                 'continuous' if len(self.ranges) > 0 else 'discrete'
#             )
#
#         return 'continuous' if len(self.ranges) > 0 else 'discrete'
#
#     @property
#     def infinite(self) -> bool:
#         """Test whether any issue is continuous (infinite outcome space)
#
#         Returns:
#             bool: continuous (== infinite) or not
#
#         Examples:
#
#             >>> from negmas import SingleIssueSpace
#             >>> SingleIssueSpace(name='issue', ranges=(0.0, 1.0), values=[2.0, 3.0, 4.0]).infinite
#             True
#             >>> SingleIssueSpace(name='issue', values=[2.0, 3.0, 4.0]).infinite
#             False
#             >>> SingleIssueSpace(name='issue', count=10).infinite
#             False
#             >>> SingleIssueSpace(name='issue', count=10, values=[20, 30, 40]).infinite
#             False
#
#         """
#         if self.everything:
#             return self.parent.infinite if self.parent is not None else len(
#                 self.ranges
#             ) > 0
#
#         return len(self.ranges) > 0
#
#     @property
#     def finite(self) -> bool:
#         """Test whether all issues are discrete (finite outcome space)
#
#         Returns:
#             bool: discrete or not
#
#         Examples:
#
#             >>> from negmas import SingleIssueSpace
#             >>> SingleIssueSpace(name='issue', ranges=(0.0, 1.0), values=[2.0, 3.0, 4.0]).finite
#             False
#             >>> SingleIssueSpace(name='issue', values=[2.0, 3.0, 4.0]).finite
#             True
#             >>> SingleIssueSpace(name='issue', count=10).finite
#             True
#             >>> SingleIssueSpace(name='issue', count=10, values=[20, 30, 40]).finite
#             True
#
#         """
#         return not self.infinite
#
#     @property
#     def all(self) -> Generator:
#         """A generator that generates all possible values.
#
#         Remarks:
#             - This function returns a generator for the case when the number of values is very large.
#             - If you need a list then use something like:
#
#
#         Examples:
#
#             >>> from negmas import SingleIssueSpace
#             >>> try:
#             ...     list(SingleIssueSpace(name='issue', ranges=(0.0, 1.0), values=[2.0, 3.0, 4.0]).all)
#             ... except ValueError:
#             ...     print('failed')
#             failed
#             >>> list(SingleIssueSpace(name='issue', values=[2.0, 3.0, 4.0]).all)
#             [2.0, 3.0, 4.0]
#             >>> list(SingleIssueSpace(name='issue', count=10).all)
#             [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
#             >>> parent = SingleIssueSpace(name='issue', count=10)
#             >>> list(SingleIssueSpace(name='issue', count=3, parent=parent).all)
#             [0, 1, 2]
#             >>> list(SingleIssueSpace(name='issue', parent=parent).all)
#             []
#             >>> list(SingleIssueSpace(name='issue', everything=True, parent=parent).all)
#             [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
#
#         """
#         if self.infinite:
#             raise ValueError(
#                 'Cannot return all possibilities of a continuous issues'
#             )
#
#         if self.everything:
#             yield from self.parent.all
#
#         else:
#             yield from self.values
#
#     def __contains__(self, item: SingleValue) -> bool:
#         """Tests if an outcome is within this issue space.
#
#
#         Examples:
#
#             >>> prices = SingleIssueSpace(name='price', ranges = [(0.0, 10.0), (20.0, 30.0)], values=[2.0, 3.0, 44])
#             >>> [_ in prices for _ in (0.0, 0.5, 1.0, 15.0, 20.0, 22.0, 40, 44.0)]
#             [False, True, True, False, False, True, False, True]
#
#         """
#         if self.everything:
#             return item in self.parent
#
#         for r in self.ranges:
#             if r[0] < item < r[1]:
#                 return True
#
#         return item in self.values
#
#     def _in_ranges(self, v: SingleValue):
#         """Checks if a value is within the ranges."""
#         for r in self.ranges:
#             if r[0] < v < r[1]:
#                 return True
#
#         return False
#
#     def _compress_ranges(self) -> None:
#         """Combines ranges and removes empty ones."""
#         self.ranges = sorted(
#             [_ for _ in self.ranges if _[0] != _[1]], key=lambda x: x[0]
#         )
#         if len(self.ranges) > 1:
#             ranges = _union_tuples(self.ranges[0], self.ranges[1])
#             for i in range(2, len(self.ranges)):
#                 if isinstance(ranges, tuple):
#                     ranges = _union_tuples(ranges, self.ranges[i])
#                 else:
#                     last_union = _union_tuples(ranges[-1], self.ranges[i])
#                     if isinstance(last_union, tuple):
#                         last_union = [last_union]
#                     ranges = ranges[:-1] + last_union
#             if isinstance(ranges, tuple):
#                 ranges = [ranges]
#             self.ranges = ranges
#
#     def simplify(self) -> None:
#         """Simplifies internal representation.
#
#         Examples:
#
#             >>> print(SingleIssueSpace(count=8, ranges= [(22.0, 34.0), (25, 41), (5.0, 7.0)]
#             ...                                     , values= [12, 11, 10, 13, 50], name='i2'))
#             i2: [(5.0, 7.0), (22.0, 41)] and {0, 1, 2, 3, 4, 5, 7, 10, 11, 12, 13, 50}
#             >>> print(SingleIssueSpace(ranges=[(0.0, 1.0), (2.0, 3.0)], values={10.0, 11.0}, name='issue 1'))
#             issue 1: [(0.0, 1.0), (2.0, 3.0)] and {10.0, 11.0}
#             >>> print(SingleIssueSpace(ranges=[(0.0, 1.0), (0.5, 3.0)], values={10.0, 11.0}, name='issue 2'))
#             issue 2: (0.0, 3.0) and {10.0, 11.0}
#             >>> print(SingleIssueSpace(ranges=[(0.0, 1.0), (0.5, 3.0)], values={0.1, 0.9}, name='issue 3'))
#             issue 3: (0.0, 3.0)
#             >>> print(SingleIssueSpace(ranges=[(0.0, 1.0), (1.0, 2.0)], values={0.3, 1.0, 2.5}, name='issue 4'))
#             issue 4: (0.0, 2.0) and {2.5}
#         """
#         if self.everything:
#             return
#
#         self._compress_ranges()
#         self.values = set(_ for _ in self.values if not self._in_ranges(_))
#         if len(self.ranges) > 1:
#             comp = []
#             for i in range(len(self.ranges) - 1):
#                 if self.ranges[i][1] == self.ranges[i + 1][0] and self.ranges[
#                     i
#                 ][
#                     1
#                 ] in self.values:
#                     self.values.remove(self.ranges[i][1])
#                     comp.append(i)
#             for i in comp:
#                 self.ranges = self.ranges[0:i] + [
#                     (self.ranges[i][0], self.ranges[i + 1][1])
#                 ] + self.ranges[
#                     i + 2:
#                     ]
#
#     @property
#     def ancestors(self) -> List['SingleIssueSpace']:
#         """Returns all the ancestors of this issue space
#
#         Examples:
#
#             >>> from negmas import SingleIssueSpace
#             >>> i1 = SingleIssueSpace(count=10, name='i1')
#             >>> i1.ancestors
#             []
#             >>> i2 = SingleIssueSpace(count=10, name='i2')
#             >>> i2.ancestors
#             []
#             >>> i_1 = SingleIssueSpace(count=10, name='i_1', parent=i1)
#             >>> [_.name for _ in i_1.ancestors]
#             ['i1']
#             >>> i_12 = SingleIssueSpace(count=10, name='i_12', parent=i_1)
#             >>> [_.name for _ in i_12.ancestors]
#             ['i_1', 'i1']
#             >>> i_123 = SingleIssueSpace(count=10, name='i_123', parent=i_12)
#             >>> [_.name for _ in i_123.ancestors]
#             ['i_12', 'i_1', 'i1']
#             >>> i_2 = SingleIssueSpace(count=10, name='i_2', parent=i2)
#             >>> [_.name for _ in i_2.ancestors]
#             ['i2']
#
#
#         """
#         if self.parent is None:
#             return []
#
#         return [self.parent] + self.parent.ancestors
#
#     def __add__(self, other: Union[SingleValue, 'SingleIssueSpace']):
#         """Finds the union of two outcome spaces.
#
#         Examples:
#
#             >>> from negmas import SingleIssueSpace
#             >>> i1 = SingleIssueSpace(count=10, ranges= [(20.0, 30.0), (40, 50.0)], values= [50], name='i1')
#             >>> i2 = SingleIssueSpace(count=8, ranges= [(22.0, 34.0), (25, 41), (5.0, 7.0)]
#             ...                                     , values= [12, 11, 10, 13, 50], name='i2')
#             >>> print(i1 + i2)
#             (i1+i2): [(5.0, 7.0), (20.0, 50.0)] and {0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 50}
#             >>> print(i1 + 144)
#             (i1+144): [(20.0, 30.0), (40, 50.0)] and {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 144, 50}
#
#         """
#         if self.everything:
#             return self.parent + other
#
#         if _is_single(other):
#             result = SingleIssueSpace(
#                 ranges=self.ranges,
#                 values=self.values,
#                 count=None,
#                 parent=self.parent,
#                 name=f'({self.name}+{other})',
#             )
#             result.values.add(other)
#         else:
#             result = SingleIssueSpace(
#                 ranges=self.ranges,
#                 values=self.values,
#                 count=None,
#                 parent=self.parent,
#                 name=f'({self.name}+{other.name})',
#             )
#             result.values = self.values.union(other.values)
#             result.ranges += other.ranges
#         result.simplify()
#         return result
#
#     def __mul__(self, other: Union[SingleValue, 'SingleIssueSpace']):
#         """Finds the intersection of two outcome spaces.
#
#         Examples:
#
#             >>> from negmas import SingleIssueSpace
#             >>> i1 = SingleIssueSpace(count=10, ranges= [(20.0, 30.0), (40, 50.0)], values= [50], name='i1')
#             >>> i2 = SingleIssueSpace(count=8, ranges= [(22.0, 34.0), (25, 41), (5.0, 7.0)]
#             ...                                     , values= [12, 11, 10, 13, 50], name='i2')
#             >>> print(i1 * i2)
#             (i1*i2): [(22.0, 30.0), (40, 41)] and {0, 1, 2, 3, 4, 5, 7, 50}
#             >>> print(i1 * 144)
#             (144*i1): set()
#             >>> print(i1 * 21.2)
#             (21.2*i1): {21.2}
#
#         """
#         if self.everything:
#             return self.parent * other
#
#         if _is_single(other):
#             if other in self:
#                 return SingleIssueSpace(
#                     ranges=None,
#                     values=[other],
#                     count=None,
#                     parent=self.parent,
#                     name=f'({other}*{self.name})',
#                 )
#
#             else:
#                 return SingleIssueSpace(
#                     ranges=None,
#                     values=None,
#                     count=None,
#                     parent=self.parent,
#                     name=f'({other}*{self.name})',
#                 )
#
#         result = SingleIssueSpace(
#             ranges=None,
#             values=None,
#             count=None,
#             parent=self.parent,
#             name=f'({self.name}*{other.name})',
#         )
#         result.values = self.values.intersection(other.values)
#         for r1 in self.ranges:
#             for r2 in other.ranges:
#                 intersection = _intersect_tuples(r1, r2)
#                 result.ranges.append(intersection)
#         result.simplify()
#         return result
#
#     def __sub__(self, other: Union[SingleValue, 'SingleIssueSpace']):
#         """An outcome space containing this outcome space minus the intersection with other.
#
#
#         Examples:
#
#             >>> from negmas import SingleIssueSpace
#             >>> i1 = SingleIssueSpace(count=10, ranges= [(20.0, 30.0), (40, 50.0)], values= [50], name='i1')
#             >>> i2 = SingleIssueSpace(count=8, ranges= [(22.0, 34.0), (25, 41), (5.0, 7.0)]
#             ...                                     , values= [12, 11, 10, 13, 50], name='i2')
#             >>> print(i1 - i2)
#             (i1-i2): [(20.0, 22.0), (41, 50.0)] and {8, 9, 6}
#             >>> print(i1 - 144)
#             (i1-144): [(20.0, 30.0), (40, 50.0)] and {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 50}
#             >>> print(i1 - 21.2)
#             (i1-21.2): [(20.0, 21.2), (21.2, 30.0), (40, 50.0)] and {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 50}
#
#
#
#         """
#         if self.everything:
#             return self.parent - other
#
#         if _is_single(other):
#             ranges = []
#             for r in self.ranges:
#                 if r[0] < other < r[1]:
#                     ranges.append((r[0], other))
#                     ranges.append((other, r[1]))
#                 else:
#                     ranges.append(r)
#             values = self.values.copy()
#             if other in values:
#                 values.remove(other)
#             return SingleIssueSpace(
#                 ranges=ranges,
#                 values=values,
#                 count=None,
#                 parent=self.parent,
#                 name=f'({self.name}-{other})',
#             )
#
#         result = SingleIssueSpace(
#             ranges=None,
#             values=None,
#             count=None,
#             parent=self.parent,
#             name=f'({self.name}-{other.name})',
#         )
#         result.values = self.values.difference(other.values)
#         other *= self
#         for r1 in self.ranges:
#             survived = r1
#             for r2 in other.ranges:
#                 if isinstance(survived, tuple):
#                     survived = _subtract_tuples(survived, r2)
#                 else:
#                     survived = [_subtract_tuples(_, r2) for _ in survived]
#             if isinstance(survived, tuple):
#                 survived = [survived]
#             result.ranges += survived
#         result.simplify()
#         return result
#
#     @property
#     def empty(self):
#         """Checks if this is an empty SingleIssueSpace.
#
#         Examples:
#
#             >>> from negmas import SingleIssueSpace
#             >>> SingleIssueSpace().empty
#             True
#             >>> SingleIssueSpace(everything=True).empty
#             True
#             >>> i1 = SingleIssueSpace(count=10)
#             >>> i1.empty
#             False
#             >>> SingleIssueSpace(everything=True, parent=i1).empty
#             False
#
#         """
#         if self.everything:
#             if self.parent is None:
#                 return True
#
#             return self.parent.empty
#
#         return len(self.ranges) == 0 and len(self.values) == 0
#
#     def __neg__(self) -> 'SingleIssueSpace':
#         """The inverse of this outcome space. Assumes that a parent outcome space is defined.
#
#         Examples:
#             >>> from negmas import SingleIssueSpace
#             >>> i1 = SingleIssueSpace(count=10, ranges= [(20.0, 30.0), (40, 50.0)], values= [50], name='i1')
#             >>> i2 = SingleIssueSpace(count=8, ranges= [(22.0, 34.0), (25, 41), (5.0, 7.0)]
#             ...                                     , values= [12, 11, 10, 13, 50], name='i2')
#             >>> try:
#             ...     print(-i2)
#             ... except RuntimeError:
#             ...     print('cannot inverse without a parent')
#             cannot inverse without a parent
#             >>> i2 = SingleIssueSpace(count=8, ranges= [(22.0, 34.0), (25, 41), (5.0, 7.0)]
#             ...                                     , values= [12, 11, 10, 13, 50], parent=i1, name='i2')
#             >>> print(-i2)
#             -i2: [(20.0, 22.0), (41, 50.0)] and {8, 9, 6}
#
#
#         """
#         if self.everything:
#             return self.parent if self.parent is not None else SingleIssueSpace(
#                 ranges=None,
#                 values=None,
#                 count=None,
#                 parent=None,
#                 name=f'-{self.name}',
#             )
#
#         if self.empty:
#             return self.parent if self.parent is not None else SingleIssueSpace(
#                 ranges=None,
#                 values=None,
#                 count=None,
#                 parent=None,
#                 everything=True,
#                 name=f'-{self.name}',
#             )
#
#         if self.parent is None:
#             raise RuntimeError(
#                 'Cannot find the inverse of an issue space that has no parent'
#             )
#
#         result = self.parent - self
#         result.rename(f'-{self.name}')
#         return result
#
#     def rand(self) -> Optional[SingleValue]:
#         """Picks a random valid value.
#
#         Examples:
#
#             >>> from negmas import SingleIssueSpace
#             >>> i1 = SingleIssueSpace(count=10, ranges= [(20.0, 30.0), (40, 50.0)], values= [50], name='i1')
#             >>> samples = [i1.rand() for _ in range(10)]
#             >>> [_ in i1 for _ in samples]
#             [True, True, True, True, True, True, True, True, True, True]
#
#         """
#         if self.everything:
#             if self.parent is None:
#                 return None
#
#             return self.parent.rand()
#
#         if self.empty:
#             return None
#
#         if len(self.ranges) == 0:
#             return random.sample(self.values)
#
#         sizes = [_[1] - _[0] for _ in self.ranges]
#         total_size = sum(sizes)
#         v = random.random() * total_size
#         total = 0.0
#         for s, r in zip(sizes, self.ranges):
#             total += s
#             if v > total:
#                 return r[0] + v - total
#
#         return self.ranges[-1][1] - EPSILON
#
#     rand_valid = rand
#
#     def rand_invalid(self) -> Optional[SingleValue]:
#         """Pick a random *invalid* value
#
#         Examples:
#
#             >>> from negmas import SingleIssueSpace
#             >>> i1 = SingleIssueSpace(count=10, ranges= [(20.0, 30.0), (40, 50.0)], values= [50], name='i1')
#             >>> samples = [i1.rand_invalid() for _ in range(10)]
#             >>> [_ in i1 for _ in samples]
#             [False, False, False, False, False, False, False, False, False, False]
#
#
#         """
#         if self.everything:
#             if self.parent is None:
#                 return None
#
#             return self.parent.rand_invalid()
#
#         if self.empty:
#             return 'anything'
#
#         if len(self.ranges) == 0:
#             if isinstance(random.sample(self.values), str):
#                 return unique_name('', add_time=False)
#
#             mx = max(self.values)
#             return random.randint(mx + 1, 2 * mx)
#
#         mx = self.ranges[0][1]
#         mn = self.ranges[0][0]
#         for r in self.ranges:
#             mx = max([mx, r[1]])
#             mn = min([mn, r[1]])
#         return random.random() * (mx - mn) + mx
#
#     def __str__(self):
#         """Returns a string representation.
#
#         Examples:
#
#             >>> i1 = SingleIssueSpace(ranges=(0.0, 1.0), values=[2.0, 3.0, 4.0], name='issue 1')
#             >>> print(i1)
#             issue 1: (0.0, 1.0) and {2.0, 3.0, 4.0}
#             >>> print(SingleIssueSpace(ranges=(0.0, 1.0), name='issue 2'))
#             issue 2: (0.0, 1.0)
#             >>> print(SingleIssueSpace(ranges=(0.0, 1.0), name='issue 3', parent=i1))
#             issue 3 (parent: issue 1): (0.0, 1.0)
#             >>> print(SingleIssueSpace(parent=i1, name='issue 4'))
#             issue 4 (parent: issue 1): set()
#             >>> print(SingleIssueSpace(parent=i1, everything=True, name='issue 4'))
#             issue 4 (parent: issue 1): everything
#             >>> print(SingleIssueSpace(values={0.0, 1.0}, name='issue 5'))
#             issue 5: {0.0, 1.0}
#             >>> print(SingleIssueSpace(ranges=[(0.0, 1.0), (2.0, 3.0)], values={10.0, 11.0}, name='issue 6'))
#             issue 6: [(0.0, 1.0), (2.0, 3.0)] and {10.0, 11.0}
#             >>> print(SingleIssueSpace(ranges=[(0.0, 1.0), (0.5, 3.0)], values={10.0, 11.0}, name='issue 6'))
#             issue 6: (0.0, 3.0) and {10.0, 11.0}
#             >>> print(SingleIssueSpace(ranges=[(0.0, 1.0), (0.5, 3.0)], values={0.1, 0.9}, name='issue 6'))
#             issue 6: (0.0, 3.0)
#
#         """
#         if self.everything:
#             base = f'everything'
#         elif len(self.ranges) == 0:
#             base = f'{self.values}'
#         elif len(self.values) == 0:
#             if len(self.ranges) == 1:
#                 base = f'{self.ranges[0]}'
#             else:
#                 base = f'{self.ranges}'
#         else:
#             if len(self.ranges) == 1:
#                 base = f'{self.ranges[0]} and {self.values}'
#             else:
#                 base = f'{self.ranges} and {self.values}'
#         if self.parent is not None:
#             parent = f' (parent: {self.parent.name})'
#         else:
#             parent = ''
#         return f'{self.name}{parent}: {base}'
#
#     def __and__(self, other):
#         return self * other
#
#     def __or__(self, other):
#         return self + other
#
#     def __xor__(self, other):
#         return self + other - (self * other)
#
#     def __invert__(self):
#         return -self
#
#     def __mod__(self, other):
#         return self - other
#
#     cardinality = n_outcomes
#     dimensionality = 1
#     n_dims = 1
#     contains = __contains__
#     union = __add__
#     instersection = __mul__
#     difference = __sub__

# class OutcomeSpace(object):
#     """An arbitrary constraint on outcomes.
#
#     Overrides standard operators so it can be used something like the following:
#
#
#
#     """
#     __slots__ = ('_parent', '_range')
#     def __init__(self, parent: Optional['OutcomeSpace']=None, **kwargs) -> None:
#         super().__init__()
#         self._parent = parent
#         self._range = copy.deepcopy(kwargs)
#
#     @property
#     def n_augmented_outcomes(self) -> Union[int, float]:
#         """Returns the total number of outcomes in a set of issues. `-1` indicates infinity"""
#
#         n = 1
#         for dim in self._range.values():
#             n *= _dim_cardinality(dim)
#             if math.isinf(n):
#                 return math.inf
#         return n
#
#     cardinality = n_augmented_outcomes
#
#     @property
#     def n_dims(self):
#         return len(self._range)
#
#     dimensionality = n_dims
#
#     @property
#     def cardinalities(self) -> Dict[str, Union[int, float]]:
#         """The cardinality of all dimensions"""
#         return {k: _dim_cardinality(cutoff_utility) for k, cutoff_utility in self._range}
#
#     @property
#     def types(self) -> Dict[str, str]:
#         """The type of the issue.
#
#         Returns:
#             str: either 'continuous' or 'discrete'
#
#         """
#         return {k: 'continuous' if math.isinf(_dim_cardinality(cutoff_utility)) else 'discrete' for k, cutoff_utility in self._range}
#
#     def is_infinite(self) -> bool:
#         """Test whether any issue is continuous (infinite outcome space)
#
#         Returns:
#             bool: continuous or not
#
#         """
#         return any(math.isinf(_) for _ in self.cardinalities)
#
#     def is_finite(self) -> bool:
#         """Test whether all issues are discrete (finite outcome space)
#
#         Returns:
#             bool: discrete or not
#
#         """
#         return not self.is_infinite()
#
#     @property
#     def all(self) -> Generator:
#         """A generator that generates all possible values.
#
#         Remarks:
#             - This function returns a generator for the case when the number of values is very large.
#             - If you need a list then use something like:
#
#
#         """
#         if self.is_infinite():
#             raise ValueError('Cannot return all possibilities of a continuous issues')
#         if len(self._range) == 0:
#             raise StopIteration
#         yield from itertools.product(_all_values for _ in self._range)
#
#     def __contains__(self, item: Union[Outcome, OutcomeRange, 'OutcomeSpace']):
#         """Tests if an outcome, outcome range, or another outcome space is within this outcome space."""
#         return outcome_in_range(item, self._range)
#
#     def __add__(self, other: Union[Outcome, OutcomeRange, 'OutcomeSpace']):
#         """Finds the union of two outcome spaces."""
#         r = OutcomeSpace(parent=self._parent)
#         r._range = range_union(self._range, other._range)
#         return r
#
#     def __mul__(self, other: Union[Outcome, OutcomeRange, 'OutcomeSpace']):
#         """Finds the intersection of two outcome spaces."""
#         r = OutcomeSpace(parent=self._parent)
#         r._range = range_intersection(self._range, other._range)
#         return r
#
#     def __sub__(self, other: Union[Outcome, OutcomeRange, 'OutcomeSpace']):
#         """An outcome space containing this outcome space minus the intersection with other."""
#
#     def __neg__(self):
#         """The inverse of this outcome space. Assumes that a parent outcome space is defined."""
#
#     def rand(self) -> Dict[str, Union[int, float, str]]:
#         """Picks a random valid value."""
#
#     rand_valid = rand
#
#     def rand_invalid(self) -> Dict[str, Union[int, float, str]]:
#         """Pick a random *invalid* value"""


def outcome_as_dict(outcome: Outcome, issue_names: List[str] = None):
    """Converts the outcome to a dict no matter what was its type"""

    if isinstance(outcome, dict):
        return outcome
    if isinstance(outcome, OutcomeType):
        return outcome.asdict()
    if issue_names is not None:
        return dict(zip(issue_names, outcome))
    return dict(zip((str(_) for _ in range(len(outcome))), outcome))


def outcome_as_tuple(outcome: Outcome):
    """Converts the outcome to a tuple no matter what was its type"""

    if isinstance(outcome, tuple):
        return outcome
    if isinstance(outcome, OutcomeType):
        return outcome.astuple()
    if isinstance(outcome, dict):
        return list(outcome.values())
    raise ValueError(f'Unknown type for outcome {type(outcome)}')
