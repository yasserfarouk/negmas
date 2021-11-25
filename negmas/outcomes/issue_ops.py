from __future__ import annotations

import copy
import itertools
import numbers
import random
import sys
import warnings
import xml.etree.ElementTree as ET
from collections import defaultdict
from functools import reduce
from operator import mul
from typing import (
    Any,
    Callable,
    Collection,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

import numpy as np

from negmas.generics import ikeys, ivalues
from negmas.helpers import PATH

from .base_issue import Issue
from .outcome_ops import outcome2dict

__all__ = [
    "generate_issues",
    "issues_from_genius",
    "issues_from_xml_str",
    "issues_to_genius",
    "issues_to_xml_str",
    "issues_from_outcomes",
    "num_outcomes",
    "enumerate_issues",
    "enumerate_discrete_issues",
    "discretize_and_enumerate_issues",
    "sample_issues",
    "sample_outcomes",
    "combine_issues",
]


def num_outcomes(issues: Collection[Issue]) -> Optional[int]:
    """
    Returns the total number of outcomes in a set of issues.
    `-1` indicates infinity
    """

    n = 1

    for issue in issues:
        n *= issue.cardinality

    return n


def enumerate_issues(
    issues: Collection["Issue"],
    max_n_outcomes: int = None,
) -> List["Outcome"]:
    """
    Enumerates the outcomes of a list of issues.

    Args:
        issues: The list of issues.
        max_n_outcomes: The maximum number of outcomes to return
    Returns:
        List of outcomes of the given type.
    """
    n = num_outcomes(issues)

    if n is None or n == float("inf"):
        return sample_issues(
            issues=issues,
            n_outcomes=max_n_outcomes,
            fail_if_not_enough=False,
            with_replacement=False,
        )

    if max_n_outcomes is not None and n > max_n_outcomes:
        values = sample_outcomes(issues=issues, n_outcomes=max_n_outcomes)
    else:
        values = enumerate_discrete_issues(issues)
    return values


def enumerate_discrete_issues(issues: Iterable[Issue]) -> List["Outcome"]:
    """Enumerates all outcomes of this set of issues if possible

    Args:
        issues: A list of issues

    Returns:
        list of outcomes
    """
    return list(tuple(_) for _ in itertools.product(*(_.all for _ in issues)))


def issues_from_outcomes(
    outcomes: List["Outcome"],
    issue_names: list[str] | None = None,
    numeric_as_ranges: bool = False,
) -> List["Issue"]:
    """
    Create a set of issues given some outcomes

    Args:

        outcomes: A list of outcomes
        issue_names: If given, will be used as issue names, otherwise random issue names will be used
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
        if isinstance(v, numbers.Integral) and not isinstance(old, numbers.Integral):
            return float(v)

        if not isinstance(v, numbers.Integral) and isinstance(old, numbers.Integral):
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
        if o is None:
            continue

        if issue_names is None:
            if isinstance(o, dict):
                issue_names = list(o.keys())
            else:
                issue_names = [f"i{_}" for _ in range(len(o))]

        if n_issues is not None and len(o) != n_issues:
            raise ValueError(
                f"Outcome {o} at {i} has {len(o)} issues but an earlier outcome had {n_issues} issues"
            )

        n_issues = len(o)
        if len(issue_names) != n_issues:
            raise ValueError(
                f"Outcome {i} ({o}) has {len(o)} values but we have {len(issue_names)} issue names"
            )

        o = outcome2dict(o, issue_names)

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


def issues_to_xml_str(issues: List["Issue"], enumerate_integer: bool = False) -> str:
    """Converts the list of issues into a well-formed xml string

    Examples:

        >>> issues = [Issue(values=10, name='i1'), Issue(values=['a', 'b', 'c'], name='i2'),
        ... Issue(values=(2.5, 3.5), name='i3')]
        >>> s = issues_to_xml_str(issues)
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

        >>> issues2, _ = issues_from_xml_str(s)
        >>> print([_.__class__.__name__ for _ in issues2])
        ['CategoricalIssue', 'CategoricalIssue', 'ContinuousIssue']
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
        output += issue._to_xml_str(indx, enumerate_integer)
    output += f"</objective>\n</utility_space>\n</negotiation_template>"

    return output


def issues_to_genius(
    issues: Iterable["Issue"], file_name: PATH, enumerate_integer: bool = False
) -> None:
    """Exports a the domain issues to a GENIUS XML file.

    Args:

        issues: The issues to be exported
        file_name (str): File name to export to

    Returns:

        A List[Issue] or Dict[Issue]


    Examples:

        >>> import pkg_resources
        >>> issues, _ = issues_from_genius(file_name = pkg_resources.resource_filename('negmas'
        ...                                      , resource_name='tests/data/Laptop/Laptop-C-domain.xml'))
        >>> issues_to_genius(issues=issues, file_name = pkg_resources.resource_filename('negmas'
        ...                                    , resource_name='tests/data/LaptopConv/Laptop-C-domain.xml'))
        >>> issues2, _ = issues_from_genius(file_name = pkg_resources.resource_filename('negmas'
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

        >>> issues, _ = issues_from_genius(file_name = pkg_resources.resource_filename('negmas'
        ...                                      , resource_name='tests/data/Laptop/Laptop-C-domain.xml')
        ...     , force_single_issue=True, keep_value_names=False, keep_issue_names=False)
        >>> print(list(issues[0].all))
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
        >>> issues_to_genius(issues=issues, enumerate_integer=True
        ...     , file_name = pkg_resources.resource_filename('negmas'
        ...                                   , resource_name='tests/data/LaptopConv/Laptop-C-domain.xml'))
        >>> issues3, _ = issues_from_genius(file_name=pkg_resources.resource_filename('negmas'
        ...                                    , resource_name='tests/data/LaptopConv/Laptop-C-domain.xml'))
        >>> print([list(issue.all) for issue in issues3])
        [['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26']]

    Remarks:
        See ``from_xml_str`` for all the parameters

    """
    with open(file_name, "w") as f:
        f.write(
            issues_to_xml_str(issues=list(issues), enumerate_integer=enumerate_integer)
        )


def issues_from_xml_str(
    xml_str: str,
    force_single_issue=False,
    force_numeric=False,
    keep_value_names=True,
    keep_issue_names=True,
    safe_parsing=True,
    n_discretization: int | None = None,
    max_n_outcomes: int = 1_000_000,
) -> tuple[list["Issue"] | None, list[dict] | None]:
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
        >>> issues, _ = issues_from_xml_str(open(domain_file_name, 'r').read()
        ... , force_single_issue=True)
        >>> issue = issues[0]
        >>> print(issue.cardinality)
        27
        >>> print(list(issue.all))
        ["Dell+60 Gb+19'' LCD", "Dell+60 Gb+20'' LCD", "Dell+60 Gb+23'' LCD", "Dell+80 Gb+19'' LCD", "Dell+80 Gb+20'' LCD", "Dell+80 Gb+23'' LCD", "Dell+120 Gb+19'' LCD", "Dell+120 Gb+20'' LCD", "Dell+120 Gb+23'' LCD", "Macintosh+60 Gb+19'' LCD", "Macintosh+60 Gb+20'' LCD", "Macintosh+60 Gb+23'' LCD", "Macintosh+80 Gb+19'' LCD", "Macintosh+80 Gb+20'' LCD", "Macintosh+80 Gb+23'' LCD", "Macintosh+120 Gb+19'' LCD", "Macintosh+120 Gb+20'' LCD", "Macintosh+120 Gb+23'' LCD", "HP+60 Gb+19'' LCD", "HP+60 Gb+20'' LCD", "HP+60 Gb+23'' LCD", "HP+80 Gb+19'' LCD", "HP+80 Gb+20'' LCD", "HP+80 Gb+23'' LCD", "HP+120 Gb+19'' LCD", "HP+120 Gb+20'' LCD", "HP+120 Gb+23'' LCD"]

        >>> issues, _ = issues_from_xml_str(open(domain_file_name, 'r').read()
        ... , force_single_issue=True, keep_value_names=False, keep_issue_names=True)
        >>> print(issues[0].name)
        Laptop-Harddisk-External Monitor
        >>> print(len(issues))
        1
        >>> print(list(issues[0].all))
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]

        >>> issues, _ = issues_from_xml_str(open(domain_file_name, 'r').read()
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

        >>> issues, _ = issues_from_xml_str(open(domain_file_name, 'r').read()
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

        >>> issues, _ = issues_from_xml_str(open(domain_file_name, 'r').read()
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


        >>> issues, _ = issues_from_xml_str(open(domain_file_name, 'r').read()
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
        >>> issues, _ = issues_from_xml_str(open(domain_file_name, 'r').read()
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

                    # if keep_value_names:
                    #     if (upper + 1 - lower) > 1_000_000:
                    #         warnings.warn(
                    #             f"Issue {issue_key} has bounds ({lower}, {upper}) which means "
                    #             f"{upper + 1 - lower} values. Consider NOT using keep_value_names"
                    #             f" to reduce memory consumption"
                    #         )
                    #     issues_dict[issue_key] = list(range(lower, upper + 1))
                    # else:
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

    for key, value in issues_dict.items():
        issues_dict[key] = Issue(
            values=value,
            name=issue_info[key]["name"]
            if keep_issue_names
            else str(issue_info[key]["index"]),
        )
    issues = list(issues_dict.values())

    if force_single_issue:
        issue_name_ = "-".join([_.name for _ in issues]) if keep_issue_names else "0"

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
                        *(
                            [str(_) for _ in issue.alli(n=n_discretization)]
                            for issue_key, issue in zip(range(len(issues)), issues)
                        )
                    )
                    all_values = list(map(lambda items: "+".join(items), all_values))
                else:
                    all_values = [str(_) for _ in issues[0].alli(n=n_discretization)]
                issues = [Issue(values=all_values, name=issue_name_)]
            else:
                if n_outcomes is None:
                    n_items = [_.cardinality for _ in issues]
                    n_outcomes = reduce(mul, n_items, 1)
                issues = [Issue(values=n_outcomes, name=issue_name_)]
        else:
            return None, None

    return issues, agents


def issues_from_genius(
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

        file_name (str): File name to import from
        force_single_issue: Combine all issues into a single issue
        force_numeric: Force the issue values to be numeric
        keep_issue_names: Use dictionaries instead of tuples to represent outcomes
        keep_value_names: keep value names in case of strings
        safe_parsing: Add more checks to parsing
        n_discretization: Number of discretization levels per issue
        max_n_outcomes: Maximum number of outcomes to allow. If more the outcomespace
                        has more outcomes, the function will fail returning
                        (None, None)

    Returns:

        A tuple of two optional lists:

            - List[Issue] containing the issues
            - List[str] containing agent names (that are sometimes stored in the genius domain)


    Examples:

        >>> import pkg_resources
        >>> issues, _ = issues_from_genius(file_name = pkg_resources.resource_filename('negmas'
        ...                                      , resource_name='tests/data/Laptop/Laptop-C-domain.xml'))
        >>> print([_.name for _ in issues])
        ['Laptop', 'Harddisk', 'External Monitor']

    Remarks:
        See ``from_xml_str`` for all the parameters

    """
    with open(file_name, encoding="utf-8") as f:
        xml_str = f.read()

        return issues_from_xml_str(
            xml_str=xml_str,
            force_single_issue=force_single_issue,
            keep_value_names=keep_value_names,
            keep_issue_names=keep_issue_names,
            safe_parsing=safe_parsing,
            n_discretization=n_discretization,
            max_n_outcomes=max_n_outcomes,
            force_numeric=force_numeric,
        )


def generate_issues(
    params: Sequence[
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

    for i, issue in enumerate(params):
        count = 1 if one_each else counts[i]  # type: ignore

        for _ in range(count):
            name = str(nxt) if int_names else names[i]  # type: ignore
            # if count > 1:
            #    name = name + f' {j}'
            nxt += 1
            result.append(Issue(values=issue, name=name))

    return result


def discretize_and_enumerate_issues(
    issues: Collection["Issue"],
    n_discretization: int = 10,
    max_n_outcomes: int = None,
) -> List["Outcome"]:
    """
    Enumerates the outcomes of a list of issues.

    Args:
        issues: The list of issues.
        max_n_outcomes: The maximum number of outcomes to return
    Returns:
        List of outcomes of the given type.
    """
    issues = [
        _ if _.is_countable() else Issue(values=_.alli(n_discretization), name=_.name)
        for _ in issues
    ]
    return enumerate_issues(issues, max_n_outcomes=max_n_outcomes)


def sample_outcomes(
    issues: Iterable["Issue"],
    n_outcomes: Optional[int] = None,
    min_per_dim=5,
    expansion_policy=None,
) -> Optional[List[Optional["Outcome"]]]:
    """Discretizes the issue space and returns either a predefined number of outcomes or uniform samples

    Args:
        issues: The issues describing the issue space to be discretized
        n_outcomes: If None then exactly `min_per_dim` bins will be used for every continuous dimension and all outcomes
        will be returned
        min_per_dim: Max levels of discretization per dimension
        expansion_policy: None or 'repeat' or 'null' or 'no'. If repeat, then some of the outcomes will be repeated
        if None or 'no' then no expansion will happen if the total number of outcomes is less than
        n_outcomes: If 'null' then expansion will be with None values

    Returns:
        List of outcomes

    Examples:

        enumberate the whole space

        >>> from negmas.outcomes import Issue
        >>> issues = [Issue(values=(0.0, 1.0), name='Price'), Issue(values=['a', 'b'], name='Name')]
        >>> sample_outcomes(issues=issues)
        [(0.0, 'a'), (0.0, 'b'), (0.25, 'a'), (0.25, 'b'), (0.5, 'a'), (0.5, 'b'), (0.75, 'a'), (0.75, 'b'), (1.0, 'a'), (1.0, 'b')]

        enumerate with sampling for very large space (we have 10 outcomes in the discretized space)

        >>> from negmas.outcomes import Issue
        >>> issues = [Issue(values=(0.0, 1.0), name='Price'), Issue(values=['a', 'b'], name='Name')]
        >>> issues[0].is_continuous()
        True
        >>> sampled=sample_outcomes(issues=issues, n_outcomes=5)
        >>> len(sampled)
        5
        >>> len(set(sampled))
        5

        >>> from negmas.outcomes import Issue
        >>> issues = [Issue(values=(0, 1), name='Price'), Issue(values=['a', 'b'], name='Name')]
        >>> issues[0].is_continuous()
        False
        >>> sampled=sample_outcomes(issues=issues, n_outcomes=5)
        >>> len(sampled)
        4
        >>> len(set(sampled))
        4

    """
    from negmas.outcomes import Issue, enumerate_discrete_issues

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
        return list(enumerate_discrete_issues(issues))

    if cardinality < n_outcomes:
        outcomes = list(enumerate_discrete_issues(issues))

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
            vals.append([issue.rand()])
            continue
        if issue.cardinality < n_per_issue:
            vals.append(issue.all)
            n_found *= issue.cardinality
        else:
            vals.append(issue.rand_outcomes(n=n_per_issue, with_replacement=False))
            n_found *= n_per_issue
        if n_found >= n_outcomes:
            n_per_issue = 1
    outcomes = itertools.product(*vals)
    return list(outcomes)[:n_outcomes]


def sample_issues(
    issues: Collection["Issue"],
    n_outcomes: int,
    with_replacement: bool = True,
    fail_if_not_enough=True,
) -> List["Outcome"]:
    """
    Samples some outcomes from the outcome space defined by the list of issues

    Args:

        issues: List of issues to sample from
        n_outcomes: The number of outcomes required
        with_replacement: Whether sampling is with replacement (allowing repetition)
        fail_if_not_enough: IF given then an exception is raised if not enough outcomes are available

    Returns:

        a list of outcomes

    Examples:

        >>> from negmas.outcomes import Issue
        >>> issues = [Issue(name='price', values=(0.0, 3.0)), Issue(name='quantity', values=10)]

        Sampling outcomes as tuples

        >>> samples = sample_issues(issues=issues, n_outcomes=10)
        >>> len(samples) == 10
        True
        >>> type(samples[0]) == tuple
        True


    """
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
        values = enumerate_discrete_issues(issues=issues)
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
                        sample_issues(
                            issues=issues,
                            n_outcomes=remaining,
                            with_replacement=True,
                            fail_if_not_enough=False,
                        )
                    )
                )
                i += 1
            values = list(tmp_values)

    return [tuple(_) for _ in values]


def combine_issues(
    issues: Iterable["Issue"],
    name: str | None = None,
    keep_value_names=True,
    issue_sep="_",
    value_sep="-",
) -> Optional["Issue"]:
    """
    Combines multiple issues into a single issue

    Args:
        issues: The issues to be combined
        name: The name of the resulting issue (If not given, combines input issue names)
        keep_value_names: If true, the values for the generated issue
                          will be a concatenation of values from earlier
                          issues separated by `value_sep`.
        issue_sep: Separator for the issue name (used only if `keep_issue_names`)
        value_sep: Separator for the issue name (used only if `keep_value_names`)

    Remarks:

        - Only works if the issues have finite cardinality

    """
    n_outcomes = num_outcomes(issues)
    if n_outcomes == float("inf"):
        return None
    name = issue_sep.join([_.name for _ in issues]) if name is None else name
    if keep_value_names:
        values = [
            value_sep.join([str(_) for _ in outcome])
            for outcomes in enumerate_issues(issues)
        ]
    else:
        values = n_outcomes
    return Issue(name=name, values=values)
