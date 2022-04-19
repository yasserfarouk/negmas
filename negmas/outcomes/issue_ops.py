from __future__ import annotations

import copy
import itertools
import numbers
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Callable, Iterable, Sequence

from negmas.outcomes.contiguous_issue import ContiguousIssue

if TYPE_CHECKING:
    from .common import Outcome

import numpy as np

from negmas.helpers import PathLike

from .base_issue import DiscreteIssue, Issue, make_issue
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


DUMMY_ISSUE_NAME = "DUMMY_ISSUE"


def num_outcomes(issues: Sequence[Issue]) -> int | float:
    """
    Returns the total number of outcomes in a set of issues.
    """

    n = 1

    for issue in issues:
        n *= issue.cardinality

    return n


def enumerate_discrete_issues(issues: Sequence[DiscreteIssue]) -> list[Outcome]:
    """
    Enumerates all outcomes of this set of discrete issues if possible.

    Args:
        issues: A list of issues

    Returns:
        list of outcomes
    """
    return list(itertools.product(*(_.all for _ in issues)))


def sample_outcomes(
    issues: Sequence[Issue],
    n_outcomes: int | None = None,
    min_per_dim: int = 5,
    expansion_policy=None,
) -> list[Outcome]:
    """
    Discretizes the issue space and returns either a predefined number of outcomes or uniform samples.

    Args:
        issues: The issues describing the issue space to be discretized
        n_outcomes: If None then exactly `min_per_dim` bins will be used for every continuous dimension and all outcomes
        will be returned
        min_per_dim: Max levels of discretization per dimension
        expansion_policy: None or 'repeat' or 'null' or 'no'. If repeat, then some of the outcomes will be repeated
        if None or 'no' then no expansion will happen if the total number of outcomes is less than
        n_outcomes: If 'null' then expansion will be with None values

    Returns:
        list of outcomes

    Examples:

        enumberate the whole space

        >>> from negmas.outcomes import make_issue
        >>> issues = [make_issue(values=(0.0, 1.0), name='Price'), make_issue(values=['a', 'b'], name='Name')]
        >>> sample_outcomes(issues=issues)
        [(0.0, 'a'), (0.0, 'b'), (0.25, 'a'), (0.25, 'b'), (0.5, 'a'), (0.5, 'b'), (0.75, 'a'), (0.75, 'b'), (1.0, 'a'), (1.0, 'b')]

        enumerate with sampling for very large space (we have 10 outcomes in the discretized space)

        >>> from negmas.outcomes import make_issue
        >>> issues = [make_issue(values=(0.0, 1.0), name='Price'), make_issue(values=['a', 'b'], name='Name')]
        >>> issues[0].is_continuous()
        True
        >>> sampled=sample_outcomes(issues=issues, n_outcomes=5)
        >>> len(sampled)
        5
        >>> len(set(sampled))
        5

        >>> from negmas.outcomes import make_issue
        >>> issues = [make_issue(values=(0, 1), name='Price'), make_issue(values=['a', 'b'], name='Name')]
        >>> issues[0].is_continuous()
        False
        >>> sampled=sample_outcomes(issues=issues, n_outcomes=5)
        >>> len(sampled)
        4
        >>> len(set(sampled))
        4

    """
    from negmas.outcomes import enumerate_discrete_issues

    issues = list(issues)
    issues = [copy.deepcopy(_) for _ in issues]
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
            n_disc += issue.cardinality

    if len(continuous) > 0:
        if n_outcomes is not None:
            n_per_issue = max(min_per_dim, int(n_outcomes - n_disc) // len(continuous))
        else:
            n_per_issue = min_per_dim

        for i, issue in enumerate(continuous):
            issues[indx[i]] = make_issue(
                name=issue.name,
                values=list(
                    np.linspace(
                        issue.min_value, issue.max_value, num=n_per_issue, endpoint=True
                    ).tolist()
                ),
            )

    cardinality = 1

    for issue in issues:
        cardinality *= issue.cardinality

    if cardinality == n_outcomes or n_outcomes is None:
        return list(
            enumerate_discrete_issues(issues)  # type: ignore I am  sure that the issues are all discrete by this point
        )

    if cardinality < n_outcomes:
        cardinality = int(cardinality)
        outcomes = list(
            enumerate_discrete_issues(issues)  # type: ignore I am  sure that the issues are all discrete by this point
        )

        if expansion_policy == "no" or expansion_policy is None:
            return outcomes
        elif expansion_policy == "null":
            return outcomes + [None] * (n_outcomes - cardinality)
        elif expansion_policy == "repeat":
            n_reps = n_outcomes // cardinality
            n_rem = n_outcomes % cardinality

            if n_reps > 1:
                for _ in range(n_reps):
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


def _sample_issues(
    issues: Sequence[Issue],
    n: int,
    with_replacement,
    n_total,
    old_values,
    trial,
    max_trials,
) -> Iterable[Outcome]:
    if trial > max_trials:
        return old_values

    remaining = n - len(old_values)
    if remaining < 0:
        return set(list(old_values)[:n])
    if remaining == 0:
        return old_values

    samples = []

    for issue in issues:
        samples.append(
            issue.rand_outcomes(
                n=remaining, with_replacement=True, fail_if_not_enough=True
            )
        )

    _v = []

    for i in range(remaining):
        _v.append([s[i] for s in samples])

    new_values = []
    for value in _v:
        new_values.append(tuple(value))
    new_values = set(new_values)
    values = old_values.union(new_values)
    remaining = n - len(values)
    if remaining < 0:
        return set(list(values)[:n])
    if remaining < 1:
        return values
    return values.union(
        _sample_issues(
            issues,
            remaining,
            with_replacement,
            n_total,
            values,
            trial + 1,
            max_trials,
        )
    )


def sample_issues(
    issues: Sequence[Issue],
    n_outcomes: int,
    with_replacement: bool = True,
    fail_if_not_enough=True,
) -> Iterable[Outcome]:
    """
    Samples some outcomes from the outcome space defined by the list of issues.

    Args:

        issues: list of issues to sample from
        n_outcomes: The number of outcomes required
        with_replacement: Whether sampling is with replacement (allowing repetition)
        fail_if_not_enough: IF given then an exception is raised if not enough outcomes are available

    Returns:

        a list of outcomes

    Examples:

        >>> from negmas.outcomes import make_issue
        >>> issues = [make_issue(name='price', values=(0.0, 3.0)), make_issue(name='quantity', values=10)]

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
        return enumerate_discrete_issues(issues=issues)  # type: ignore I know that these issues are discrete
    if n_total is None and n_outcomes is None:
        raise ValueError(
            f"Cannot sample unknown number of outcomes from continuous outcome spaces"
        )
    return list(
        _sample_issues(issues, n_outcomes, with_replacement, n_total, set(), 0, 10)
    )


def enumerate_issues(
    issues: Sequence[Issue],
    max_cardinality: int | None = None,
) -> list[Outcome]:
    """
    Enumerates the outcomes of a list of issues.

    Args:
        issues: The list of issues.
        max_cardinality: The maximum number of outcomes to return
    Returns:
        list of outcomes of the given type.
    """
    n = num_outcomes(issues)

    if n is None or n == float("inf"):
        if max_cardinality is None:
            raise ValueError(
                "Cannot enumerate continuous issues without specifying `max_cardinality`"
            )
        return list(
            sample_issues(
                issues=issues,
                n_outcomes=max_cardinality,
                fail_if_not_enough=False,
                with_replacement=False,
            )
        )

    if max_cardinality is not None and n > max_cardinality:
        return sample_outcomes(issues=issues, n_outcomes=max_cardinality)
    else:
        return list(tuple(_) for _ in itertools.product(*(_.all for _ in issues)))


def issues_from_outcomes(
    outcomes: Sequence[Outcome] | int,
    numeric_as_ranges: bool = True,
    issue_names: list[str] | None = None,
) -> tuple[DiscreteIssue]:
    """
    Create a set of issues given some outcomes.

    Args:

        outcomes: A list of outcomes or the number of outcomes
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

    if isinstance(outcomes, int):
        outcomes = [(_,) for _ in range(outcomes)]

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

        o_dict = outcome2dict(o, issue_names)

        if names is not None and not all(a == b for a, b in zip(names, o_dict.keys())):
            raise ValueError(
                f"Outcome {o} at {i} has issues {list(o_dict.keys())} but an earlier outcome had issues {names}"
            )

        for i, v in enumerate(o):
            k = issue_names[i]
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
        return tuple(  # type: ignore (seems  ok but not sure)
            make_issue(values=(v[0], v[-1]), name=n)
            if len(v) > 0
            and (isinstance(v[0], int))
            and all(a == b + 1 for a, b in zip(v[1:], v[:-1]))
            else make_issue(values=v, name=n)
            for n, v in values.items()
        )
    else:
        return tuple(make_issue(values=v, name=n) for n, v in values.items())  # type: ignore (seems  ok but not sure)


def issues_to_xml_str(issues: Sequence[Issue]) -> str:
    """
    Converts the list of issues into a well-formed xml string.

    Examples:

        >>> issues = [make_issue(values=10, name='i1'), make_issue(values=['a', 'b', 'c'], name='i2'),
        ... make_issue(values=(2.5, 3.5), name='i3')]
        >>> s = issues_to_xml_str(issues)
        >>> print(s.strip())
        <negotiation_template>
        <utility_space number_of_issues="3">
        <objective description="" etype="objective" index="0" name="root" type="objective">
            <issue etype="discrete" index="1" name="i1" type="discrete" vtype="integer">
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
        output += issue._to_xml_str(indx)
    output += f"</objective>\n</utility_space>\n</negotiation_template>"

    return output


def issues_to_genius(issues: Sequence[Issue], file_name: PathLike | str) -> None:
    """
    Exports a the domain issues to a GENIUS XML file.

    Args:

        issues: The issues to be exported
        file_name (str): File name to export to

    Returns:

        A tuple[Issue, ...] or dict[Issue]


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
        ...                                      , resource_name='tests/data/Laptop/Laptop-C-domain.xml'))
        >>> print([list(issue.all) for issue in issues])
        [['Dell', 'Macintosh', 'HP'], ['60 Gb', '80 Gb', '120 Gb'], ["19'' LCD", "20'' LCD", "23'' LCD"]]

    Remarks:
        See ``from_xml_str`` for all the parameters

    """
    with open(file_name, "w") as f:
        f.write(issues_to_xml_str(issues=issues))


def issues_from_xml_str(
    xml_str: str,
    safe_parsing=True,
    n_discretization: int | None = None,
) -> tuple[Sequence[Issue] | None, Sequence[str] | None]:
    """
    Exports a list/dict of issues from a GENIUS XML file.

    Args:

        xml_str (str): The string containing GENIUS style XML domain issue definitions
        safe_parsing (bool): Turn on extra checks
        n_discretization (Optional[int]): If not None, real valued issues are discretized with the given
        number of values
        max_cardinality (int): Maximum number of outcomes allowed (effective only if force_single_issue is True)

    Returns:

        - tuple[Issue, ...] The issues (note that issue names will be stored in the name attribute of each issue if keep_issue_names)
        - list[dict] A list of agent information dicts each contains 'agent', 'class', 'utility_file_name'

    Examples:

        >>> import pkg_resources
        >>> domain_file_name = pkg_resources.resource_filename('negmas'
        ...                                      , resource_name='tests/data/Laptop/Laptop-C-domain.xml')
        >>> issues, _ = issues_from_xml_str(open(domain_file_name, 'r').read())
        >>> print([_.cardinality for _ in issues])
        [3, 3, 3]

        >>> domain_file_name = pkg_resources.resource_filename('negmas'
        ...                              , resource_name='tests/data/fuzzyagent/single_issue_domain.xml')
        >>> issues, _ = issues_from_xml_str(open(domain_file_name, 'r').read())
        >>> len(issues)
        1
        >>> type(issues)
        <class 'tuple'>
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
    issues_dict: dict[int | str, Any] = {}
    issue_info = {}

    for child in utility_space:
        if child.tag == "issue":
            indx = int(child.attrib["index"]) - 1
            issue_name = str(child.attrib["name"])
            issue_info[issue_name] = {"name": issue_name, "index": indx}
            info = {"type": "discrete", "etype": "discrete", "vtype": "discrete"}

            for a in ("type", "etype", "vtype"):
                info[a] = child.attrib.get(a, info[a])
            mytype = info["type"]

            if mytype == "discrete":
                issues_dict[issue_name] = []

                for item in child:
                    if item.tag == "item":
                        item_name = item.attrib.get("value", None)

                        if (
                            item_name not in issues_dict[issue_name]
                        ):  # ignore repeated items
                            issues_dict[issue_name].append(item_name)

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
                    issues_dict[issue_name] = lower, upper
                else:
                    lower, upper = float(lower_), float(upper_)
                    if n_discretization is None:
                        issues_dict[issue_name] = lower, upper
                    else:
                        issues_dict[issue_name] = n_discretization
            else:
                # I should add the real-valued issues_dict code here
                raise ValueError(f"Unknown type: {mytype}")
        else:
            raise ValueError(f"Unknown child for objective: {child.tag}")

    issues_by_index = dict()
    for key, value in issues_dict.items():
        indx = issue_info[key]["index"]
        issues_by_index[indx] = make_issue(values=value, name=issue_info[key]["name"])
    n_issues = max(issues_by_index.keys()) + 1
    issues = []
    for i in range(n_issues):
        if i not in issues_by_index.keys():
            if safe_parsing:
                raise ValueError(
                    f"No issue with index {i} is found even though we have issues with higher indices"
                )
            issues.append(ContiguousIssue((1, 1), name=DUMMY_ISSUE_NAME))
            continue
        issues.append(issues_by_index[i])

    return tuple(issues), tuple(agents)


def issues_from_genius(
    file_name: PathLike | str,
    safe_parsing=True,
    n_discretization: int | None = None,
) -> tuple[Sequence[Issue] | None, Sequence[str] | None]:
    """
    Imports a the domain issues from a GENIUS XML file.

    Args:

        file_name (str): File name to import from
        safe_parsing: Add more checks to parsing
        n_discretization: Number of discretization levels per issue

    Returns:

        A tuple of two optional lists:

            - tuple[Issue, ...] containing the issues
            - list[str] containing agent names (that are sometimes stored in the genius domain)


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
            safe_parsing=safe_parsing,
            n_discretization=n_discretization,
        )


def generate_issues(
    params: Sequence[
        int | list[str] | tuple[int, int] | Callable | tuple[float, float]
    ],
    counts: list[int] | None = None,
    names: list[str] | None = None,
) -> tuple[Issue, ...]:
    """
    Generates a set of issues with given parameters. Each is optionally repeated.

    Args:

        params: The parameters of the issues
        counts: The number of times to repeat each of the `issues`
        names: The names to assign to the issues. If None, then string representations of integers
               starting from zero will be used.

    Returns:
        list['Issue']: The list of issues with given conditions

    """
    one_each = counts is None
    int_names = names is None
    result: list[Issue] = []
    nxt = 0

    for i, issue in enumerate(params):
        count = 1 if one_each else counts[i]  # type: ignore

        for _ in range(count):
            name = str(nxt) if int_names else names[i]  # type: ignore
            # if count > 1:
            #    name = name + f' {j}'
            nxt += 1
            result.append(make_issue(values=issue, name=name))

    return tuple(result)


def discretize_and_enumerate_issues(
    issues: Iterable[Issue],
    n_discretization: int | None = 10,
    max_cardinality: int | None = None,
) -> list[Outcome]:
    """
    Enumerates the outcomes of a list of issues.

    Args:
        issues: The list of issues.
        max_cardinality: The maximum number of outcomes to return
    Returns:
        list of outcomes of the given type.
    """
    issues = [
        _
        if _.is_finite()
        else make_issue(values=list(_.value_generator(n_discretization)), name=_.name)
        for _ in issues
    ]
    return enumerate_issues(issues, max_cardinality=max_cardinality)


def combine_issues(
    issues: Sequence[Issue],
    name: str | None = None,
    keep_value_names=True,
    issue_sep="_",
    value_sep="-",
) -> Issue | None:
    """
    Combines multiple issues into a single issue.

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
            value_sep.join([str(_) for _ in outcomes])
            for outcomes in enumerate_issues(issues)
        ]
    else:
        values = n_outcomes
    return make_issue(name=name, values=values)
