"""Defines import/export functionality
"""
import pathlib
import functools
import operator
import os
import shutil
import xml.etree.ElementTree as ET
from os import listdir
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union, Sequence

import numpy as np

from .helpers import PATH
from .generics import ivalues
from .negotiators import Negotiator
from .outcomes import Issue, enumerate_outcomes
from .sao import SAOMechanism
from .utilities import UtilityFunction, make_discounted_ufun

__all__ = [
    "load_genius_domain",
    "load_genius_domain_from_folder",
    "convert_genius_domain_from_folder",
    "convert_genius_domain",
    "find_domain_and_utility_files",
    "get_domain_issues",
]


def get_domain_issues(
    domain_file_name: str,
    force_single_issue=False,
    max_n_outcomes: int = 1_000_000,
    n_discretization: Optional[int] = None,
    keep_issue_names=True,
    keep_value_names=True,
    safe_parsing=False,
) -> Union[Dict[str, Issue], List[Issue]]:
    """
    Returns the issues of a given XML domain (Genius Format)

    Args:
        domain_file_name:
        force_single_issue:
        max_n_outcomes:
        n_discretization:
        keep_issue_names:
        keep_value_names:
        safe_parsing:

    Returns:
        List or Dict of issues

    """
    issues, issues_details = None, None
    if domain_file_name is not None:
        domain_file_name = str(domain_file_name)
        issues_details, _ = Issue.from_genius(
            domain_file_name,
            force_single_issue=False,
            keep_issue_names=True,
            keep_value_names=True,
            safe_parsing=safe_parsing,
            n_discretization=n_discretization,
        )
        if force_single_issue:
            issues, _ = Issue.from_genius(
                domain_file_name,
                force_single_issue=force_single_issue,
                keep_issue_names=keep_issue_names,
                keep_value_names=keep_value_names,
                max_n_outcomes=max_n_outcomes,
                n_discretization=n_discretization,
            )
            if issues is None:
                return []
        else:
            issues, _ = Issue.from_genius(
                domain_file_name,
                force_single_issue=force_single_issue,
                keep_issue_names=keep_issue_names,
                keep_value_names=keep_value_names,
                safe_parsing=safe_parsing,
                n_discretization=n_discretization,
            )
    return issues if not force_single_issue else [issues]


def load_genius_domain(
    domain_file_name: PATH,
    utility_file_names: Optional[List[str]] = None,
    agent_factories: Optional[
        Union[Callable[[], Negotiator], List[Callable[[], Negotiator]]]
    ] = None,
    force_single_issue=False,
    force_numeric=False,
    cache_and_discretize_outcomes=False,
    max_n_outcomes: int = 1_000_000,
    n_discretization: Optional[int] = None,
    keep_issue_names=True,
    keep_value_names=True,
    normalize_utilities=False,
    normalize_max_only=False,
    n_steps=None,
    time_limit=3 * 60,  # GENIUS uses 3min time limit by default
    max_n_agents=None,
    dynamic_entry=True,
    safe_parsing=False,
    ignore_reserved=False,
    ignore_discount=False,
    **kwargs,
) -> Tuple[Optional[SAOMechanism], List[dict], Union[Dict[str, Issue], List[Issue]]]:
    """
    Loads a genius domain, creates appropriate negotiators if necessary

    Args:
        domain_file_name:
        utility_file_names:
        agent_factories:
        force_single_issue:
        cache_and_discretize_outcomes:
        max_n_outcomes:
        n_discretization:
        keep_issue_names:
        keep_value_names:
        normalize_utilities:
        normalize_max_only:
        n_steps:
        time_limit:
        max_n_agents:
        dynamic_entry:
        safe_parsing:
        ignore_reserved:
        ignore_discount:
        kwargs: Extra arguments to pass verbatim to SAOMechanism constructor

    Returns:
        - mechanism (SAOMechanism): A mechanism for the given issues
        - agent_info (List[Dict]): All Negotiator functions from the given file
        - issues Union[Issue, Dict[str, Issue], List[Issue]]] : The issues

    """
    issues, issues_details, mechanism = None, None, None
    if domain_file_name is not None:
        domain_file_name = str(domain_file_name)
        issues_details, _ = Issue.from_genius(
            domain_file_name,
            force_single_issue=False,
            keep_issue_names=True,
            keep_value_names=True,
            safe_parsing=safe_parsing,
            n_discretization=n_discretization,
        )
        if force_single_issue:
            issues, _ = Issue.from_genius(
                domain_file_name,
                force_single_issue=force_single_issue,
                keep_issue_names=keep_issue_names,
                keep_value_names=keep_value_names,
                max_n_outcomes=max_n_outcomes,
                n_discretization=n_discretization,
                force_numeric=force_numeric,
            )
            if issues is None:
                return None, [], []
        else:
            issues, _ = Issue.from_genius(
                domain_file_name,
                force_single_issue=force_single_issue,
                keep_issue_names=keep_issue_names,
                keep_value_names=keep_value_names,
                safe_parsing=safe_parsing,
                n_discretization=n_discretization,
                force_numeric=force_numeric,
            )

    agent_info = []
    if utility_file_names is None:
        utility_file_names = []
    utility_file_names = [str(_) for _ in utility_file_names]
    for ufname in utility_file_names:
        utility, discount_factor = UtilityFunction.from_genius(
            file_name=ufname,
            force_single_issue=force_single_issue,
            keep_issue_names=keep_issue_names,
            keep_value_names=keep_value_names,
            normalize_utility=normalize_utilities,
            normalize_max_only=normalize_max_only,
            domain_issues=issues_details,
            safe_parsing=safe_parsing,
            max_n_outcomes=max_n_outcomes,
            ignore_discount=ignore_discount,
            ignore_reserved=ignore_reserved,
            force_numeric=force_numeric,
        )
        agent_info.append(
            {
                "ufun": utility,
                "ufun_name": ufname,
                "reserved_value_func": utility.reserved_value
                if utility is not None
                else 0.0,
                "discount_factor": discount_factor,
            }
        )
    outcomes = None
    try:
        if force_single_issue or cache_and_discretize_outcomes or len(issues) == 1:
            n_outcomes: float = functools.reduce(
                operator.mul,
                (
                    float(_.cardinality)
                    if not _.is_uncountable()
                    else float(n_discretization)
                    if n_discretization is not None
                    else np.inf
                    for _ in ivalues(issues)
                ),
                1.0,
            )
            if n_outcomes < max_n_outcomes:
                outcomes = enumerate_outcomes(
                    issues, astype=dict if keep_issue_names else tuple
                )
    except ValueError:
        pass
    if domain_file_name is not None:
        mechanism_name = (
            domain_file_name.split("/")[-1][:-4]
            # .replace("-domain", "")
            # .replace("_domain", "")
            # .replace("domain", "")
        )
        kwargs["outcome_type"] = dict if keep_issue_names else tuple
        kwargs["avoid_ultimatum"] = False
        mechanism = SAOMechanism(
            issues=issues,
            outcomes=outcomes,
            n_steps=n_steps,
            time_limit=time_limit,
            max_n_agents=max_n_agents,
            dynamic_entry=dynamic_entry,
            name=mechanism_name,
            **kwargs,
        )
        if agent_info is not None and len(agent_info) > 0:
            for info in agent_info:
                info["ufun"] = (
                    info["ufun"]
                    if info["discount_factor"] is None or info["discount_factor"] == 1.0
                    else make_discounted_ufun(
                        ufun=info["ufun"],
                        ami=mechanism.ami,
                        discount_per_round=info["discount_factor"],
                        power_per_round=1.0,
                    )
                )

    if agent_factories is not None and agent_info is not None and len(agent_info) > 0:
        if not isinstance(agent_factories, Iterable):
            agent_factories = [agent_factories] * len(agent_info)
        agents = [factory() for factory in agent_factories[0 : len(agent_info)]]
        for a, info in zip(agents, agent_info):
            mechanism.add(a, ufun=info["ufun"])

    return mechanism, agent_info, (issues if not force_single_issue else [issues])


def load_genius_domain_from_folder(
    folder_name: str,
    agent_factories: Optional[
        Union[Callable[[], Negotiator], List[Callable[[], Negotiator]]]
    ] = None,
    force_single_issue=False,
    force_numeric=False,
    cache_and_discretize_outcomes=False,
    max_n_outcomes: int = 1_000_000,
    n_discretization: Optional[int] = None,
    keep_issue_names=True,
    keep_value_names=True,
    normalize_utilities=False,
    normalize_max_only=False,
    n_steps=None,
    time_limit=60,  # GENIUS uses 3min time limit by default
    max_n_agents=None,
    dynamic_entry=True,
    safe_parsing=False,
    ignore_reserved=False,
    ignore_discount=False,
    **kwargs,
) -> Tuple[Optional[SAOMechanism], List[dict], Union[Dict[str, Issue], List[Issue]]]:
    """
    Loads a genius domain from a folder. See ``load_genius_domain`` for more details.

    Args:
        folder_name:
        agent_factories:
        force_single_issue:
        cache_and_discretize_outcomes:
        max_n_outcomes:
        n_discretization:
        keep_issue_names:
        keep_value_names:
        normalize_utilities:
        normalize_max_only:
        n_steps:
        time_limit:
        max_n_agents:
        dynamic_entry:
        safe_parsing:
        ignore_reserved:
        ignore_discount:
        kwargs: other arguments to pass to SAOMechanism constructor verbatim

    Returns:
        - mechanism (SAOMechanism): A mechanism for the given issues
        - agent_info (List[Dict]): All Negotiator utility functions from the given file
        - issues Union[Issue, Dict[str, Issue], List[Issue]]] : The issues

    Examples:

        >>> import pkg_resources
        >>> from negmas import *
        >>> folder_name = pkg_resources.resource_filename('negmas', resource_name='tests/data/10issues')
        >>> mechanism, negotiators, issues = load_genius_domain_from_folder(folder_name
        ...                             , force_single_issue=False, keep_issue_names=False
        ...                             , keep_value_names=False, normalize_utilities=False)
        >>> print(len(issues))
        10
        >>> print(len(negotiators))
        2
        >>> print([type(a['ufun']) for a in negotiators])
        [<class 'negmas.utilities.nonlinear.HyperRectangleUtilityFunction'>, <class 'negmas.utilities.nonlinear.HyperRectangleUtilityFunction'>]
        >>> print(negotiators[0]['ufun'].outcome_ranges[0])
        {1: (7.0, 9.0), 3: (2.0, 7.0), 5: (0.0, 8.0), 8: (0.0, 7.0)}
        >>> print(negotiators[0]['ufun'].mappings[0])
        97.0
        >>> u = negotiators[0]['ufun']
        >>> print(u(tuple([0.0] * len(issues))))
        0.0
        >>> print(u(tuple([0.5] * len(issues))))
        186.0

        Try loading and running a domain with predetermined agents:
        >>> mechanism, agents, issues = load_genius_domain_from_folder(
        ...                             pkg_resources.resource_filename('negmas', resource_name='tests/data/Laptop')
        ...                             , agent_factories=AspirationNegotiator
        ...                             , force_single_issue=True, keep_issue_names=False
        ...                             , keep_value_names=False)
        >>> state = mechanism.run()
        >>> state.agreement is not None
        True


        >>> mechanism, negotiators, issues = load_genius_domain_from_folder(
        ...                             pkg_resources.resource_filename('negmas', resource_name='tests/data/Laptop'))

        >>> len(issues), len(negotiators)
        (3, 2)

        >>> [type(a['ufun']) for a in negotiators]
        [<class 'negmas.utilities.linear.LinearUtilityAggregationFunction'>, <class 'negmas.utilities.linear.LinearUtilityAggregationFunction'>]

        >>> mechanism, negotiators, issues = load_genius_domain_from_folder(
        ...                             pkg_resources.resource_filename('negmas', resource_name='tests/data/Laptop')
        ...                             , force_single_issue=True, keep_issue_names=False
        ...                             , keep_value_names=False)
        >>> len(issues), len(negotiators)
        (1, 2)
        >>> [type(a['ufun']) for a in negotiators]
        [<class 'negmas.utilities.nonlinear.MappingUtilityFunction'>, <class 'negmas.utilities.nonlinear.MappingUtilityFunction'>]

    """
    folder_name = str(folder_name)
    files = sorted(listdir(folder_name))
    domain_file_name = None
    utility_file_names = []
    for f in files:
        if not f.endswith(".xml") or f.endswith("pareto.xml"):
            continue
        full_name = folder_name + "/" + f
        root = ET.parse(full_name).getroot()

        if root.tag == "negotiation_template":
            domain_file_name = full_name
        elif root.tag == "utility_space":
            utility_file_names.append(full_name)
    if domain_file_name is None:
        return None, [], []
    return load_genius_domain(
        domain_file_name=domain_file_name,
        utility_file_names=utility_file_names,
        agent_factories=agent_factories,
        force_numeric=force_numeric,
        force_single_issue=force_single_issue,
        cache_and_discretize_outcomes=cache_and_discretize_outcomes,
        max_n_outcomes=max_n_outcomes,
        n_discretization=n_discretization,
        keep_issue_names=keep_issue_names,
        keep_value_names=keep_value_names,
        normalize_utilities=normalize_utilities,
        normalize_max_only=normalize_max_only,
        n_steps=n_steps,
        time_limit=time_limit,
        max_n_agents=max_n_agents,
        dynamic_entry=dynamic_entry,
        safe_parsing=safe_parsing,
        ignore_reserved=ignore_reserved,
        ignore_discount=ignore_discount,
        **kwargs,
    )


def find_domain_and_utility_files(folder_name) -> Tuple[Optional[PATH], List[PATH]]:
    """Finds the domain and utility_function files in a folder
    """
    files = sorted(listdir(folder_name))
    domain_file_name = None
    utility_file_names = []
    folder_name = str(folder_name)
    for f in files:
        if not f.endswith(".xml") or f.endswith("pareto.xml"):
            continue
        full_name = folder_name + "/" + f
        root = ET.parse(full_name).getroot()

        if root.tag == "negotiation_template":
            domain_file_name = full_name
        elif root.tag == "utility_space":
            utility_file_names.append(full_name)
    return domain_file_name, utility_file_names


def convert_genius_domain(
    src_domain_file_name: Optional[PATH],
    dst_domain_file_name: Optional[PATH],
    src_utility_file_names: Sequence[PATH] = tuple(),
    dst_utility_file_names: Sequence[PATH] = tuple(),
    force_single_issue=False,
    cache_and_discretize_outcomes=False,
    max_n_outcomes: int = 1_000_000,
    n_discretization: Optional[int] = None,
    keep_issue_names=True,
    keep_value_names=True,
    normalize_utilities=False,
    normalize_max_only=False,
    safe_parsing=False,
) -> bool:
    assert len(src_utility_file_names) == len(dst_utility_file_names)
    if (
        not force_single_issue
        and not cache_and_discretize_outcomes
        and keep_issue_names
        and keep_value_names
        and not normalize_utilities
    ):
        # no need to do anything, just copy
        if src_domain_file_name and dst_domain_file_name:
            shutil.copy(src=src_domain_file_name, dst=dst_domain_file_name)
        for src, dst in zip(src_utility_file_names, dst_utility_file_names):
            shutil.copy(src=src, dst=dst)
        return True
    issues, issues_details = None, None

    if src_domain_file_name is not None and dst_domain_file_name is not None:
        issues_details, _ = Issue.from_genius(
            src_domain_file_name,
            force_single_issue=False,
            keep_issue_names=True,
            keep_value_names=True,
            safe_parsing=safe_parsing,
            n_discretization=n_discretization,
        )
        if force_single_issue:
            issues, _ = Issue.from_genius(
                src_domain_file_name,
                force_single_issue=force_single_issue,
                keep_issue_names=keep_issue_names,
                keep_value_names=keep_value_names,
                max_n_outcomes=max_n_outcomes,
                n_discretization=n_discretization,
            )
        else:
            issues = issues_details
        if issues is None:
            return False

        Issue.to_genius(
            issues=issues, file_name=dst_domain_file_name, enumerate_integer=True
        )

    if src_utility_file_names is None:
        src_utility_file_names = []

    for ufname, dstfname in zip(src_utility_file_names, dst_utility_file_names):
        utility, discount_factor = UtilityFunction.from_genius(
            file_name=ufname,
            force_single_issue=force_single_issue,
            keep_issue_names=keep_issue_names,
            keep_value_names=keep_value_names,
            normalize_utility=normalize_utilities,
            normalize_max_only=normalize_max_only,
            domain_issues=issues_details,
            safe_parsing=safe_parsing,
            max_n_outcomes=max_n_outcomes,
        )
        if utility is None:
            return False
        UtilityFunction.to_genius(
            u=utility,
            discount_factor=discount_factor,
            issues=issues,
            file_name=dstfname,
        )

    return True


def convert_genius_domain_from_folder(
    src_folder_name: PATH, dst_folder_name: PATH, **kwargs,
) -> bool:
    """Loads a genius domain from a folder. See ``load_genius_domain`` for more details.



    """
    src_folder_name = pathlib.Path(src_folder_name)
    os.makedirs(dst_folder_name, exist_ok=True)
    files = sorted(listdir(src_folder_name))
    domain_file_name = None
    utility_file_names = []
    for f in files:
        if not f.endswith(".xml") or f.endswith("pareto.xml"):
            continue
        full_name = src_folder_name / f
        root = ET.parse(str(full_name)).getroot()

        if root.tag == "negotiation_template":
            domain_file_name = full_name
        elif root.tag == "utility_space":
            utility_file_names.append(full_name)
    success = convert_genius_domain(
        src_domain_file_name=domain_file_name,
        dst_domain_file_name=os.path.join(
            dst_folder_name, os.path.basename(domain_file_name)
        ),
        src_utility_file_names=utility_file_names,
        dst_utility_file_names=[
            os.path.join(dst_folder_name, os.path.basename(_))
            for _ in utility_file_names
        ],
        **kwargs,
    )

    if not success:
        if len(os.listdir(dst_folder_name)) == 0:
            shutil.rmtree(dst_folder_name, ignore_errors=True)

    return success
