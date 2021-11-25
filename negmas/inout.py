"""
Defines import/export functionality
"""
from __future__ import annotations

import os
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import reduce
from os import PathLike, listdir
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple, Type, Union

from _pytest.mark.structures import normalize_mark_list
from networkx.algorithms import operators

from negmas.helpers.timeout import force_single_thread
from negmas.outcomes.categorical_issue import CategoricalIssue
from negmas.outcomes.contiguous_issue import ContiguousIssue
from negmas.preferences.nonlinear import MappingUtilityFunction

from .helpers import PATH
from .mechanisms import Mechanism
from .negotiators import Negotiator
from .outcomes import Issue, OutcomeSpace, issues_from_genius, issues_to_genius
from .preferences import (
    DiscountedUtilityFunction,
    UtilityFunction,
    make_discounted_ufun,
    normalize,
)
from .sao import SAOMechanism

__all__ = [
    "Domain",
    "load_genius_domain",
    "load_genius_domain_from_folder",
    "convert_genius_domain_from_folder",
    "convert_genius_domain",
    "find_domain_and_utility_files",
    "get_domain_issues",
]


@dataclass
class Domain:
    """
    A class representing a negotiation domain
    """

    agenda: OutcomeSpace
    ufuns: tuple[UtilityFunction, ...]
    mechanism_type: Type[Mechanism] | None
    mechanism_params: dict

    def to_genius_files(self, domain_path: Path, ufun_paths: list[Path]):
        """
        Save domain and ufun files to the `path` as XML.
        """
        domain_path = Path(domain_path)
        ufun_paths = [Path(_) for _ in ufun_paths]
        if len(self.ufuns) != len(ufun_paths):
            raise ValueError(
                f"I have {len(self.ufuns)} ufuns but {len(ufun_paths)} paths were passed!!"
            )
        domain_path.parent.mkdir(parents=True, exist_ok=True)
        self.agenda.to_genius(domain_path)
        for ufun, path in zip(self.ufuns, ufun_paths):
            ufun.to_genius(path, issues=self.issues)
        return self

    def to_genius_folder(self, path: Path):
        """
        Save domain and ufun files to the `path` as XML.
        """
        path.mkdir(parents=True, exist_ok=True)
        domain_name = self.agenda.name.split("/")[-1]
        ufun_names = [_.name.split("/")[-1] for _ in self.ufuns]
        self.agenda.to_genius(path / domain_name)
        for ufun, name in zip(self.ufuns, ufun_names):
            ufun.to_genius(path / name, issues=self.issues)
        return self

    @property
    def n_negotiators(self) -> int:
        return len(self.ufuns)

    @property
    def n_issues(self) -> int:
        return len(self.agenda.issues)

    @property
    def issues(self) -> list[Issue]:
        return self.agenda.issues

    @property
    def issue_names(self) -> list[str]:
        return self.agenda.issue_names

    def to_single_issue(self, numeric=False, stringify=False) -> "Domain":
        """
        Forces the domain to have a single issue with all possible outcomes

        Args:
            numeric: If given, the output issue will be a `ContiguousIssue` otherwise it will be a `CategoricalIssue`
            stringify:  If given, the output issue will have string values. Checked only if `numeric` is `False`

        Remarks:
            - maps the agenda and ufuns to work correctly together
            - Only works if the outcome space is finite
        """
        if not self.agenda.is_finite:
            raise ValueError(
                f"Cannot convert an infinite outcome space to a single issue"
            )
        outcomes = self.agenda.enumerate_discrete()
        values = (
            range(len(outcomes))
            if numeric
            else [str(_) for _ in outcomes]
            if stringify
            else outcomes
        )
        self.agenda.issues = [
            ContiguousIssue(len(outcomes), name="-".join(self.issue_names))
            if numeric
            else CategoricalIssue(values, name="-".join(self.issue_names))
        ]
        if numeric or stringify:
            values = [(_,) for _ in values]
        ufuns = []
        for u in self.ufuns:
            if isinstance(u, DiscountedUtilityFunction):
                usave = u
                v = u.ufun
                while isinstance(v, DiscountedUtilityFunction):
                    u, v = v, v.ufun
                u.ufun = MappingUtilityFunction(
                    mapping=dict(zip(((_,) for _ in values), [v(_) for _ in outcomes])),
                    reserved_value=v.reserved_value,
                    name=v.name,
                )
                ufuns.append(usave)
                continue
            ufuns.append(
                MappingUtilityFunction(
                    mapping=dict(zip(((_,) for _ in values), [u(_) for _ in outcomes])),
                    reserved_value=u.reserved_value,
                    name=u.name,
                )
            )
        self.ufuns = tuple(ufuns)
        return self

    def make_session(
        self,
        negotiators: Callable[[], Negotiator] | list[Negotiator] | None = None,
        n_steps: int | None = None,
        time_limit: float | None = None,
        roles: list[str] | None = None,
        **kwargs,
    ):
        """
        Generates a ready to run mechanism session for this domain.
        """
        if not self.mechanism_type:
            raise ValueError(
                "Cannot create the domain because it has no `mechanism_type`"
            )

        args = self.mechanism_params
        args.update(kwargs)
        m = self.mechanism_type(
            issues=self.agenda.issues, n_steps=n_steps, time_limit=time_limit, **args
        )
        if not negotiators:
            return m
        if isinstance(negotiators, Callable):
            negotiators = [negotiators() for _ in range(self.n_negotiators)]
        if len(self.ufuns) != len(negotiators) or len(negotiators) < 1:
            raise ValueError(
                f"Invalid ufuns ({self.ufuns}) or negotiators ({self.negotiators})"
            )
        if not roles:
            roles = ["agent"] * len(negotiators)
        for n, r, u in zip(negotiators, roles, self.ufuns):
            m.add(n, preferences=u, role=r)
        return m

    def normalize(
        self,
        rng: Tuple[float | None, float | None] = (0.0, 1.0),
        epsilon: float = 1e-6,
        infeasible_cutoff: float = float("-inf"),
    ) -> "Domain":
        """Normalizes a utility function to the given range

        Args:
            ufun: The utility function to normalize
            outcomes: A collection of outcomes to normalize for
            rng: range to normalize to. Default is [0, 1]
            epsilon: A small number specifying the resolution
            infeasible_cutoff: A value under which any utility is considered infeasible and is not used in normalization
        """
        for _ in self.ufuns:
            _ = normalize(
                _,
                outcomes=self.agenda.discrete_outcomes(),
                rng=rng,
                infeasible_cutoff=infeasible_cutoff,
                epsilon=epsilon,
            )

        self.ufuns = tuple(
            normalize(
                _,
                outcomes=self.agenda.discrete_outcomes(),
                rng=rng,
                infeasible_cutoff=infeasible_cutoff,
                epsilon=epsilon,
            )
            for _ in self.ufuns
        )
        return self

    def discretize(self, levels: int = 10):
        """Discretize all issues"""
        self.agenda.discretize(levels, inplace=True)
        return self

    def remove_discounting(self):
        """Removes discounting from all ufuns"""
        ufuns = []
        for u in self.ufuns:
            while isinstance(u, DiscountedUtilityFunction):
                u = u.ufun
            ufuns.append(u)
        self.ufuns = tuple(ufuns)
        return self

    def remove_reserved_values(self, r: float = float("-inf")):
        """Removes reserved values from all ufuns replaacing it with `r`"""
        for u in self.ufuns:
            u.reserved_value = r
        return self

    @staticmethod
    def from_genius_folder(
        path: PathLike | str,
        force_numeric=False,
        ignore_discount=False,
        ignore_reserved=False,
        safe_parsing=True,
    ) -> "Domain" | None:
        return load_genius_domain_from_folder(
            folder_name=str(path),
            force_numeric=force_numeric,
            ignore_discount=ignore_discount,
            ignore_reserved=ignore_reserved,
            safe_parsing=safe_parsing,
        )

    @staticmethod
    def from_genius_files(
        domain: PathLike | str,
        ufuns: List[PathLike | str],
        force_numeric=False,
        ignore_discount=False,
        ignore_reserved=False,
        safe_parsing=True,
    ) -> "Domain" | None:
        return load_genius_domain(
            domain,
            [str(_) for _ in ufuns],
            force_numeric=force_numeric,
            safe_parsing=safe_parsing,
            ignore_discount=ignore_discount,
            ignore_reserved=ignore_reserved,
            normalize_utilities=False,
            normalize_max_only=False,
        )


def get_domain_issues(
    domain_file_name: str,
    max_n_outcomes: int = 1_000_000,
    n_discretization: Optional[int] = None,
    safe_parsing=False,
) -> List[Issue]:
    """
    Returns the issues of a given XML domain (Genius Format)

    Args:
        domain_file_name: Name of the file
        n_discretization: Number of discrete levels per continuous variable.
        max_n_outcomes: Maximum number of outcomes in the outcome space after discretization. Used only if `n_discretization` is given.
        safe_parsing: Apply more checks while parsing

    Returns:
        List of issues

    """
    issues, issues_details = None, None
    if domain_file_name is not None:
        domain_file_name = str(domain_file_name)
        issues, _ = issues_from_genius(
            domain_file_name,
            safe_parsing=safe_parsing,
            n_discretization=n_discretization,
        )
    return issues


def load_genius_domain(
    domain_file_name: PATH,
    utility_file_names: Optional[List[str]] = None,
    force_numeric=False,
    ignore_discount=False,
    ignore_reserved=False,
    safe_parsing=True,
    **kwargs,
) -> Domain:
    """
    Loads a genius domain, creates appropriate negotiators if necessary

    Args:
        domain_file_name: XML file containing Genius-formatted domain spec
        utility_file_names: XML files containing Genius-fromatted ufun spec
        ignore_reserved: Sets the reserved_value of all ufuns to -inf
        ignore_discount: Ignores discounting
        safe_parsing: Applies more stringent checks during parsing
        kwargs: Extra arguments to pass verbatim to SAOMechanism constructor

    Returns:
        A `Domain` ready to run

    """

    issues = None
    if domain_file_name is not None:
        domain_file_name = str(domain_file_name)
        issues, _ = issues_from_genius(
            domain_file_name,
            safe_parsing=safe_parsing,
            force_numeric=force_numeric,
        )

    agent_info = []
    if utility_file_names is None:
        utility_file_names = []
    utility_file_names = [str(_) for _ in utility_file_names]
    for ufname in utility_file_names:
        utility, discount_factor = UtilityFunction.from_genius(
            file_name=ufname,
            issues=issues,
            safe_parsing=safe_parsing,
            ignore_discount=ignore_discount,
            ignore_reserved=ignore_reserved,
            force_numeric=force_numeric,
            name=str(ufname),
        )
        agent_info.append(
            {
                "ufun": utility,
                "ufun_name": ufname,
                "reserved_value_func": utility.reserved_value
                if utility is not None
                else float("-inf"),
                "discount_factor": discount_factor,
            }
        )
    if domain_file_name is not None:
        kwargs["avoid_ultimatum"] = False
        kwargs["dynamic_entry"] = False
        kwargs["max_n_agents"] = None
        if not ignore_discount:
            for info in agent_info:
                info["ufun"] = (
                    info["ufun"]
                    if info["discount_factor"] is None or info["discount_factor"] == 1.0
                    else make_discounted_ufun(
                        ufun=info["ufun"],
                        discount_per_round=info["discount_factor"],
                        power_per_round=1.0,
                    )
                )

    return Domain(
        agenda=OutcomeSpace(issues, name=domain_file_name),
        ufuns=[_["ufun"] for _ in agent_info],
        mechanism_type=SAOMechanism,
        mechanism_params=kwargs,
    )


def load_genius_domain_from_folder(
    folder_name: str,
    force_numeric=False,
    ignore_reserved=False,
    ignore_discount=False,
    safe_parsing=False,
    **kwargs,
) -> Domain | None:
    """
    Loads a genius domain from a folder. See ``load_genius_domain`` for more details.

    Args:
        folder_name: A folder containing one XML domain file and one or more ufun files in Genius format
        ignore_reserved: Sets the reserved_value of all ufuns to -inf
        ignore_discount: Ignores discounting
        safe_parsing: Applies more stringent checks during parsing
        kwargs: Extra arguments to pass verbatim to SAOMechanism constructor

    Returns:
        A domain ready for `make_session`

    Examples:

        >>> import pkg_resources
        >>> from negmas import *

        Try loading and running a domain with predetermined agents:
        >>> domain = load_genius_domain_from_folder(
        ...                             pkg_resources.resource_filename('negmas', resource_name='tests/data/Laptop'))
        >>> mechanism = domain.make_session(AspirationNegotiator, n_steps=100)
        >>> state = mechanism.run()
        >>> state.agreement is not None
        True


        Try loading a domain and check the resulting ufuns
        >>> domain = load_genius_domain_from_folder(
        ...                             pkg_resources.resource_filename('negmas', resource_name='tests/data/Laptop'))

        >>> domain.n_issues, domain.n_negotiators
        (3, 2)

        >>> [type(_) for _ in domain.ufuns]
        [<class 'negmas.preferences.linear.LinearUtilityAggregationFunction'>, <class 'negmas.preferences.linear.LinearUtilityAggregationFunction'>]

        Try loading a domain forcing a single issue space
        >>> domain = load_genius_domain_from_folder(
        ...                             pkg_resources.resource_filename('negmas', resource_name='tests/data/Laptop')
        ...                             ).to_single_issue()
        >>> domain.n_issues, domain.n_negotiators
        (1, 2)
        >>> [type(_) for _ in domain.ufuns]
        [<class 'negmas.preferences.nonlinear.MappingUtilityFunction'>, <class 'negmas.preferences.nonlinear.MappingUtilityFunction'>]

        Try loading a domain with nonlinear ufuns:
        >>> folder_name = pkg_resources.resource_filename('negmas', resource_name='tests/data/10issues')
        >>> domain = load_genius_domain_from_folder(folder_name)
        >>> print(domain.n_issues)
        10
        >>> print(domain.n_negotiators)
        2
        >>> print([type(u) for u in domain.ufuns])
        [<class 'negmas.preferences.nonlinear.HyperRectangleUtilityFunction'>, <class 'negmas.preferences.nonlinear.HyperRectangleUtilityFunction'>]
        >>> u = domain.ufuns[0]
        >>> print(u.outcome_ranges[0])
        {'c1-i9': (7.0, 9.0), 'c1-i7': (2.0, 7.0), 'c1-i5': (0.0, 8.0), 'c1-i2': (0.0, 7.0)}
        >>> print(u.mappings[0])
        97.0
        >>> print(u(dict(zip(domain.issue_names,[0.0] * domain.n_issues))))
        0.0
        >>> print(u(dict(zip(domain.issue_names,[0.5] * domain.n_issues))))
        186.0



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
        return None
    return load_genius_domain(
        domain_file_name=domain_file_name,
        utility_file_names=utility_file_names,
        force_numeric=force_numeric,
        safe_parsing=safe_parsing,
        ignore_reserved=ignore_reserved,
        ignore_discount=ignore_discount,
        **kwargs,
    )


def find_domain_and_utility_files(folder_name) -> Tuple[Optional[PATH], List[PATH]]:
    """Finds the domain and utility_function files in a folder"""
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
    cache_and_discretize_outcomes=False,
    max_n_outcomes: int = 1_000_000,
    n_discretization: Optional[int] = None,
    normalize_utilities=False,
    normalize_max_only=False,
    safe_parsing=False,
    force_single_issue=False,
    keep_issue_names=True,
    keep_value_names=True,
) -> bool:
    if not cache_and_discretize_outcomes and not normalize_utilities:
        # no need to do anything, just copy
        if src_domain_file_name and dst_domain_file_name:
            shutil.copy(src=src_domain_file_name, dst=dst_domain_file_name)
        for src, dst in zip(src_utility_file_names, dst_utility_file_names):
            shutil.copy(src=src, dst=dst)
        return True
    domain = Domain.from_genius_files(
        src_domain_file_name, src_utility_file_names, safe_parsing=safe_parsing
    )
    if normalize_utilities:
        domain.normalize(rng=(None, 1.0) if normalize_max_only else (0.0, 1.0))
    if cache_and_discretize_outcomes:
        if not n_discretization:
            return False
        cardinalities = [
            _.cardinality if _.is_countable() else n_discretization
            for _ in domain.issues
        ]
        n = 1
        for c in cardinalities:
            n *= c
        if n > max_n_outcomes:
            return False
        domain.discretize()
    if force_single_issue:
        domain.to_single_issue(numeric=not keep_value_names)
    try:
        domain.to_genius_files(dst_domain_file_name, dst_utility_file_names)
    except ValueError:
        return False

    return True


def convert_genius_domain_from_folder(
    src_folder_name: PATH,
    dst_folder_name: PATH,
    **kwargs,
) -> bool:
    """
    Loads a genius domain from a folder and saves it to another after transformatin.

    See ``load_genius_domain`` for more details of the transformations
    """
    src_folder_name = Path(src_folder_name)
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
