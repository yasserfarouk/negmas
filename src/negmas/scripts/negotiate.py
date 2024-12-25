from __future__ import annotations

from datetime import datetime
from copy import deepcopy
import pandas as pd
import sys
import random
from typing import Any, Optional
from pathlib import Path
from time import perf_counter
from negmas.inout import serialize

import matplotlib
import typer
import functools
from matplotlib import pyplot as plt
from pandas.core.window.numba_ import Callable
from rich import print
from stringcase import titlecase

from negmas.genius.ginfo import get_java_class
from negmas.genius.negotiator import GeniusNegotiator
from negmas.helpers import get_class, instantiate
from negmas.helpers.inout import dump
from negmas.helpers.strings import camel_case, humanize_time, shortest_unique_names
from negmas.helpers.types import get_full_type_name
from negmas.inout import Scenario
from negmas.mechanisms import Mechanism
from negmas.negotiators.negotiator import Negotiator
from negmas.outcomes.outcome_space import CartesianOutcomeSpace
from negmas.preferences.ops import (
    calc_reserved_value,
    kalai_points,
    make_rank_ufun,
    max_welfare_points,
    nash_points,
    pareto_frontier,
)
from negmas.preferences.generators import generate_multi_issue_ufuns
from negmas.serialization import PYTHON_CLASS_IDENTIFIER

app = typer.Typer()

GENIUSMARKER = "genius"
ANLMARKER = "anl"


def get_screen_resolution() -> tuple[int, int]:
    from tkinter import Tk

    # creating tkinter window
    root = Tk()
    # getting screen's height in pixels
    height = root.winfo_screenheight()
    # getting screen's width in pixels
    width = root.winfo_screenwidth()
    return (width, height)


def get_protocol(name: str) -> type[Mechanism]:
    if name.lower() == "sao":
        return get_class("negmas.sao.mechanism.SAOMechanism")
    if name.lower() == "tau":
        return get_class("negmas.gb.mechanisms.TAUMechanism")
    if name.lower() == "gtau":
        return get_class("negmas.gb.mechanisms.GeneralizedTAUMechanism")
    if name.lower() == "gao":
        return get_class("negmas.gb.mechanisms.GAOMechanism")
    if "." not in name:
        name = f"negmas.{name}"
    return get_class(name)


def get_proper_class_name(s: str) -> str:
    # assert s.startswith(GENIUSMARKER)
    if s.startswith(GENIUSMARKER):
        return s.split(GENIUSMARKER)[-1]
    if s.startswith(ANLMARKER):
        return s.split(ANLMARKER)[-1]
    raise RuntimeError(f"{s} does not start with a known marker")


def create_adapter(adapter_type, negotiator_type, name):
    return adapter_type(name=name, base=negotiator_type(name=name))


def make_genius_negotiator(*args, java_class_name: str, **kwargs):
    return GeniusNegotiator(*args, **kwargs, java_class_name=java_class_name)


def make_anl_negotiator(class_name: str, **kwargs):
    return instantiate(class_name, module_name="anl_agents", **kwargs)


def get_negotiator(class_name: str) -> type[Negotiator] | Callable[[str], Negotiator]:
    if class_name.startswith(GENIUSMARKER):
        for sp in (".", ":"):
            x = sp.join(class_name.split(sp)[1:])
            if x:
                class_name = x
                break
        java_class = get_java_class(class_name)
        if java_class is None:
            raise ValueError(
                f"Cannot find java class name for genius negotiator of type {class_name}"
            )
        return functools.partial(make_genius_negotiator, java_class_name=java_class)

    if class_name.startswith(f"{ANLMARKER}."):
        for sp in (".", ":"):
            x = sp.join(class_name.split(sp)[1:])
            if x:
                class_name = x
                break
        if class_name.startswith(f"{ANLMARKER}."):
            class_name = class_name[len(ANLMARKER) + 1 :]
        return functools.partial(make_anl_negotiator, class_name=class_name)
    if "/" in class_name:
        adapter_name, _, negotiator_name = class_name.partition("/")
        adapter_type = get_adapter(adapter_name)
        negotiator_type = get_negotiator(negotiator_name)
        return functools.partial(create_adapter, adapter_type, negotiator_type)
    if "." not in class_name:
        if "_" in class_name:
            class_name = titlecase(camel_case(class_name))
        try:
            return get_class(f"negmas.{class_name}")
        except Exception:
            if not class_name.endswith("Negotiator"):
                class_name = f"{class_name}Negotiator"
            class_name = f"negmas.{class_name}"
    return get_class(class_name)


def get_adapter(
    name: str, base_name="NegotiatorAdapter"
) -> type[Negotiator] | Callable[[str], Negotiator]:
    if "." not in name:
        if "_" in name:
            name = titlecase(camel_case(name))
        if not name.endswith(base_name):
            name = f"{name}{base_name}"
        name = f"negmas.gb.adapters.tau.{name}"
    return get_class(name)


def shorten_protocol_name(name: str) -> str:
    name = name.split(".")[-1]
    return name.replace("Mechanism", "").replace("Protocol", "")


def dist(x: tuple[float, ...], lst: list[tuple[float, ...]]):
    if not lst:
        return float("nan")
    return min(sum((a - b) ** 2 for a, b in zip(x, p, strict=True)) for p in lst)


def diff(x: tuple[float, ...], lst: list[tuple[float, ...]]):
    if not lst:
        return float("nan")
    s = sum(x)
    return min(abs(sum(_) - s) for _ in lst)


@app.command()
def run(
    scenario: Optional[Path] = typer.Option(
        default=None,
        show_default="Generate A new Scenario",
        help="The scenario to negotiate about",
    ),
    protocol: str = typer.Option(
        "SAO",
        "--protocol",
        "--mechanism",
        "-p",
        "-m",
        help="The protocol (Mechanism to use)",
        rich_help_panel="Basic Options",
    ),
    negotiators: list[str] = typer.Option(
        ["AspirationNegotiator", "NaiveTitForTatNegotiator"],
        "--agent",
        "--negotiator",
        "-n",
        "-a",
        help="Negotiator (agent) type. To use an adapter type, put the adapter name first separated from the negotiator name by a slash (e.g. TAUAdapter/AspirationNegotiator)",
        rich_help_panel="Basic Options",
    ),
    extend_negotiators: bool = typer.Option(
        False,
        "--extend-negotiators",
        "-E",
        help="Extend the negotiator list to cover all ufuns",
        rich_help_panel="Basic Options",
    ),
    truncate_ufuns: bool = typer.Option(
        False,
        "--truncate-ufuns",
        "-T",
        help="Use the first n. negotiator ufuns only",
        rich_help_panel="Basic Options",
    ),
    extra_params: str = typer.Option(
        "",
        "--params",
        help="Mechanism initialization parameters as comma-separated `key=value` pairs.",
        rich_help_panel="Basic Options",
    ),
    share_ufuns: bool = typer.Option(
        False,
        help="Share partner ufuns using private-data.",
        rich_help_panel="Basic Options",
    ),
    share_reserved_values: bool = typer.Option(
        False,
        help="Share partner reserved-values using private-data.",
        rich_help_panel="Basic Options",
    ),
    # Deadline
    steps: Optional[int] = typer.Option(  # type: ignore
        None,
        "--steps",
        "-s",
        help="Number of Steps allowed in the negotiation",
        rich_help_panel="Deadline",
    ),
    timelimit: Optional[float] = typer.Option(
        None,
        "--time",
        "--timelimit",
        "-t",
        help="Number of Seconds allowed in the negotiation",
        rich_help_panel="Deadline",
    ),
    # Given Scenario
    reserved: list[float] = typer.Option(
        None,
        "--reserved",
        "-r",
        help="Reserved values to override the ones in the scenario. Must be the same length as the ufuns.",
        rich_help_panel="Scenario Overrides",
    ),
    fraction: list[float] = typer.Option(
        None,
        "--fraction",
        "-f",
        help="Rational factions to use for generating reserved values to override the ones in the scenario. Must be the same length as the ufuns.",
        rich_help_panel="Scenario Overrides",
    ),
    discount: bool = typer.Option(
        True,
        "--discount/--no-discount",
        "-d/-D",
        help="Load Discount Factor",
        rich_help_panel="Scenario Overrides",
    ),
    normalize: bool = typer.Option(
        True,
        "--normalize",
        "/-N",
        help="Normalize ufuns to the range (0-1)",
        rich_help_panel="Scenario Overrides",
    ),
    # used in case no domain is given only
    issues: Optional[int] = typer.Option(
        None, "--issues", "-i", help="N. Issues", rich_help_panel="Generated Scenario"
    ),
    values_min: int = typer.Option(
        2,
        help="Minimum allowed n. values per issue",
        rich_help_panel="Generated Scenario",
    ),
    values_max: int = typer.Option(
        50,
        help="Maximum allowed n. values per issue",
        rich_help_panel="Generated Scenario",
    ),
    size: Optional[list[int]] = typer.Option(
        None,
        "--size",
        "-z",
        help="Sizes of issues in order (overrides values-min, values-max)",
        rich_help_panel="Generated Scenario",
    ),
    reserved_values_min: float = typer.Option(
        0.0, help="Min Allowed Reserved value", rich_help_panel="Generated Scenario"
    ),
    reserved_values_max: float = typer.Option(
        1.0, help="Max allowed reserved value", rich_help_panel="Generated Scenario"
    ),
    rational: bool = typer.Option(
        True,
        "--rational/--irrational-ok",
        "-R/-I",
        help="Gurantee Some Rational Outcomes",
        rich_help_panel="Generated Scenario",
    ),
    rational_fraction: Optional[list[float]] = typer.Option(
        None,
        "--rational-fraction",
        "-F",
        help="Reservation fractions",
        rich_help_panel="Generated Scenario",
    ),
    reservation_selector: str = typer.Option(
        "min",
        help="Reservation value selector if both reserved-values and rational-fraction are given: min|max|first|last",
        rich_help_panel="Generated Scenario",
    ),
    python_class_identifier: str = typer.Option(
        PYTHON_CLASS_IDENTIFIER,
        help="Python class identifier in the saved files",
        rich_help_panel="Output control",
    ),
    issue_name: Optional[list[str]] = typer.Option(
        None, help="Issue Names", rich_help_panel="Generated Scenario"
    ),
    os_name: Optional[str] = typer.Option(
        None, help="Outcome Space Name", rich_help_panel="Generated Scenario"
    ),
    ufun_names: Optional[list[str]] = typer.Option(
        None, help="Names of Ufuns", rich_help_panel="Generated Scenario"
    ),
    numeric: bool = typer.Option(
        False, help="Numeric Issues", rich_help_panel="Generated Scenario"
    ),
    linear: bool = typer.Option(
        True,
        "--linear/--non-linear",
        help="Linear Ufuns",
        rich_help_panel="Generated Scenario",
    ),
    pareto_generator: Optional[list[str]] = typer.Option(
        None,
        help="One or more Pareto Generator methods. See negmas.preferences.generator for possible values",
        rich_help_panel="Generated Scenario",
    ),
    # Output Control
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Make verbose", rich_help_panel="Output Control"
    ),
    verbosity: int = typer.Option(
        0,
        help="Verbosity level (higher=more verbose)",
        rich_help_panel="Output Control",
    ),
    progress: bool = typer.Option(
        True, help="Show Progress Bar", rich_help_panel="Output Control"
    ),
    history: bool = typer.Option(
        False, help="Print History", rich_help_panel="Output Control"
    ),
    stats: bool = typer.Option(
        True, help="Generate Statistics", rich_help_panel="Output Control"
    ),
    rank_stats: Optional[bool] = typer.Option(
        None, help="Generate Rank Statistics", rich_help_panel="Output Control"
    ),
    compact_stats: bool = typer.Option(
        True,
        "--compact-stats/--detailed-stats",
        "-c/-C",
        help="Show distances",
        rich_help_panel="Output Control",
    ),
    # Plotting
    plot: bool = typer.Option(True, help="Generate Plot", rich_help_panel="Plotting"),
    only2d: bool = typer.Option(
        False,
        "--only2d/--with-offers",
        "-2/-0",
        help="Only 2D Plot",
        rich_help_panel="Plotting",
    ),
    plot_backend: str = typer.Option(
        "",
        help="Backend used for plotting. See matplotlib backends.",
        rich_help_panel="Plotting",
    ),
    plot_interactive: bool = typer.Option(
        True, help="Make the plot interactive", rich_help_panel="Plotting"
    ),
    plot_show: bool = typer.Option(
        True, help="Show the plot", rich_help_panel="Plotting"
    ),
    simple_offers_view: Optional[bool] = typer.Option(
        None, help="Simple Offers View", rich_help_panel="Plotting"
    ),
    annotations: bool = typer.Option(
        False, help="Show Annotations", rich_help_panel="Plotting"
    ),
    agreement: bool = typer.Option(
        False, help="Show Agreement", rich_help_panel="Plotting"
    ),
    pareto_dist: bool = typer.Option(
        True, help="Show Pareto Distance", rich_help_panel="Plotting"
    ),
    nash_dist: bool = typer.Option(
        True, help="Show Nash Distance", rich_help_panel="Plotting"
    ),
    kalai_dist: bool = typer.Option(
        True, help="Show Kalai Distance", rich_help_panel="Plotting"
    ),
    max_welfare_dist: bool = typer.Option(
        True, help="Show Max Welfare Distance", rich_help_panel="Plotting"
    ),
    max_rel_welfare_dist: bool = typer.Option(
        False, help="Show Max Relative Welfare Distance", rich_help_panel="Plotting"
    ),
    end_reason: bool = typer.Option(
        True, help="Show End Reason", rich_help_panel="Plotting"
    ),
    show_reserved: bool = typer.Option(
        True, help="Show Reserved Value Lines", rich_help_panel="Plotting"
    ),
    total_time: bool = typer.Option(
        True, help="Show Time Limit", rich_help_panel="Plotting"
    ),
    relative_time: bool = typer.Option(
        True, help="Show Relative Time", rich_help_panel="Plotting"
    ),
    show_n_steps: bool = typer.Option(
        True, help="Show N. Steps", rich_help_panel="Plotting"
    ),
    # Saving to Disk
    save_path: Optional[Path] = typer.Option(
        None,
        help="Path to save results to",
        show_default="Do not Save",  # type: ignore
        rich_help_panel="Saving to Disk",
    ),
    save_history: bool = typer.Option(
        True, help="Save Negotiation Histroy", rich_help_panel="Saving to Disk"
    ),
    save_stats: bool = typer.Option(
        True, help="Save Statistics", rich_help_panel="Saving to Disk"
    ),
    save_type: str = typer.Option(
        "yml", help="Scenario Format:yml|xml", rich_help_panel="Saving to Disk"
    ),
    save_compact: bool = typer.Option(
        True, help="Compact file", rich_help_panel="Saving to Disk"
    ),
    plot_path: Optional[Path] = typer.Option(
        None, help="Path to save the plot to.", rich_help_panel="Plotting"
    ),
    # Advanced
    fast: Optional[bool] = typer.Option(
        None, help="Avoid slow operations", rich_help_panel="Advanced"
    ),
    path: list[Path] = typer.Option(
        list(),
        help="One or more extra paths to look for negotiator and mechanism classes.",
        rich_help_panel="Advanced",
    ),
    raise_exceptions: bool = typer.Option(
        False, help="Raise Exceptions on Failure", rich_help_panel="Advanced"
    ),
):
    kwargs = dict()
    if extra_params:
        extra_params = "dict(" + extra_params + ")"
        kwargs = eval(extra_params)

    if verbose and verbosity < 1:
        verbosity = 1
    for p in path:
        sys.path.append(str(p))
    adapter_names = shortest_unique_names(
        [_.split("/")[0] if "/" in _ else "" for _ in negotiators]
    )
    steps: int | float
    # timelimit: int | float
    if steps is None:
        steps = float("inf")
    if timelimit is None:
        timelimit = float("inf")
    if scenario is None:
        n_ufuns = len(negotiators)
        if not n_ufuns:
            print("[red]You must either specify a domain or negotiators[/red]")
            exit(1)
        if not issues:
            issues = random.randint(1, 3)
        gparams: dict[str, Any] = dict(
            n_issues=issues,
            n_values=(values_min, values_max) if not size else None,
            sizes=size,
            n_ufuns=n_ufuns,
            reserved_values=(reserved_values_min, reserved_values_max),
            rational_fractions=tuple(rational_fraction) if rational_fraction else None,
            issue_names=tuple(issue_name) if issue_name else None,
            os_name=os_name,
            numeric=numeric,
            linear=linear,
            ufun_names=tuple(ufun_names) if ufun_names else None,
            reservation_selector=dict(
                min=min, max=max, first=lambda a, _: a, last=lambda _, b: b
            )[reservation_selector],
            guarantee_rational=rational,
        )
        if pareto_generator:
            gparams["pareto_generators"] = pareto_generator
        ufuns = generate_multi_issue_ufuns(**gparams)
        os = ufuns[0].outcome_space
        assert isinstance(os, CartesianOutcomeSpace)
        if verbosity > 0:
            print(
                f"[purple]Generated a domain with {len(os.issues)} issues and {os.cardinality} outcomes[/purple]"
            )
        current_scenario = Scenario(os, ufuns, mechanism_type=get_protocol(protocol))
    else:
        current_scenario = Scenario.from_genius_folder(
            scenario, ignore_reserved=False, ignore_discount=not discount
        )
    assert current_scenario is not None
    saved_scenario: Scenario = deepcopy(current_scenario)
    if not current_scenario:
        print(f"Failed to load scenario from {scenario}")
        return
    if normalize:
        current_scenario.normalize()

    assert current_scenario is not None
    current_scenario.mechanism_type = get_protocol(protocol)
    if verbosity > 0:
        print(f"Scenario: {scenario}")
        print(
            f"Mechanism: {shorten_protocol_name(get_full_type_name(current_scenario.mechanism_type))}"
        )
        print(f"steps: {steps}\ntimelimit: {timelimit}")
    if (
        truncate_ufuns
        and len(current_scenario.ufuns) > len(negotiators)
        and len(negotiators) > 1
    ):
        current_scenario.ufuns = current_scenario.ufuns[: len(negotiators)]

    if (
        extend_negotiators
        and len(negotiators) > 0
        and len(negotiators) != len(current_scenario.ufuns)
    ):
        if len(negotiators) < len(current_scenario.ufuns):
            if verbosity > 0:
                print(
                    f"Found {len(current_scenario.ufuns)} ufuns and {len(negotiators)} negotiators. Will add negotiators of the last type to match the n. ufuns"
                )
            negotiators = negotiators + (
                [negotiators[-1]] * (len(current_scenario.ufuns) - len(negotiators))
            )

        if len(negotiators) > len(current_scenario.ufuns):
            if verbosity > 0:
                print(
                    f"Found {len(current_scenario.ufuns)} ufuns and {len(negotiators)} negotiators. Will ignore the last n. negotiators"
                )
            negotiators = negotiators[: len(current_scenario.ufuns)]

    negotiator_names = shortest_unique_names(negotiators, guarantee_unique=True)
    if share_ufuns:
        assert (
            len(current_scenario.ufuns) == 2 and len(negotiators) == 2
        ), "Sharing ufuns in multilateral negotiations is not yet supported"
        opp_ufuns = list(reversed(deepcopy(current_scenario.ufuns)))
        if not share_reserved_values:
            for u in opp_ufuns:
                u.reserved_value = float("nan")
    else:
        opp_ufuns = [None] * len(negotiators)
    agents = [
        get_negotiator(_)(name=name)  # type: ignore
        if ou is None
        else get_negotiator(_)(name=name, private_info=dict(opponent_ufun=ou))  # type: ignore
        for _, name, ou in zip(negotiators, negotiator_names, opp_ufuns, strict=True)  # type: ignore
    ]
    if len(agents) < 2:
        print(
            f"At least 2 negotiators are needed: found {[_.__class__.__name__ for _ in agents]}"
        )
        return
    if reserved and not extend_negotiators:
        assert len(reserved) == len(negotiators), f"{reserved=} but {negotiators=}"
    if reserved:
        for u, r in zip(current_scenario.ufuns, reserved):
            u.reserved_value = r
    if fraction:
        if len(fraction) < len(negotiators):
            fraction += [1.0] * (len(negotiators) - len(fraction))
        for u, f in zip(current_scenario.ufuns, fraction):
            u.reserved_value = calc_reserved_value(u, f)
    if (
        not extend_negotiators
        and len(agents) > 0
        and len(agents) != len(current_scenario.ufuns)
    ):
        print(
            f"You passed {len(agents)} agents for a negotiation with {len(current_scenario.ufuns)} ufuns. pass --extend-negotiators to adjust the agent number"
        )
        exit(1)
    if save_path:
        current_scenario.dumpas(save_path, save_type, save_compact)

    session = current_scenario.make_session(
        agents,
        n_steps=steps,
        time_limit=timelimit,
        verbosity=verbosity - 1,
        share_ufuns=share_ufuns,
        share_reserved_values=share_reserved_values,
        **dict(ignore_negotiator_exceptions=not raise_exceptions),
        **kwargs,
    )
    if len(session.negotiators) < 2:
        print(
            f"At least 2 negotiators are needed: Only the following could join {[_.__class__.__name__ for _ in session.negotiators]}"
        )
        return
    if verbosity > 0:
        print(f"Adapters: {', '.join(adapter_names)}")
    if verbosity > 1:
        print(f"Negotiators: {', '.join(negotiator_names)}")
    results = dict()
    runner = session.run_with_progress if progress else session.run
    _start = perf_counter()
    state = runner()
    duration = perf_counter() - _start
    current_scenario = saved_scenario
    if verbosity > 1:
        print(f"Time: {humanize_time(duration, show_ms=True, show_us=True)}")
        print(f"Steps: {session.current_step}")
        print(state)
    print(f"Agreement: {state.agreement}")
    advantages = [
        u(state.agreement)
        - (u.reserved_value if u.reserved_value is not None else float("inf"))
        for u in current_scenario.ufuns
    ]
    utilities_final = [u(state.agreement) for u in current_scenario.ufuns]
    print(f"Utilities: {utilities_final}")
    print(f"Advantages: {advantages}")
    fast = fast or fast is None and current_scenario.outcome_space.cardinality > 10_000
    if fast:
        if simple_offers_view is None:
            simple_offers_view = True
    stats = (
        stats or stats is None and current_scenario.outcome_space.cardinality <= 10_000
    )
    rank_stats = (
        rank_stats
        or rank_stats is None
        and current_scenario.outcome_space.cardinality <= 1000
    )

    if stats or rank_stats:
        pareto, pareto_outcomes = session.pareto_frontier()
    else:
        pareto, pareto_outcomes = tuple(), list()
    results["negotiators"] = [get_full_type_name(type(x)) for x in session.negotiators]
    results["agreement"] = state.agreement
    results["utilities"] = utilities_final
    results["advantages"] = advantages
    results["negotiator_names"] = [x.name for x in session.negotiators]
    results["negotiator_ids"] = [x.id for x in session.negotiators]
    results["final_state"] = serialize(
        session.state, python_class_identifier=python_class_identifier
    )
    results["step"] = session.current_step
    results["relative_time"] = session.relative_time
    results["n_steps"] = session.n_steps
    results["time_limit"] = session.time_limit
    results["negotiator_times"] = session.negotiator_times
    results["time"] = str(datetime.now())
    stats_dict = dict()
    if not compact_stats:
        stats_dict["pareto"] = pareto
        stats_dict["pareto_outcomes"] = pareto_outcomes

    if stats:
        utils = tuple(u(state.agreement) for u in current_scenario.ufuns)

        def find_stats(name, f, utils=utils):
            pts = f(frontier=pareto, frontier_outcomes=pareto_outcomes)
            val = dist(utils, list(a for a, _ in pts))
            if not compact_stats:
                stats_dict[f"{name} Points"] = pts
            stats_dict[f"{name} Distance"] = val
            if verbosity > 0:
                print(f"{name} Points: {pts}")
            print(f"{name} Distance: {val}")

        for name, f in (
            ("Nash", session.nash_points),
            ("Kalai", session.kalai_points),
            ("Modified Kalai", session.modified_kalai_points),
            ("Max Welfare", session.max_welfare_points),
            # ("Max Relative Welfare", session.max_relative_welfare_points),
        ):
            find_stats(name, f)
    if rank_stats:
        ranks_ufuns = tuple(make_rank_ufun(_) for _ in current_scenario.ufuns)
        ranks = tuple(_(state.agreement) for _ in ranks_ufuns)
        print(f"Agreement Relative Ranks: {ranks}")
        pareto_save = pareto_outcomes
        alloutcomes = session.discrete_outcomes()
        pareto, pareto_indices = pareto_frontier(
            ranks_ufuns, outcomes=alloutcomes, sort_by_welfare=True
        )
        pareto_outcomes = [alloutcomes[_] for _ in pareto_indices]
        if len(pareto_save) != len(pareto_outcomes) or any(
            a != b
            for a, b in zip(sorted(pareto_save), sorted(pareto_outcomes), strict=True)
        ):
            print(
                f"[bold red]Ordinal pareto outcomes do not match cardinal pareto outcomes[/bold red]\nOrdinal: {pareto_outcomes}\nCardinal: {pareto_save}"
            )

        def find_rank_stats(name, f, ranks=ranks, **kwargs):
            utils_indices = f(
                frontier=pareto,
                ufuns=ranks_ufuns,
                outcome_space=current_scenario.outcome_space,
                **kwargs,
            )
            pts = tuple((a, pareto_outcomes[b]) for a, b in utils_indices)
            nutils = list(a for a, _ in utils_indices)
            val = dist(ranks, nutils)
            if not compact_stats:
                stats_dict[f"{name} Points"] = pts
            stats_dict[f"{name} Distance"] = val
            if verbosity > 0:
                print(f"{name} Points: {pts}")
            print(f"{name} Distance: {val}")

        for name, f, kwargs in (
            ("Ordinal Nash", nash_points, dict()),
            ("Ordinal Kalai", kalai_points, dict(subtract_reserved_value=True)),
            (
                "Ordinal Modified Kalai",
                kalai_points,
                dict(subtract_reserved_value=False),
            ),
            ("Ordinal Max Welfare", max_welfare_points, dict()),
            # ("Ordinal Max Relative Welfare", max_relative_welfare_points),
        ):
            find_rank_stats(name, f, **kwargs)
    if save_path:
        save_path.mkdir(exist_ok=True, parents=True)
        dump(results, save_path / "session.json")
    if save_path and save_stats and (stats or rank_stats):
        save_path.mkdir(exist_ok=True, parents=True)
        dump(stats_dict, save_path / "stats.json")

    if history:
        if hasattr(session, "full_trace"):
            hist = session.full_trace  # type: ignore full_trace is defined for SAO and GBM
        else:
            hist = session.history
        print(hist)
    if plot_path:
        plot_path = Path(plot_path).absolute()
        plot_path.parent.mkdir(parents=True, exist_ok=True)
    elif save_path:
        plot_path = save_path / "session.png"
        plot_path.parent.mkdir(parents=True, exist_ok=True)
    if plot:
        if plot_backend:
            matplotlib.use(plot_backend)
            matplotlib.interactive(plot_interactive)
        session.plot(
            save_fig=plot_path is not None,
            path=plot_path.parent if plot_path else None,
            fig_name=plot_path.name if plot_path else None,
            only2d=only2d,
            show_agreement=agreement,
            show_pareto_distance=pareto_dist,
            show_nash_distance=nash_dist,
            show_kalai_distance=kalai_dist,
            show_max_welfare_distance=max_welfare_dist,
            show_max_relative_welfare_distance=max_rel_welfare_dist,
            show_end_reason=end_reason,
            show_annotations=annotations,
            show_reserved=show_reserved,
            show_total_time=total_time,
            show_relative_time=relative_time,
            show_n_steps=show_n_steps,
            fast=fast,
            simple_offers_view=simple_offers_view,
        )
        if plot_show:
            mng = plt.get_current_fig_manager()
            mng.resize(1024, 860)  # type: ignore
            mng.full_screen_toggle()  # type: ignore
            plt.show()
    if save_path and save_history:
        if hasattr(session, "full_trace"):
            hist = pd.DataFrame(
                session.full_trace,  # type: ignore
                columns=[  # type: ignore
                    "time",
                    "relative_time",
                    "step",
                    "negotiator",
                    "offer",
                    "responses",
                    "state",
                ],
            )
        else:
            hist = pd.DataFrame.from_records(
                [
                    serialize(_, python_class_identifier=python_class_identifier)
                    for _ in session.history
                ]
            )
        dump(hist, save_path / "history.csv", compact=True, sort_keys=False)


if __name__ == "__main__":
    app()
