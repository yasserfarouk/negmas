from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

import matplotlib
import typer
from matplotlib import functools
from matplotlib import pyplot as plt
from pandas.core.window.numba_ import Callable
from rich import print
from stringcase import titlecase

from negmas.genius.ginfo import get_java_class
from negmas.genius.negotiator import GeniusNegotiator
from negmas.helpers import get_class
from negmas.helpers.strings import camel_case, humanize_time, shortest_unique_names
from negmas.helpers.types import get_full_type_name
from negmas.inout import Scenario
from negmas.mechanisms import Mechanism
from negmas.negotiators.negotiator import Negotiator
from negmas.preferences.ops import (
    calc_reserved_value,
    kalai_points,
    make_rank_ufun,
    max_welfare_points,
    nash_points,
    pareto_frontier,
)

app = typer.Typer()

GENIUSMARKER = "genius"


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


def get_genius_proper_class_name(s: str) -> str:
    assert s.startswith(GENIUSMARKER)
    return s.split(GENIUSMARKER)[-1]


def create_adapter(adapter_type, negotiator_type, name):
    return adapter_type(name=name, base=negotiator_type(name=name))


def make_genius_negotiator(*args, java_class_name: str, **kwargs):
    return GeniusNegotiator(*args, **kwargs, java_class_name=java_class_name)


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
        except:
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
    domain: Path,
    protocol: str = typer.Option(
        "SAO",
        "--protocol",
        "--mechanism",
        "-p",
        "-m",
        help="The protocol (Mechanism to use)",
    ),
    negotiators: list[str] = typer.Option(
        ["aspiration", "aspiration"],
        "--agent",
        "--negotiator",
        "-n",
        "-a",
        help="Negotiator (agent) type. To use an adapter type, put the adapter name first separated from the negotiator name by a slash (e.g. TAUAdapter/AspirationNegotiator)",
    ),
    path: list[Path] = list(),
    reserved: list[float] = typer.Option(None, "--reserved", "-r"),
    fraction: list[float] = typer.Option(None, "--fraction", "-f"),
    normalize: bool = True,
    steps: int = typer.Option(None, "--steps", "-s"),  # type: ignore
    timelimit: int = typer.Option(None, "--time", "--timelimit", "-t"),  # type: ignore
    plot: bool = True,
    fast: bool = None,  # type: ignore
    plot_path: Path = None,  # type: ignore
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    verbosity: int = 0,
    progress: bool = True,
    plot_backend: str = "",
    plot_interactive: bool = True,
    history: bool = False,
    stats: bool = True,
    rank_stats: bool = None,  # type: ignore
    discount: bool = True,
    extend_negotiators: bool = False,
    truncate_ufuns: bool = False,
    only2d: bool = False,
    show_agreement: bool = False,
    show_pareto_distance: bool = True,
    show_nash_distance: bool = True,
    show_kalai_distance: bool = True,
    show_max_welfare_distance: bool = True,
    show_max_relative_welfare_distance: bool = False,
    show_end_reason: bool = True,
    show_annotations: bool = False,
    show_reserved: bool = True,
    show_total_time: bool = True,
    show_relative_time: bool = True,
    show_n_steps: bool = True,
    simple_offers_view: bool = None,  # type: ignore
    raise_exceptions: bool = False,
    extra_params: str = "",
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
    timelimit: int | float
    if steps is None:
        steps = float("inf")
    if timelimit is None:
        timelimit = float("inf")
    scenario = Scenario.from_genius_folder(
        domain, ignore_reserved=False, ignore_discount=not discount
    )
    if not scenario:
        print(f"Failed to load scenario from {domain}")
        return
    if normalize:
        scenario.normalize()

    assert scenario is not None
    scenario.mechanism_type = get_protocol(protocol)
    if verbosity > 0:
        print(f"Scenario: {domain}")
        print(
            f"Mechanism: {shorten_protocol_name(get_full_type_name(scenario.mechanism_type))}"
        )
        print(f"steps: {steps}\ntimelimit: {timelimit}")
    if (
        truncate_ufuns
        and len(scenario.ufuns) > len(negotiators)
        and len(negotiators) > 1
    ):
        scenario.ufuns = scenario.ufuns[: len(negotiators)]

    if (
        extend_negotiators
        and len(negotiators) > 0
        and len(negotiators) != len(scenario.ufuns)
    ):
        if len(negotiators) < len(scenario.ufuns):
            if verbosity > 0:
                print(
                    f"Found {len(scenario.ufuns)} ufuns and {len(negotiators)} negotiators. Will add negotiators of the last type to match the n. ufuns"
                )
            negotiators = negotiators + (
                [negotiators[-1]] * (len(scenario.ufuns) - len(negotiators))
            )

        if len(negotiators) > len(scenario.ufuns):
            if verbosity > 0:
                print(
                    f"Found {len(scenario.ufuns)} ufuns and {len(negotiators)} negotiators. Will ignore the last n. negotiators"
                )
            negotiators = negotiators[: len(scenario.ufuns)]

    negotiator_names = shortest_unique_names(negotiators, guarantee_unique=True)
    agents = [
        get_negotiator(_)(name=name) for _, name in zip(negotiators, negotiator_names, strict=True)  # type: ignore
    ]
    if len(agents) < 2:
        print(
            f"At least 2 negotiators are needed: found {[_.__class__.__name__ for _ in agents]}"
        )
        return
    if reserved and not extend_negotiators:
        assert len(reserved) == len(negotiators), f"{reserved=} but {negotiators=}"
    if reserved:
        for u, r in zip(scenario.ufuns, reserved):
            u.reserved_value = r
    if fraction:
        if len(fraction) < len(negotiators):
            fraction += [1.0] * (len(negotiators) - len(fraction))
        for u, f in zip(scenario.ufuns, fraction):
            u.reserved_value = calc_reserved_value(u, f)
    if (
        not extend_negotiators
        and len(agents) > 0
        and len(agents) != len(scenario.ufuns)
    ):
        print(
            f"You passed {len(agents)} agents for a negotiation with {len(scenario.ufuns)} ufuns. pass --extend-negotiators to adjust the agent number"
        )
        exit(1)
    session = scenario.make_session(
        agents,
        n_steps=steps,
        time_limit=timelimit,
        verbosity=verbosity - 1,
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
    runner = session.run_with_progress if progress else session.run
    _start = perf_counter()
    state = runner()
    duration = perf_counter() - _start
    if verbosity > 1:
        print(f"Time: {humanize_time(duration, show_ms=True, show_us=True)}")
        print(f"Steps: {session.current_step}")
        print(state)
    print(f"Agreement: {state.agreement}")
    print(f"Utilities: {[u(state.agreement) for u in scenario.ufuns]}")
    print(
        f"Advantages: {[u(state.agreement) - (u.reserved_value if u.reserved_value is not None else float('inf')) for u in scenario.ufuns]}"
    )
    fast = fast or fast is None and scenario.outcome_space.cardinality > 10_000
    if fast:
        # rank_stats = stats = False
        if simple_offers_view is None:
            simple_offers_view = True
    rank_stats = (
        rank_stats or rank_stats is None and scenario.outcome_space.cardinality < 10_000
    )

    if stats or rank_stats:
        pareto, pareto_outcomes = session.pareto_frontier()
    else:
        pareto, pareto_outcomes = [], []
    if stats:
        utils = tuple(u(state.agreement) for u in scenario.ufuns)
        print(f"Agreement Utilities: {utils}")
        pts = session.nash_points(frontier=pareto, frontier_outcomes=pareto_outcomes)
        print(
            f"Nash Points: {pts} -- Distance = {dist(utils, list(a for a, _ in pts))}"
        )
        pts = session.kalai_points(frontier=pareto, frontier_outcomes=pareto_outcomes)
        print(
            f"Kalai Points: {pts} -- Distance = {dist(utils, list(a for a, _ in pts))}"
        )
        pts = session.modified_kalai_points(
            frontier=pareto, frontier_outcomes=pareto_outcomes
        )
        print(
            f"Modified Kalai Points: {pts} -- Distance = {dist(utils, list(a for a, _ in pts))}"
        )
        pts = session.max_welfare_points(
            frontier=pareto, frontier_outcomes=pareto_outcomes
        )
        print(
            f"Max. Welfare Points: {pts} -- Diff = {diff(utils, list(a for a, _ in pts))}"
        )
        # pts = session.max_relative_welfare_points(frontier=pareto, frontier_outcomes=pareto_outcomes)
        # print(
        #     f"Max. Relative Points: {pts} -- Distance = {dist(utils, list(a for a, _ in pts))}"
        # )
    if rank_stats:
        ranks_ufuns = tuple(make_rank_ufun(_) for _ in scenario.ufuns)
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
        utils_indices = nash_points(
            frontier=pareto, ufuns=ranks_ufuns, outcome_space=scenario.outcome_space
        )
        pts = tuple((a, pareto_outcomes[b]) for a, b in utils_indices)
        nutils = list(a for a, _ in utils_indices)
        print(f"Ordinal Nash Points: {pts} -- Distance = {dist(ranks, nutils)}")
        pts = kalai_points(
            frontier=pareto,
            ufuns=ranks_ufuns,
            outcome_space=scenario.outcome_space,
            subtract_reserved_value=True,
        )
        pts = tuple((a, pareto_outcomes[b]) for a, b in utils_indices)
        nutils = list(a for a, _ in utils_indices)
        print(f"Ordinal Kalai Points: {pts} -- Distance = {dist(ranks, nutils)}")
        pts = kalai_points(
            frontier=pareto,
            ufuns=ranks_ufuns,
            outcome_space=scenario.outcome_space,
            subtract_reserved_value=False,
        )
        pts = tuple((a, pareto_outcomes[b]) for a, b in utils_indices)
        nutils = list(a for a, _ in utils_indices)
        print(
            f"Ordinal Modified Kalai Points: {pts} -- Distance = {dist(ranks, nutils)}"
        )
        pts = max_welfare_points(
            frontier=pareto, ufuns=ranks_ufuns, outcome_space=scenario.outcome_space
        )
        pts = tuple((a, pareto_outcomes[b]) for a, b in utils_indices)
        nutils = list(a for a, _ in utils_indices)
        print(f"Ordinal Max. Welfare Points: {pts} -- Diff = {diff(ranks, nutils)}")
        # pts = max_relative_welfare_points(
        #     frontier=pareto, ufuns=ranks_ufuns, outcome_space=scenario.outcome_space
        # )
        # pts = tuple((a, pareto_outcomes[b]) for a, b in utils_indices)
        # nutils = list(a for a, _ in utils_indices)
        # print(f"Ordinal Max. Relative Points: {pts} -- Distance = {dist(ranks, nutils)}")
    if history:
        if hasattr(session, "full_trace"):
            print(session.full_trace)  # type: ignore full_trace is defined for SAO and GBM
        else:
            print(session.history)
    if plot_path:
        plot_path = Path(plot_path).absolute()
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
            show_agreement=show_agreement,
            show_pareto_distance=show_pareto_distance,
            show_nash_distance=show_nash_distance,
            show_kalai_distance=show_kalai_distance,
            show_max_welfare_distance=show_max_welfare_distance,
            show_max_relative_welfare_distance=show_max_relative_welfare_distance,
            show_end_reason=show_end_reason,
            show_annotations=show_annotations,
            show_reserved=show_reserved,
            show_total_time=show_total_time,
            show_relative_time=show_relative_time,
            show_n_steps=show_n_steps,
            fast=fast,
            simple_offers_view=simple_offers_view,
        )
        mng = plt.get_current_fig_manager()
        mng.resize(1024, 860)
        mng.full_screen_toggle()
        plt.show()


if __name__ == "__main__":
    app()
