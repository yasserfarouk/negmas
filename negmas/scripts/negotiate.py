from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Optional

import matplotlib
import typer
from matplotlib import pyplot as plt
from rich import print
from stringcase import titlecase

from negmas.helpers import get_class
from negmas.helpers.strings import camel_case, humanize_time, shortest_unique_names
from negmas.helpers.types import get_full_type_name
from negmas.inout import Scenario
from negmas.mechanisms import Mechanism
from negmas.negotiators.negotiator import Negotiator

app = typer.Typer()


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
    if name.lower() == "gao":
        return get_class("negmas.gb.mechanisms.GAOMechanism")
    if "." not in name:
        name = f"negmas.{name}"
    return get_class(name)


def get_negotiator(name: str) -> type[Negotiator]:
    if "." not in name:
        if "_" in name:
            name = titlecase(camel_case(name))
        if not name.endswith("Negotiator"):
            name = f"{name}Negotiator"
        name = f"negmas.{name}"
    return get_class(name)


def shorten_protocol_name(name: str) -> str:
    name = name.split(".")[-1]
    return name.replace("Mechanism", "").replace("Protocol", "")


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
        help="Negotiator (agent) type",
    ),
    reserved: list[float] = typer.Option(None, "--reserved", "-r"),
    normalize: bool = True,
    steps: int = typer.Option(None, "--steps", "-s"),  # type: ignore
    timelimit: int = typer.Option(None, "--time", "--timelimit", "-t"),  # type: ignore
    plot: bool = True,
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    progress: bool = False,
    plot_backend: str = "",
    plot_interactive: bool = True,
    history: bool = False,
    stats: bool = True,
):
    if reserved is not None:
        assert len(reserved) == len(negotiators)
    negotiator_names = shortest_unique_names(negotiators)
    steps: int | float
    timelimit: int | float
    if steps is None:
        steps = float("inf")
    if timelimit is None:
        timelimit = float("inf")
    scenario = Scenario.from_genius_folder(domain, ignore_reserved=reserved is not None)
    if not scenario:
        print(f"Failed to load scenario from {domain}")
        return
    if normalize:
        scenario.normalize()
    if reserved is not None:
        for u, r in zip(scenario.ufuns, reserved):
            u.reserved_value = r

    assert scenario is not None
    scenario.mechanism_type = get_protocol(protocol)

    agents = [
        get_negotiator(_)(name=name) for _, name in zip(negotiators, negotiator_names)
    ]
    session = scenario.make_session(agents, n_steps=steps, time_limit=timelimit)
    if verbose:
        print(f"Scenario: {domain}")
        print(
            f"Mechanism: {shorten_protocol_name(get_full_type_name(scenario.mechanism_type))}"
        )
        print(f"Negotiators: {', '.join(negotiator_names)}")
        print(f"steps: {steps}\ntimelimit: {timelimit}")
    runner = session.run_with_progress if progress else session.run
    _start = perf_counter()
    state = runner()
    duration = perf_counter() - _start
    if verbose:
        print(f"Time: {humanize_time(duration, show_ms=True, show_us=True)}")
        print(f"Steps: {session.current_step}")
        print(state)
    print(f"Agreement: {state.agreement}")
    if stats:
        pareto, pareto_outcomes = session.pareto_frontier()
        print(
            f"Nash Points: {session.nash_points(frontier=pareto, frontier_outcomes=pareto_outcomes)}"
        )
        print(
            f"Kalai Points: {session.kalai_points(frontier=pareto, frontier_outcomes=pareto_outcomes)}"
        )
        print(
            f"Max. Welfare Points: {session.max_welfare_points(frontier=pareto, frontier_outcomes=pareto_outcomes)}"
        )
        print(
            f"Max. Relative Points: {session.max_relative_welfare_points(frontier=pareto, frontier_outcomes=pareto_outcomes)}"
        )
    if history:
        if hasattr(session, "full_trace"):
            print(session.full_trace)  # type: ignore full_trace is defined for SAO and GBM
        else:
            print(session.history)
    if plot:
        if plot_backend:
            matplotlib.use(plot_backend)
            matplotlib.interactive(plot_interactive)
        session.plot()
        mng = plt.get_current_fig_manager()
        mng.resize(1024, 860)
        mng.full_screen_toggle()
        plt.show()


if __name__ == "__main__":
    app()
