"""
Core logic for negotiate CLI commands.

This module contains shared functionality used by both:
- negotiate.py (standalone Typer-based CLI)
- app.py (Click-based negmas negotiate subcommand)

The goal is to eliminate code duplication while maintaining backward compatibility.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from rich import print
from tabulate import tabulate

if TYPE_CHECKING:
    from negmas.registry import NegotiatorInfo, ScenarioInfo


def show_negotiator_match_table(
    spec: str, matches: list[tuple[str, NegotiatorInfo]], selected_index: int
) -> None:
    """
    Display a table of matching negotiators with a visual indicator of which one was selected.

    Args:
        spec: The original negotiator specification string
        matches: List of (key, NegotiatorInfo) tuples for matching negotiators
        selected_index: Index in matches list that was selected (will show → marker)
    """
    print(
        f"\n[yellow]Multiple negotiators match '{spec}':[/yellow] (showing {len(matches)})"
    )

    table_data = []
    for i, (key, info) in enumerate(matches):
        marker = "→" if i == selected_index else " "
        row = [
            marker,
            info.short_name,
            info.source,
            info.full_type_name.split(".")[-1]
            if "." in info.full_type_name
            else info.full_type_name,
            key[:16] + "..." if len(key) > 19 else key,
        ]
        table_data.append(row)

    print(
        tabulate(
            table_data,
            headers=[" ", "Short Name", "Source", "Class", "Key"],
            tablefmt="simple",
        )
    )
    print()


def show_scenario_match_table(
    spec: str, matches: list[tuple[str, ScenarioInfo]], selected_index: int
) -> None:
    """
    Display a table of matching scenarios with a visual indicator of which one was selected.

    Args:
        spec: The original scenario specification string
        matches: List of (name, ScenarioInfo) tuples for matching scenarios
        selected_index: Index in matches list that was selected (will show → marker)
    """
    print(
        f"\n[yellow]Multiple scenarios match '{spec}':[/yellow] (showing {len(matches)})"
    )

    table_data = []
    for i, (name, info) in enumerate(matches):
        marker = "→" if i == selected_index else " "
        tags_str = ", ".join(sorted(info.tags)) if info.tags else ""
        row = [
            marker,
            name,
            info.source,
            tags_str[:30] + "..." if len(tags_str) > 33 else tags_str,
            str(info.n_negotiators) if hasattr(info, "n_negotiators") else "?",
        ]
        table_data.append(row)

    print(
        tabulate(
            table_data, headers=[" ", "Name", "Source", "Tags", "N"], tablefmt="simple"
        )
    )
    print()


def resolve_negotiator_from_registry(
    spec: str, verbose: bool = False, on_multiple_matches: str = "warn"
) -> str | None:
    """
    Resolve a negotiator specification to a full type name using the registry.

    Supports:
    - Short names: "AspirationNegotiator"
    - Source-qualified: "negmas@AspirationNegotiator" or "negmas>AspirationNegotiator"
    - Full paths: "negmas.sao.AspirationNegotiator" (returned as-is)

    Priority when multiple matches found:
    1. source='negmas' (native NegMAS implementations)
    2. Non-genius sources (native implementations)
    3. Any source

    Args:
        spec: Negotiator specification string
        verbose: If True, print resolution information
        on_multiple_matches: What to do when multiple matches found:
            - "fail": Show table and return None (caller should exit)
            - "warn": Show table and use prioritized match
            - "silent": Use prioritized match without output

    Returns:
        Full type name (e.g., "negmas.sao.AspirationNegotiator") or None if not found
    """
    from negmas import negotiator_registry

    # If spec contains separator, parse source and name
    source_filter = None
    search_name = spec

    # Handle '@' separator (preferred)
    if "@" in spec:
        parts = spec.split("@", 1)
        if len(parts) == 2:
            source_filter, search_name = parts

    # Handle '>' separator (deprecated but supported)
    elif ">" in spec:
        parts = spec.split(">", 1)
        if len(parts) == 2:
            source_filter, search_name = parts
            if verbose:
                print(
                    f"[yellow]Note: '>' separator is deprecated. Use '@' instead: {source_filter}@{search_name}[/yellow]"
                )

    # If spec already looks like a full path, return it as-is
    if "." in search_name and not source_filter:
        return spec

    # Query registry
    if source_filter:
        results = negotiator_registry.query(
            source=source_filter, short_name=search_name
        )
    else:
        results = negotiator_registry.query(short_name=search_name)

    if not results:
        return None

    matches = list(results.items())

    # Single match - return it
    if len(matches) == 1:
        _, info = matches[0]
        return info.full_type_name

    # Multiple matches - apply priority and selection logic
    def priority_key(item):
        """Sort key: prioritize negmas source, then non-genius sources."""
        _, info = item
        if info.source == "negmas":
            return 0
        elif info.source != "genius":
            return 1
        else:
            return 2

    sorted_matches = sorted(matches, key=priority_key)
    selected_index = 0
    _, selected_info = sorted_matches[selected_index]

    # Handle multiple matches based on mode
    if on_multiple_matches == "fail":
        show_negotiator_match_table(spec, sorted_matches, selected_index)
        print(
            f"[red]Error: Multiple matches found for '{spec}'. Use source@name format to specify which one.[/red]"
        )
        print(
            "[dim]Examples: negmas@AspirationNegotiator, anl@AspirationNegotiator[/dim]"
        )
        return None
    elif on_multiple_matches == "warn":
        show_negotiator_match_table(spec, sorted_matches, selected_index)
        print(
            f"[green]Using: {selected_info.source}@{selected_info.short_name}[/green]"
        )
    # else: silent mode - don't show anything

    return selected_info.full_type_name


def resolve_scenario_from_registry(
    spec: str, verbose: bool = False, on_multiple_matches: str = "warn"
) -> Path | None:
    """
    Resolve a scenario specification to a path using the registry.

    Supports:
    - Short names: "CameraB"
    - Source-qualified: "negmas@CameraB" or "negmas>CameraB"
    - Paths: "/path/to/scenario" (returned as Path)

    Priority when multiple matches found:
    1. source='negmas' (native NegMAS scenarios)
    2. Non-genius sources
    3. Any source

    Args:
        spec: Scenario specification string
        verbose: If True, print resolution information
        on_multiple_matches: What to do when multiple matches found:
            - "fail": Show table and return None (caller should exit)
            - "warn": Show table and use prioritized match
            - "silent": Use prioritized match without output

    Returns:
        Path to scenario directory, or None if not found
    """
    from negmas import scenario_registry

    # If spec is already a path and exists, return it
    path_spec = Path(spec)
    if path_spec.exists():
        return path_spec

    # If spec contains separator, parse source and name
    source_filter = None
    search_name = spec

    # Handle '@' separator (preferred)
    if "@" in spec:
        parts = spec.split("@", 1)
        if len(parts) == 2:
            source_filter, search_name = parts

    # Handle '>' separator (deprecated but supported)
    elif ">" in spec:
        parts = spec.split(">", 1)
        if len(parts) == 2:
            source_filter, search_name = parts
            if verbose:
                print(
                    f"[yellow]Note: '>' separator is deprecated. Use '@' instead: {source_filter}@{search_name}[/yellow]"
                )

    # Query registry (ScenarioInfo uses 'name' field, not 'short_name')
    if source_filter:
        results = scenario_registry.query(source=source_filter, name=search_name)
    else:
        results = scenario_registry.query(name=search_name)

    if not results:
        return None

    matches = [(name, info) for name, info in results.items()]

    # Single match - return it
    if len(matches) == 1:
        _, info = matches[0]
        return info.path

    # Multiple matches - apply priority and selection logic
    def priority_key(item):
        """Sort key: prioritize negmas source, then non-genius sources."""
        _, info = item
        if info.source == "negmas":
            return 0
        elif info.source != "genius":
            return 1
        else:
            return 2

    sorted_matches = sorted(matches, key=priority_key)
    selected_index = 0
    _, selected_info = sorted_matches[selected_index]

    # Handle multiple matches based on mode
    if on_multiple_matches == "fail":
        show_scenario_match_table(spec, sorted_matches, selected_index)
        print(
            f"[red]Error: Multiple matches found for '{spec}'. Use source@name format to specify which one.[/red]"
        )
        print("[dim]Examples: negmas@CameraB, custom@CameraB[/dim]")
        return None
    elif on_multiple_matches == "warn":
        show_scenario_match_table(spec, sorted_matches, selected_index)
        print(f"[green]Using: {selected_info.source}@{selected_info.name}[/green]")
    # else: silent mode - don't show anything

    return selected_info.path


def print_negotiator_info(negotiators: list, specs: list[str]) -> None:
    """
    Print information about the negotiators being used in the negotiation.

    Always prints the full type path of each negotiator along with its name.

    Args:
        negotiators: List of instantiated negotiator objects
        specs: List of original specification strings used to create the negotiators
    """
    from negmas.helpers.types import get_full_type_name

    print("\n[bold cyan]Negotiators:[/bold cyan]")
    for i, (neg, spec) in enumerate(zip(negotiators, specs, strict=True)):
        full_type = get_full_type_name(type(neg))
        name = neg.name if hasattr(neg, "name") else f"Agent{i}"

        # Show both the spec used and the resolved full type
        if spec != full_type:
            print(f"  [{i + 1}] {name}")
            print(f"      Spec: {spec}")
            print(f"      Type: {full_type}")
        else:
            print(f"  [{i + 1}] {name}: {full_type}")
    print()


def list_scenarios(source_filter: str | None = None) -> None:
    """
    List all available scenarios from the scenario registry.

    Args:
        source_filter: Optional source to filter by (e.g., 'negmas', 'genius')
    """
    from negmas import scenario_registry

    print("\n[bold]Available Scenarios in Registry[/bold]\n")

    # Query with source filter if provided
    if source_filter:
        results = scenario_registry.query(source=source_filter)
        print(f"Filtered by source: [cyan]{source_filter}[/cyan]")
    else:
        results = dict(scenario_registry)

    if not results:
        print(
            f"[yellow]No scenarios found{f' with source {source_filter}' if source_filter else ''}[/yellow]"
        )
        return

    print(f"Total: {len(results)} scenarios\n")

    # Display table
    table_data = []
    for name in sorted(results.keys()):
        info = results[name]
        tags_str = ", ".join(sorted(info.tags)) if info.tags else ""
        n_negotiators = (
            str(info.n_negotiators) if hasattr(info, "n_negotiators") else "?"
        )
        row = [
            name,
            info.source,
            tags_str[:40] + "..." if len(tags_str) > 43 else tags_str,
            n_negotiators,
        ]
        table_data.append(row)

    print(
        tabulate(table_data, headers=["Name", "Source", "Tags", "N"], tablefmt="simple")
    )

    print("\n[dim]Usage examples:[/dim]")
    print("  negotiate -S negmas@CameraB -n AspirationNegotiator -n RandomNegotiator")
    print("  negotiate -S CameraB -n AspirationNegotiator -n RandomNegotiator -s 100")
