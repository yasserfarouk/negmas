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


# ============================================================================
# Negotiator Factory Functions
# ============================================================================


def get_protocol(name: str):
    """
    Get a mechanism (protocol) class by name.

    Args:
        name: Protocol name (e.g., 'SAO', 'TAU', 'GTAU', 'GAO') or full class path

    Returns:
        The mechanism class
    """
    from negmas.helpers import get_class

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
    """Extract the class name from a marker-prefixed string."""
    if s.startswith("genius"):
        return s.split("genius")[-1]
    if s.startswith("anl"):
        return s.split("anl")[-1]
    raise RuntimeError(f"{s} does not start with a known marker")


def create_adapter(adapter_type, negotiator_type, name: str):
    """Create an adapter wrapping a negotiator."""
    return adapter_type(name=name, base=negotiator_type(name=name))


def make_genius_negotiator(*args, java_class_name: str, **kwargs):
    """Create a Genius negotiator via the bridge."""
    from negmas.genius.negotiator import GeniusNegotiator

    return GeniusNegotiator(*args, **kwargs, java_class_name=java_class_name)


def make_anl_negotiator(class_name: str, **kwargs):
    """Create a negotiator from the anl_agents package."""
    from negmas.helpers import instantiate

    return instantiate(class_name, module_name="anl_agents", **kwargs)


def make_llm_negotiator(class_name: str, **kwargs):
    """Create a negotiator from negmas-llm package."""
    try:
        from importlib import import_module

        import_module("negmas_llm")
    except ImportError:
        print("[red]Error: negmas-llm package is not installed.[/red]")
        print("[yellow]To use LLM negotiators, install the package:[/yellow]")
        print("  pip install negmas-llm")
        print("  or: uv add negmas-llm")
        raise SystemExit(1)

    from negmas.helpers import instantiate

    return instantiate(class_name, module_name="negmas_llm", **kwargs)


def make_negolog_negotiator(class_name: str, **kwargs):
    """Create a negotiator from negmas-negolog package."""
    try:
        from importlib import import_module

        import_module("negmas_negolog")
    except ImportError:
        print("[red]Error: negmas-negolog package is not installed.[/red]")
        print("[yellow]To use Negolog negotiators, install the package:[/yellow]")
        print("  pip install negmas-negolog")
        print("  or: uv add negmas-negolog")
        raise SystemExit(1)

    from negmas.helpers import instantiate

    return instantiate(class_name, module_name="negmas_negolog", **kwargs)


def make_ga_negotiator(class_name: str, **kwargs):
    """Create a negotiator from negmas-genius-agents package."""
    try:
        from importlib import import_module

        import_module("genius_agents")
    except ImportError:
        print("[red]Error: negmas-genius-agents package is not installed.[/red]")
        print("[yellow]To use Genius Agents negotiators, install the package:[/yellow]")
        print("  pip install negmas-genius-agents")
        print("  or: uv add negmas-genius-agents")
        raise SystemExit(1)

    from negmas.helpers import instantiate

    return instantiate(class_name, module_name="genius_agents", **kwargs)


def parse_component_spec(spec: str) -> tuple[str, dict[str, Any]]:
    """
    Parse a component specification like 'GTimeDependentOffering(e=0.2, k=0.0)'.

    Returns:
        A tuple of (class_name, kwargs_dict)
    """
    spec = spec.strip()
    if "(" not in spec:
        return spec, {}

    # Extract class name and params
    paren_idx = spec.index("(")
    class_name = spec[:paren_idx].strip()
    params_str = spec[paren_idx + 1 : -1].strip()  # Remove ( and )

    if not params_str:
        return class_name, {}

    # Parse key=value pairs
    kwargs: dict[str, Any] = {}
    # Handle nested parentheses by tracking depth
    depth = 0
    current_param = ""
    for char in params_str + ",":
        if char in "([{":
            depth += 1
            current_param += char
        elif char in ")]}":
            depth -= 1
            current_param += char
        elif char == "," and depth == 0:
            if current_param.strip():
                key, _, value = current_param.partition("=")
                key = key.strip()
                value = value.strip()
                # Try to evaluate the value
                try:
                    kwargs[key] = eval(value)
                except Exception:
                    kwargs[key] = value
            current_param = ""
        else:
            current_param += char

    return class_name, kwargs


def get_component(class_name: str, module_prefix: str = "negmas.gb.components"):
    """
    Get a component class by name.

    Args:
        class_name: The class name (e.g., 'GTimeDependentOffering', 'ACNext')
        module_prefix: The module prefix to search in

    Returns:
        The component class
    """
    from negmas.helpers import get_class

    # Try direct import from negmas first
    try:
        return get_class(f"negmas.{class_name}")
    except Exception:
        pass

    # Try from gb.components
    try:
        return get_class(f"{module_prefix}.{class_name}")
    except Exception:
        pass

    # Try from gb.components.genius
    try:
        return get_class(f"{module_prefix}.genius.{class_name}")
    except Exception:
        pass

    # Try from sao.components
    try:
        return get_class(f"negmas.sao.components.{class_name}")
    except Exception:
        pass

    raise ValueError(f"Cannot find component class: {class_name}")


def make_boa_negotiator(spec: str, is_map: bool = False, **kwargs):
    """
    Create a BOA or MAP negotiator from a specification string.

    Args:
        spec: Component specification like 'offering=GTimeDependentOffering(e=0.2),acceptance=GACNext'
        is_map: If True, create a MAPNegotiator, otherwise create a BOANegotiator
        **kwargs: Additional arguments passed to the negotiator

    Returns:
        A configured BOA or MAP negotiator

    Examples:
        - 'offering=GTimeDependentOffering(e=0.2),acceptance=GACNext'
        - 'offering=TimeBasedOfferingPolicy,acceptance=ACNext,model=GHardHeadedFrequencyModel'
    """
    from negmas.gb.negotiators.modular import MAPNegotiator, make_boa

    # First pass: collect all component specs
    component_specs: dict[str, tuple[str, dict[str, Any]]] = {}

    # Parse comma-separated component specs
    # Handle nested parentheses
    depth = 0
    current_spec = ""
    specs = []
    for char in spec + ",":
        if char in "([{":
            depth += 1
            current_spec += char
        elif char in ")]}":
            depth -= 1
            current_spec += char
        elif char == "," and depth == 0:
            if current_spec.strip():
                specs.append(current_spec.strip())
            current_spec = ""
        else:
            current_spec += char

    for component_spec in specs:
        if "=" not in component_spec:
            raise ValueError(
                f"Invalid component spec '{component_spec}'. Expected 'key=ComponentClass(args)'"
            )

        key, _, value = component_spec.partition("=")
        key = key.strip().lower()
        value = value.strip()

        class_name, component_kwargs = parse_component_spec(value)
        component_specs[key] = (class_name, component_kwargs)

    # Second pass: instantiate components in the right order
    # Offering policy first (acceptance may depend on it)
    components: dict[str, Any] = {}

    # Instantiate offering policy first
    if "offering" in component_specs:
        class_name, component_kwargs = component_specs["offering"]
        component_class = get_component(class_name)
        components["offering"] = component_class(**component_kwargs)

    # Instantiate acceptance policy (may need offering policy)
    if "acceptance" in component_specs:
        class_name, component_kwargs = component_specs["acceptance"]
        component_class = get_component(class_name)

        # Check if acceptance policy needs offering_policy or offering_strategy parameter
        import inspect

        sig = inspect.signature(component_class.__init__)
        if "offering" in components:
            if "offering_policy" in sig.parameters:
                component_kwargs["offering_policy"] = components["offering"]
            elif "offering_strategy" in sig.parameters:
                component_kwargs["offering_strategy"] = components["offering"]

        components["acceptance"] = component_class(**component_kwargs)

    # Instantiate model
    if "model" in component_specs:
        class_name, component_kwargs = component_specs["model"]
        component_class = get_component(class_name)
        components["model"] = component_class(**component_kwargs)

    # Instantiate any other components
    for key, (class_name, component_kwargs) in component_specs.items():
        if key not in components:
            component_class = get_component(class_name)
            components[key] = component_class(**component_kwargs)

    # Extract known component types
    offering = components.pop("offering", None)
    acceptance = components.pop("acceptance", None)
    model = components.pop("model", None)

    if is_map:
        # For MAP, we may have multiple models and extra components
        models = [model] if model else None
        model_names = ["model"] if model else None

        # Any remaining components go to extra_components
        extra_components = list(components.values()) if components else None
        extra_component_names = list(components.keys()) if components else None

        return MAPNegotiator(
            offering=offering,
            acceptance=acceptance,
            models=models,
            model_names=model_names,
            extra_components=extra_components,
            extra_component_names=extra_component_names,
            **kwargs,
        )
    else:
        return make_boa(offering=offering, acceptance=acceptance, model=model, **kwargs)


# Track whether we started the genius bridge ourselves
_genius_bridge_started_by_cli = False


def ensure_genius_bridge_running() -> bool:
    """
    Ensures the Genius bridge is running, starting it if necessary.

    Returns:
        True if the bridge is running, False otherwise.

    Side effects:
        Sets _genius_bridge_started_by_cli to True if we started the bridge.
    """
    global _genius_bridge_started_by_cli

    from negmas.genius.bridge import genius_bridge_is_running, init_genius_bridge
    from negmas.genius.common import DEFAULT_JAVA_PORT

    if genius_bridge_is_running(DEFAULT_JAVA_PORT):
        return True

    # Try to start the bridge
    print("[yellow]Genius bridge not running. Attempting to start...[/yellow]")
    port = init_genius_bridge(port=DEFAULT_JAVA_PORT, die_on_exit=True)

    if port > 0:
        print(f"[green]Genius bridge started on port {port}[/green]")
        _genius_bridge_started_by_cli = True
        return True
    elif port == -1:
        # Bridge was already running (race condition)
        return True
    else:
        print("[red]Failed to start Genius bridge.[/red]")
        print(
            "[yellow]Please ensure the bridge is installed and Java is available.[/yellow]"
        )
        from negmas.genius.bridge import genius_bridge_is_installed

        if not genius_bridge_is_installed():
            print(
                "[yellow]Run 'negmas genius-setup' to install the Genius bridge.[/yellow]"
            )
        return False
