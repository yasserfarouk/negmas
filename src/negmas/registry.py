"""NegMAS Registry System for Mechanisms, Negotiators, Components, and Scenarios.

Overview
--------
The registry system provides a centralized way to register, discover, and query
negotiation-related classes and scenarios. It enables:

1. **Discovery**: Find all available implementations without knowing their locations
2. **Querying**: Filter implementations by properties (e.g., find all ANAC negotiators)
3. **Extensibility**: External libraries can register their own implementations
4. **Metadata**: Store and retrieve information about each implementation

Architecture
------------
The registry consists of four separate registries, each for a different type:

- ``mechanism_registry``: Negotiation mechanisms (e.g., SAOMechanism, TAUMechanism)
- ``negotiator_registry``: Negotiation agents (e.g., AspirationNegotiator, NaiveTitForTatNegotiator)
- ``component_registry``: BOA components (acceptance strategies, offering strategies, opponent models)
- ``scenario_registry``: Negotiation scenarios (domain + utility function definitions)

Key Design: Registry Keys
-------------------------
For class registries (mechanisms, negotiators, components), keys work as follows:

- **Primary key**: ``short_name`` - A human-readable name (e.g., "AspirationNegotiator")
- **Fallback key**: ``full_type_name`` - Fully qualified name (e.g., "negmas.sao.AspirationNegotiator")

When registering a class:

1. If no ``short_name`` is provided, the class name (``cls.__name__``) is used
2. If the ``short_name`` already exists for a *different* class, the new class is
   registered under its ``full_type_name`` instead (to avoid silent overwrites)
3. Both names can be used to retrieve the class via ``get_class()``

This design provides:

- **Convenience**: Use short names like "SAOMechanism" for common lookups
- **Safety**: No silent overwrites when names clash (e.g., two libraries both
  defining "MyNegotiator")
- **Uniqueness**: The ``full_type_name`` is always unique and can be used when
  short names are ambiguous

For scenario registries, keys are the resolved absolute path (always unique).

Usage Examples
--------------

Registering Classes
~~~~~~~~~~~~~~~~~~~

Using decorators (recommended)::

    from negmas import register_negotiator, register_mechanism

    @register_negotiator(
        bilateral_only=True,
        learns=True,
        tags={"custom", "learning"},
    )
    class MyLearningNegotiator(SAONegotiator):
        '''A custom negotiator that learns.'''
        pass

    @register_mechanism(requires_deadline=False, tags={"custom"})
    class MyMechanism(Mechanism):
        pass

Using direct registration::

    from negmas import negotiator_registry, NegotiatorInfo

    class AnotherNegotiator(SAONegotiator):
        pass

    negotiator_registry.register(
        AnotherNegotiator,
        short_name="another",  # Optional custom name
        tags={"experimental"},
    )

Discovering and Querying
~~~~~~~~~~~~~~~~~~~~~~~~

List all registered items::

    from negmas import negotiator_registry, mechanism_registry

    # List all negotiators
    for name, info in negotiator_registry.items():
        print(f"{name}: {info.full_type_name}")

    # List just the names
    print(list(negotiator_registry.keys()))

Query by properties::

    # Find negotiators that learn
    learners = negotiator_registry.query(learns=True)

    # Find ANAC 2019 negotiators
    anac2019 = negotiator_registry.query(anac_year=2019)

    # Find mechanisms that don't require a deadline
    flexible = mechanism_registry.query(requires_deadline=False)

Query by tags::

    # Find items with ALL specified tags
    builtin_sao = negotiator_registry.query(tags=["builtin", "sao"])

    # Find items with ANY of the specified tags
    genius_or_builtin = negotiator_registry.query(any_tags=["genius", "builtin"])

    # Exclude items with certain tags
    no_genius = negotiator_registry.query(exclude_tags=["genius"])

    # Combined tag filtering
    results = negotiator_registry.query(
        tags=["sao"],  # Must have "sao"
        any_tags=["anac-2018", "anac-2019"],  # Must have one of these
        exclude_tags=["experimental"],  # Must not have this
    )

Get a specific class::

    from negmas import negotiator_registry

    # By short name
    cls = negotiator_registry.get_class("AspirationNegotiator")

    # By full type name (useful when short names clash)
    cls = negotiator_registry.get_class("negmas.sao.negotiators.AspirationNegotiator")

Working with Scenarios
~~~~~~~~~~~~~~~~~~~~~~

Register individual scenarios::

    from negmas import register_scenario

    info = register_scenario(
        "/path/to/my/scenario",
        name="MyScenario",
        tags={"custom", "bilateral"},
        n_negotiators=2,
    )

Register all scenarios from a directory::

    from negmas import register_all_scenarios

    # Recursively find and register all loadable scenarios
    scenarios = register_all_scenarios(
        "/path/to/scenarios",
        tags={"my-project"},
    )
    print(f"Registered {len(scenarios)} scenarios")

Query scenarios::

    from negmas import scenario_registry

    # Find bilateral scenarios
    bilateral = scenario_registry.query(tags=["bilateral"])

    # Find XML format scenarios
    xml_scenarios = scenario_registry.query(format="xml")

    # Get scenario by name (may return multiple if name is not unique)
    camera_scenarios = scenario_registry.get_by_name("CameraB")

Built-in Registrations
~~~~~~~~~~~~~~~~~~~~~~

NegMAS automatically registers:

- Built-in mechanisms (SAOMechanism, TAUMechanism, etc.)
- Built-in negotiators (AspirationNegotiator, NaiveTitForTatNegotiator, etc.)
- Genius negotiators (imported from Genius library)
- BOA components (acceptance, offering, opponent modeling strategies)
- Built-in scenarios (CameraB, etc.)

These are registered when you import from negmas and can be discovered immediately::

    from negmas import negotiator_registry

    # See all registered negotiators
    print(f"Total negotiators: {len(negotiator_registry)}")

    # Filter by source
    builtin = negotiator_registry.query(tags=["builtin"])
    genius = negotiator_registry.query(tags=["genius"])
    print(f"Built-in: {len(builtin)}, Genius: {len(genius)}")

CLI Access
----------

The registry is also accessible via the command line::

    # List all mechanisms
    negmas registry list mechanisms

    # List negotiators with filters
    negmas registry list negotiators --tag genius --tag anac-2019

    # Show registry statistics
    negmas registry stats

    # Search by name pattern
    negmas registry search "Aspiration*"

    # List all tags
    negmas registry tags --count
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    pass

__all__ = [
    "RegistryInfo",
    "Registry",
    "MechanismInfo",
    "NegotiatorInfo",
    "ComponentInfo",
    "ScenarioInfo",
    "ScenarioRegistry",
    "mechanism_registry",
    "negotiator_registry",
    "component_registry",
    "scenario_registry",
    "register_mechanism",
    "register_negotiator",
    "register_component",
    "register_scenario",
    "register_all_scenarios",
    "get_registered_class",
]


T = TypeVar("T")


@dataclass
class RegistryInfo:
    """Base class for registration information.

    Attributes:
        short_name: A short, human-readable name for the class.
        full_type_name: The fully qualified class name (e.g., 'negmas.sao.SAOMechanism').
        cls: The actual class object.
        tags: A set of string tags for categorization and filtering.
        extra: Additional key-value pairs for custom properties.
    """

    short_name: str
    full_type_name: str
    cls: type
    tags: set[str] = field(default_factory=set)
    extra: dict[str, Any] = field(default_factory=dict)

    def has_tag(self, tag: str) -> bool:
        """Check if this item has a specific tag.

        Args:
            tag: The tag to check for.

        Returns:
            True if the tag is present, False otherwise.
        """
        return tag in self.tags

    def has_any_tag(self, tags: set[str] | list[str] | tuple[str, ...]) -> bool:
        """Check if this item has any of the specified tags.

        Args:
            tags: The tags to check for.

        Returns:
            True if any of the tags is present, False otherwise.
        """
        return bool(self.tags & set(tags))

    def has_all_tags(self, tags: set[str] | list[str] | tuple[str, ...]) -> bool:
        """Check if this item has all of the specified tags.

        Args:
            tags: The tags to check for.

        Returns:
            True if all of the tags are present, False otherwise.
        """
        return set(tags) <= self.tags


@dataclass
class MechanismInfo(RegistryInfo):
    """Registration information for mechanisms.

    Attributes:
        requires_deadline: Whether the mechanism requires a deadline (n_steps or time_limit).
            TAU mechanisms do not require a deadline because they have implicit termination.
    """

    requires_deadline: bool = True


@dataclass
class NegotiatorInfo(RegistryInfo):
    """Registration information for negotiators.

    Attributes:
        bilateral_only: Whether the negotiator only works in bilateral negotiations.
        requires_opponent_ufun: Whether the negotiator requires access to opponent's utility function.
        learns: Whether the negotiator learns from repeated negotiations.
        anac_year: The ANAC competition year for Genius negotiators (None for non-Genius).
        supports_uncertainty: Whether the negotiator supports uncertain preferences.
        supports_discounting: Whether the negotiator supports time-discounted utilities.
    """

    bilateral_only: bool = False
    requires_opponent_ufun: bool = False
    learns: bool = False
    anac_year: int | None = None
    supports_uncertainty: bool = False
    supports_discounting: bool = False


@dataclass
class ComponentInfo(RegistryInfo):
    """Registration information for components.

    Attributes:
        component_type: The type of component (e.g., 'acceptance', 'offering', 'model').
    """

    component_type: str = "generic"


@dataclass
class ScenarioInfo:
    """Registration information for negotiation scenarios.

    Unlike other registry entries, scenarios are paths to files/folders rather than classes.

    Attributes:
        name: A short name for the scenario (may not be unique - typically folder/file name).
        path: The full path to the scenario file or folder.
        tags: A set of string tags for categorization and filtering.
        normalized: Whether the scenario utilities are normalized (if known).
        n_outcomes: The number of possible outcomes (if known).
        n_negotiators: The number of negotiators in the scenario (if known).
        anac: Whether this scenario is from an ANAC competition.
        file: True if this is a single file scenario, False if it's a folder.
        format: The scenario format ('xml', 'json', 'yaml').
        has_stats: Whether the scenario has pre-computed statistics.
        has_plot: Whether the scenario has a pre-generated plot.
        extra: Additional key-value pairs for custom properties.
    """

    name: str
    path: Path
    tags: set[str] = field(default_factory=set)
    normalized: bool | None = None
    n_outcomes: int | None = None
    n_negotiators: int | None = None
    anac: bool | None = None
    file: bool = False
    format: str = "xml"
    has_stats: bool = False
    has_plot: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def has_tag(self, tag: str) -> bool:
        """Check if this scenario has a specific tag.

        Args:
            tag: The tag to check for.

        Returns:
            True if the tag is present, False otherwise.
        """
        return tag in self.tags

    def has_any_tag(self, tags: set[str] | list[str] | tuple[str, ...]) -> bool:
        """Check if this scenario has any of the specified tags.

        Args:
            tags: The tags to check for.

        Returns:
            True if any of the tags is present, False otherwise.
        """
        return bool(self.tags & set(tags))

    def has_all_tags(self, tags: set[str] | list[str] | tuple[str, ...]) -> bool:
        """Check if this scenario has all of the specified tags.

        Args:
            tags: The tags to check for.

        Returns:
            True if all of the tags are present, False otherwise.
        """
        return set(tags) <= self.tags


class Registry(dict[str, RegistryInfo]):
    """A registry for storing and querying registered classes.

    This is a dictionary subclass that provides additional query methods
    for finding classes by their properties.
    """

    def __init__(self, info_class: type[RegistryInfo]):
        """Initialize the registry.

        Args:
            info_class: The RegistryInfo subclass used for entries in this registry.
        """
        super().__init__()
        self._info_class = info_class
        self._by_class: dict[type, str] = {}

    def register(
        self,
        cls: type,
        short_name: str | None = None,
        tags: set[str] | list[str] | tuple[str, ...] | None = None,
        **kwargs,
    ) -> None:
        """Register a class in the registry.

        Args:
            cls: The class to register.
            short_name: A short name for the class. If None, uses the class name.
            tags: Optional set of tags for categorization and filtering.
            **kwargs: Additional properties for the RegistryInfo.

        Note:
            If the short_name already exists with a different class, the new class
            will be registered under its full type name instead to avoid clashes.
        """
        if short_name is None:
            short_name = cls.__name__

        full_type_name = f"{cls.__module__}.{cls.__qualname__}"

        # Check for name clash
        if short_name in self:
            existing = self[short_name]
            if existing.full_type_name != full_type_name:
                # Clash detected - use full type name for this registration
                short_name = full_type_name

        # Normalize tags to a set
        tags_set = set(tags) if tags is not None else set()

        # Create the info object
        info = self._info_class(
            short_name=short_name,
            full_type_name=full_type_name,
            cls=cls,
            tags=tags_set,
            **kwargs,
        )

        # Store by short name only
        self[short_name] = info
        self._by_class[cls] = short_name

    def unregister(self, cls_or_name: type | str) -> bool:
        """Remove a class from the registry.

        Args:
            cls_or_name: Either the class itself or its registered short name.

        Returns:
            True if the class was found and removed, False if it wasn't registered.

        Example:
            # Unregister by class
            registry.unregister(MyNegotiator)

            # Unregister by name
            registry.unregister("MyNegotiator")
        """
        if isinstance(cls_or_name, str):
            # Given a name, find the info and get the class
            info = self.get(cls_or_name)
            if info is None:
                return False
            short_name = cls_or_name
            cls = info.cls
        else:
            # Given a class, find the short name
            cls = cls_or_name
            short_name = self._by_class.get(cls)
            if short_name is None:
                return False

        # Remove from both dictionaries
        del self[short_name]
        del self._by_class[cls]
        return True

    def get_by_class(self, cls: type) -> T | None:
        """Get the registration info for a class.

        Args:
            cls: The class to look up.

        Returns:
            The RegistryInfo for the class, or None if not registered.
        """
        short_name = self._by_class.get(cls)
        if short_name is None:
            return None
        return self.get(short_name)

    def is_registered(self, cls: type) -> bool:
        """Check if a class is registered.

        Args:
            cls: The class to check.

        Returns:
            True if the class is registered, False otherwise.
        """
        return cls in self._by_class

    def query(
        self,
        *,
        tags: set[str] | list[str] | tuple[str, ...] | None = None,
        any_tags: set[str] | list[str] | tuple[str, ...] | None = None,
        exclude_tags: set[str] | list[str] | tuple[str, ...] | None = None,
        **criteria,
    ) -> dict[str, T]:
        """Query the registry for classes matching the given criteria.

        Args:
            tags: If provided, only return items that have ALL of these tags.
            any_tags: If provided, only return items that have ANY of these tags.
            exclude_tags: If provided, exclude items that have ANY of these tags.
            **criteria: Attribute-value pairs to match exactly.

        Returns:
            A dictionary of matching entries (short_name -> info).

        Example:
            # Find all genius negotiators from ANAC 2019
            negotiator_registry.query(tags={"genius"}, anac_year=2019)

            # Find any builtin or genius negotiators
            negotiator_registry.query(any_tags={"builtin", "genius"})

            # Find builtin negotiators that are not bilateral-only
            negotiator_registry.query(tags={"builtin"}, bilateral_only=False)

            # Find all negotiators except genius ones
            negotiator_registry.query(exclude_tags={"genius"})
        """
        results = {}
        for key, info in self.items():
            match = True

            # Check required tags (all must be present)
            if tags is not None and not info.has_all_tags(tags):
                continue

            # Check any_tags (at least one must be present)
            if any_tags is not None and not info.has_any_tag(any_tags):
                continue

            # Check exclude_tags (none must be present)
            if exclude_tags is not None and info.has_any_tag(exclude_tags):
                continue

            # Check attribute criteria
            for attr, value in criteria.items():
                if not hasattr(info, attr):
                    match = False
                    break
                if getattr(info, attr) != value:
                    match = False
                    break
            if match:
                results[key] = info
        return results

    def list_tags(self) -> set[str]:
        """List all unique tags used across all registered items.

        Returns:
            A set of all unique tags.
        """
        all_tags: set[str] = set()
        for info in self.values():
            all_tags |= info.tags
        return all_tags

    def query_by_tag(self, tag: str) -> dict[str, T]:
        """Query the registry for classes with a specific tag.

        This is a convenience method equivalent to query(tags={tag}).

        Args:
            tag: The tag to filter by.

        Returns:
            A dictionary of matching entries (short_name -> info).
        """
        return self.query(tags={tag})

    def list_all(self) -> list[str]:
        """List all registered short names.

        Returns:
            A list of all registered short names.
        """
        return list(self.keys())

    def get_class(self, name: str) -> type | None:
        """Get the class for a registered name.

        Args:
            name: The short name or full type name.

        Returns:
            The class, or None if not found.
        """
        info = self.get(name)
        if info is None:
            return None
        return info.cls


class ScenarioRegistry(dict[str, ScenarioInfo]):
    """A registry for storing and querying registered scenarios.

    Unlike the regular Registry, this stores scenario paths rather than classes.
    Scenarios are identified by a unique key (typically the path string).
    """

    def __init__(self):
        """Initialize the scenario registry."""
        super().__init__()
        self._by_name: dict[str, list[str]] = {}  # name -> list of keys (not unique)

    def register(
        self,
        path: str | Path,
        name: str | None = None,
        tags: set[str] | list[str] | tuple[str, ...] | None = None,
        normalized: bool | None = None,
        n_outcomes: int | None = None,
        n_negotiators: int | None = None,
        anac: bool | None = None,
        has_stats: bool = False,
        has_plot: bool = False,
        **extra,
    ) -> ScenarioInfo:
        """Register a scenario in the registry.

        Args:
            path: The path to the scenario file or folder.
            name: A short name for the scenario. If None, uses the file/folder name.
            tags: Optional set of tags for categorization and filtering.
            normalized: Whether the scenario utilities are normalized.
            n_outcomes: The number of possible outcomes.
            n_negotiators: The number of negotiators in the scenario.
            anac: Whether this scenario is from an ANAC competition.
            has_stats: Whether the scenario has pre-computed statistics.
            has_plot: Whether the scenario has a pre-generated plot.
            **extra: Additional properties to store.

        Returns:
            The ScenarioInfo object that was created.
        """
        path = Path(path).resolve()

        # Determine if it's a file or folder
        is_file = path.is_file()

        # Determine format from extension or folder contents
        if is_file:
            suffix = path.suffix.lower()
            if suffix == ".xml":
                fmt = "xml"
            elif suffix == ".json":
                fmt = "json"
            elif suffix in (".yml", ".yaml"):
                fmt = "yaml"
            else:
                fmt = "unknown"
        else:
            # Check folder contents
            if list(path.glob("*.xml")):
                fmt = "xml"
            elif list(path.glob("*.json")):
                fmt = "json"
            elif list(path.glob("*.yml")) or list(path.glob("*.yaml")):
                fmt = "yaml"
            else:
                fmt = "unknown"

        # Use path stem as name if not provided
        if name is None:
            name = path.stem if is_file else path.name

        # Auto-generate tags from path components
        auto_tags: set[str] = set()

        # Add format tag
        if fmt != "unknown":
            auto_tags.add(fmt)

        # Combine with provided tags
        tags_set = set(tags) if tags is not None else set()
        all_tags = auto_tags | tags_set

        # Use the path string as the unique key
        key = str(path)

        # Create the info object
        info = ScenarioInfo(
            name=name,
            path=path,
            tags=all_tags,
            normalized=normalized,
            n_outcomes=n_outcomes,
            n_negotiators=n_negotiators,
            anac=anac,
            file=is_file,
            format=fmt,
            has_stats=has_stats,
            has_plot=has_plot,
            extra=extra,
        )

        # Store by path
        self[key] = info

        # Index by name for quick lookup
        if name not in self._by_name:
            self._by_name[name] = []
        if key not in self._by_name[name]:
            self._by_name[name].append(key)

        return info

    def unregister(self, path_or_name: str | Path) -> bool:
        """Remove a scenario from the registry.

        Args:
            path_or_name: Either the path (as string or Path) or the scenario name.
                If a name is given and multiple scenarios have that name, all are removed.

        Returns:
            True if at least one scenario was removed, False if none were found.

        Example:
            # Unregister by path
            registry.unregister("/path/to/scenario")

            # Unregister by name (removes all with that name)
            registry.unregister("Laptop")
        """
        # If it's a Path object, resolve it to match how register() stores keys
        if isinstance(path_or_name, Path):
            path_str = str(path_or_name.resolve())
        else:
            path_str = path_or_name

        # First check if it's a direct key (path)
        if path_str in self:
            info = self[path_str]
            name = info.name

            # Remove from main dict
            del self[path_str]

            # Remove from name index
            if name in self._by_name:
                if path_str in self._by_name[name]:
                    self._by_name[name].remove(path_str)
                if not self._by_name[name]:
                    del self._by_name[name]
            return True

        # Otherwise, check if it's a name
        if path_str in self._by_name:
            keys = self._by_name[path_str].copy()
            for key in keys:
                if key in self:
                    del self[key]
            del self._by_name[path_str]
            return len(keys) > 0

        return False

    def get_by_name(self, name: str) -> list[ScenarioInfo]:
        """Get all scenarios with a given name.

        Note that scenario names are not unique, so this returns a list.

        Args:
            name: The scenario name to look up.

        Returns:
            A list of ScenarioInfo objects with that name (may be empty).
        """
        keys = self._by_name.get(name, [])
        return [self[key] for key in keys if key in self]

    def query(
        self,
        *,
        tags: set[str] | list[str] | tuple[str, ...] | None = None,
        any_tags: set[str] | list[str] | tuple[str, ...] | None = None,
        exclude_tags: set[str] | list[str] | tuple[str, ...] | None = None,
        format: str | None = None,
        anac: bool | None = None,
        normalized: bool | None = None,
        file: bool | None = None,
        **criteria,
    ) -> dict[str, ScenarioInfo]:
        """Query the registry for scenarios matching the given criteria.

        Args:
            tags: If provided, only return items that have ALL of these tags.
            any_tags: If provided, only return items that have ANY of these tags.
            exclude_tags: If provided, exclude items that have ANY of these tags.
            format: Filter by scenario format ('xml', 'json', 'yaml').
            anac: Filter by ANAC status.
            normalized: Filter by normalized status.
            file: Filter by file (True) vs folder (False).
            **criteria: Additional attribute-value pairs to match exactly.

        Returns:
            A dictionary of matching entries (path -> info).
        """
        results = {}
        for key, info in self.items():
            # Check required tags (all must be present)
            if tags is not None and not info.has_all_tags(tags):
                continue

            # Check any_tags (at least one must be present)
            if any_tags is not None and not info.has_any_tag(any_tags):
                continue

            # Check exclude_tags (none must be present)
            if exclude_tags is not None and info.has_any_tag(exclude_tags):
                continue

            # Check format
            if format is not None and info.format != format:
                continue

            # Check anac
            if anac is not None and info.anac != anac:
                continue

            # Check normalized
            if normalized is not None and info.normalized != normalized:
                continue

            # Check file
            if file is not None and info.file != file:
                continue

            # Check additional criteria
            match = True
            for attr, value in criteria.items():
                if not hasattr(info, attr):
                    match = False
                    break
                if getattr(info, attr) != value:
                    match = False
                    break
            if match:
                results[key] = info
        return results

    def list_tags(self) -> set[str]:
        """List all unique tags used across all registered scenarios.

        Returns:
            A set of all unique tags.
        """
        all_tags: set[str] = set()
        for info in self.values():
            all_tags |= info.tags
        return all_tags

    def query_by_tag(self, tag: str) -> dict[str, ScenarioInfo]:
        """Query the registry for scenarios with a specific tag.

        This is a convenience method equivalent to query(tags={tag}).

        Args:
            tag: The tag to filter by.

        Returns:
            A dictionary of matching entries (path -> info).
        """
        return self.query(tags={tag})

    def list_all(self) -> list[str]:
        """List all registered scenario paths.

        Returns:
            A list of all registered scenario paths (as strings).
        """
        return list(self.keys())

    def list_names(self) -> list[str]:
        """List all unique scenario names.

        Returns:
            A list of all unique scenario names.
        """
        return list(self._by_name.keys())


# Global registries
mechanism_registry: Registry[MechanismInfo] = Registry(MechanismInfo)
negotiator_registry: Registry[NegotiatorInfo] = Registry(NegotiatorInfo)
component_registry: Registry[ComponentInfo] = Registry(ComponentInfo)
scenario_registry: ScenarioRegistry = ScenarioRegistry()


def register_mechanism(
    cls: type | None = None,
    *,
    short_name: str | None = None,
    requires_deadline: bool = True,
    tags: set[str] | list[str] | tuple[str, ...] | None = None,
    **extra,
):
    """Decorator to register a mechanism class.

    Can be used with or without arguments:
        @register_mechanism
        class MyMechanism(Mechanism):
            pass

        @register_mechanism(short_name="my_mech", requires_deadline=False, tags={"sao"})
        class MyMechanism(Mechanism):
            pass

    Args:
        cls: The class to register (when used without parentheses).
        short_name: A short name for the mechanism.
        requires_deadline: Whether the mechanism requires a deadline.
        tags: Set of tags for categorization (e.g., {"sao", "builtin"}).
        **extra: Additional properties to store.

    Returns:
        The decorated class (unchanged).
    """

    def decorator(cls: type) -> type:
        mechanism_registry.register(
            cls,
            short_name=short_name,
            tags=tags,
            requires_deadline=requires_deadline,
            extra=extra,
        )
        return cls

    if cls is not None:
        # Called without parentheses: @register_mechanism
        return decorator(cls)
    # Called with parentheses: @register_mechanism(...)
    return decorator


def register_negotiator(
    cls: type | None = None,
    *,
    short_name: str | None = None,
    bilateral_only: bool = False,
    requires_opponent_ufun: bool = False,
    learns: bool = False,
    anac_year: int | None = None,
    supports_uncertainty: bool = False,
    supports_discounting: bool = False,
    tags: set[str] | list[str] | tuple[str, ...] | None = None,
    **extra,
):
    """Decorator to register a negotiator class.

    Can be used with or without arguments:
        @register_negotiator
        class MyNegotiator(Negotiator):
            pass

        @register_negotiator(bilateral_only=True, learns=True, tags={"builtin", "aspiration"})
        class MyNegotiator(Negotiator):
            pass

    Args:
        cls: The class to register (when used without parentheses).
        short_name: A short name for the negotiator.
        bilateral_only: Whether the negotiator only works bilaterally.
        requires_opponent_ufun: Whether the negotiator needs opponent's utility.
        learns: Whether the negotiator learns from repeated negotiations.
        anac_year: ANAC competition year (for Genius negotiators).
        supports_uncertainty: Whether uncertain preferences are supported.
        supports_discounting: Whether time-discounted utilities are supported.
        tags: Set of tags for categorization (e.g., {"builtin", "aspiration"}).
        **extra: Additional properties to store.

    Returns:
        The decorated class (unchanged).
    """

    def decorator(cls: type) -> type:
        negotiator_registry.register(
            cls,
            short_name=short_name,
            tags=tags,
            bilateral_only=bilateral_only,
            requires_opponent_ufun=requires_opponent_ufun,
            learns=learns,
            anac_year=anac_year,
            supports_uncertainty=supports_uncertainty,
            supports_discounting=supports_discounting,
            extra=extra,
        )
        return cls

    if cls is not None:
        return decorator(cls)
    return decorator


def register_component(
    cls: type | None = None,
    *,
    short_name: str | None = None,
    component_type: str = "generic",
    tags: set[str] | list[str] | tuple[str, ...] | None = None,
    **extra,
):
    """Decorator to register a component class.

    Can be used with or without arguments:
        @register_component
        class MyComponent(Component):
            pass

        @register_component(component_type="acceptance", tags={"builtin", "rational"})
        class MyAcceptancePolicy(AcceptancePolicy):
            pass

    Args:
        cls: The class to register (when used without parentheses).
        short_name: A short name for the component.
        component_type: The type of component (e.g., 'acceptance', 'offering', 'model').
        tags: Set of tags for categorization (e.g., {"builtin", "rational"}).
        **extra: Additional properties to store.

    Returns:
        The decorated class (unchanged).
    """

    def decorator(cls: type) -> type:
        component_registry.register(
            cls,
            short_name=short_name,
            tags=tags,
            component_type=component_type,
            extra=extra,
        )
        return cls

    if cls is not None:
        return decorator(cls)
    return decorator


def register_scenario(
    path: str | Path,
    name: str | None = None,
    tags: set[str] | list[str] | tuple[str, ...] | None = None,
    normalized: bool | None = None,
    n_outcomes: int | None = None,
    n_negotiators: int | None = None,
    anac: bool | None = None,
    has_stats: bool = False,
    has_plot: bool = False,
    **extra,
) -> ScenarioInfo:
    """Register a scenario in the global scenario registry.

    This function registers a negotiation scenario (a file or folder containing
    domain and utility definitions) so it can be discovered and used by name.

    Args:
        path: The path to the scenario file or folder.
        name: A short name for the scenario. If None, uses the file/folder name.
        tags: Optional set of tags for categorization and filtering.
        normalized: Whether the scenario utilities are normalized.
        n_outcomes: The number of possible outcomes.
        n_negotiators: The number of negotiators in the scenario.
        anac: Whether this scenario is from an ANAC competition.
        has_stats: Whether the scenario has pre-computed statistics.
        has_plot: Whether the scenario has a pre-generated plot.
        **extra: Additional properties to store.

    Returns:
        The ScenarioInfo object that was created.

    Example:
        # Register a scenario folder
        register_scenario(
            "/path/to/MyScenario",
            tags={"custom", "bilateral"},
            n_negotiators=2,
        )

        # Register with auto-detection
        register_scenario("/path/to/domain.xml")
    """
    return scenario_registry.register(
        path=path,
        name=name,
        tags=tags,
        normalized=normalized,
        n_outcomes=n_outcomes,
        n_negotiators=n_negotiators,
        anac=anac,
        has_stats=has_stats,
        has_plot=has_plot,
        **extra,
    )


def register_all_scenarios(
    path: str | Path,
    tags: set[str] | list[str] | tuple[str, ...] | None = None,
    recursive: bool = True,
    registry: ScenarioRegistry | None = None,
) -> list[ScenarioInfo]:
    """Register all loadable scenarios from a directory.

    This function recursively searches for valid negotiation scenarios in the
    given directory and registers them in the scenario registry. It attempts
    to load each folder as a scenario, and if that fails, searches for YAML
    files and subfolders that might be scenarios.

    The function extracts all available information from each scenario:
    - n_outcomes: Number of possible outcomes
    - n_negotiators: Number of negotiators (from utility functions)
    - normalized: Whether utilities are normalized to [0, 1]
    - format: The scenario format (xml, json, yaml)

    Scenarios are automatically tagged based on their properties:
    - "bilateral": 2 negotiators
    - "multilateral": 3+ negotiators
    - Format tag: "xml", "json", or "yaml"

    Args:
        path: The root directory to search for scenarios.
        tags: Optional tags to add to all registered scenarios.
        recursive: If True, recursively search subdirectories.
        registry: The registry to use. If None, uses the global scenario_registry.

    Returns:
        A list of ScenarioInfo objects for successfully registered scenarios.

    Raises:
        ValueError: If the path does not exist or is not a directory.

    Example:
        # Register all scenarios from a directory
        from negmas import register_all_scenarios

        scenarios = register_all_scenarios(
            "/path/to/scenarios",
            tags={"custom", "my-project"},
        )
        print(f"Registered {len(scenarios)} scenarios")

        # Register without recursion
        scenarios = register_all_scenarios(
            "/path/to/scenarios",
            recursive=False,
        )
    """
    from negmas.inout import Scenario

    path = Path(path)
    if not path.exists():
        raise ValueError(f"Path does not exist: {path}")
    if not path.is_dir():
        raise ValueError(f"Path is not a directory: {path}")

    if registry is None:
        registry = scenario_registry

    base_tags = set(tags) if tags else set()
    registered: list[ScenarioInfo] = []

    def _try_register_scenario(scenario_path: Path) -> ScenarioInfo | None:
        """Try to load and register a single scenario."""
        try:
            scenario = Scenario.load(scenario_path)
            if scenario is None:
                return None

            # Extract information from loaded scenario
            n_negotiators = scenario.n_negotiators
            n_outcomes: int | None = None
            try:
                cardinality = scenario.outcome_space.cardinality
                if isinstance(cardinality, int) or (
                    isinstance(cardinality, float) and cardinality.is_integer()
                ):
                    n_outcomes = int(cardinality)
            except Exception:
                pass

            # Check if normalized
            normalized: bool | None = None
            try:
                normalized = scenario.is_normalized()
            except Exception:
                pass

            # Determine format from files in directory
            fmt = _detect_scenario_format(scenario_path)

            # Build tags
            scenario_tags = set(base_tags)
            if n_negotiators == 2:
                scenario_tags.add("bilateral")
            elif n_negotiators is not None and n_negotiators > 2:
                scenario_tags.add("multilateral")
            if fmt != "unknown":
                scenario_tags.add(fmt)

            # Register the scenario
            info = registry.register(
                path=scenario_path,
                name=scenario_path.name,
                tags=scenario_tags,
                normalized=normalized,
                n_outcomes=n_outcomes,
                n_negotiators=n_negotiators,
            )
            return info

        except Exception:
            return None

    def _search_directory(dir_path: Path) -> None:
        """Search a directory for scenarios."""
        # First, try to load this directory as a scenario
        if Scenario.is_loadable(dir_path):
            info = _try_register_scenario(dir_path)
            if info is not None:
                registered.append(info)
                return  # Don't recurse into successfully loaded scenarios

        if not recursive:
            return

        # If not a scenario, search children
        try:
            children = list(dir_path.iterdir())
        except PermissionError:
            return

        # Try YAML files first (single-file scenarios)
        for child in children:
            if child.is_file() and child.suffix.lower() in (".yml", ".yaml"):
                # YAML files might be standalone scenarios - try parent dir
                pass  # YAML scenarios are typically in folders, not standalone

        # Then try subdirectories
        for child in children:
            if child.is_dir() and not child.name.startswith((".", "_")):
                _search_directory(child)

    _search_directory(path)
    return registered


def _detect_scenario_format(path: Path) -> str:
    """Detect the format of a scenario from its files.

    Args:
        path: Path to the scenario folder.

    Returns:
        The format string: 'xml', 'json', 'yaml', or 'unknown'.
    """
    if path.is_file():
        suffix = path.suffix.lower()
        if suffix == ".xml":
            return "xml"
        elif suffix == ".json":
            return "json"
        elif suffix in (".yml", ".yaml"):
            return "yaml"
        return "unknown"

    # Check folder contents
    if list(path.glob("*.xml")):
        return "xml"
    if list(path.glob("*.json")):
        return "json"
    if list(path.glob("*.yml")) or list(path.glob("*.yaml")):
        return "yaml"
    return "unknown"


def get_registered_class(
    name: str,
    registry: (
        Registry[MechanismInfo] | Registry[NegotiatorInfo] | Registry[ComponentInfo]
    ),
) -> type | None:
    """Get a registered class by name from a registry.

    Args:
        name: The short name or full type name.
        registry: The registry to search.

    Returns:
        The class, or None if not found.
    """
    return registry.get_class(name)


# Auto-register built-in classes when this module is imported
# This is done by importing registry_init which calls _register_all()
from negmas import registry_init as _registry_init  # noqa: F401, E402
