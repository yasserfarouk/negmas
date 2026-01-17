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
from typing import TYPE_CHECKING, Any, Generic, TypeVar

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
    "save_registry",
    "load_registry",
    "clear_registry",
    "DEFAULT_REGISTRY_PATH",
]


T = TypeVar("T", bound="RegistryInfo")


@dataclass
class RegistryInfo:
    """Base class for registration information.

    Attributes:
        key: The unique registry key (e.g., 'AspirationNegotiator#a1b2c3d4').
            This is automatically generated and guaranteed unique within the registry.
        short_name: A short, human-readable name for the class.
        full_type_name: The fully qualified class name (e.g., 'negmas.sao.SAOMechanism').
        cls: The actual class object.
        source: The origin of this registration ('negmas' for built-in, library name for
            external, or 'unknown' if not specified).
        params: Constructor parameters for creating instances via Registry.create().
        tags: A set of string tags for categorization and filtering.
        extra: Additional key-value pairs for custom properties.
    """

    key: str
    short_name: str
    full_type_name: str
    cls: type
    source: str = "unknown"
    params: dict[str, Any] = field(default_factory=dict)
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

    All mechanism properties are now expressed via tags:
    - "requires-deadline": Mechanism requires n_steps or time_limit
    - "propose": Requires negotiators to have propose capability
    - "respond": Requires negotiators to have respond capability
    - "sao", "gb", "tau", "st": Protocol type tags
    """

    pass


@dataclass
class NegotiatorInfo(RegistryInfo):
    """Registration information for negotiators.

    All negotiator properties are now expressed via tags:
    - "bilateral-only": Only works in bilateral negotiations
    - "requires-opponent-ufun": Requires access to opponent's utility function
    - "learning": Learns from repeated negotiations
    - "anac-YYYY": ANAC competition year (e.g., "anac-2019")
    - "supports-uncertainty": Supports uncertain preferences
    - "supports-discounting": Supports time-discounted utilities
    """

    pass


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

    Boolean properties are now expressed via tags:
    - "normalized": Utilities are normalized to [0, 1]
    - "anac": From an ANAC competition
    - "file": Single file scenario (vs folder)
    - "has-stats": Has pre-computed statistics
    - "has-plot": Has a pre-generated plot
    - "xml", "json", "yaml": Format tags

    Attributes:
        name: A short name for the scenario (may not be unique - typically folder/file name).
        path: The full path to the scenario file or folder.
        source: The origin of this registration ('negmas' for built-in, library name for
            external, or 'unknown' if not specified).
        tags: A set of string tags for categorization and filtering.
        n_outcomes: The number of possible outcomes (if known).
        n_negotiators: The number of negotiators in the scenario (if known).
        opposition_level: The opposition level between negotiators (0=cooperative, 1=competitive).
        extra: Additional key-value pairs for custom properties.
    """

    name: str
    path: Path
    source: str = "unknown"
    tags: set[str] = field(default_factory=set)
    n_outcomes: int | None = None
    n_negotiators: int | None = None
    opposition_level: float | None = None
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


class Registry(Generic[T], dict[str, T]):
    """A registry for storing and querying registered classes.

    This is a dictionary subclass that provides additional query methods
    for finding classes by their properties.

    Type Parameters:
        T: The RegistryInfo subclass used for entries in this registry.
    """

    def __init__(self, info_class: type[T]):
        """Initialize the registry.

        Args:
            info_class: The RegistryInfo subclass used for entries in this registry.
        """
        super().__init__()
        self._info_class = info_class
        self._by_class: dict[type, list[str]] = {}

    def register(
        self,
        cls: type | str,
        short_name: str | None = None,
        source: str = "unknown",
        params: dict[str, Any] | None = None,
        tags: set[str] | list[str] | tuple[str, ...] | None = None,
        **kwargs,
    ) -> str:
        """Register a class in the registry.

        Args:
            cls: The class to register, or its full type name string
                (e.g., 'negmas.sao.AspirationNegotiator'). If a string is provided,
                the class will be resolved using negmas helpers.
            short_name: A human-readable name for this registration. Can differ from
                the class name to create "virtual" negotiators with different params.
                If None, uses the class name. Examples:
                - 'AspirationNegotiator' (same as class)
                - 'AggressiveAspiration' (virtual negotiator with specific params)
            source: The origin of this registration ('negmas' for built-in, library
                name for external, or 'unknown' if not specified).
            params: Constructor parameters for creating instances via create().
                This enables "virtual" negotiators that share a class but have
                different default parameters.
            tags: Optional set of tags for categorization and filtering.
            **kwargs: Additional properties for the RegistryInfo. Deprecated boolean
                params are converted to tags automatically.

        Returns:
            The unique key assigned to this registration.

        Note:
            The same class can be registered multiple times with different names
            and parameters (creating "virtual" negotiators). Each registration gets
            a unique key in the format '{short_name}#{uuid8}'.

        Example:
            # Register with class
            key1 = registry.register(AspirationNegotiator)

            # Register with full type name string
            key2 = registry.register('negmas.sao.AspirationNegotiator')

            # Create a "virtual" negotiator with custom params
            key3 = registry.register(
                AspirationNegotiator,
                short_name='AggressiveAspiration',
                params={'aspiration_type': 'boulware', 'max_aspiration': 0.95},
            )
        """
        import uuid

        from negmas.helpers import get_class, get_full_type_name

        # Resolve class from string if needed
        if isinstance(cls, str):
            full_type_name = cls
            resolved_cls = get_class(cls)
            if resolved_cls is None:
                raise ValueError(f"Could not resolve class from: {cls}")
            cls = resolved_cls
        else:
            full_type_name = get_full_type_name(cls)

        if short_name is None:
            short_name = cls.__name__

        # Generate a unique key using UUID
        uuid_suffix = uuid.uuid4().hex[:8]
        unique_key = f"{short_name}#{uuid_suffix}"

        # Normalize tags to a set
        tags_set = set(tags) if tags is not None else set()

        # Normalize params to a dict
        params_dict = dict(params) if params is not None else {}

        # Handle deprecated boolean params -> convert to tags
        # Mechanism-related
        if "requires_deadline" in kwargs:
            val = kwargs.pop("requires_deadline")
            if val:
                tags_set.add("requires-deadline")

        # Negotiator-related
        if "bilateral_only" in kwargs:
            val = kwargs.pop("bilateral_only")
            if val:
                tags_set.add("bilateral-only")

        if "requires_opponent_ufun" in kwargs:
            val = kwargs.pop("requires_opponent_ufun")
            if val:
                tags_set.add("requires-opponent-ufun")

        if "learns" in kwargs:
            val = kwargs.pop("learns")
            if val:
                tags_set.add("learning")

        if "anac_year" in kwargs:
            val = kwargs.pop("anac_year")
            if val is not None:
                tags_set.add(f"anac-{val}")

        if "supports_uncertainty" in kwargs:
            val = kwargs.pop("supports_uncertainty")
            if val:
                tags_set.add("supports-uncertainty")

        if "supports_discounting" in kwargs:
            val = kwargs.pop("supports_discounting")
            if val:
                tags_set.add("supports-discounting")

        # Component-related: component_type is still a field, also add as tag
        if "component_type" in kwargs:
            component_type = kwargs.get("component_type", "generic")
            if component_type != "generic":
                tags_set.add(component_type)

        # Handle 'extra' dict if provided - merge into kwargs or store in info
        extra_dict = kwargs.pop("extra", {})

        # Create the info object
        info = self._info_class(
            key=unique_key,
            short_name=short_name,
            full_type_name=full_type_name,
            cls=cls,
            source=source,
            params=params_dict,
            tags=tags_set,
            extra=extra_dict,
            **kwargs,
        )

        # Store by the unique key
        self[unique_key] = info

        # Track class -> keys mapping (one class can have multiple registrations)
        if cls not in self._by_class:
            self._by_class[cls] = []
        self._by_class[cls].append(unique_key)

        return unique_key

    def unregister(self, cls_or_key: type | str) -> bool:
        """Remove a registration from the registry.

        Args:
            cls_or_key: Either the class itself, a full type name string, or a
                registered key. If a class or full type name is given and it has
                multiple registrations, ALL are removed. If a key is given, only
                that specific registration is removed.

        Returns:
            True if at least one registration was removed, False if none were found.

        Example:
            # Unregister by key (removes only that registration)
            registry.unregister("MyNegotiator#a1b2c3d4")

            # Unregister by class (removes ALL registrations of this class)
            registry.unregister(MyNegotiator)

            # Unregister by full type name (removes ALL registrations)
            registry.unregister("mymodule.MyNegotiator")
        """
        if isinstance(cls_or_key, str):
            # First check if it's a direct key (contains '#')
            if "#" in cls_or_key:
                info = self.get(cls_or_key)
                if info is None:
                    return False
                key = cls_or_key
                cls = info.cls

                # Remove from main dict
                del self[key]

                # Remove from class index
                if cls in self._by_class:
                    if key in self._by_class[cls]:
                        self._by_class[cls].remove(key)
                    if not self._by_class[cls]:
                        del self._by_class[cls]
                return True
            else:
                # It might be a full_type_name - try to resolve to class
                from negmas.helpers import get_class

                resolved_cls = get_class(cls_or_key)
                if resolved_cls is not None and resolved_cls in self._by_class:
                    # Unregister by resolved class
                    return self.unregister(resolved_cls)
                return False
        else:
            # Given a class, remove ALL registrations of this class
            cls = cls_or_key
            keys = self._by_class.get(cls)
            if not keys:
                return False

            # Remove all registrations
            for key in keys.copy():
                if key in self:
                    del self[key]
            del self._by_class[cls]
            return True

    def get_by_class(self, cls: type | str) -> T | None:
        """Get the first registration info for a class.

        Args:
            cls: The class to look up, or its full type name string.

        Returns:
            The first RegistryInfo for the class, or None if not registered.
            If the class has multiple registrations, returns the first one.
            Use get_all_by_class() to get all registrations.
        """
        if isinstance(cls, str):
            from negmas.helpers import get_class

            try:
                resolved_cls = get_class(cls)
            except (ModuleNotFoundError, AttributeError, ImportError):
                return None
            if resolved_cls is None:
                return None
            cls = resolved_cls

        keys = self._by_class.get(cls)
        if not keys:
            return None
        return self.get(keys[0])

    def get_all_by_class(self, cls: type | str) -> list[T]:
        """Get all registration info for a class.

        Args:
            cls: The class to look up, or its full type name string.

        Returns:
            A list of all RegistryInfo objects for this class, or empty list if not registered.
        """
        if isinstance(cls, str):
            from negmas.helpers import get_class

            try:
                resolved_cls = get_class(cls)
            except (ModuleNotFoundError, AttributeError, ImportError):
                return []
            if resolved_cls is None:
                return []
            cls = resolved_cls

        keys = self._by_class.get(cls, [])
        return [self[key] for key in keys if key in self]

    def is_registered(self, cls: type | str) -> bool:
        """Check if a class is registered.

        Args:
            cls: The class to check, or its full type name string.

        Returns:
            True if the class has at least one registration, False otherwise.
        """
        if isinstance(cls, str):
            from negmas.helpers import get_class

            try:
                resolved_cls = get_class(cls)
            except (ModuleNotFoundError, AttributeError, ImportError):
                return False
            if resolved_cls is None:
                return False
            cls = resolved_cls

        return cls in self._by_class and len(self._by_class[cls]) > 0

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
        """List all registered keys.

        Returns:
            A list of all registered keys.
        """
        return list(self.keys())

    def get_by_short_name(self, short_name: str) -> list[T]:
        """Get all registrations with a given short name.

        Since keys are now unique (with UUID suffix), multiple registrations
        can have the same short_name. This method returns all of them.

        Args:
            short_name: The short name to look up.

        Returns:
            A list of RegistryInfo objects with that short name (may be empty).
        """
        return [info for info in self.values() if info.short_name == short_name]

    def get_class(self, key_or_name: str) -> type | None:
        """Get the class for a registered key or name.

        Args:
            key_or_name: The unique key (e.g., 'MyNegotiator#a1b2c3d4'),
                short name (e.g., 'MyNegotiator'), or full type name
                (e.g., 'mymodule.MyNegotiator').

        Returns:
            The class, or None if not found. If multiple registrations match
            a short name, returns the first one found.
        """
        # First try direct key lookup
        info = self.get(key_or_name)
        if info is not None:
            return info.cls

        # Then try by short_name
        matches = self.get_by_short_name(key_or_name)
        if matches:
            return matches[0].cls

        # Finally try by full_type_name
        for info in self.values():
            if info.full_type_name == key_or_name:
                return info.cls

        return None

    def create(self, key_or_name: str, **override_params) -> Any:
        """Create an instance of a registered class.

        This method instantiates a class using its stored params, merged with
        any override parameters provided.

        Args:
            key_or_name: The unique key (e.g., 'MyNegotiator#a1b2c3d4'),
                short name (e.g., 'MyNegotiator'), or full type name.
            **override_params: Parameters to override the stored params.

        Returns:
            An instance of the registered class.

        Raises:
            KeyError: If no registration is found for the given key or name.
            TypeError: If the class cannot be instantiated with the params.

        Example:
            # Register with default params
            key = registry.register(MyNegotiator, params={'alpha': 0.5})

            # Create with stored params
            neg1 = registry.create(key)  # uses alpha=0.5

            # Create with overridden params
            neg2 = registry.create(key, alpha=0.9)  # uses alpha=0.9
        """
        # First try direct key lookup
        info = self.get(key_or_name)

        # Then try by short_name
        if info is None:
            matches = self.get_by_short_name(key_or_name)
            if matches:
                info = matches[0]

        # Finally try by full_type_name
        if info is None:
            for item in self.values():
                if item.full_type_name == key_or_name:
                    info = item
                    break

        if info is None:
            raise KeyError(f"No registration found for: {key_or_name}")

        # Merge stored params with overrides
        merged_params = {**info.params, **override_params}

        # Create and return the instance
        return info.cls(**merged_params)

    def register_many(self, registrations: list[dict[str, Any]]) -> list[str]:
        """Register multiple classes in a single call.

        Args:
            registrations: A list of dictionaries, each containing the arguments
                for a single registration. Each dict must have a 'cls' key with
                the class to register. Other keys are passed to register().

        Returns:
            A list of unique keys assigned to the registrations.

        Example:
            keys = registry.register_many([
                {'cls': MyNegotiator, 'short_name': 'my_neg', 'source': 'mylib'},
                {'cls': OtherNegotiator, 'params': {'alpha': 0.5}},
            ])
        """
        keys = []
        for reg in registrations:
            reg_copy = dict(reg)
            cls = reg_copy.pop("cls")
            key = self.register(cls, **reg_copy)
            keys.append(key)
        return keys

    def unregister_many(self, keys_or_classes: list[str | type]) -> int:
        """Remove multiple registrations in a single call.

        Args:
            keys_or_classes: A list of keys or classes to unregister.

        Returns:
            The number of registrations that were removed.

        Example:
            count = registry.unregister_many([
                'MyNegotiator#a1b2c3d4',
                OtherNegotiator,  # removes all registrations of this class
            ])
        """
        count = 0
        for item in keys_or_classes:
            if self.unregister(item):
                count += 1
        return count


def _matches_numeric_filter(
    value: int | float | None,
    filter_value: int | float | tuple[int | float | None, int | float | None],
) -> bool:
    """Check if a value matches a numeric filter.

    Args:
        value: The value to check (can be None).
        filter_value: Either an exact value to match, or a (min, max) tuple for range.
            In the tuple, None means unbounded on that end.

    Returns:
        True if the value matches the filter, False otherwise.
        Returns False if value is None (unknown values don't match).

    Examples:
        _matches_numeric_filter(100, 100)  # True (exact match)
        _matches_numeric_filter(100, 50)   # False
        _matches_numeric_filter(100, (50, 150))  # True (in range)
        _matches_numeric_filter(100, (None, 150))  # True (no min)
        _matches_numeric_filter(100, (50, None))   # True (no max)
        _matches_numeric_filter(None, 100)  # False (unknown value)
    """
    if value is None:
        return False

    if isinstance(filter_value, tuple):
        min_val, max_val = filter_value
        if min_val is not None and value < min_val:
            return False
        if max_val is not None and value > max_val:
            return False
        return True
    else:
        return value == filter_value


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
        source: str = "unknown",
        tags: set[str] | list[str] | tuple[str, ...] | None = None,
        n_outcomes: int | None = None,
        n_negotiators: int | None = None,
        opposition_level: float | None = None,
        # Deprecated params - converted to tags
        normalized: bool | None = None,
        anac: bool | None = None,
        has_stats: bool = False,
        has_plot: bool = False,
        **extra,
    ) -> ScenarioInfo:
        """Register a scenario in the registry.

        Args:
            path: The path to the scenario file or folder.
            name: A short name for the scenario. If None, uses the file/folder name.
            source: The origin of this registration ('negmas' for built-in, library
                name for external, or 'unknown' if not specified).
            tags: Optional set of tags for categorization and filtering.
            n_outcomes: The number of possible outcomes.
            n_negotiators: The number of negotiators in the scenario.
            opposition_level: The opposition level between negotiators (0-1).
            normalized: DEPRECATED. Use tags={'normalized'} instead.
            anac: DEPRECATED. Use tags={'anac'} instead.
            has_stats: DEPRECATED. Use tags={'has-stats'} instead.
            has_plot: DEPRECATED. Use tags={'has-plot'} instead.
            **extra: Additional properties to store.

        Returns:
            The ScenarioInfo object that was created, or the existing one if already registered.

        Note:
            If the path is already registered, returns the existing registration without
            modifying it. Registrations never overwrite existing entries.
        """
        import warnings

        path = Path(path).resolve()

        # Use the path string as the primary key
        primary_key = str(path)

        # Check if this path is already registered
        if primary_key in self:
            # Already registered - return existing info
            return self[primary_key]

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

        # Add file tag if it's a single file
        if is_file:
            auto_tags.add("file")

        # Handle deprecated boolean params -> convert to tags
        if normalized is not None:
            warnings.warn(
                "Parameter 'normalized' is deprecated. Use tags={'normalized'} instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if normalized:
                auto_tags.add("normalized")

        if anac is not None:
            warnings.warn(
                "Parameter 'anac' is deprecated. Use tags={'anac'} instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if anac:
                auto_tags.add("anac")

        if has_stats:
            warnings.warn(
                "Parameter 'has_stats' is deprecated. Use tags={'has-stats'} instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            auto_tags.add("has-stats")

        if has_plot:
            warnings.warn(
                "Parameter 'has_plot' is deprecated. Use tags={'has-plot'} instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            auto_tags.add("has-plot")

        # Combine with provided tags
        tags_set = set(tags) if tags is not None else set()
        all_tags = auto_tags | tags_set

        # Try to load pre-computed info from _info.yaml
        info_file = path / "_info.yaml" if path.is_dir() else None
        if info_file is not None and info_file.exists():
            try:
                from negmas.inout import load

                info_data = load(info_file)
                if n_outcomes is None and "n_outcomes" in info_data:
                    n_outcomes = info_data["n_outcomes"]
                if n_negotiators is None and "n_negotiators" in info_data:
                    n_negotiators = info_data["n_negotiators"]
                if opposition_level is None and "opposition_level" in info_data:
                    opposition_level = info_data["opposition_level"]
                # Also check for format in info file
                if "format" in info_data and info_data["format"] != "unknown":
                    all_tags.add(info_data["format"])
            except Exception:
                pass  # Ignore errors reading info file

        # Create the info object
        info = ScenarioInfo(
            name=name,
            path=path,
            source=source,
            tags=all_tags,
            n_outcomes=n_outcomes,
            n_negotiators=n_negotiators,
            opposition_level=opposition_level,
            extra=extra,
        )

        # Store by path (primary key is always unique since it's the resolved path)
        self[primary_key] = info

        # Index by name for quick lookup
        if name not in self._by_name:
            self._by_name[name] = []
        if primary_key not in self._by_name[name]:
            self._by_name[name].append(primary_key)

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
        n_outcomes: int | tuple[int | None, int | None] | None = None,
        n_negotiators: int | tuple[int | None, int | None] | None = None,
        opposition_level: float | tuple[float | None, float | None] | None = None,
        # Deprecated params - use tags instead
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
            n_outcomes: Filter by number of outcomes. Can be exact value or (min, max) tuple.
            n_negotiators: Filter by number of negotiators. Can be exact value or (min, max) tuple.
            opposition_level: Filter by opposition level. Can be exact value or (min, max) tuple.
            format: DEPRECATED. Use tags={'xml'} or tags={'json'} instead.
            anac: DEPRECATED. Use tags={'anac'} instead.
            normalized: DEPRECATED. Use tags={'normalized'} instead.
            file: DEPRECATED. Use tags={'file'} instead.
            **criteria: Additional attribute-value pairs to match exactly.

        Returns:
            A dictionary of matching entries (path -> info).
        """
        import warnings

        # Handle deprecated params
        if format is not None:
            warnings.warn(
                f"Parameter 'format' is deprecated. Use tags={{'{format}'}} instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            # Convert to tag filter
            if tags is None:
                tags = {format}
            else:
                tags = set(tags) | {format}

        if anac is not None:
            warnings.warn(
                "Parameter 'anac' is deprecated. Use tags={'anac'} instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if anac:
                if tags is None:
                    tags = {"anac"}
                else:
                    tags = set(tags) | {"anac"}

        if normalized is not None:
            warnings.warn(
                "Parameter 'normalized' is deprecated. Use tags={'normalized'} instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if normalized:
                if tags is None:
                    tags = {"normalized"}
                else:
                    tags = set(tags) | {"normalized"}

        if file is not None:
            warnings.warn(
                "Parameter 'file' is deprecated. Use tags={'file'} instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if file:
                if tags is None:
                    tags = {"file"}
                else:
                    tags = set(tags) | {"file"}

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

            # Check numeric filters
            if n_outcomes is not None:
                if not _matches_numeric_filter(info.n_outcomes, n_outcomes):
                    continue

            if n_negotiators is not None:
                if not _matches_numeric_filter(info.n_negotiators, n_negotiators):
                    continue

            if opposition_level is not None:
                if not _matches_numeric_filter(info.opposition_level, opposition_level):
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

    def register_many(self, registrations: list[dict[str, Any]]) -> list[ScenarioInfo]:
        """Register multiple scenarios in a single call.

        Args:
            registrations: A list of dictionaries, each containing the arguments
                for a single registration. Each dict must have a 'path' key with
                the path to the scenario. Other keys are passed to register().

        Returns:
            A list of ScenarioInfo objects for the registered scenarios.

        Example:
            infos = registry.register_many([
                {'path': '/path/to/scenario1', 'source': 'mylib', 'tags': {'custom'}},
                {'path': '/path/to/scenario2', 'n_negotiators': 2},
            ])
        """
        results = []
        for reg in registrations:
            reg_copy = dict(reg)
            path = reg_copy.pop("path")
            info = self.register(path, **reg_copy)
            results.append(info)
        return results

    def unregister_many(self, paths_or_names: list[str | Path]) -> int:
        """Remove multiple scenarios in a single call.

        Args:
            paths_or_names: A list of paths or names to unregister.
                If a name is given and multiple scenarios have that name,
                all are removed.

        Returns:
            The number of scenarios that were removed.

        Example:
            count = registry.unregister_many([
                '/path/to/scenario1',
                'MyScenario',  # removes all with this name
            ])
        """
        count = 0
        for item in paths_or_names:
            if self.unregister(item):
                count += 1
        return count

    def load(self, key_or_name: str | Path) -> Any:
        """Load a registered scenario.

        This method loads the scenario from disk using its registered path.

        Args:
            key_or_name: The path (as registered) or scenario name.
                If a name is given and multiple scenarios have that name,
                loads the first one found.

        Returns:
            The loaded Scenario object.

        Raises:
            KeyError: If no registration is found for the given key or name.
            Exception: If the scenario cannot be loaded.

        Example:
            # Load by path (key)
            scenario = registry.load('/path/to/scenario')

            # Load by name
            scenario = registry.load('Laptop')
        """
        from negmas.inout import Scenario

        # Convert Path to string for lookup
        if isinstance(key_or_name, Path):
            key_str = str(key_or_name.resolve())
        else:
            key_str = key_or_name

        # First try direct key lookup (path)
        info = self.get(key_str)

        # Then try by name
        if info is None:
            infos = self.get_by_name(key_str)
            if infos:
                info = infos[0]

        if info is None:
            raise KeyError(f"No scenario found for: {key_or_name}")

        # Load and return the scenario
        return Scenario.load(info.path)


# Global registries
mechanism_registry: Registry[MechanismInfo] = Registry(MechanismInfo)
negotiator_registry: Registry[NegotiatorInfo] = Registry(NegotiatorInfo)
component_registry: Registry[ComponentInfo] = Registry(ComponentInfo)
scenario_registry: ScenarioRegistry = ScenarioRegistry()


def register_mechanism(
    cls: type | None = None,
    *,
    short_name: str | None = None,
    source: str = "unknown",
    params: dict[str, Any] | None = None,
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
        source: The origin of this registration ('negmas' for built-in, library
            name for external, or 'unknown' if not specified).
        params: Default constructor parameters for creating instances.
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
            source=source,
            params=params,
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
    source: str = "unknown",
    params: dict[str, Any] | None = None,
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
        source: The origin of this registration ('negmas' for built-in, library
            name for external, or 'unknown' if not specified).
        params: Default constructor parameters for creating instances.
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
            source=source,
            params=params,
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
    source: str = "unknown",
    params: dict[str, Any] | None = None,
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
        source: The origin of this registration ('negmas' for built-in, library
            name for external, or 'unknown' if not specified).
        params: Default constructor parameters for creating instances.
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
            source=source,
            params=params,
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
    source: str = "unknown",
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
        source: The origin of this registration ('negmas' for built-in, library
            name for external, or 'unknown' if not specified).
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
        source=source,
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


# Default registry save/load path
DEFAULT_REGISTRY_PATH = Path.home() / "negmas" / "registry"


def save_registry(
    path: str | Path | None = None,
    *,
    include_mechanisms: bool = True,
    include_negotiators: bool = True,
    include_components: bool = True,
    include_scenarios: bool = True,
) -> Path:
    """Save all registries to a folder.

    This function serializes the global registries (mechanisms, negotiators,
    components, and scenarios) to JSON files in the specified folder.

    Note: Only metadata is saved. Class registrations store the full type name,
    so the classes must be importable when loading. Scenario registrations store
    the path, so the scenario files must exist when loading.

    Args:
        path: The folder to save to. Defaults to ~/negmas/registry.
        include_mechanisms: Whether to save the mechanism registry.
        include_negotiators: Whether to save the negotiator registry.
        include_components: Whether to save the component registry.
        include_scenarios: Whether to save the scenario registry.

    Returns:
        The path where the registries were saved.

    Example:
        # Save to default location
        save_registry()

        # Save to custom location
        save_registry("/path/to/my/registry")

        # Save only negotiators and scenarios
        save_registry(include_mechanisms=False, include_components=False)
    """
    import json

    if path is None:
        path = DEFAULT_REGISTRY_PATH
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    def _serialize_registry_info(info: RegistryInfo) -> dict[str, Any]:
        """Serialize a RegistryInfo to a dictionary."""
        return {
            "key": info.key,
            "short_name": info.short_name,
            "full_type_name": info.full_type_name,
            "source": info.source,
            "params": info.params,
            "tags": list(info.tags),
            "extra": info.extra,
        }

    def _serialize_mechanism_info(info: MechanismInfo) -> dict[str, Any]:
        """Serialize a MechanismInfo to a dictionary."""
        data = _serialize_registry_info(info)
        # Note: requires_deadline is now a tag, not a field
        return data

    def _serialize_negotiator_info(info: NegotiatorInfo) -> dict[str, Any]:
        """Serialize a NegotiatorInfo to a dictionary."""
        data = _serialize_registry_info(info)
        # Note: All boolean properties are now tags, not fields
        return data

    def _serialize_component_info(info: ComponentInfo) -> dict[str, Any]:
        """Serialize a ComponentInfo to a dictionary."""
        data = _serialize_registry_info(info)
        data["component_type"] = info.component_type
        return data

    def _serialize_scenario_info(info: ScenarioInfo) -> dict[str, Any]:
        """Serialize a ScenarioInfo to a dictionary."""
        return {
            "name": info.name,
            "path": str(info.path),
            "source": info.source,
            "tags": list(info.tags),
            "n_outcomes": info.n_outcomes,
            "n_negotiators": info.n_negotiators,
            "opposition_level": info.opposition_level,
            "extra": info.extra,
        }

    if include_mechanisms:
        mechanisms_data = {
            key: _serialize_mechanism_info(info)
            for key, info in mechanism_registry.items()
        }
        with open(path / "mechanisms.json", "w") as f:
            json.dump(mechanisms_data, f, indent=2)

    if include_negotiators:
        negotiators_data = {
            key: _serialize_negotiator_info(info)
            for key, info in negotiator_registry.items()
        }
        with open(path / "negotiators.json", "w") as f:
            json.dump(negotiators_data, f, indent=2)

    if include_components:
        components_data = {
            key: _serialize_component_info(info)
            for key, info in component_registry.items()
        }
        with open(path / "components.json", "w") as f:
            json.dump(components_data, f, indent=2)

    if include_scenarios:
        scenarios_data = {
            key: _serialize_scenario_info(info)
            for key, info in scenario_registry.items()
        }
        with open(path / "scenarios.json", "w") as f:
            json.dump(scenarios_data, f, indent=2)

    return path


def load_registry(
    path: str | Path | None = None,
    *,
    include_mechanisms: bool = True,
    include_negotiators: bool = True,
    include_components: bool = True,
    include_scenarios: bool = True,
    clear_existing: bool = False,
) -> dict[str, int]:
    """Load registries from a folder.

    This function loads previously saved registries from JSON files and
    adds them to the global registries.

    Note: Class registrations require that the classes are importable.
    If a class cannot be imported, that registration is skipped with a warning.

    Args:
        path: The folder to load from. Defaults to ~/negmas/registry.
        include_mechanisms: Whether to load the mechanism registry.
        include_negotiators: Whether to load the negotiator registry.
        include_components: Whether to load the component registry.
        include_scenarios: Whether to load the scenario registry.
        clear_existing: If True, clear existing registrations before loading.

    Returns:
        A dictionary with counts of loaded items:
        {'mechanisms': N, 'negotiators': N, 'components': N, 'scenarios': N}

    Raises:
        FileNotFoundError: If the path does not exist.

    Example:
        # Load from default location
        counts = load_registry()
        print(f"Loaded {counts['negotiators']} negotiators")

        # Load from custom location, replacing existing
        load_registry("/path/to/my/registry", clear_existing=True)

        # Load only scenarios
        load_registry(
            include_mechanisms=False,
            include_negotiators=False,
            include_components=False,
        )
    """
    import json
    import warnings

    from negmas.helpers import get_class

    if path is None:
        path = DEFAULT_REGISTRY_PATH
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Registry path does not exist: {path}")

    counts: dict[str, int] = {
        "mechanisms": 0,
        "negotiators": 0,
        "components": 0,
        "scenarios": 0,
    }

    def _load_class(full_type_name: str) -> type | None:
        """Try to load a class from its full type name."""
        try:
            return get_class(full_type_name)
        except (ModuleNotFoundError, AttributeError, ImportError) as e:
            warnings.warn(f"Could not load class {full_type_name}: {e}")
            return None

    if include_mechanisms:
        mechanisms_file = path / "mechanisms.json"
        if mechanisms_file.exists():
            if clear_existing:
                mechanism_registry.clear()
                mechanism_registry._by_class.clear()

            with open(mechanisms_file) as f:
                mechanisms_data = json.load(f)

            for key, data in mechanisms_data.items():
                cls = _load_class(data["full_type_name"])
                if cls is None:
                    continue

                # Check if already registered with this key
                if key in mechanism_registry:
                    continue

                mechanism_registry.register(
                    cls,
                    short_name=data["short_name"],
                    source=data.get("source", "unknown"),
                    params=data.get("params", {}),
                    tags=set(data.get("tags", [])),
                    extra=data.get("extra", {}),
                )
                counts["mechanisms"] += 1

    if include_negotiators:
        negotiators_file = path / "negotiators.json"
        if negotiators_file.exists():
            if clear_existing:
                negotiator_registry.clear()
                negotiator_registry._by_class.clear()

            with open(negotiators_file) as f:
                negotiators_data = json.load(f)

            for key, data in negotiators_data.items():
                cls = _load_class(data["full_type_name"])
                if cls is None:
                    continue

                if key in negotiator_registry:
                    continue

                negotiator_registry.register(
                    cls,
                    short_name=data["short_name"],
                    source=data.get("source", "unknown"),
                    params=data.get("params", {}),
                    tags=set(data.get("tags", [])),
                    extra=data.get("extra", {}),
                )
                counts["negotiators"] += 1

    if include_components:
        components_file = path / "components.json"
        if components_file.exists():
            if clear_existing:
                component_registry.clear()
                component_registry._by_class.clear()

            with open(components_file) as f:
                components_data = json.load(f)

            for key, data in components_data.items():
                cls = _load_class(data["full_type_name"])
                if cls is None:
                    continue

                if key in component_registry:
                    continue

                component_registry.register(
                    cls,
                    short_name=data["short_name"],
                    source=data.get("source", "unknown"),
                    params=data.get("params", {}),
                    tags=set(data.get("tags", [])),
                    component_type=data.get("component_type", "generic"),
                    extra=data.get("extra", {}),
                )
                counts["components"] += 1

    if include_scenarios:
        scenarios_file = path / "scenarios.json"
        if scenarios_file.exists():
            if clear_existing:
                scenario_registry.clear()
                scenario_registry._by_name.clear()

            with open(scenarios_file) as f:
                scenarios_data = json.load(f)

            for key, data in scenarios_data.items():
                scenario_path = Path(data["path"])

                # Check if path exists (optional - scenarios might be on different machines)
                # We register anyway to preserve the metadata

                if key in scenario_registry:
                    continue

                scenario_registry.register(
                    path=scenario_path,
                    name=data["name"],
                    source=data.get("source", "unknown"),
                    tags=set(data.get("tags", [])),
                    n_outcomes=data.get("n_outcomes"),
                    n_negotiators=data.get("n_negotiators"),
                    opposition_level=data.get("opposition_level"),
                    **data.get("extra", {}),
                )
                counts["scenarios"] += 1

    return counts


def clear_registry(
    *,
    include_mechanisms: bool = True,
    include_negotiators: bool = True,
    include_components: bool = True,
    include_scenarios: bool = True,
) -> None:
    """Clear all global registries.

    This removes all registrations from the specified registries.

    Args:
        include_mechanisms: Whether to clear the mechanism registry.
        include_negotiators: Whether to clear the negotiator registry.
        include_components: Whether to clear the component registry.
        include_scenarios: Whether to clear the scenario registry.

    Example:
        # Clear all registries
        clear_registry()

        # Clear only negotiators
        clear_registry(
            include_mechanisms=False,
            include_components=False,
            include_scenarios=False,
        )
    """
    if include_mechanisms:
        mechanism_registry.clear()
        mechanism_registry._by_class.clear()

    if include_negotiators:
        negotiator_registry.clear()
        negotiator_registry._by_class.clear()

    if include_components:
        component_registry.clear()
        component_registry._by_class.clear()

    if include_scenarios:
        scenario_registry.clear()
        scenario_registry._by_name.clear()


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
