"""Registration system for mechanisms, negotiators, and components.

This module provides a registration system that allows mechanisms, negotiators,
and components to be automatically registered and discovered. This is particularly
useful for:
- External libraries that want to register their own implementations
- Discovery of available implementations
- Querying implementations by their properties

Example:
    # Registering a custom negotiator
    @register_negotiator(
        short_name="my_negotiator",
        bilateral_only=True,
        learns=True,
    )
    class MyNegotiator(SAONegotiator):
        pass

    # Discovering all registered negotiators
    from negmas import negotiator_registry
    for name, info in negotiator_registry.items():
        print(f"{name}: {info}")

    # Querying by properties
    bilateral_negotiators = negotiator_registry.query(bilateral_only=True)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    pass

__all__ = [
    "RegistryInfo",
    "Registry",
    "MechanismInfo",
    "NegotiatorInfo",
    "ComponentInfo",
    "mechanism_registry",
    "negotiator_registry",
    "component_registry",
    "register_mechanism",
    "register_negotiator",
    "register_component",
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
        extra: Additional key-value pairs for custom properties.
    """

    short_name: str
    full_type_name: str
    cls: type
    extra: dict[str, Any] = field(default_factory=dict)


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

    def register(self, cls: type, short_name: str | None = None, **kwargs) -> None:
        """Register a class in the registry.

        Args:
            cls: The class to register.
            short_name: A short name for the class. If None, uses the class name.
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

        # Create the info object
        info = self._info_class(
            short_name=short_name, full_type_name=full_type_name, cls=cls, **kwargs
        )

        # Store by short name only
        self[short_name] = info
        self._by_class[cls] = short_name

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

    def query(self, **criteria) -> dict[str, T]:
        """Query the registry for classes matching the given criteria.

        Args:
            **criteria: Attribute-value pairs to match.

        Returns:
            A dictionary of matching entries (short_name -> info).
        """
        results = {}
        for key, info in self.items():
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


# Global registries
mechanism_registry: Registry[MechanismInfo] = Registry(MechanismInfo)
negotiator_registry: Registry[NegotiatorInfo] = Registry(NegotiatorInfo)
component_registry: Registry[ComponentInfo] = Registry(ComponentInfo)


def register_mechanism(
    cls: type | None = None,
    *,
    short_name: str | None = None,
    requires_deadline: bool = True,
    **extra,
):
    """Decorator to register a mechanism class.

    Can be used with or without arguments:
        @register_mechanism
        class MyMechanism(Mechanism):
            pass

        @register_mechanism(short_name="my_mech", requires_deadline=False)
        class MyMechanism(Mechanism):
            pass

    Args:
        cls: The class to register (when used without parentheses).
        short_name: A short name for the mechanism.
        requires_deadline: Whether the mechanism requires a deadline.
        **extra: Additional properties to store.

    Returns:
        The decorated class (unchanged).
    """

    def decorator(cls: type) -> type:
        mechanism_registry.register(
            cls, short_name=short_name, requires_deadline=requires_deadline, extra=extra
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
    **extra,
):
    """Decorator to register a negotiator class.

    Can be used with or without arguments:
        @register_negotiator
        class MyNegotiator(Negotiator):
            pass

        @register_negotiator(bilateral_only=True, learns=True)
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
        **extra: Additional properties to store.

    Returns:
        The decorated class (unchanged).
    """

    def decorator(cls: type) -> type:
        negotiator_registry.register(
            cls,
            short_name=short_name,
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
    **extra,
):
    """Decorator to register a component class.

    Can be used with or without arguments:
        @register_component
        class MyComponent(Component):
            pass

        @register_component(component_type="acceptance")
        class MyAcceptancePolicy(AcceptancePolicy):
            pass

    Args:
        cls: The class to register (when used without parentheses).
        short_name: A short name for the component.
        component_type: The type of component (e.g., 'acceptance', 'offering', 'model').
        **extra: Additional properties to store.

    Returns:
        The decorated class (unchanged).
    """

    def decorator(cls: type) -> type:
        component_registry.register(
            cls, short_name=short_name, component_type=component_type, extra=extra
        )
        return cls

    if cls is not None:
        return decorator(cls)
    return decorator


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
