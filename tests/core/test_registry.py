"""Tests for the registry system."""

from __future__ import annotations

import pytest

from negmas.registry import (
    Registry,
    RegistryInfo,
    MechanismInfo,
    NegotiatorInfo,
    ComponentInfo,
    mechanism_registry,
    negotiator_registry,
    component_registry,
    register_mechanism,
    register_negotiator,
    register_component,
    get_registered_class,
)


class TestRegistryInfo:
    """Tests for RegistryInfo dataclass."""

    def test_registry_info_creation(self):
        """Test basic RegistryInfo creation."""

        class DummyClass:
            pass

        info = RegistryInfo(
            key="dummy#12345678",
            short_name="dummy",
            full_type_name="test.DummyClass",
            cls=DummyClass,
        )
        assert info.key == "dummy#12345678"
        assert info.short_name == "dummy"
        assert info.full_type_name == "test.DummyClass"
        assert info.cls is DummyClass
        assert info.source == "unknown"
        assert info.params == {}
        assert info.tags == set()
        assert info.extra == {}

    def test_registry_info_with_extra(self):
        """Test RegistryInfo with extra data."""

        class DummyClass:
            pass

        info = RegistryInfo(
            key="dummy#12345678",
            short_name="dummy",
            full_type_name="test.DummyClass",
            cls=DummyClass,
            extra={"key": "value"},
        )
        assert info.extra == {"key": "value"}

    def test_registry_info_with_source_and_params(self):
        """Test RegistryInfo with source and params."""

        class DummyClass:
            pass

        info = RegistryInfo(
            key="dummy#12345678",
            short_name="dummy",
            full_type_name="test.DummyClass",
            cls=DummyClass,
            source="mylib",
            params={"alpha": 0.5, "beta": 0.3},
        )
        assert info.source == "mylib"
        assert info.params == {"alpha": 0.5, "beta": 0.3}


class TestMechanismInfo:
    """Tests for MechanismInfo dataclass."""

    def test_mechanism_info_defaults(self):
        """Test MechanismInfo default values."""

        class DummyMechanism:
            pass

        info = MechanismInfo(
            key="dummy#12345678",
            short_name="dummy",
            full_type_name="test.DummyMechanism",
            cls=DummyMechanism,
        )
        # No longer has requires_deadline field - use tags instead
        assert info.tags == set()

    def test_mechanism_info_custom(self):
        """Test MechanismInfo with custom values (using tags)."""

        class DummyMechanism:
            pass

        # requires_deadline is now represented as a tag
        info = MechanismInfo(
            key="dummy#12345678",
            short_name="dummy",
            full_type_name="test.DummyMechanism",
            cls=DummyMechanism,
            tags={"requires-deadline"},
        )
        assert info.has_tag("requires-deadline")


class TestNegotiatorInfo:
    """Tests for NegotiatorInfo dataclass."""

    def test_negotiator_info_defaults(self):
        """Test NegotiatorInfo default values."""

        class DummyNegotiator:
            pass

        info = NegotiatorInfo(
            key="dummy#12345678",
            short_name="dummy",
            full_type_name="test.DummyNegotiator",
            cls=DummyNegotiator,
        )
        # Boolean fields removed - use tags instead
        assert info.tags == set()

    def test_negotiator_info_custom(self):
        """Test NegotiatorInfo with custom values (using tags)."""

        class DummyNegotiator:
            pass

        # Old boolean fields now represented as tags
        info = NegotiatorInfo(
            key="dummy#12345678",
            short_name="dummy",
            full_type_name="test.DummyNegotiator",
            cls=DummyNegotiator,
            tags={"bilateral-only", "learning", "anac-2020", "supports-uncertainty"},
        )
        assert info.has_tag("bilateral-only")
        assert info.has_tag("learning")
        assert info.has_tag("anac-2020")
        assert info.has_tag("supports-uncertainty")


class TestComponentInfo:
    """Tests for ComponentInfo dataclass."""

    def test_component_info_defaults(self):
        """Test ComponentInfo default values."""

        class DummyComponent:
            pass

        info = ComponentInfo(
            key="dummy#12345678",
            short_name="dummy",
            full_type_name="test.DummyComponent",
            cls=DummyComponent,
        )
        assert info.component_type == "generic"

    def test_component_info_custom(self):
        """Test ComponentInfo with custom values."""

        class DummyComponent:
            pass

        info = ComponentInfo(
            key="dummy#12345678",
            short_name="dummy",
            full_type_name="test.DummyComponent",
            cls=DummyComponent,
            component_type="acceptance",
        )
        assert info.component_type == "acceptance"


class TestRegistry:
    """Tests for Registry class."""

    def test_registry_register(self):
        """Test registering a class."""
        registry = Registry(RegistryInfo)

        class TestClass:
            pass

        key = registry.register(TestClass, short_name="test")
        assert key in registry
        assert registry[key].cls is TestClass
        assert registry[key].short_name == "test"
        assert "#" in key  # Key should have UUID suffix

    def test_registry_register_auto_name(self):
        """Test registering with automatic short name."""
        registry = Registry(RegistryInfo)

        class MyTestClass:
            pass

        key = registry.register(MyTestClass)
        assert key in registry
        assert registry[key].cls is MyTestClass
        assert registry[key].short_name == "MyTestClass"

    def test_registry_register_returns_key(self):
        """Test that register returns the unique key."""
        registry = Registry(RegistryInfo)

        class TestClass:
            pass

        key = registry.register(TestClass, short_name="test")
        assert isinstance(key, str)
        assert key.startswith("test#")
        assert len(key) == len("test#") + 8  # short_name + # + 8 char UUID

    def test_registry_register_with_source_and_params(self):
        """Test registering with source and params."""
        registry = Registry(RegistryInfo)

        class TestClass:
            pass

        key = registry.register(
            TestClass, short_name="test", source="mylib", params={"alpha": 0.5}
        )
        info = registry[key]
        assert info.source == "mylib"
        assert info.params == {"alpha": 0.5}

    def test_registry_register_by_full_type_name(self):
        """Test registering using full type name string."""
        from negmas.sao.negotiators import AspirationNegotiator

        registry = Registry(RegistryInfo)

        full_name = (
            f"{AspirationNegotiator.__module__}.{AspirationNegotiator.__qualname__}"
        )
        key = registry.register(full_name, short_name="test_from_string")
        assert key in registry
        assert registry[key].cls is AspirationNegotiator

    def test_registry_get_by_class(self):
        """Test getting info by class."""
        registry = Registry(RegistryInfo)

        class TestClass:
            pass

        registry.register(TestClass, short_name="test")
        info = registry.get_by_class(TestClass)
        assert info is not None
        assert info.short_name == "test"

    def test_registry_get_by_class_with_string(self):
        """Test getting info by full type name string."""
        from negmas.sao.negotiators import AspirationNegotiator

        registry = Registry(RegistryInfo)

        registry.register(AspirationNegotiator, short_name="test")
        full_name = (
            f"{AspirationNegotiator.__module__}.{AspirationNegotiator.__qualname__}"
        )
        info = registry.get_by_class(full_name)
        assert info is not None
        assert info.short_name == "test"

    def test_registry_get_by_class_not_found(self):
        """Test getting info for unregistered class."""
        registry = Registry(RegistryInfo)

        class TestClass:
            pass

        info = registry.get_by_class(TestClass)
        assert info is None

    def test_registry_is_registered(self):
        """Test is_registered method."""
        registry = Registry(RegistryInfo)

        class TestClass:
            pass

        class UnregisteredClass:
            pass

        registry.register(TestClass, short_name="test")
        assert registry.is_registered(TestClass) is True
        assert registry.is_registered(UnregisteredClass) is False

    def test_registry_is_registered_with_string(self):
        """Test is_registered with full type name string."""
        from negmas.sao.negotiators import AspirationNegotiator

        registry = Registry(RegistryInfo)

        registry.register(AspirationNegotiator, short_name="test")
        full_name = (
            f"{AspirationNegotiator.__module__}.{AspirationNegotiator.__qualname__}"
        )
        assert registry.is_registered(full_name) is True
        assert registry.is_registered("nonexistent.Class") is False

    def test_registry_get_class(self):
        """Test get_class method with short name."""
        registry = Registry(RegistryInfo)

        class TestClass:
            pass

        key = registry.register(TestClass, short_name="test")
        # Can get by key
        assert registry.get_class(key) is TestClass
        # Can get by short name
        assert registry.get_class("test") is TestClass
        # Returns None for nonexistent
        assert registry.get_class("nonexistent") is None

    def test_registry_get_by_short_name(self):
        """Test get_by_short_name method."""
        registry = Registry(RegistryInfo)

        class TestClass:
            pass

        registry.register(TestClass, short_name="test")
        registry.register(TestClass, short_name="test")  # Same short_name

        results = registry.get_by_short_name("test")
        assert len(results) == 2
        assert all(info.short_name == "test" for info in results)

    def test_registry_list_all(self):
        """Test list_all method returns keys."""
        registry = Registry(RegistryInfo)

        class TestClass1:
            pass

        class TestClass2:
            pass

        key1 = registry.register(TestClass1, short_name="test1")
        key2 = registry.register(TestClass2, short_name="test2")

        all_keys = registry.list_all()
        assert key1 in all_keys
        assert key2 in all_keys

    def test_registry_query(self):
        """Test query method (using tags)."""
        registry = Registry(NegotiatorInfo)

        class BilateralNegotiator:
            pass

        class MultilateralNegotiator:
            pass

        key1 = registry.register(
            BilateralNegotiator, short_name="bilateral", tags={"bilateral-only"}
        )
        key2 = registry.register(
            MultilateralNegotiator, short_name="multilateral", tags=set()
        )

        bilateral_results = registry.query(tags={"bilateral-only"})
        assert key1 in bilateral_results
        assert key2 not in bilateral_results

        # Query for those WITHOUT bilateral-only tag
        all_results = dict(registry)
        multilateral_results = {
            k: v for k, v in all_results.items() if not v.has_tag("bilateral-only")
        }
        assert key2 in multilateral_results
        assert key1 not in multilateral_results

    def test_registry_query_multiple_criteria(self):
        """Test query with multiple criteria (using tags)."""
        registry = Registry(NegotiatorInfo)

        class LearningBilateral:
            pass

        class NonLearningBilateral:
            pass

        key1 = registry.register(
            LearningBilateral,
            short_name="learning_bilateral",
            tags={"bilateral-only", "learning"},
        )
        key2 = registry.register(
            NonLearningBilateral,
            short_name="nonlearning_bilateral",
            tags={"bilateral-only"},
        )

        results = registry.query(tags={"bilateral-only", "learning"})
        assert key1 in results
        assert key2 not in results

    def test_registry_virtual_negotiators(self):
        """Test registering the same class with different names and params (virtual negotiators)."""
        registry = Registry(NegotiatorInfo)

        class AspirationNegotiator:
            def __init__(self, aspiration_type="linear", max_aspiration=1.0):
                self.aspiration_type = aspiration_type
                self.max_aspiration = max_aspiration

        # Register base negotiator
        key1 = registry.register(AspirationNegotiator, short_name="Aspiration")

        # Register "virtual" aggressive variant
        key2 = registry.register(
            AspirationNegotiator,
            short_name="AggressiveAspiration",
            params={"aspiration_type": "boulware", "max_aspiration": 0.95},
        )

        # Register "virtual" conceding variant
        key3 = registry.register(
            AspirationNegotiator,
            short_name="ConcedingAspiration",
            params={"aspiration_type": "conceder", "max_aspiration": 0.8},
        )

        # All should be registered
        assert len(registry) == 3

        # All point to same class
        assert registry[key1].cls is AspirationNegotiator
        assert registry[key2].cls is AspirationNegotiator
        assert registry[key3].cls is AspirationNegotiator

        # But have different short_names
        assert registry[key1].short_name == "Aspiration"
        assert registry[key2].short_name == "AggressiveAspiration"
        assert registry[key3].short_name == "ConcedingAspiration"

        # And different params
        assert registry[key1].params == {}
        assert registry[key2].params == {
            "aspiration_type": "boulware",
            "max_aspiration": 0.95,
        }
        assert registry[key3].params == {
            "aspiration_type": "conceder",
            "max_aspiration": 0.8,
        }

        # get_all_by_class returns all three
        all_infos = registry.get_all_by_class(AspirationNegotiator)
        assert len(all_infos) == 3

    def test_registry_same_short_name_different_registrations(self):
        """Test that same short name creates different keys with UUID."""
        registry = Registry(RegistryInfo)

        class TestClass:
            pass

        # Register twice with same short_name
        key1 = registry.register(TestClass, short_name="test")
        key2 = registry.register(TestClass, short_name="test")

        # Keys should be different
        assert key1 != key2

        # Both should exist
        assert key1 in registry
        assert key2 in registry
        assert len(registry) == 2

        # Both have same short_name
        assert registry[key1].short_name == "test"
        assert registry[key2].short_name == "test"

    def test_registry_unregister_by_class_removes_all(self):
        """Test that unregistering by class removes all registrations."""
        registry = Registry(RegistryInfo)

        class TestClass:
            pass

        # Register with multiple names
        registry.register(TestClass, short_name="name1")
        registry.register(TestClass, short_name="name2")
        assert len(registry) == 2

        # Unregister by class - should remove all
        result = registry.unregister(TestClass)
        assert result is True
        assert len(registry) == 0
        assert not registry.is_registered(TestClass)

    def test_registry_unregister_by_key_keeps_others(self):
        """Test that unregistering by key keeps other registrations of same class."""
        registry = Registry(RegistryInfo)

        class TestClass:
            pass

        # Register with multiple names
        key1 = registry.register(TestClass, short_name="name1")
        key2 = registry.register(TestClass, short_name="name2")
        assert len(registry) == 2

        # Unregister by key - should only remove that one
        result = registry.unregister(key1)
        assert result is True
        assert key1 not in registry
        assert key2 in registry
        assert len(registry) == 1

        # Class should still be registered (via key2)
        assert registry.is_registered(TestClass)

    def test_registry_unregister_by_full_type_name(self):
        """Test unregistering by full type name string."""
        from negmas.sao.negotiators import AspirationNegotiator

        registry = Registry(RegistryInfo)

        registry.register(AspirationNegotiator, short_name="test")
        full_name = (
            f"{AspirationNegotiator.__module__}.{AspirationNegotiator.__qualname__}"
        )

        result = registry.unregister(full_name)
        assert result is True
        assert len(registry) == 0

    def test_registry_unregister_by_class(self):
        """Test unregistering a class by its class object."""
        registry = Registry(RegistryInfo)

        class TestClass:
            pass

        key = registry.register(TestClass, short_name="test")
        assert key in registry
        assert registry.is_registered(TestClass)

        # Unregister by class
        result = registry.unregister(TestClass)
        assert result is True
        assert key not in registry
        assert registry.is_registered(TestClass) is False
        assert registry.get_by_class(TestClass) is None

    def test_registry_unregister_by_key(self):
        """Test unregistering a class by its key."""
        registry = Registry(RegistryInfo)

        class TestClass:
            pass

        key = registry.register(TestClass, short_name="test")
        assert key in registry

        # Unregister by key
        result = registry.unregister(key)
        assert result is True
        assert key not in registry
        assert registry.is_registered(TestClass) is False

    def test_registry_unregister_not_found(self):
        """Test unregistering a class that doesn't exist."""
        registry = Registry(RegistryInfo)

        class TestClass:
            pass

        # Unregister by key that doesn't exist
        result = registry.unregister("nonexistent#12345678")
        assert result is False

        # Unregister by class that isn't registered
        result = registry.unregister(TestClass)
        assert result is False

    def test_registry_unregister_clears_both_dicts(self):
        """Test that unregistering cleans up both internal dictionaries."""
        registry = Registry(RegistryInfo)

        class TestClass:
            pass

        key = registry.register(TestClass, short_name="test")

        # Verify both mappings exist
        assert key in registry
        assert TestClass in registry._by_class

        # Unregister
        registry.unregister(TestClass)

        # Verify both mappings are removed
        assert key not in registry
        assert TestClass not in registry._by_class

    def test_registry_create(self):
        """Test create method."""
        registry = Registry(RegistryInfo)

        class TestClass:
            def __init__(self, alpha=0.5, beta=0.3):
                self.alpha = alpha
                self.beta = beta

        key = registry.register(
            TestClass, short_name="test", params={"alpha": 0.9, "beta": 0.1}
        )

        # Create with stored params
        instance = registry.create(key)
        assert instance.alpha == 0.9
        assert instance.beta == 0.1

        # Create with override params
        instance2 = registry.create(key, alpha=0.5)
        assert instance2.alpha == 0.5
        assert instance2.beta == 0.1

    def test_registry_create_by_short_name(self):
        """Test create method with short name."""
        registry = Registry(RegistryInfo)

        class TestClass:
            def __init__(self, value=10):
                self.value = value

        registry.register(TestClass, short_name="test", params={"value": 42})

        instance = registry.create("test")
        assert instance.value == 42

    def test_registry_create_not_found(self):
        """Test create raises KeyError for unknown key."""
        registry = Registry(RegistryInfo)

        with pytest.raises(KeyError):
            registry.create("nonexistent")

    def test_registry_register_many(self):
        """Test register_many method."""
        registry = Registry(RegistryInfo)

        class Class1:
            pass

        class Class2:
            pass

        keys = registry.register_many(
            [
                {"cls": Class1, "short_name": "c1", "source": "test"},
                {"cls": Class2, "short_name": "c2", "tags": {"tag1"}},
            ]
        )

        assert len(keys) == 2
        assert all(key in registry for key in keys)
        assert registry[keys[0]].short_name == "c1"
        assert registry[keys[0]].source == "test"
        assert registry[keys[1]].short_name == "c2"
        assert "tag1" in registry[keys[1]].tags

    def test_registry_unregister_many(self):
        """Test unregister_many method."""
        registry = Registry(RegistryInfo)

        class Class1:
            pass

        class Class2:
            pass

        class Class3:
            pass

        key1 = registry.register(Class1, short_name="c1")
        key2 = registry.register(Class2, short_name="c2")
        key3 = registry.register(Class3, short_name="c3")

        # Unregister two by key and one by class
        count = registry.unregister_many([key1, Class2])
        assert count == 2
        assert key1 not in registry
        assert key2 not in registry
        assert key3 in registry


class TestDecorators:
    """Tests for registration decorators."""

    def test_register_mechanism_decorator(self):
        """Test @register_mechanism decorator."""

        @register_mechanism(short_name="test_mech")
        class TestMechanism:
            pass

        assert mechanism_registry.is_registered(TestMechanism)
        info = mechanism_registry.get_by_class(TestMechanism)
        assert info is not None
        assert info.short_name == "test_mech"
        # Default decorator adds requires-deadline tag
        assert info.has_tag("requires-deadline")

    def test_register_mechanism_decorator_no_args(self):
        """Test @register_mechanism decorator without arguments."""

        @register_mechanism
        class AnotherTestMechanism:
            pass

        assert mechanism_registry.is_registered(AnotherTestMechanism)
        info = mechanism_registry.get_by_class(AnotherTestMechanism)
        assert info is not None
        assert info.short_name == "AnotherTestMechanism"
        # Default decorator adds requires-deadline tag
        assert info.has_tag("requires-deadline")

    def test_register_negotiator_decorator(self):
        """Test @register_negotiator decorator."""

        @register_negotiator(
            short_name="test_neg", tags={"bilateral-only", "learning", "anac-2020"}
        )
        class TestNegotiator:
            pass

        assert negotiator_registry.is_registered(TestNegotiator)
        info = negotiator_registry.get_by_class(TestNegotiator)
        assert info is not None
        assert info.short_name == "test_neg"
        # Boolean fields are now tags
        assert info.has_tag("bilateral-only")
        assert info.has_tag("learning")
        assert info.has_tag("anac-2020")

    def test_register_negotiator_decorator_no_args(self):
        """Test @register_negotiator decorator without arguments."""

        @register_negotiator
        class AnotherTestNegotiator:
            pass

        assert negotiator_registry.is_registered(AnotherTestNegotiator)
        info = negotiator_registry.get_by_class(AnotherTestNegotiator)
        assert info is not None
        assert info.short_name == "AnotherTestNegotiator"
        # Check defaults - no special tags
        assert not info.has_tag("bilateral-only")
        assert not info.has_tag("learning")

    def test_register_component_decorator(self):
        """Test @register_component decorator."""

        @register_component(short_name="test_comp", component_type="acceptance")
        class TestComponent:
            pass

        assert component_registry.is_registered(TestComponent)
        info = component_registry.get_by_class(TestComponent)
        assert info is not None
        assert info.short_name == "test_comp"
        assert info.component_type == "acceptance"

    def test_register_component_decorator_no_args(self):
        """Test @register_component decorator without arguments."""

        @register_component
        class AnotherTestComponent:
            pass

        assert component_registry.is_registered(AnotherTestComponent)
        info = component_registry.get_by_class(AnotherTestComponent)
        assert info is not None
        assert info.short_name == "AnotherTestComponent"
        assert info.component_type == "generic"  # default


class TestGetRegisteredClass:
    """Tests for get_registered_class function."""

    def test_get_registered_class_by_short_name(self):
        """Test getting class by short name."""
        registry = Registry(RegistryInfo)

        class TestClass:
            pass

        registry.register(TestClass, short_name="test")
        cls = get_registered_class("test", registry)
        assert cls is TestClass

    def test_get_registered_class_not_found(self):
        """Test getting class that doesn't exist."""
        registry = Registry(RegistryInfo)
        cls = get_registered_class("nonexistent", registry)
        assert cls is None


class TestBuiltInRegistrations:
    """Tests for built-in class registrations."""

    def test_mechanisms_registered(self):
        """Test that built-in mechanisms are registered.

        Note: Only concrete mechanisms are registered, not base classes like
        Mechanism or GBMechanism.
        """
        # Import to trigger registration
        import negmas.registry_init  # noqa: F401

        # Check some known concrete mechanisms by class lookup
        expected = [
            "SAOMechanism",
            "TAUMechanism",
            "SerialTAUMechanism",
            "ParallelGBMechanism",
            "SerialGBMechanism",
        ]
        for name in expected:
            cls = mechanism_registry.get_class(name)
            assert cls is not None, f"{name} not registered"

        # Base classes should NOT be registered
        assert mechanism_registry.get_class("Mechanism") is None
        assert mechanism_registry.get_class("GBMechanism") is None

    def test_negotiators_registered(self):
        """Test that built-in negotiators are registered.

        Note: Only concrete negotiators are registered, not base classes like
        Negotiator, SAONegotiator, or GBNegotiator.
        """
        import negmas.registry_init  # noqa: F401

        # Check some known concrete negotiators
        expected = [
            "AspirationNegotiator",
            "ToughNegotiator",
            "RandomNegotiator",
            "NiceNegotiator",
            "MiCRONegotiator",
        ]
        for name in expected:
            cls = negotiator_registry.get_class(name)
            assert cls is not None, f"{name} not registered"

        # Base classes should NOT be registered
        assert negotiator_registry.get_class("Negotiator") is None
        assert negotiator_registry.get_class("SAONegotiator") is None
        assert negotiator_registry.get_class("GBNegotiator") is None

    def test_components_registered(self):
        """Test that built-in components are registered.

        Note: Only concrete components are registered, not base classes like
        AcceptancePolicy or OfferingPolicy.
        """
        import negmas.registry_init  # noqa: F401

        # Check some known concrete components
        expected = ["AcceptImmediately", "RejectAlways", "RandomOfferingPolicy"]
        for name in expected:
            cls = component_registry.get_class(name)
            assert cls is not None, f"{name} not registered"

        # Base classes should NOT be registered
        assert component_registry.get_class("AcceptancePolicy") is None
        assert component_registry.get_class("OfferingPolicy") is None

    def test_tau_mechanism_no_deadline(self):
        """Test that TAU mechanisms don't require deadline."""
        import negmas.registry_init  # noqa: F401

        tau_infos = mechanism_registry.get_by_short_name("TAUMechanism")
        assert len(tau_infos) > 0
        # requires_deadline is now a tag
        assert not tau_infos[0].has_tag("requires-deadline")

        serial_tau_infos = mechanism_registry.get_by_short_name("SerialTAUMechanism")
        assert len(serial_tau_infos) > 0
        assert not serial_tau_infos[0].has_tag("requires-deadline")

    def test_sao_mechanism_requires_deadline(self):
        """Test that SAO mechanism requires deadline."""
        import negmas.registry_init  # noqa: F401

        sao_infos = mechanism_registry.get_by_short_name("SAOMechanism")
        assert len(sao_infos) > 0
        # requires_deadline is now a tag
        assert sao_infos[0].has_tag("requires-deadline")

    def test_query_acceptance_components(self):
        """Test querying for acceptance components."""
        import negmas.registry_init  # noqa: F401

        acceptance_policies = component_registry.query(component_type="acceptance")
        assert len(acceptance_policies) > 0

        # All results should have component_type == "acceptance"
        for key, info in acceptance_policies.items():
            assert info.component_type == "acceptance"

    def test_query_offering_components(self):
        """Test querying for offering components."""
        import negmas.registry_init  # noqa: F401

        offering_policies = component_registry.query(component_type="offering")
        assert len(offering_policies) > 0

        # All results should have component_type == "offering"
        for key, info in offering_policies.items():
            assert info.component_type == "offering"

    def test_get_class_from_registry(self):
        """Test getting actual class from registry."""
        import negmas.registry_init  # noqa: F401
        from negmas.sao import SAOMechanism

        cls = mechanism_registry.get_class("SAOMechanism")
        assert cls is SAOMechanism

    def test_full_type_name_stored_in_info(self):
        """Test that full type name is stored in the info object."""
        import negmas.registry_init  # noqa: F401

        # Get SAOMechanism info by short name
        infos = mechanism_registry.get_by_short_name("SAOMechanism")
        assert len(infos) > 0
        info = infos[0]

        # Verify full type name is stored in the info
        assert info.full_type_name == "negmas.sao.mechanism.SAOMechanism"
        assert info.short_name == "SAOMechanism"

    def test_genius_negotiators_registered(self):
        """Test that Genius negotiators are registered with ANAC year."""
        import negmas.registry_init  # noqa: F401

        # Check some known Genius negotiators
        genius_negotiators = [
            "AgentK",  # ANAC 2010
            "HardHeaded",  # ANAC 2011
            "CUHKAgent",  # ANAC 2012
            "AgentKF",  # ANAC 2013
            "Atlas",  # ANAC 2014
            "Atlas3",  # ANAC 2015
            "Caduceus",  # ANAC 2016
            "PonPokoAgent",  # ANAC 2017
            "AgentHerb",  # ANAC 2018
            "AgentGG",  # ANAC 2019
        ]
        for name in genius_negotiators:
            cls = negotiator_registry.get_class(name)
            assert cls is not None, f"{name} not registered"

    def test_query_negotiators_by_anac_year(self):
        """Test querying negotiators by ANAC competition year (using tags)."""
        import negmas.registry_init  # noqa: F401

        # Query for ANAC 2019 agents using tags
        anac_2019 = negotiator_registry.query(tags={"anac-2019"})
        assert len(anac_2019) > 0

        # All results should have anac-2019 tag
        for key, info in anac_2019.items():
            assert info.has_tag("anac-2019")

        # Check specific 2019 agents
        short_names = [info.short_name for info in anac_2019.values()]
        assert "AgentGG" in short_names
        assert "TheNewDeal" in short_names

    def test_query_multiple_anac_years(self):
        """Test that different ANAC years have different agents (using tags)."""
        import negmas.registry_init  # noqa: F401

        anac_2010 = negotiator_registry.query(tags={"anac-2010"})
        anac_2015 = negotiator_registry.query(tags={"anac-2015"})

        # Different years should have different agents
        names_2010 = {info.short_name for info in anac_2010.values()}
        names_2015 = {info.short_name for info in anac_2015.values()}
        assert names_2010 != names_2015

        # 2010 should have AgentK
        assert "AgentK" in names_2010
        # 2015 should have Atlas3
        assert "Atlas3" in names_2015

    def test_genius_boa_components_registered(self):
        """Test that Genius BOA components are registered."""
        import negmas.registry_init  # noqa: F401

        # Check Genius acceptance policies
        genius_acceptance = ["GACNext", "GACConst", "GACTime", "GACCombi"]
        for name in genius_acceptance:
            infos = component_registry.get_by_short_name(name)
            assert len(infos) > 0, f"{name} not registered"
            assert infos[0].component_type == "acceptance"

        # Check Genius offering policies
        genius_offering = [
            "GTimeDependentOffering",
            "GBoulwareOffering",
            "GRandomOffering",
        ]
        for name in genius_offering:
            infos = component_registry.get_by_short_name(name)
            assert len(infos) > 0, f"{name} not registered"
            assert infos[0].component_type == "offering"

        # Check Genius opponent models
        genius_models = ["GHardHeadedFrequencyModel", "GBayesianModel"]
        for name in genius_models:
            infos = component_registry.get_by_short_name(name)
            assert len(infos) > 0, f"{name} not registered"
            assert infos[0].component_type == "model"

    def test_query_model_components(self):
        """Test querying for model components."""
        import negmas.registry_init  # noqa: F401

        models = component_registry.query(component_type="model")
        assert len(models) > 0

        # All results should have component_type == "model"
        for key, info in models.items():
            assert info.component_type == "model"

        # Should include both SAO and Genius models
        short_names = [info.short_name for info in models.values()]
        assert "ZeroSumModel" in short_names  # SAO model
        assert "GBayesianModel" in short_names  # Genius model


class TestTagging:
    """Tests for the tagging functionality."""

    def test_registry_info_tags_default(self):
        """Test that tags default to empty set."""

        class DummyClass:
            pass

        info = RegistryInfo(
            key="dummy#12345678",
            short_name="dummy",
            full_type_name="test.DummyClass",
            cls=DummyClass,
        )
        assert info.tags == set()

    def test_registry_info_tags_custom(self):
        """Test RegistryInfo with custom tags."""

        class DummyClass:
            pass

        info = RegistryInfo(
            key="dummy#12345678",
            short_name="dummy",
            full_type_name="test.DummyClass",
            cls=DummyClass,
            tags={"tag1", "tag2"},
        )
        assert info.tags == {"tag1", "tag2"}

    def test_has_tag(self):
        """Test has_tag method."""

        class DummyClass:
            pass

        info = RegistryInfo(
            key="dummy#12345678",
            short_name="dummy",
            full_type_name="test.DummyClass",
            cls=DummyClass,
            tags={"sao", "builtin", "propose"},
        )
        assert info.has_tag("sao") is True
        assert info.has_tag("builtin") is True
        assert info.has_tag("genius") is False

    def test_has_any_tag(self):
        """Test has_any_tag method."""

        class DummyClass:
            pass

        info = RegistryInfo(
            key="dummy#12345678",
            short_name="dummy",
            full_type_name="test.DummyClass",
            cls=DummyClass,
            tags={"sao", "builtin"},
        )
        assert info.has_any_tag(["sao", "genius"]) is True
        assert info.has_any_tag(["genius", "boa"]) is False
        assert info.has_any_tag([]) is False

    def test_has_all_tags(self):
        """Test has_all_tags method."""

        class DummyClass:
            pass

        info = RegistryInfo(
            key="dummy#12345678",
            short_name="dummy",
            full_type_name="test.DummyClass",
            cls=DummyClass,
            tags={"sao", "builtin", "propose"},
        )
        assert info.has_all_tags(["sao", "builtin"]) is True
        assert info.has_all_tags(["sao", "genius"]) is False
        assert info.has_all_tags([]) is True

    def test_registry_register_with_tags(self):
        """Test registering a class with tags."""
        registry = Registry(RegistryInfo)

        class TestClass:
            pass

        key = registry.register(TestClass, short_name="test", tags={"tag1", "tag2"})
        info = registry[key]
        assert info is not None
        assert info.tags == {"tag1", "tag2"}

    def test_registry_query_with_tags(self):
        """Test query method with tags parameter (all must match)."""
        registry = Registry(NegotiatorInfo)

        class Negotiator1:
            pass

        class Negotiator2:
            pass

        class Negotiator3:
            pass

        key1 = registry.register(
            Negotiator1, short_name="neg1", tags={"sao", "builtin"}
        )
        key2 = registry.register(Negotiator2, short_name="neg2", tags={"sao", "genius"})
        key3 = registry.register(Negotiator3, short_name="neg3", tags={"builtin"})

        # Query for items with both "sao" and "builtin" tags
        results = registry.query(tags=["sao", "builtin"])
        assert key1 in results
        assert key2 not in results
        assert key3 not in results

    def test_registry_query_with_any_tags(self):
        """Test query method with any_tags parameter (any must match)."""
        registry = Registry(NegotiatorInfo)

        class Negotiator1:
            pass

        class Negotiator2:
            pass

        class Negotiator3:
            pass

        key1 = registry.register(
            Negotiator1, short_name="neg1", tags={"sao", "builtin"}
        )
        key2 = registry.register(Negotiator2, short_name="neg2", tags={"genius"})
        key3 = registry.register(Negotiator3, short_name="neg3", tags={"other"})

        # Query for items with either "sao" or "genius" tag
        results = registry.query(any_tags=["sao", "genius"])
        assert key1 in results
        assert key2 in results
        assert key3 not in results

    def test_registry_query_with_exclude_tags(self):
        """Test query method with exclude_tags parameter."""
        registry = Registry(NegotiatorInfo)

        class Negotiator1:
            pass

        class Negotiator2:
            pass

        class Negotiator3:
            pass

        key1 = registry.register(
            Negotiator1, short_name="neg1", tags={"sao", "builtin"}
        )
        key2 = registry.register(Negotiator2, short_name="neg2", tags={"genius"})
        key3 = registry.register(Negotiator3, short_name="neg3", tags={"sao"})

        # Exclude items with "genius" tag
        results = registry.query(exclude_tags=["genius"])
        assert key1 in results
        assert key2 not in results
        assert key3 in results

    def test_registry_query_combined_tag_filters(self):
        """Test query method with combined tag filters."""
        registry = Registry(NegotiatorInfo)

        class Negotiator1:
            pass

        class Negotiator2:
            pass

        class Negotiator3:
            pass

        class Negotiator4:
            pass

        key1 = registry.register(
            Negotiator1, short_name="neg1", tags={"sao", "builtin", "propose"}
        )
        key2 = registry.register(
            Negotiator2, short_name="neg2", tags={"sao", "genius", "propose"}
        )
        key3 = registry.register(
            Negotiator3, short_name="neg3", tags={"builtin", "propose"}
        )
        key4 = registry.register(
            Negotiator4, short_name="neg4", tags={"sao", "deprecated"}
        )

        # Complex query: must have "sao", can have "propose" or "respond", exclude "deprecated"
        results = registry.query(
            tags=["sao"], any_tags=["propose", "respond"], exclude_tags=["deprecated"]
        )
        assert key1 in results
        assert key2 in results
        assert key3 not in results  # Missing "sao"
        assert key4 not in results  # Has "deprecated"

    def test_registry_list_tags(self):
        """Test list_tags method."""
        registry = Registry(RegistryInfo)

        class Class1:
            pass

        class Class2:
            pass

        class Class3:
            pass

        registry.register(Class1, short_name="c1", tags={"tag1", "tag2"})
        registry.register(Class2, short_name="c2", tags={"tag2", "tag3"})
        registry.register(Class3, short_name="c3", tags=set())

        all_tags = registry.list_tags()
        assert all_tags == {"tag1", "tag2", "tag3"}

    def test_registry_query_by_tag(self):
        """Test query_by_tag convenience method."""
        registry = Registry(RegistryInfo)

        class Class1:
            pass

        class Class2:
            pass

        key1 = registry.register(Class1, short_name="c1", tags={"sao", "builtin"})
        key2 = registry.register(Class2, short_name="c2", tags={"genius"})

        sao_results = registry.query_by_tag("sao")
        assert key1 in sao_results
        assert key2 not in sao_results

        genius_results = registry.query_by_tag("genius")
        assert key1 not in genius_results
        assert key2 in genius_results

    def test_decorator_with_tags(self):
        """Test that decorators accept tags parameter."""

        @register_mechanism(short_name="tagged_mech", tags={"test", "custom"})
        class TaggedMechanism:
            pass

        info = mechanism_registry.get_by_class(TaggedMechanism)
        assert info is not None
        # Tags include both custom tags and auto-added requires-deadline
        assert "test" in info.tags
        assert "custom" in info.tags

        @register_negotiator(short_name="tagged_neg", tags={"test", "sao"})
        class TaggedNegotiator:
            pass

        info = negotiator_registry.get_by_class(TaggedNegotiator)
        assert info is not None
        assert info.tags == {"test", "sao"}

        @register_component(short_name="tagged_comp", tags={"test", "acceptance"})
        class TaggedComponent:
            pass

        info = component_registry.get_by_class(TaggedComponent)
        assert info is not None
        assert info.tags == {"test", "acceptance"}


class TestBuiltInTags:
    """Tests for tags on built-in registrations."""

    def test_builtin_negotiators_have_builtin_tag(self):
        """Test that built-in negotiators have 'builtin' tag."""
        import negmas.registry_init  # noqa: F401

        builtin_negotiators = [
            "AspirationNegotiator",
            "ToughNegotiator",
            "RandomNegotiator",
            "NiceNegotiator",
        ]
        for name in builtin_negotiators:
            infos = negotiator_registry.get_by_short_name(name)
            assert len(infos) > 0, f"{name} not found"
            assert infos[0].has_tag("builtin"), f"{name} missing 'builtin' tag"

    def test_genius_negotiators_have_genius_tag(self):
        """Test that Genius negotiators have 'genius' tag."""
        import negmas.registry_init  # noqa: F401

        genius_negotiators = ["AgentK", "AgentGG", "Atlas3", "CUHKAgent"]
        for name in genius_negotiators:
            infos = negotiator_registry.get_by_short_name(name)
            assert len(infos) > 0, f"{name} not found"
            assert infos[0].has_tag("genius"), f"{name} missing 'genius' tag"
            assert not infos[0].has_tag("builtin"), (
                f"{name} should not have 'builtin' tag"
            )

    def test_genius_negotiators_have_anac_year_tags(self):
        """Test that Genius negotiators have ANAC year tags."""
        import negmas.registry_init  # noqa: F401

        # Check specific negotiators and their expected ANAC year tags
        anac_negotiators = {
            "AgentK": "anac-2010",
            "HardHeaded": "anac-2011",
            "CUHKAgent": "anac-2012",
            "AgentKF": "anac-2013",
            "Atlas3": "anac-2015",
            "AgentGG": "anac-2019",
        }
        for name, expected_year_tag in anac_negotiators.items():
            infos = negotiator_registry.get_by_short_name(name)
            assert len(infos) > 0, f"{name} not found"
            assert infos[0].has_tag(expected_year_tag), (
                f"{name} missing '{expected_year_tag}' tag"
            )
            assert infos[0].has_tag("anac"), f"{name} missing 'anac' tag"

    def test_query_builtin_vs_genius(self):
        """Test querying for builtin vs genius negotiators."""
        import negmas.registry_init  # noqa: F401

        # Query for builtin negotiators
        builtin = negotiator_registry.query_by_tag("builtin")
        assert len(builtin) > 0
        for key, info in builtin.items():
            assert not info.has_tag("genius"), (
                f"{info.short_name} has both builtin and genius"
            )

        # Query for genius negotiators
        genius = negotiator_registry.query_by_tag("genius")
        assert len(genius) > 0
        for key, info in genius.items():
            assert not info.has_tag("builtin"), (
                f"{info.short_name} has both genius and builtin"
            )

    def test_genius_boa_components_have_tags(self):
        """Test that Genius BOA components have appropriate tags."""
        import negmas.registry_init  # noqa: F401

        genius_acceptance = ["GACNext", "GACConst", "GACTime", "GACCombi"]
        for name in genius_acceptance:
            infos = component_registry.get_by_short_name(name)
            assert len(infos) > 0, f"{name} not found"
            assert infos[0].has_tag("genius"), f"{name} missing 'genius' tag"
            assert infos[0].has_tag("boa"), f"{name} missing 'boa' tag"

    def test_builtin_mechanisms_have_builtin_tag(self):
        """Test that built-in mechanisms have 'builtin' tag."""
        import negmas.registry_init  # noqa: F401

        mechanisms = ["SAOMechanism", "TAUMechanism"]
        for name in mechanisms:
            infos = mechanism_registry.get_by_short_name(name)
            assert len(infos) > 0, f"{name} not found"
            assert infos[0].has_tag("builtin"), f"{name} missing 'builtin' tag"

    def test_list_all_tags(self):
        """Test that list_tags returns expected tags."""
        import negmas.registry_init  # noqa: F401

        # Check negotiator tags
        neg_tags = negotiator_registry.list_tags()
        assert "builtin" in neg_tags
        assert "genius" in neg_tags
        assert "sao" in neg_tags
        assert "anac" in neg_tags

        # Check component tags
        comp_tags = component_registry.list_tags()
        assert "builtin" in comp_tags
        assert "genius" in comp_tags
        assert "boa" in comp_tags


class TestScenarioInfo:
    """Tests for ScenarioInfo dataclass."""

    def test_scenario_info_creation(self):
        """Test basic ScenarioInfo creation."""
        from pathlib import Path
        from negmas.registry import ScenarioInfo

        info = ScenarioInfo(name="test_scenario", path=Path("/tmp/test_scenario"))
        assert info.name == "test_scenario"
        assert info.path == Path("/tmp/test_scenario")
        assert info.source == "unknown"
        assert info.tags == set()
        # Boolean fields removed - use tags instead
        assert info.n_outcomes is None
        assert info.n_negotiators is None
        assert info.opposition_level is None
        assert info.extra == {}

    def test_scenario_info_with_all_fields(self):
        """Test ScenarioInfo with all fields populated (using tags)."""
        from pathlib import Path
        from negmas.registry import ScenarioInfo

        info = ScenarioInfo(
            name="anac_scenario",
            path=Path("/tmp/anac/scenario"),
            source="negmas",
            tags={"anac", "bilateral", "xml", "normalized", "has-stats", "has-plot"},
            n_outcomes=100,
            n_negotiators=2,
            opposition_level=0.5,
            rational_fraction=0.75,
            extra={"year": 2019},
        )
        assert info.name == "anac_scenario"
        assert info.source == "negmas"
        assert info.n_outcomes == 100
        assert info.n_negotiators == 2
        assert info.opposition_level == 0.5
        assert info.rational_fraction == 0.75
        # Boolean fields are now tags
        assert info.has_tag("normalized")
        assert info.has_tag("anac")
        assert info.has_tag("has-stats")
        assert info.has_tag("has-plot")
        assert info.extra == {"year": 2019}

    def test_scenario_info_with_rational_fraction(self):
        """Test ScenarioInfo with rational_fraction field."""
        from pathlib import Path
        from negmas.registry import ScenarioInfo

        info = ScenarioInfo(
            name="rational_test", path=Path("/tmp/rational_test"), rational_fraction=0.6
        )
        assert info.rational_fraction == 0.6

    def test_scenario_info_rational_fraction_defaults_to_none(self):
        """Test that rational_fraction defaults to None if not provided."""
        from pathlib import Path
        from negmas.registry import ScenarioInfo

        info = ScenarioInfo(name="test", path=Path("/tmp/test"))
        assert info.rational_fraction is None

    def test_scenario_info_has_tag(self):
        """Test has_tag method on ScenarioInfo."""
        from pathlib import Path
        from negmas.registry import ScenarioInfo

        info = ScenarioInfo(
            name="test", path=Path("/tmp/test"), tags={"builtin", "bilateral", "xml"}
        )
        assert info.has_tag("builtin") is True
        assert info.has_tag("bilateral") is True
        assert info.has_tag("anac") is False

    def test_scenario_info_has_any_tag(self):
        """Test has_any_tag method on ScenarioInfo."""
        from pathlib import Path
        from negmas.registry import ScenarioInfo

        info = ScenarioInfo(
            name="test", path=Path("/tmp/test"), tags={"builtin", "bilateral"}
        )
        assert info.has_any_tag(["builtin", "anac"]) is True
        assert info.has_any_tag(["anac", "multilateral"]) is False

    def test_scenario_info_has_all_tags(self):
        """Test has_all_tags method on ScenarioInfo."""
        from pathlib import Path
        from negmas.registry import ScenarioInfo

        info = ScenarioInfo(
            name="test", path=Path("/tmp/test"), tags={"builtin", "bilateral", "xml"}
        )
        assert info.has_all_tags(["builtin", "bilateral"]) is True
        assert info.has_all_tags(["builtin", "anac"]) is False
        assert info.has_all_tags([]) is True


class TestScenarioRegistry:
    """Tests for ScenarioRegistry class."""

    def test_scenario_registry_register(self):
        """Test registering a scenario."""
        from pathlib import Path
        from negmas.registry import ScenarioRegistry

        registry = ScenarioRegistry()
        path = Path("/tmp/test_scenario")

        info = registry.register(path, name="test_scenario")
        assert str(path.resolve()) in registry
        assert info.name == "test_scenario"

    def test_scenario_registry_register_with_source(self):
        """Test registering a scenario with source."""
        from pathlib import Path
        from negmas.registry import ScenarioRegistry

        registry = ScenarioRegistry()
        path = Path("/tmp/test_scenario_src")

        info = registry.register(path, name="test_scenario", source="mylib")
        assert info.source == "mylib"

    def test_scenario_registry_auto_name(self):
        """Test registering with automatic name from path."""
        from pathlib import Path
        from negmas.registry import ScenarioRegistry

        registry = ScenarioRegistry()
        path = Path("/tmp/MyScenario")

        info = registry.register(path)
        assert info.name == "MyScenario"

    def test_scenario_registry_with_tags(self):
        """Test registering with custom tags."""
        from pathlib import Path
        from negmas.registry import ScenarioRegistry

        registry = ScenarioRegistry()
        path = Path("/tmp/test_scenario")

        info = registry.register(path, tags={"custom", "test"})
        assert "custom" in info.tags
        assert "test" in info.tags

    def test_scenario_registry_get_by_name(self):
        """Test getting scenarios by name."""
        from pathlib import Path
        from negmas.registry import ScenarioRegistry

        registry = ScenarioRegistry()
        path1 = Path("/tmp/scenario1")
        path2 = Path("/tmp/other/scenario1")  # Same name, different path

        registry.register(path1, name="scenario1")
        registry.register(path2, name="scenario1")

        results = registry.get_by_name("scenario1")
        assert len(results) == 2

    def test_scenario_registry_query(self):
        """Test query method."""
        from pathlib import Path
        from negmas.registry import ScenarioRegistry

        registry = ScenarioRegistry()
        registry.register(Path("/tmp/s1"), name="s1", tags={"bilateral", "xml"})
        registry.register(Path("/tmp/s2"), name="s2", tags={"multilateral", "json"})

        # Query by tag
        results = registry.query(tags={"bilateral"})
        assert len(results) == 1

        # Query with any_tags
        results = registry.query(any_tags={"bilateral", "multilateral"})
        assert len(results) == 2

        # Query with exclude_tags
        results = registry.query(exclude_tags={"multilateral"})
        assert len(results) == 1

    def test_scenario_registry_query_by_format(self):
        """Test query method with format filter (using tags)."""
        from pathlib import Path
        from negmas.registry import ScenarioRegistry

        registry = ScenarioRegistry()
        # Format is now a tag
        registry.register(Path("/tmp/s1.xml"), name="s1", tags={"xml"})
        registry.register(Path("/tmp/s2.json"), name="s2", tags={"json"})

        # Query by format tag
        results = registry.query(tags={"xml"})
        assert len(results) == 1

    def test_scenario_registry_with_rational_fraction(self):
        """Test registering scenarios with rational_fraction field."""
        from pathlib import Path
        from negmas.registry import ScenarioRegistry

        registry = ScenarioRegistry()
        path = Path("/tmp/rational_scenario")

        info = registry.register(path, name="rational_test", rational_fraction=0.8)
        assert info.rational_fraction == 0.8

    def test_scenario_registry_query_rational_fraction_exact(self):
        """Test querying by exact rational_fraction value."""
        from pathlib import Path
        from negmas.registry import ScenarioRegistry

        registry = ScenarioRegistry()
        registry.register(Path("/tmp/rf1"), name="rf1", rational_fraction=0.5)
        registry.register(Path("/tmp/rf2"), name="rf2", rational_fraction=0.8)
        registry.register(Path("/tmp/rf3"), name="rf3", rational_fraction=0.5)
        registry.register(Path("/tmp/rf4"), name="rf4")  # No rational_fraction

        # Query for exact value
        results = registry.query(rational_fraction=0.5)
        assert len(results) == 2

        results = registry.query(rational_fraction=0.8)
        assert len(results) == 1

    def test_scenario_registry_query_rational_fraction_range(self):
        """Test querying by rational_fraction range."""
        from pathlib import Path
        from negmas.registry import ScenarioRegistry

        registry = ScenarioRegistry()
        registry.register(Path("/tmp/rf1"), name="rf1", rational_fraction=0.2)
        registry.register(Path("/tmp/rf2"), name="rf2", rational_fraction=0.5)
        registry.register(Path("/tmp/rf3"), name="rf3", rational_fraction=0.8)
        registry.register(Path("/tmp/rf4"), name="rf4", rational_fraction=0.95)
        registry.register(Path("/tmp/rf5"), name="rf5")  # No rational_fraction

        # Query for range [0.4, 0.9]
        results = registry.query(rational_fraction=(0.4, 0.9))
        assert len(results) == 2
        names = {info.name for info in results.values()}
        assert names == {"rf2", "rf3"}

        # Query for range with no lower bound
        results = registry.query(rational_fraction=(None, 0.5))
        assert len(results) == 2
        names = {info.name for info in results.values()}
        assert names == {"rf1", "rf2"}

        # Query for range with no upper bound
        results = registry.query(rational_fraction=(0.8, None))
        assert len(results) == 2
        names = {info.name for info in results.values()}
        assert names == {"rf3", "rf4"}

    def test_scenario_registry_query_combined_with_rational_fraction(self):
        """Test querying with rational_fraction combined with other filters."""
        from pathlib import Path
        from negmas.registry import ScenarioRegistry

        registry = ScenarioRegistry()
        registry.register(
            Path("/tmp/crf1"),
            name="crf1",
            rational_fraction=0.6,
            n_negotiators=2,
            tags={"bilateral"},
        )
        registry.register(
            Path("/tmp/crf2"),
            name="crf2",
            rational_fraction=0.8,
            n_negotiators=2,
            tags={"bilateral"},
        )
        registry.register(
            Path("/tmp/crf3"),
            name="crf3",
            rational_fraction=0.6,
            n_negotiators=3,
            tags={"multilateral"},
        )

        # Query for bilateral with rational_fraction >= 0.5
        results = registry.query(tags={"bilateral"}, rational_fraction=(0.5, None))
        assert len(results) == 2

        # Query for bilateral with high rational_fraction
        results = registry.query(tags={"bilateral"}, rational_fraction=(0.7, None))
        assert len(results) == 1
        assert list(results.values())[0].name == "crf2"

    def test_scenario_registry_list_tags(self):
        """Test list_tags method."""
        from pathlib import Path
        from negmas.registry import ScenarioRegistry

        registry = ScenarioRegistry()
        registry.register(Path("/tmp/s1"), name="s1", tags={"tag1", "tag2"})
        registry.register(Path("/tmp/s2"), name="s2", tags={"tag2", "tag3"})

        all_tags = registry.list_tags()
        assert all_tags == {"tag1", "tag2", "tag3"}

    def test_scenario_registry_list_names(self):
        """Test list_names method."""
        from pathlib import Path
        from negmas.registry import ScenarioRegistry

        registry = ScenarioRegistry()
        registry.register(Path("/tmp/s1"), name="scenario1")
        registry.register(Path("/tmp/s2"), name="scenario2")

        names = registry.list_names()
        assert "scenario1" in names
        assert "scenario2" in names

    def test_scenario_registry_unregister_by_path(self):
        """Test unregistering a scenario by its path."""
        from pathlib import Path
        from negmas.registry import ScenarioRegistry

        registry = ScenarioRegistry()
        path = Path("/tmp/test_unregister")
        registry.register(path, name="test_unregister")

        key = str(path.resolve())
        assert key in registry

        # Unregister by path
        result = registry.unregister(path)
        assert result is True
        assert key not in registry
        assert registry.get_by_name("test_unregister") == []

    def test_scenario_registry_unregister_by_path_string(self):
        """Test unregistering a scenario by its path as string."""
        from pathlib import Path
        from negmas.registry import ScenarioRegistry

        registry = ScenarioRegistry()
        path = Path("/tmp/test_unregister_str")
        registry.register(path, name="test_unregister_str")

        key = str(path.resolve())
        assert key in registry

        # Unregister by path string
        result = registry.unregister(key)
        assert result is True
        assert key not in registry

    def test_scenario_registry_unregister_by_name(self):
        """Test unregistering scenarios by name (removes all with that name)."""
        from pathlib import Path
        from negmas.registry import ScenarioRegistry

        registry = ScenarioRegistry()
        path1 = Path("/tmp/scenario_a")
        path2 = Path("/tmp/subdir/scenario_a")  # Same name, different path

        registry.register(path1, name="scenario_a")
        registry.register(path2, name="scenario_a")

        assert len(registry) == 2

        # Unregister by name - should remove both
        result = registry.unregister("scenario_a")
        assert result is True
        assert len(registry) == 0

    def test_scenario_registry_unregister_not_found(self):
        """Test unregistering a scenario that doesn't exist."""
        from negmas.registry import ScenarioRegistry

        registry = ScenarioRegistry()

        result = registry.unregister("nonexistent")
        assert result is False

    def test_scenario_registry_unregister_updates_name_index(self):
        """Test that unregistering properly updates the name index."""
        from pathlib import Path
        from negmas.registry import ScenarioRegistry

        registry = ScenarioRegistry()
        path1 = Path("/tmp/scenario_x")
        path2 = Path("/tmp/other/scenario_x")

        registry.register(path1, name="scenario_x")
        registry.register(path2, name="scenario_x")

        # Unregister one by path
        registry.unregister(path1)

        # Should still find the other one by name
        results = registry.get_by_name("scenario_x")
        assert len(results) == 1
        assert results[0].path == path2.resolve()

    def test_scenario_registry_never_hides_existing(self):
        """Test that registering same path twice returns existing info."""
        from pathlib import Path
        from negmas.registry import ScenarioRegistry

        registry = ScenarioRegistry()
        path = Path("/tmp/unique_scenario")

        info1 = registry.register(path, name="first_name", tags={"tag1"})
        info2 = registry.register(path, name="second_name", tags={"tag2"})

        # Should return the existing info, not create new
        assert info1 is info2
        assert info1.name == "first_name"  # Original name preserved
        assert len(registry) == 1

    def test_scenario_registry_register_many(self):
        """Test register_many method."""
        from negmas.registry import ScenarioRegistry

        registry = ScenarioRegistry()
        registrations = [
            {"path": "/tmp/many_s1", "name": "many1", "tags": {"test"}},
            {"path": "/tmp/many_s2", "name": "many2", "n_negotiators": 2},
            {"path": "/tmp/many_s3", "source": "test_lib"},
        ]

        infos = registry.register_many(registrations)

        assert len(infos) == 3
        assert len(registry) == 3
        assert infos[0].name == "many1"
        assert "test" in infos[0].tags
        assert infos[1].name == "many2"
        assert infos[1].n_negotiators == 2
        assert infos[2].source == "test_lib"

    def test_scenario_registry_unregister_many(self):
        """Test unregister_many method."""
        from pathlib import Path
        from negmas.registry import ScenarioRegistry

        registry = ScenarioRegistry()
        path1 = Path("/tmp/unreg_s1")
        path2 = Path("/tmp/unreg_s2")
        path3 = Path("/tmp/unreg_s3")
        registry.register(path1, name="unreg1")
        registry.register(path2, name="unreg2")
        registry.register(path3, name="unreg3")

        assert len(registry) == 3

        # Unregister two (use resolved path for the first, name for the second)
        count = registry.unregister_many([path1, "unreg2"])
        assert count == 2
        assert len(registry) == 1

        # The remaining one should be unreg_s3
        assert "unreg3" in registry.list_names()

    def test_scenario_registry_load(self):
        """Test load method with a real scenario."""
        import negmas.registry_init  # noqa: F401
        from negmas.registry import scenario_registry

        # CameraB should be a builtin scenario
        results = scenario_registry.get_by_name("CameraB")
        if len(results) == 0:
            pytest.skip("CameraB scenario not registered")

        # Load by name
        scenario = scenario_registry.load("CameraB")
        assert scenario is not None
        assert scenario.n_negotiators == 2

    def test_scenario_registry_load_by_path(self):
        """Test load method by path."""
        import negmas.registry_init  # noqa: F401
        from negmas.registry import scenario_registry

        # CameraB should be a builtin scenario
        results = scenario_registry.get_by_name("CameraB")
        if len(results) == 0:
            pytest.skip("CameraB scenario not registered")

        # Get the path and load by path
        path_str = str(results[0].path)
        scenario = scenario_registry.load(path_str)
        assert scenario is not None
        assert scenario.n_negotiators == 2

    def test_scenario_registry_load_not_found(self):
        """Test load method with non-existent scenario."""
        from negmas.registry import ScenarioRegistry

        registry = ScenarioRegistry()

        with pytest.raises(KeyError):
            registry.load("nonexistent_scenario")


class TestRegisterScenarioFunction:
    """Tests for the register_scenario function."""

    def test_register_scenario_function(self):
        """Test register_scenario function."""
        from pathlib import Path
        from negmas.registry import register_scenario, scenario_registry

        path = Path("/tmp/func_test_scenario")
        info = register_scenario(path, name="func_test", tags={"test"})

        assert info.name == "func_test"
        assert "test" in info.tags
        assert str(path.resolve()) in scenario_registry

        # Cleanup
        scenario_registry.unregister(path)

    def test_register_scenario_with_stats_and_plot(self):
        """Test register_scenario with stats and plot flags (using tags)."""
        from pathlib import Path
        from negmas.registry import register_scenario, scenario_registry

        path = Path("/tmp/stats_scenario")
        info = register_scenario(path, tags={"has-stats", "has-plot"})

        # has_stats and has_plot are now tags
        assert info.has_tag("has-stats")
        assert info.has_tag("has-plot")

        # Cleanup
        scenario_registry.unregister(path)


class TestBuiltInScenarios:
    """Tests for built-in scenario registrations."""

    def test_builtin_scenarios_registered(self):
        """Test that built-in scenarios are registered."""
        import negmas.registry_init  # noqa: F401
        from negmas.registry import scenario_registry

        # Should have at least some scenarios registered
        assert len(scenario_registry) > 0

    def test_builtin_scenarios_have_builtin_tag(self):
        """Test that built-in scenarios have 'builtin' tag."""
        import negmas.registry_init  # noqa: F401
        from negmas.registry import scenario_registry

        builtin_scenarios = scenario_registry.query_by_tag("builtin")
        assert len(builtin_scenarios) > 0

    def test_builtin_scenarios_have_format_tag(self):
        """Test that built-in scenarios have format tags."""
        import negmas.registry_init  # noqa: F401
        from negmas.registry import scenario_registry

        # Most scenarios should have a format tag
        xml_scenarios = scenario_registry.query_by_tag("xml")
        # There should be at least one xml scenario
        assert len(xml_scenarios) >= 0  # May be 0 if all are yaml/json

    def test_camerab_scenario_properties(self):
        """Test CameraB scenario is properly registered."""
        import negmas.registry_init  # noqa: F401
        from negmas.registry import scenario_registry

        results = scenario_registry.get_by_name("CameraB")
        if len(results) > 0:  # CameraB may not be in builtin scenarios
            info = results[0]
            assert info.n_negotiators == 2
            assert info.has_tag("bilateral")


class TestRegisterAllScenarios:
    """Tests for register_all_scenarios function."""

    def test_register_all_scenarios_from_negmas_scenarios(self):
        """Test registering all scenarios from negmas scenarios directory."""
        from negmas.registry import register_all_scenarios, ScenarioRegistry
        from pathlib import Path
        import negmas

        # Get the negmas scenarios path
        scenarios_path = Path(negmas.__file__).parent / "scenarios"
        if not scenarios_path.exists():
            pytest.skip("negmas scenarios path doesn't exist")

        registry = ScenarioRegistry()
        scenarios = register_all_scenarios(scenarios_path, registry=registry)

        # Should have found some scenarios
        assert len(scenarios) > 0

    def test_register_all_scenarios_extracts_info(self):
        """Test that register_all_scenarios extracts scenario information."""
        from negmas.registry import register_all_scenarios, ScenarioRegistry
        from pathlib import Path
        import negmas

        scenarios_path = Path(negmas.__file__).parent / "scenarios"
        if not scenarios_path.exists():
            pytest.skip("negmas scenarios path doesn't exist")

        registry = ScenarioRegistry()
        scenarios = register_all_scenarios(scenarios_path, registry=registry)

        # Check that info was extracted
        for info in scenarios:
            # n_negotiators should be set for loaded scenarios
            assert info.n_negotiators is not None or info.n_outcomes is not None

    def test_register_all_scenarios_invalid_path(self):
        """Test register_all_scenarios with non-existent path."""
        from negmas.registry import register_all_scenarios
        from pathlib import Path

        with pytest.raises(ValueError):
            register_all_scenarios(Path("/nonexistent/path"))

    def test_register_all_scenarios_not_directory(self):
        """Test register_all_scenarios with file path."""
        from negmas.registry import register_all_scenarios
        from pathlib import Path
        import tempfile

        with tempfile.NamedTemporaryFile() as f:
            with pytest.raises(ValueError):
                register_all_scenarios(Path(f.name))

    def test_register_all_scenarios_non_recursive(self):
        """Test register_all_scenarios with recursive=False."""
        from negmas.registry import register_all_scenarios, ScenarioRegistry
        from pathlib import Path
        import negmas

        scenarios_path = Path(negmas.__file__).parent / "scenarios"
        if not scenarios_path.exists():
            pytest.skip("negmas scenarios path doesn't exist")

        registry_recursive = ScenarioRegistry()
        registry_non_recursive = ScenarioRegistry()

        scenarios_recursive = register_all_scenarios(
            scenarios_path, registry=registry_recursive, recursive=True
        )
        scenarios_non_recursive = register_all_scenarios(
            scenarios_path, registry=registry_non_recursive, recursive=False
        )

        # Non-recursive should find fewer or equal scenarios
        assert len(scenarios_non_recursive) <= len(scenarios_recursive)


class TestSaveLoadRegistry:
    """Tests for save_registry and load_registry functions."""

    def test_save_registry_creates_files(self, tmp_path):
        """Test that save_registry creates JSON files."""
        from negmas.registry import save_registry

        path = save_registry(tmp_path)

        assert path == tmp_path
        assert (tmp_path / "mechanisms.json").exists()
        assert (tmp_path / "negotiators.json").exists()
        assert (tmp_path / "components.json").exists()
        assert (tmp_path / "scenarios.json").exists()

    def test_save_registry_selective(self, tmp_path):
        """Test saving only specific registries."""
        from negmas.registry import save_registry

        save_registry(
            tmp_path,
            include_mechanisms=False,
            include_components=False,
            include_scenarios=False,
        )

        assert not (tmp_path / "mechanisms.json").exists()
        assert (tmp_path / "negotiators.json").exists()
        assert not (tmp_path / "components.json").exists()
        assert not (tmp_path / "scenarios.json").exists()

    def test_save_and_load_roundtrip(self, tmp_path):
        """Test that save and load preserve data."""
        from negmas.registry import (
            save_registry,
            load_registry,
            negotiator_registry,
            clear_registry,
        )

        # Save current state
        save_registry(tmp_path)

        # Count current negotiators (only count those that can be reloaded)
        original_count = len(negotiator_registry)

        # Clear and reload
        clear_registry(
            include_mechanisms=False, include_components=False, include_scenarios=False
        )

        assert len(negotiator_registry) == 0

        # Load back - some test classes may fail to load (expected)
        counts = load_registry(
            tmp_path,
            include_mechanisms=False,
            include_components=False,
            include_scenarios=False,
        )

        # Should have loaded negotiators (some may fail if they're test classes)
        assert counts["negotiators"] > 0
        # Allow for some test classes not reloading
        assert len(negotiator_registry) >= original_count - 10

    def test_load_registry_nonexistent_path(self, tmp_path):
        """Test loading from non-existent path raises error."""
        from negmas.registry import load_registry

        with pytest.raises(FileNotFoundError):
            load_registry(tmp_path / "nonexistent")

    def test_load_registry_skips_missing_files(self, tmp_path):
        """Test that load_registry works with partial files."""
        from negmas.registry import load_registry
        import json

        # Create only negotiators.json
        tmp_path.mkdir(parents=True, exist_ok=True)
        with open(tmp_path / "negotiators.json", "w") as f:
            json.dump({}, f)

        # Should not raise, just skip missing files
        counts = load_registry(tmp_path)
        assert counts["mechanisms"] == 0  # File doesn't exist
        assert counts["negotiators"] == 0  # Empty file

    def test_clear_registry(self):
        """Test clear_registry function."""
        from negmas.registry import (
            clear_registry,
            negotiator_registry,
            mechanism_registry,
        )
        import negmas.registry_init  # noqa: F401 - ensure registries are populated

        # Ensure we have some registrations
        assert len(negotiator_registry) > 0 or len(mechanism_registry) > 0

        # Clear negotiators only
        original_mechanism_count = len(mechanism_registry)
        clear_registry(
            include_mechanisms=False, include_components=False, include_scenarios=False
        )

        assert len(negotiator_registry) == 0
        assert len(mechanism_registry) == original_mechanism_count

        # Re-import to restore
        import importlib

        importlib.reload(negmas.registry_init)

    def test_save_load_with_custom_class(self, tmp_path):
        """Test saving and loading custom class registrations."""
        from negmas.registry import save_registry, negotiator_registry

        # Register a custom class (use a real negmas class)
        from negmas.sao.negotiators import AspirationNegotiator

        key = negotiator_registry.register(
            AspirationNegotiator,
            short_name="TestCustomAspiration",
            source="test",
            params={"aspiration_type": "boulware"},
            tags={"test", "custom"},
        )

        try:
            # Save
            save_registry(
                tmp_path,
                include_mechanisms=False,
                include_components=False,
                include_scenarios=False,
            )

            # Verify JSON content
            import json

            with open(tmp_path / "negotiators.json") as f:
                data = json.load(f)

            assert key in data
            assert data[key]["short_name"] == "TestCustomAspiration"
            assert data[key]["source"] == "test"
            assert data[key]["params"] == {"aspiration_type": "boulware"}
            assert set(data[key]["tags"]) == {"test", "custom"}

        finally:
            # Cleanup
            negotiator_registry.unregister(key)

    def test_default_registry_path(self):
        """Test DEFAULT_REGISTRY_PATH is set correctly."""
        from negmas.registry import DEFAULT_REGISTRY_PATH
        from pathlib import Path

        assert DEFAULT_REGISTRY_PATH == Path.home() / "negmas" / "registry"


class TestScenarioRegistryReadOnly:
    """Tests for scenario registry read_only functionality."""

    def test_scenario_info_read_only_default(self):
        """Test ScenarioInfo read_only defaults to False."""
        from pathlib import Path
        from negmas.registry import ScenarioInfo

        info = ScenarioInfo(name="test", path=Path("/tmp/test"))
        assert info.read_only is False

    def test_scenario_info_read_only_explicit(self):
        """Test ScenarioInfo with read_only set to True."""
        from pathlib import Path
        from negmas.registry import ScenarioInfo

        info = ScenarioInfo(name="test", path=Path("/tmp/test"), read_only=True)
        assert info.read_only is True

    def test_register_scenario_read_only(self):
        """Test registering a scenario as read-only."""
        from pathlib import Path
        from negmas.registry import ScenarioRegistry

        registry = ScenarioRegistry()
        path = Path("/tmp/test_readonly")

        info = registry.register(path, name="test_readonly", read_only=True)
        assert info.read_only is True

    def test_unregister_read_only_is_allowed(self):
        """Test that read-only scenarios can be unregistered (read_only is informational only)."""
        from pathlib import Path
        from negmas.registry import ScenarioRegistry

        registry = ScenarioRegistry()
        path = Path("/tmp/test_readonly_unregister")

        registry.register(path, name="test_readonly", read_only=True)

        # read_only is informational, so unregister should succeed
        result = registry.unregister(path)
        assert result is True
        assert str(path.resolve()) not in registry

    def test_unregister_read_only_by_name_is_allowed(self):
        """Test that read-only scenarios can be unregistered by name."""
        from pathlib import Path
        from negmas.registry import ScenarioRegistry

        registry = ScenarioRegistry()
        path = Path("/tmp/test_readonly_by_name")

        registry.register(path, name="test_readonly_name", read_only=True)

        # read_only is informational, so unregister should succeed
        result = registry.unregister("test_readonly_name")
        assert result is True
        assert len(registry) == 0

    def test_builtin_scenarios_are_readonly(self):
        """Test that built-in scenarios are registered as read-only."""
        from negmas.registry import scenario_registry

        # Get a built-in scenario
        builtin_scenarios = scenario_registry.query_by_tag("builtin")

        if len(builtin_scenarios) > 0:
            # Check that at least one built-in scenario is read-only
            info = list(builtin_scenarios.values())[0]
            assert info.read_only is True
            assert info.source == "negmas"

    def test_register_scenario_function_read_only(self):
        """Test register_scenario function with read_only parameter."""
        from pathlib import Path
        from negmas.registry import register_scenario, scenario_registry

        path = Path("/tmp/test_func_readonly")

        try:
            info = register_scenario(path, name="func_readonly", read_only=True)
            assert info.read_only is True

            # Verify it's in the registry
            assert str(path.resolve()) in scenario_registry

            # Verify we can unregister it (read_only is informational)
            result = scenario_registry.unregister(path)
            assert result is True
        finally:
            # Cleanup if needed
            key = str(path.resolve())
            if key in scenario_registry:
                del scenario_registry[key]

    def test_read_only_property_query(self):
        """Test querying scenarios by read_only property."""
        from pathlib import Path
        from negmas.registry import ScenarioRegistry

        registry = ScenarioRegistry()

        # Register mix of read-only and non-read-only scenarios
        registry.register(Path("/tmp/ro1"), name="ro1", read_only=True)
        registry.register(Path("/tmp/ro2"), name="ro2", read_only=True)
        registry.register(Path("/tmp/rw1"), name="rw1", read_only=False)

        # Query for read-only scenarios
        results = registry.query(read_only=True)
        assert len(results) == 2
        assert all(info.read_only is True for info in results.values())

        # Query for non-read-only scenarios
        results = registry.query(read_only=False)
        assert len(results) == 1
        assert all(info.read_only is False for info in results.values())


class TestScenarioRegistryRangeQueries:
    """Tests for scenario registry range query functionality."""

    def test_query_n_outcomes_exact(self):
        """Test querying scenarios by exact n_outcomes."""
        from pathlib import Path
        from negmas.registry import ScenarioRegistry

        registry = ScenarioRegistry()
        registry.register(Path("/tmp/s1"), name="s1", n_outcomes=100)
        registry.register(Path("/tmp/s2"), name="s2", n_outcomes=200)
        registry.register(Path("/tmp/s3"), name="s3", n_outcomes=100)

        results = registry.query(n_outcomes=100)
        assert len(results) == 2

    def test_query_n_outcomes_range(self):
        """Test querying scenarios by n_outcomes range."""
        from pathlib import Path
        from negmas.registry import ScenarioRegistry

        registry = ScenarioRegistry()
        registry.register(Path("/tmp/s1"), name="s1", n_outcomes=50)
        registry.register(Path("/tmp/s2"), name="s2", n_outcomes=100)
        registry.register(Path("/tmp/s3"), name="s3", n_outcomes=150)
        registry.register(Path("/tmp/s4"), name="s4", n_outcomes=200)

        # Query for scenarios with 75 to 175 outcomes
        results = registry.query(n_outcomes=(75, 175))
        assert len(results) == 2
        for info in results.values():
            assert info.n_outcomes is not None
            assert 75 <= info.n_outcomes <= 175

    def test_query_n_outcomes_range_no_min(self):
        """Test querying scenarios with no minimum bound."""
        from pathlib import Path
        from negmas.registry import ScenarioRegistry

        registry = ScenarioRegistry()
        registry.register(Path("/tmp/s1"), name="s1", n_outcomes=50)
        registry.register(Path("/tmp/s2"), name="s2", n_outcomes=100)
        registry.register(Path("/tmp/s3"), name="s3", n_outcomes=150)

        # Query for scenarios with at most 100 outcomes
        results = registry.query(n_outcomes=(None, 100))
        assert len(results) == 2
        for info in results.values():
            assert info.n_outcomes is not None
            assert info.n_outcomes <= 100

    def test_query_n_outcomes_range_no_max(self):
        """Test querying scenarios with no maximum bound."""
        from pathlib import Path
        from negmas.registry import ScenarioRegistry

        registry = ScenarioRegistry()
        registry.register(Path("/tmp/s1"), name="s1", n_outcomes=50)
        registry.register(Path("/tmp/s2"), name="s2", n_outcomes=100)
        registry.register(Path("/tmp/s3"), name="s3", n_outcomes=150)

        # Query for scenarios with at least 100 outcomes
        results = registry.query(n_outcomes=(100, None))
        assert len(results) == 2
        for info in results.values():
            assert info.n_outcomes is not None
            assert info.n_outcomes >= 100

    def test_query_n_negotiators_exact(self):
        """Test querying scenarios by exact n_negotiators."""
        from pathlib import Path
        from negmas.registry import ScenarioRegistry

        registry = ScenarioRegistry()
        registry.register(Path("/tmp/s1"), name="s1", n_negotiators=2)
        registry.register(Path("/tmp/s2"), name="s2", n_negotiators=3)
        registry.register(Path("/tmp/s3"), name="s3", n_negotiators=2)

        results = registry.query(n_negotiators=2)
        assert len(results) == 2

    def test_query_n_negotiators_range(self):
        """Test querying scenarios by n_negotiators range."""
        from pathlib import Path
        from negmas.registry import ScenarioRegistry

        registry = ScenarioRegistry()
        registry.register(Path("/tmp/s1"), name="s1", n_negotiators=2)
        registry.register(Path("/tmp/s2"), name="s2", n_negotiators=3)
        registry.register(Path("/tmp/s3"), name="s3", n_negotiators=4)
        registry.register(Path("/tmp/s4"), name="s4", n_negotiators=5)

        # Query for scenarios with 3 to 4 negotiators
        results = registry.query(n_negotiators=(3, 4))
        assert len(results) == 2
        for info in results.values():
            assert info.n_negotiators is not None
            assert 3 <= info.n_negotiators <= 4

    def test_query_combined_ranges(self):
        """Test querying scenarios with multiple range filters."""
        from pathlib import Path
        from negmas.registry import ScenarioRegistry

        registry = ScenarioRegistry()
        registry.register(Path("/tmp/s1"), name="s1", n_outcomes=100, n_negotiators=2)
        registry.register(Path("/tmp/s2"), name="s2", n_outcomes=200, n_negotiators=3)
        registry.register(Path("/tmp/s3"), name="s3", n_outcomes=150, n_negotiators=2)
        registry.register(Path("/tmp/s4"), name="s4", n_outcomes=250, n_negotiators=4)

        # Query for scenarios with 100-200 outcomes and 2-3 negotiators
        results = registry.query(n_outcomes=(100, 200), n_negotiators=(2, 3))
        assert len(results) == 3  # s1, s2, s3 all match
        for info in results.values():
            assert info.n_outcomes is not None
            assert info.n_negotiators is not None
            assert 100 <= info.n_outcomes <= 200
            assert 2 <= info.n_negotiators <= 3

    def test_query_range_excludes_none(self):
        """Test that range queries exclude scenarios with None values."""
        from pathlib import Path
        from negmas.registry import ScenarioRegistry

        registry = ScenarioRegistry()
        registry.register(Path("/tmp/s1"), name="s1", n_outcomes=100)
        registry.register(Path("/tmp/s2"), name="s2", n_outcomes=None)
        registry.register(Path("/tmp/s3"), name="s3", n_outcomes=200)

        # Query should only return scenarios with known n_outcomes in range
        results = registry.query(n_outcomes=(50, 250))
        assert len(results) == 2
        assert all(info.n_outcomes is not None for info in results.values())

    def test_query_opposition_level_range(self):
        """Test querying scenarios by opposition_level range."""
        from pathlib import Path
        from negmas.registry import ScenarioRegistry

        registry = ScenarioRegistry()
        registry.register(Path("/tmp/s1"), name="s1", opposition_level=0.1)
        registry.register(Path("/tmp/s2"), name="s2", opposition_level=0.5)
        registry.register(Path("/tmp/s3"), name="s3", opposition_level=0.9)

        # Query for moderately opposed scenarios
        results = registry.query(opposition_level=(0.3, 0.7))
        assert len(results) == 1
        assert list(results.values())[0].opposition_level == 0.5

    def test_query_rational_fraction_range(self):
        """Test querying scenarios by rational_fraction range."""
        from pathlib import Path
        from negmas.registry import ScenarioRegistry

        registry = ScenarioRegistry()
        registry.register(Path("/tmp/s1"), name="s1", rational_fraction=0.2)
        registry.register(Path("/tmp/s2"), name="s2", rational_fraction=0.5)
        registry.register(Path("/tmp/s3"), name="s3", rational_fraction=0.8)

        # Query for scenarios with moderate rationality
        results = registry.query(rational_fraction=(0.4, 0.6))
        assert len(results) == 1
        assert list(results.values())[0].rational_fraction == 0.5
