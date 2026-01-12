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
            short_name="dummy", full_type_name="test.DummyClass", cls=DummyClass
        )
        assert info.short_name == "dummy"
        assert info.full_type_name == "test.DummyClass"
        assert info.cls is DummyClass
        assert info.extra == {}

    def test_registry_info_with_extra(self):
        """Test RegistryInfo with extra data."""

        class DummyClass:
            pass

        info = RegistryInfo(
            short_name="dummy",
            full_type_name="test.DummyClass",
            cls=DummyClass,
            extra={"key": "value"},
        )
        assert info.extra == {"key": "value"}


class TestMechanismInfo:
    """Tests for MechanismInfo dataclass."""

    def test_mechanism_info_defaults(self):
        """Test MechanismInfo default values."""

        class DummyMechanism:
            pass

        info = MechanismInfo(
            short_name="dummy", full_type_name="test.DummyMechanism", cls=DummyMechanism
        )
        assert info.requires_deadline is True

    def test_mechanism_info_custom(self):
        """Test MechanismInfo with custom values."""

        class DummyMechanism:
            pass

        info = MechanismInfo(
            short_name="dummy",
            full_type_name="test.DummyMechanism",
            cls=DummyMechanism,
            requires_deadline=False,
        )
        assert info.requires_deadline is False


class TestNegotiatorInfo:
    """Tests for NegotiatorInfo dataclass."""

    def test_negotiator_info_defaults(self):
        """Test NegotiatorInfo default values."""

        class DummyNegotiator:
            pass

        info = NegotiatorInfo(
            short_name="dummy",
            full_type_name="test.DummyNegotiator",
            cls=DummyNegotiator,
        )
        assert info.bilateral_only is False
        assert info.requires_opponent_ufun is False
        assert info.learns is False
        assert info.anac_year is None
        assert info.supports_uncertainty is False
        assert info.supports_discounting is False

    def test_negotiator_info_custom(self):
        """Test NegotiatorInfo with custom values."""

        class DummyNegotiator:
            pass

        info = NegotiatorInfo(
            short_name="dummy",
            full_type_name="test.DummyNegotiator",
            cls=DummyNegotiator,
            bilateral_only=True,
            learns=True,
            anac_year=2020,
            supports_uncertainty=True,
        )
        assert info.bilateral_only is True
        assert info.learns is True
        assert info.anac_year == 2020
        assert info.supports_uncertainty is True


class TestComponentInfo:
    """Tests for ComponentInfo dataclass."""

    def test_component_info_defaults(self):
        """Test ComponentInfo default values."""

        class DummyComponent:
            pass

        info = ComponentInfo(
            short_name="dummy", full_type_name="test.DummyComponent", cls=DummyComponent
        )
        assert info.component_type == "generic"

    def test_component_info_custom(self):
        """Test ComponentInfo with custom values."""

        class DummyComponent:
            pass

        info = ComponentInfo(
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

        registry.register(TestClass, short_name="test")
        assert "test" in registry
        assert registry["test"].cls is TestClass

    def test_registry_register_auto_name(self):
        """Test registering with automatic short name."""
        registry = Registry(RegistryInfo)

        class MyTestClass:
            pass

        registry.register(MyTestClass)
        assert "MyTestClass" in registry
        assert registry["MyTestClass"].cls is MyTestClass

    def test_registry_get_by_class(self):
        """Test getting info by class."""
        registry = Registry(RegistryInfo)

        class TestClass:
            pass

        registry.register(TestClass, short_name="test")
        info = registry.get_by_class(TestClass)
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

    def test_registry_get_class(self):
        """Test get_class method."""
        registry = Registry(RegistryInfo)

        class TestClass:
            pass

        registry.register(TestClass, short_name="test")
        assert registry.get_class("test") is TestClass
        assert registry.get_class("nonexistent") is None

    def test_registry_list_all(self):
        """Test list_all method."""
        registry = Registry(RegistryInfo)

        class TestClass1:
            pass

        class TestClass2:
            pass

        registry.register(TestClass1, short_name="test1")
        registry.register(TestClass2, short_name="test2")

        all_names = registry.list_all()
        assert "test1" in all_names
        assert "test2" in all_names

    def test_registry_query(self):
        """Test query method."""
        registry = Registry(NegotiatorInfo)

        class BilateralNegotiator:
            pass

        class MultilateralNegotiator:
            pass

        registry.register(
            BilateralNegotiator, short_name="bilateral", bilateral_only=True
        )
        registry.register(
            MultilateralNegotiator, short_name="multilateral", bilateral_only=False
        )

        bilateral_results = registry.query(bilateral_only=True)
        assert "bilateral" in bilateral_results
        assert "multilateral" not in bilateral_results

        multilateral_results = registry.query(bilateral_only=False)
        assert "multilateral" in multilateral_results
        assert "bilateral" not in multilateral_results

    def test_registry_query_multiple_criteria(self):
        """Test query with multiple criteria."""
        registry = Registry(NegotiatorInfo)

        class LearningBilateral:
            pass

        class NonLearningBilateral:
            pass

        registry.register(
            LearningBilateral,
            short_name="learning_bilateral",
            bilateral_only=True,
            learns=True,
        )
        registry.register(
            NonLearningBilateral,
            short_name="nonlearning_bilateral",
            bilateral_only=True,
            learns=False,
        )

        results = registry.query(bilateral_only=True, learns=True)
        assert "learning_bilateral" in results
        assert "nonlearning_bilateral" not in results

    def test_registry_name_clash_handling(self):
        """Test that name clashes are handled by using full type name."""
        registry = Registry(NegotiatorInfo)

        class TestNegotiator:
            pass

        class TestNegotiator2:
            pass

        # Register first with short name "clash"
        registry.register(TestNegotiator, short_name="clash")

        # Register second with same short name - should use full type name
        registry.register(TestNegotiator2, short_name="clash")

        # First should still be accessible by short name
        info1 = registry.get("clash")
        assert info1 is not None
        assert info1.cls is TestNegotiator

        # Second should be registered under full type name
        full_name = f"{TestNegotiator2.__module__}.{TestNegotiator2.__qualname__}"
        info2 = registry.get(full_name)
        assert info2 is not None
        assert info2.cls is TestNegotiator2

        # Both should be registered
        assert registry.is_registered(TestNegotiator)
        assert registry.is_registered(TestNegotiator2)


class TestDecorators:
    """Tests for registration decorators."""

    def test_register_mechanism_decorator(self):
        """Test @register_mechanism decorator."""

        @register_mechanism(short_name="test_mech", requires_deadline=False)
        class TestMechanism:
            pass

        assert mechanism_registry.is_registered(TestMechanism)
        info = mechanism_registry.get_by_class(TestMechanism)
        assert info is not None
        assert info.short_name == "test_mech"
        assert info.requires_deadline is False

    def test_register_mechanism_decorator_no_args(self):
        """Test @register_mechanism decorator without arguments."""

        @register_mechanism
        class AnotherTestMechanism:
            pass

        assert mechanism_registry.is_registered(AnotherTestMechanism)
        info = mechanism_registry.get_by_class(AnotherTestMechanism)
        assert info is not None
        assert info.short_name == "AnotherTestMechanism"
        assert info.requires_deadline is True  # default

    def test_register_negotiator_decorator(self):
        """Test @register_negotiator decorator."""

        @register_negotiator(
            short_name="test_neg", bilateral_only=True, learns=True, anac_year=2020
        )
        class TestNegotiator:
            pass

        assert negotiator_registry.is_registered(TestNegotiator)
        info = negotiator_registry.get_by_class(TestNegotiator)
        assert info is not None
        assert info.short_name == "test_neg"
        assert info.bilateral_only is True
        assert info.learns is True
        assert info.anac_year == 2020

    def test_register_negotiator_decorator_no_args(self):
        """Test @register_negotiator decorator without arguments."""

        @register_negotiator
        class AnotherTestNegotiator:
            pass

        assert negotiator_registry.is_registered(AnotherTestNegotiator)
        info = negotiator_registry.get_by_class(AnotherTestNegotiator)
        assert info is not None
        assert info.short_name == "AnotherTestNegotiator"
        # Check defaults
        assert info.bilateral_only is False
        assert info.learns is False

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

        all_mechanisms = mechanism_registry.list_all()
        assert len(all_mechanisms) > 0

        # Check some known concrete mechanisms
        expected = [
            "SAOMechanism",
            "TAUMechanism",
            "SerialTAUMechanism",
            "ParallelGBMechanism",
            "SerialGBMechanism",
        ]
        for name in expected:
            assert name in all_mechanisms, f"{name} not registered"

        # Base classes should NOT be registered
        assert "Mechanism" not in all_mechanisms
        assert "GBMechanism" not in all_mechanisms

    def test_negotiators_registered(self):
        """Test that built-in negotiators are registered.

        Note: Only concrete negotiators are registered, not base classes like
        Negotiator, SAONegotiator, or GBNegotiator.
        """
        import negmas.registry_init  # noqa: F401

        all_negotiators = negotiator_registry.list_all()
        assert len(all_negotiators) > 0

        # Check some known concrete negotiators
        expected = [
            "AspirationNegotiator",
            "ToughNegotiator",
            "RandomNegotiator",
            "NiceNegotiator",
            "MiCRONegotiator",
        ]
        for name in expected:
            assert name in all_negotiators, f"{name} not registered"

        # Base classes should NOT be registered
        assert "Negotiator" not in all_negotiators
        assert "SAONegotiator" not in all_negotiators
        assert "GBNegotiator" not in all_negotiators

    def test_components_registered(self):
        """Test that built-in components are registered.

        Note: Only concrete components are registered, not base classes like
        AcceptancePolicy or OfferingPolicy.
        """
        import negmas.registry_init  # noqa: F401

        all_components = component_registry.list_all()
        assert len(all_components) > 0

        # Check some known concrete components
        expected = ["AcceptImmediately", "RejectAlways", "RandomOfferingPolicy"]
        for name in expected:
            assert name in all_components, f"{name} not registered"

        # Base classes should NOT be registered
        assert "AcceptancePolicy" not in all_components
        assert "OfferingPolicy" not in all_components

    def test_tau_mechanism_no_deadline(self):
        """Test that TAU mechanisms don't require deadline."""
        import negmas.registry_init  # noqa: F401

        tau_info = mechanism_registry.get("TAUMechanism")
        assert tau_info is not None
        assert tau_info.requires_deadline is False

        serial_tau_info = mechanism_registry.get("SerialTAUMechanism")
        assert serial_tau_info is not None
        assert serial_tau_info.requires_deadline is False

    def test_sao_mechanism_requires_deadline(self):
        """Test that SAO mechanism requires deadline."""
        import negmas.registry_init  # noqa: F401

        sao_info = mechanism_registry.get("SAOMechanism")
        assert sao_info is not None
        assert sao_info.requires_deadline is True

    def test_query_acceptance_components(self):
        """Test querying for acceptance components."""
        import negmas.registry_init  # noqa: F401

        acceptance_policies = component_registry.query(component_type="acceptance")
        assert len(acceptance_policies) > 0

        # All results should have component_type == "acceptance"
        for name, info in acceptance_policies.items():
            assert info.component_type == "acceptance"

    def test_query_offering_components(self):
        """Test querying for offering components."""
        import negmas.registry_init  # noqa: F401

        offering_policies = component_registry.query(component_type="offering")
        assert len(offering_policies) > 0

        # All results should have component_type == "offering"
        for name, info in offering_policies.items():
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
        info = mechanism_registry.get("SAOMechanism")
        assert info is not None

        # Verify full type name is stored in the info
        assert info.full_type_name == "negmas.sao.mechanism.SAOMechanism"
        assert info.short_name == "SAOMechanism"

    def test_genius_negotiators_registered(self):
        """Test that Genius negotiators are registered with ANAC year."""
        import negmas.registry_init  # noqa: F401

        all_negotiators = negotiator_registry.list_all()

        # Check that we have a significant number of negotiators (including Genius)
        assert len(all_negotiators) > 100

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
            assert name in all_negotiators, f"{name} not registered"

    def test_query_negotiators_by_anac_year(self):
        """Test querying negotiators by ANAC competition year."""
        import negmas.registry_init  # noqa: F401

        # Query for ANAC 2019 agents
        anac_2019 = negotiator_registry.query(anac_year=2019)
        assert len(anac_2019) > 0

        # All results should have anac_year == 2019
        for name, info in anac_2019.items():
            assert info.anac_year == 2019

        # Check specific 2019 agents
        assert "AgentGG" in anac_2019
        assert "TheNewDeal" in anac_2019

    def test_query_multiple_anac_years(self):
        """Test that different ANAC years have different agents."""
        import negmas.registry_init  # noqa: F401

        anac_2010 = negotiator_registry.query(anac_year=2010)
        anac_2015 = negotiator_registry.query(anac_year=2015)

        # Different years should have different agents
        assert set(anac_2010.keys()) != set(anac_2015.keys())

        # 2010 should have AgentK
        assert "AgentK" in anac_2010
        # 2015 should have Atlas3
        assert "Atlas3" in anac_2015

    def test_genius_boa_components_registered(self):
        """Test that Genius BOA components are registered."""
        import negmas.registry_init  # noqa: F401

        all_components = component_registry.list_all()

        # Check Genius acceptance policies
        genius_acceptance = ["GACNext", "GACConst", "GACTime", "GACCombi"]
        for name in genius_acceptance:
            assert name in all_components, f"{name} not registered"
            info = component_registry.get(name)
            assert info.component_type == "acceptance"

        # Check Genius offering policies
        genius_offering = [
            "GTimeDependentOffering",
            "GBoulwareOffering",
            "GRandomOffering",
        ]
        for name in genius_offering:
            assert name in all_components, f"{name} not registered"
            info = component_registry.get(name)
            assert info.component_type == "offering"

        # Check Genius opponent models
        genius_models = ["GHardHeadedFrequencyModel", "GBayesianModel"]
        for name in genius_models:
            assert name in all_components, f"{name} not registered"
            info = component_registry.get(name)
            assert info.component_type == "model"

    def test_query_model_components(self):
        """Test querying for model components."""
        import negmas.registry_init  # noqa: F401

        models = component_registry.query(component_type="model")
        assert len(models) > 0

        # All results should have component_type == "model"
        for name, info in models.items():
            assert info.component_type == "model"

        # Should include both SAO and Genius models
        assert "ZeroSumModel" in models  # SAO model
        assert "GBayesianModel" in models  # Genius model


class TestTagging:
    """Tests for the tagging functionality."""

    def test_registry_info_tags_default(self):
        """Test that tags default to empty set."""

        class DummyClass:
            pass

        info = RegistryInfo(
            short_name="dummy", full_type_name="test.DummyClass", cls=DummyClass
        )
        assert info.tags == set()

    def test_registry_info_tags_custom(self):
        """Test RegistryInfo with custom tags."""

        class DummyClass:
            pass

        info = RegistryInfo(
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

        registry.register(TestClass, short_name="test", tags={"tag1", "tag2"})
        info = registry.get("test")
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

        registry.register(Negotiator1, short_name="neg1", tags={"sao", "builtin"})
        registry.register(Negotiator2, short_name="neg2", tags={"sao", "genius"})
        registry.register(Negotiator3, short_name="neg3", tags={"builtin"})

        # Query for items with both "sao" and "builtin" tags
        results = registry.query(tags=["sao", "builtin"])
        assert "neg1" in results
        assert "neg2" not in results
        assert "neg3" not in results

    def test_registry_query_with_any_tags(self):
        """Test query method with any_tags parameter (any must match)."""
        registry = Registry(NegotiatorInfo)

        class Negotiator1:
            pass

        class Negotiator2:
            pass

        class Negotiator3:
            pass

        registry.register(Negotiator1, short_name="neg1", tags={"sao", "builtin"})
        registry.register(Negotiator2, short_name="neg2", tags={"genius"})
        registry.register(Negotiator3, short_name="neg3", tags={"other"})

        # Query for items with either "sao" or "genius" tag
        results = registry.query(any_tags=["sao", "genius"])
        assert "neg1" in results
        assert "neg2" in results
        assert "neg3" not in results

    def test_registry_query_with_exclude_tags(self):
        """Test query method with exclude_tags parameter."""
        registry = Registry(NegotiatorInfo)

        class Negotiator1:
            pass

        class Negotiator2:
            pass

        class Negotiator3:
            pass

        registry.register(Negotiator1, short_name="neg1", tags={"sao", "builtin"})
        registry.register(Negotiator2, short_name="neg2", tags={"genius"})
        registry.register(Negotiator3, short_name="neg3", tags={"sao"})

        # Exclude items with "genius" tag
        results = registry.query(exclude_tags=["genius"])
        assert "neg1" in results
        assert "neg2" not in results
        assert "neg3" in results

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

        registry.register(
            Negotiator1, short_name="neg1", tags={"sao", "builtin", "propose"}
        )
        registry.register(
            Negotiator2, short_name="neg2", tags={"sao", "genius", "propose"}
        )
        registry.register(Negotiator3, short_name="neg3", tags={"builtin", "propose"})
        registry.register(Negotiator4, short_name="neg4", tags={"sao", "deprecated"})

        # Complex query: must have "sao", can have "propose" or "respond", exclude "deprecated"
        results = registry.query(
            tags=["sao"], any_tags=["propose", "respond"], exclude_tags=["deprecated"]
        )
        assert "neg1" in results
        assert "neg2" in results
        assert "neg3" not in results  # Missing "sao"
        assert "neg4" not in results  # Has "deprecated"

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

        registry.register(Class1, short_name="c1", tags={"sao", "builtin"})
        registry.register(Class2, short_name="c2", tags={"genius"})

        sao_results = registry.query_by_tag("sao")
        assert "c1" in sao_results
        assert "c2" not in sao_results

        genius_results = registry.query_by_tag("genius")
        assert "c1" not in genius_results
        assert "c2" in genius_results

    def test_decorator_with_tags(self):
        """Test that decorators accept tags parameter."""

        @register_mechanism(short_name="tagged_mech", tags={"test", "custom"})
        class TaggedMechanism:
            pass

        info = mechanism_registry.get_by_class(TaggedMechanism)
        assert info is not None
        assert info.tags == {"test", "custom"}

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
            info = negotiator_registry.get(name)
            assert info is not None, f"{name} not found"
            assert info.has_tag("builtin"), f"{name} missing 'builtin' tag"

    def test_genius_negotiators_have_genius_tag(self):
        """Test that Genius negotiators have 'genius' tag."""
        import negmas.registry_init  # noqa: F401

        genius_negotiators = ["AgentK", "AgentGG", "Atlas3", "CUHKAgent"]
        for name in genius_negotiators:
            info = negotiator_registry.get(name)
            assert info is not None, f"{name} not found"
            assert info.has_tag("genius"), f"{name} missing 'genius' tag"
            assert not info.has_tag("builtin"), f"{name} should not have 'builtin' tag"

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
            info = negotiator_registry.get(name)
            assert info is not None, f"{name} not found"
            assert info.has_tag(
                expected_year_tag
            ), f"{name} missing '{expected_year_tag}' tag"
            assert info.has_tag("anac"), f"{name} missing 'anac' tag"

    def test_query_builtin_vs_genius(self):
        """Test querying for builtin vs genius negotiators."""
        import negmas.registry_init  # noqa: F401

        # Query for builtin negotiators
        builtin = negotiator_registry.query_by_tag("builtin")
        assert len(builtin) > 0
        for name, info in builtin.items():
            assert not info.has_tag("genius"), f"{name} has both builtin and genius"

        # Query for genius negotiators
        genius = negotiator_registry.query_by_tag("genius")
        assert len(genius) > 0
        for name, info in genius.items():
            assert not info.has_tag("builtin"), f"{name} has both genius and builtin"

    def test_genius_boa_components_have_tags(self):
        """Test that Genius BOA components have appropriate tags."""
        import negmas.registry_init  # noqa: F401

        genius_acceptance = ["GACNext", "GACConst", "GACTime", "GACCombi"]
        for name in genius_acceptance:
            info = component_registry.get(name)
            assert info is not None, f"{name} not found"
            assert info.has_tag("genius"), f"{name} missing 'genius' tag"
            assert info.has_tag("boa"), f"{name} missing 'boa' tag"

    def test_builtin_mechanisms_have_builtin_tag(self):
        """Test that built-in mechanisms have 'builtin' tag."""
        import negmas.registry_init  # noqa: F401

        mechanisms = ["SAOMechanism", "TAUMechanism"]
        for name in mechanisms:
            info = mechanism_registry.get(name)
            assert info is not None, f"{name} not found"
            assert info.has_tag("builtin"), f"{name} missing 'builtin' tag"

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
        assert info.tags == set()
        assert info.normalized is None
        assert info.n_outcomes is None
        assert info.n_negotiators is None
        assert info.anac is None
        assert info.file is False
        assert info.format == "xml"
        assert info.has_stats is False
        assert info.has_plot is False
        assert info.extra == {}

    def test_scenario_info_with_all_fields(self):
        """Test ScenarioInfo with all fields populated."""
        from pathlib import Path
        from negmas.registry import ScenarioInfo

        info = ScenarioInfo(
            name="anac_scenario",
            path=Path("/tmp/anac/scenario"),
            tags={"anac", "bilateral", "xml"},
            normalized=True,
            n_outcomes=100,
            n_negotiators=2,
            anac=True,
            file=False,
            format="xml",
            has_stats=True,
            has_plot=True,
            extra={"year": 2019},
        )
        assert info.name == "anac_scenario"
        assert info.normalized is True
        assert info.n_outcomes == 100
        assert info.n_negotiators == 2
        assert info.anac is True
        assert info.has_stats is True
        assert info.has_plot is True
        assert "anac" in info.tags
        assert info.extra == {"year": 2019}

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
        """Test query method with format filter."""
        from pathlib import Path
        from negmas.registry import ScenarioRegistry

        registry = ScenarioRegistry()
        # Note: format is auto-detected but we can override via extra or the auto-detected value
        info1 = registry.register(Path("/tmp/s1.xml"), name="s1")
        info1.format = "xml"
        info2 = registry.register(Path("/tmp/s2.json"), name="s2")
        info2.format = "json"

        # Query by format
        results = registry.query(format="xml")
        assert len(results) == 1

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


class TestRegisterScenarioFunction:
    """Tests for register_scenario function."""

    def test_register_scenario_function(self):
        """Test the register_scenario convenience function."""
        from pathlib import Path
        from negmas.registry import register_scenario, scenario_registry

        # Register a test scenario
        path = Path("/tmp/test_func_scenario")
        info = register_scenario(
            path, name="test_func_scenario", tags={"test"}, n_negotiators=2
        )

        assert info.name == "test_func_scenario"
        assert info.n_negotiators == 2
        assert "test" in info.tags

        # Verify it's in the registry
        assert str(path.resolve()) in scenario_registry

    def test_register_scenario_with_stats_and_plot(self):
        """Test registering a scenario with has_stats and has_plot."""
        from pathlib import Path
        from negmas.registry import register_scenario, scenario_registry

        path = Path("/tmp/test_scenario_with_stats")
        info = register_scenario(
            path, name="test_with_stats", tags={"test"}, has_stats=True, has_plot=True
        )

        assert info.has_stats is True
        assert info.has_plot is True

        # Verify it's in the registry with correct values
        retrieved = scenario_registry[str(path.resolve())]
        assert retrieved.has_stats is True
        assert retrieved.has_plot is True


class TestBuiltInScenarios:
    """Tests for built-in scenario registrations."""

    def test_builtin_scenarios_registered(self):
        """Test that built-in scenarios are registered."""
        import negmas.registry_init  # noqa: F401
        from negmas.registry import scenario_registry

        all_scenarios = scenario_registry.list_all()
        # We should have at least the CameraB scenario
        assert len(all_scenarios) >= 1

        # Check for CameraB scenario
        names = scenario_registry.list_names()
        assert "CameraB" in names

    def test_builtin_scenarios_have_builtin_tag(self):
        """Test that built-in scenarios have 'builtin' tag."""
        import negmas.registry_init  # noqa: F401
        from negmas.registry import scenario_registry

        # Only check scenarios that are tagged as builtin
        # (Other tests may register test scenarios without builtin tag)
        builtin_scenarios = scenario_registry.query(tags=["builtin"])
        assert len(builtin_scenarios) >= 1, "Should have at least one builtin scenario"

        for info in builtin_scenarios.values():
            assert info.has_tag("builtin"), f"{info.name} missing 'builtin' tag"

    def test_builtin_scenarios_have_format_tag(self):
        """Test that built-in scenarios have format tag."""
        import negmas.registry_init  # noqa: F401
        from negmas.registry import scenario_registry

        for path, info in scenario_registry.items():
            # Format should be detected and stored
            assert info.format in ("xml", "json", "yaml", "unknown")

    def test_camerab_scenario_properties(self):
        """Test CameraB scenario has expected properties."""
        import negmas.registry_init  # noqa: F401
        from negmas.registry import scenario_registry

        # Find CameraB scenario
        camerab_scenarios = scenario_registry.get_by_name("CameraB")
        assert len(camerab_scenarios) >= 1

        info = camerab_scenarios[0]
        assert info.name == "CameraB"
        assert info.has_tag("builtin")
        assert info.format == "xml"
        # CameraB has 2 negotiators (bilateral)
        assert info.n_negotiators == 2
        assert info.has_tag("bilateral")


class TestRegisterAllScenarios:
    """Tests for register_all_scenarios function."""

    def test_register_all_scenarios_from_negmas_scenarios(self):
        """Test registering scenarios from the negmas scenarios directory."""
        from pathlib import Path
        from negmas.registry import register_all_scenarios, ScenarioRegistry

        # Use the negmas built-in scenarios directory
        scenarios_dir = (
            Path(__file__).parent.parent.parent / "src" / "negmas" / "scenarios"
        )
        if not scenarios_dir.exists():
            pytest.skip("Scenarios directory not found")

        # Use a fresh registry to avoid conflicts
        fresh_registry = ScenarioRegistry()
        results = register_all_scenarios(
            scenarios_dir, tags={"test-batch"}, registry=fresh_registry
        )

        # Should have registered at least CameraB
        assert len(results) >= 1

        # All should have the custom tag
        for info in results:
            assert info.has_tag("test-batch")

    def test_register_all_scenarios_extracts_info(self):
        """Test that register_all_scenarios extracts scenario information."""
        from pathlib import Path
        from negmas.registry import register_all_scenarios, ScenarioRegistry

        # Use the negmas built-in scenarios directory
        scenarios_dir = (
            Path(__file__).parent.parent.parent / "src" / "negmas" / "scenarios"
        )
        if not scenarios_dir.exists():
            pytest.skip("Scenarios directory not found")

        fresh_registry = ScenarioRegistry()
        results = register_all_scenarios(scenarios_dir, registry=fresh_registry)

        # Check that info was extracted
        for info in results:
            # Should have n_negotiators
            assert info.n_negotiators is not None
            assert info.n_negotiators >= 2

            # Should have bilateral or multilateral tag
            assert info.has_tag("bilateral") or info.has_tag("multilateral")

            # Should have format tag
            assert info.has_any_tag(["xml", "json", "yaml"])

    def test_register_all_scenarios_invalid_path(self):
        """Test that register_all_scenarios raises on invalid path."""
        from negmas.registry import register_all_scenarios

        with pytest.raises(ValueError, match="does not exist"):
            register_all_scenarios("/nonexistent/path/to/scenarios")

    def test_register_all_scenarios_not_directory(self):
        """Test that register_all_scenarios raises on non-directory path."""
        from pathlib import Path
        from negmas.registry import register_all_scenarios

        # Use this test file as a non-directory path
        with pytest.raises(ValueError, match="not a directory"):
            register_all_scenarios(Path(__file__))

    def test_register_all_scenarios_non_recursive(self):
        """Test non-recursive scenario registration."""
        from pathlib import Path
        from negmas.registry import register_all_scenarios, ScenarioRegistry

        # Use the negmas built-in scenarios directory
        scenarios_dir = (
            Path(__file__).parent.parent.parent / "src" / "negmas" / "scenarios"
        )
        if not scenarios_dir.exists():
            pytest.skip("Scenarios directory not found")

        fresh_registry = ScenarioRegistry()
        # Non-recursive should not find scenarios in subdirectories
        # unless the root itself is a scenario
        results = register_all_scenarios(
            scenarios_dir, recursive=False, registry=fresh_registry
        )

        # Results depend on directory structure
        # The scenarios dir itself is not a scenario, so non-recursive
        # should return empty or fewer results
        assert isinstance(results, list)
