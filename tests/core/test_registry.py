"""Tests for the registry system."""

from __future__ import annotations


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
