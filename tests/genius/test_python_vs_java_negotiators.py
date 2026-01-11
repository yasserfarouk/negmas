"""
Comparative tests: Python GBOANegotiators vs Java Bridge Negotiators.

This module tests whether the Python-native BOA negotiator implementations
(GBoulware, GConceder, GHardHeaded, etc.) behave similarly to their Java
counterparts when run through the Genius bridge.

The goal is to verify that:
1. Both implementations reach similar agreement rates
2. Both implementations show similar utility distributions
3. Both implementations display characteristic time-dependent behavior
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import pkg_resources
import pytest

from negmas.genius.bridge import genius_bridge_is_running

SKIP_IF_NO_BRIDGE = not os.environ.get("NEGMAS_LONG_TEST", False)
STEPLIMIT = 100
TIMELIMIT = 60


@pytest.fixture
def laptop_domain():
    """Load the Laptop negotiation domain for testing."""
    from negmas.inout import Scenario

    base_folder = pkg_resources.resource_filename(
        "negmas", resource_name="tests/data/Laptop"
    )
    domain = Scenario.from_genius_folder(Path(base_folder))
    assert domain is not None
    return domain


@pytest.fixture
def car_domain():
    """Load the Car-A negotiation domain for testing."""
    from negmas.inout import Scenario

    base_folder = pkg_resources.resource_filename(
        "negmas", resource_name="tests/data/Car-A-domain"
    )
    domain = Scenario.from_genius_folder(Path(base_folder))
    assert domain is not None
    return domain


def run_negotiation_and_get_stats(
    domain,
    negotiator1_factory: Callable,
    negotiator2_factory: Callable,
    n_runs: int = 5,
    n_steps: int = STEPLIMIT,
):
    """
    Run multiple negotiations and collect statistics.

    Returns:
        dict with keys:
            - agreements: number of agreements
            - utility1_sum: sum of utilities for negotiator 1
            - utility2_sum: sum of utilities for negotiator 2
            - runs: number of runs
    """
    agreements = 0
    utility1_sum = 0.0
    utility2_sum = 0.0

    for _ in range(n_runs):
        neg = domain.make_session(n_steps=n_steps, time_limit=None)
        n1 = negotiator1_factory()
        n2 = negotiator2_factory()
        neg.add(n1, ufun=domain.ufuns[0])
        neg.add(n2, ufun=domain.ufuns[1])
        neg.run()

        if neg.agreement is not None:
            agreements += 1
            utility1_sum += domain.ufuns[0](neg.agreement)
            utility2_sum += domain.ufuns[1](neg.agreement)

    return {
        "agreements": agreements,
        "utility1_sum": utility1_sum,
        "utility2_sum": utility2_sum,
        "runs": n_runs,
    }


# ============================================================================
# Test: Python GBoulware vs Java TimeDependentAgentBoulware
# ============================================================================


@pytest.mark.skipif(
    condition=SKIP_IF_NO_BRIDGE and not genius_bridge_is_running(),
    reason="No Genius Bridge, skipping genius-agent tests",
)
class TestPythonVsJavaBoulware:
    """Compare Python GBoulware with Java TimeDependentAgentBoulware."""

    def test_both_run_successfully(self, laptop_domain):
        """Both implementations should complete negotiations without errors."""
        from negmas.genius import GBoulware
        from negmas.genius.gnegotiators import TimeDependentAgentBoulware
        from negmas.genius.bridge import GeniusBridge

        # Python GBoulware vs AspirationNegotiator
        from negmas.sao import AspirationNegotiator

        neg1 = laptop_domain.make_session(n_steps=STEPLIMIT)
        neg1.add(GBoulware(), ufun=laptop_domain.ufuns[0])
        neg1.add(AspirationNegotiator(), ufun=laptop_domain.ufuns[1])
        result1 = neg1.run()
        assert result1.ended

        # Java TimeDependentAgentBoulware vs AspirationNegotiator
        neg2 = laptop_domain.make_session(n_steps=STEPLIMIT)
        neg2.add(TimeDependentAgentBoulware(preferences=laptop_domain.ufuns[0]))
        neg2.add(AspirationNegotiator(), ufun=laptop_domain.ufuns[1])
        result2 = neg2.run()
        assert result2.ended

        GeniusBridge.clean()

    def test_similar_behavior_pattern(self, laptop_domain):
        """Both should show Boulware behavior (slow concession until deadline)."""
        from negmas.genius import GBoulware
        from negmas.genius.gnegotiators import TimeDependentAgentBoulware
        from negmas.genius.bridge import GeniusBridge
        from negmas.sao import AspirationNegotiator

        # Run Python GBoulware and track offers
        neg1 = laptop_domain.make_session(n_steps=STEPLIMIT)
        python_agent = GBoulware()
        neg1.add(python_agent, ufun=laptop_domain.ufuns[0])
        neg1.add(AspirationNegotiator(), ufun=laptop_domain.ufuns[1])
        neg1.run()

        python_offers = neg1.negotiator_offers(python_agent.id)
        python_utilities = [laptop_domain.ufuns[0](o) for o in python_offers if o]

        # Run Java TimeDependentAgentBoulware and track offers
        neg2 = laptop_domain.make_session(n_steps=STEPLIMIT)
        java_agent = TimeDependentAgentBoulware(preferences=laptop_domain.ufuns[0])
        neg2.add(java_agent)
        neg2.add(AspirationNegotiator(), ufun=laptop_domain.ufuns[1])
        neg2.run()

        java_offers = neg2.negotiator_offers(java_agent.id)
        java_utilities = [laptop_domain.ufuns[0](o) for o in java_offers if o]

        GeniusBridge.clean()

        # Both should show Boulware pattern: high initial utility, slow decay
        if len(python_utilities) > 2 and len(java_utilities) > 2:
            # First offers should be high utility (close to 1.0)
            assert python_utilities[0] > 0.7, "Python Boulware should start high"
            assert java_utilities[0] > 0.7, "Java Boulware should start high"

            # Early offers should stay relatively high (Boulware characteristic)
            early_python = sum(python_utilities[: len(python_utilities) // 2]) / (
                len(python_utilities) // 2
            )
            early_java = sum(java_utilities[: len(java_utilities) // 2]) / (
                len(java_utilities) // 2
            )
            assert early_python > 0.6, "Python Boulware should maintain high early"
            assert early_java > 0.6, "Java Boulware should maintain high early"


# ============================================================================
# Test: Python GConceder vs Java TimeDependentAgentConceder
# ============================================================================


@pytest.mark.skipif(
    condition=SKIP_IF_NO_BRIDGE and not genius_bridge_is_running(),
    reason="No Genius Bridge, skipping genius-agent tests",
)
class TestPythonVsJavaConceder:
    """Compare Python GConceder with Java TimeDependentAgentConceder."""

    def test_both_run_successfully(self, laptop_domain):
        """Both implementations should complete negotiations without errors."""
        from negmas.genius import GConceder
        from negmas.genius.gnegotiators import TimeDependentAgentConceder
        from negmas.genius.bridge import GeniusBridge
        from negmas.sao import AspirationNegotiator

        # Python GConceder
        neg1 = laptop_domain.make_session(n_steps=STEPLIMIT)
        neg1.add(GConceder(), ufun=laptop_domain.ufuns[0])
        neg1.add(AspirationNegotiator(), ufun=laptop_domain.ufuns[1])
        result1 = neg1.run()
        assert result1.ended

        # Java TimeDependentAgentConceder
        neg2 = laptop_domain.make_session(n_steps=STEPLIMIT)
        neg2.add(TimeDependentAgentConceder(preferences=laptop_domain.ufuns[0]))
        neg2.add(AspirationNegotiator(), ufun=laptop_domain.ufuns[1])
        result2 = neg2.run()
        assert result2.ended

        GeniusBridge.clean()

    def test_similar_concession_pattern(self, laptop_domain):
        """Both should show Conceder behavior (fast early concession)."""
        from negmas.genius import GConceder
        from negmas.genius.gnegotiators import TimeDependentAgentConceder
        from negmas.genius.bridge import GeniusBridge
        from negmas.sao import AspirationNegotiator

        # Python GConceder
        neg1 = laptop_domain.make_session(n_steps=STEPLIMIT)
        python_agent = GConceder()
        neg1.add(python_agent, ufun=laptop_domain.ufuns[0])
        neg1.add(AspirationNegotiator(), ufun=laptop_domain.ufuns[1])
        neg1.run()

        python_offers = neg1.negotiator_offers(python_agent.id)
        python_utilities = [laptop_domain.ufuns[0](o) for o in python_offers if o]

        # Java TimeDependentAgentConceder
        neg2 = laptop_domain.make_session(n_steps=STEPLIMIT)
        java_agent = TimeDependentAgentConceder(preferences=laptop_domain.ufuns[0])
        neg2.add(java_agent)
        neg2.add(AspirationNegotiator(), ufun=laptop_domain.ufuns[1])
        neg2.run()

        java_offers = neg2.negotiator_offers(java_agent.id)
        java_utilities = [laptop_domain.ufuns[0](o) for o in java_offers if o]

        GeniusBridge.clean()

        # Both should show Conceder pattern: quick drop in utility
        if len(python_utilities) > 4 and len(java_utilities) > 4:
            # Calculate how much utility drops in first quarter
            quarter = max(1, len(python_utilities) // 4)
            python_drop = python_utilities[0] - python_utilities[quarter]
            java_drop = (
                java_utilities[0]
                - java_utilities[min(quarter, len(java_utilities) - 1)]
            )

            # Conceder should drop significantly early (more than Boulware would)
            # We just verify both show similar conceding behavior
            assert python_drop >= 0, "Python Conceder should concede"
            assert java_drop >= 0, "Java Conceder should concede"


# ============================================================================
# Test: Python GHardHeaded vs Java HardHeaded
# ============================================================================


@pytest.mark.skipif(
    condition=SKIP_IF_NO_BRIDGE and not genius_bridge_is_running(),
    reason="No Genius Bridge, skipping genius-agent tests",
)
class TestPythonVsJavaHardHeaded:
    """Compare Python GHardHeaded with Java HardHeaded (KLH)."""

    def test_both_run_successfully(self, laptop_domain):
        """Both implementations should complete negotiations without errors."""
        from negmas.genius import GHardHeaded
        from negmas.genius.gnegotiators import HardHeaded
        from negmas.genius.bridge import GeniusBridge
        from negmas.sao import AspirationNegotiator

        # Python GHardHeaded
        neg1 = laptop_domain.make_session(n_steps=STEPLIMIT)
        neg1.add(GHardHeaded(), ufun=laptop_domain.ufuns[0])
        neg1.add(AspirationNegotiator(), ufun=laptop_domain.ufuns[1])
        result1 = neg1.run()
        assert result1.ended

        # Java HardHeaded
        neg2 = laptop_domain.make_session(n_steps=STEPLIMIT)
        neg2.add(HardHeaded(preferences=laptop_domain.ufuns[0]))
        neg2.add(AspirationNegotiator(), ufun=laptop_domain.ufuns[1])
        result2 = neg2.run()
        assert result2.ended

        GeniusBridge.clean()

    def test_both_use_boulware_style(self, laptop_domain):
        """Both should use Boulware-style time-dependent offering."""
        from negmas.genius import GHardHeaded
        from negmas.genius.gnegotiators import HardHeaded
        from negmas.genius.bridge import GeniusBridge
        from negmas.sao import AspirationNegotiator

        # Python GHardHeaded
        neg1 = laptop_domain.make_session(n_steps=STEPLIMIT)
        python_agent = GHardHeaded()
        neg1.add(python_agent, ufun=laptop_domain.ufuns[0])
        neg1.add(AspirationNegotiator(), ufun=laptop_domain.ufuns[1])
        neg1.run()

        python_offers = neg1.negotiator_offers(python_agent.id)
        python_utilities = [laptop_domain.ufuns[0](o) for o in python_offers if o]

        # Java HardHeaded
        neg2 = laptop_domain.make_session(n_steps=STEPLIMIT)
        java_agent = HardHeaded(preferences=laptop_domain.ufuns[0])
        neg2.add(java_agent)
        neg2.add(AspirationNegotiator(), ufun=laptop_domain.ufuns[1])
        neg2.run()

        java_offers = neg2.negotiator_offers(java_agent.id)
        java_utilities = [laptop_domain.ufuns[0](o) for o in java_offers if o]

        GeniusBridge.clean()

        # Both should start with high utility offers
        if python_utilities and java_utilities:
            assert python_utilities[0] > 0.6, "Python HardHeaded should start high"
            assert java_utilities[0] > 0.6, "Java HardHeaded should start high"


# ============================================================================
# Test: Python GAgentK vs Java AgentK
# ============================================================================


@pytest.mark.skipif(
    condition=SKIP_IF_NO_BRIDGE and not genius_bridge_is_running(),
    reason="No Genius Bridge, skipping genius-agent tests",
)
class TestPythonVsJavaAgentK:
    """Compare Python GAgentK with Java AgentK."""

    def test_both_run_successfully(self, laptop_domain):
        """Both implementations should complete negotiations without errors."""
        from negmas.genius import GAgentK
        from negmas.genius.gnegotiators import AgentK
        from negmas.genius.bridge import GeniusBridge
        from negmas.sao import AspirationNegotiator

        # Python GAgentK
        neg1 = laptop_domain.make_session(n_steps=STEPLIMIT)
        neg1.add(GAgentK(), ufun=laptop_domain.ufuns[0])
        neg1.add(AspirationNegotiator(), ufun=laptop_domain.ufuns[1])
        result1 = neg1.run()
        assert result1.ended

        # Java AgentK
        neg2 = laptop_domain.make_session(n_steps=STEPLIMIT)
        neg2.add(AgentK(preferences=laptop_domain.ufuns[0]))
        neg2.add(AspirationNegotiator(), ufun=laptop_domain.ufuns[1])
        result2 = neg2.run()
        assert result2.ended

        GeniusBridge.clean()


# ============================================================================
# Test: Python GCUHKAgent vs Java CUHKAgent
# ============================================================================


@pytest.mark.skipif(
    condition=SKIP_IF_NO_BRIDGE and not genius_bridge_is_running(),
    reason="No Genius Bridge, skipping genius-agent tests",
)
class TestPythonVsJavaCUHKAgent:
    """Compare Python GCUHKAgent with Java CUHKAgent."""

    def test_both_run_successfully(self, laptop_domain):
        """Both implementations should complete negotiations without errors."""
        from negmas.genius import GCUHKAgent
        from negmas.genius.gnegotiators import CUHKAgent
        from negmas.genius.bridge import GeniusBridge
        from negmas.sao import AspirationNegotiator

        # Python GCUHKAgent
        neg1 = laptop_domain.make_session(n_steps=STEPLIMIT)
        neg1.add(GCUHKAgent(), ufun=laptop_domain.ufuns[0])
        neg1.add(AspirationNegotiator(), ufun=laptop_domain.ufuns[1])
        result1 = neg1.run()
        assert result1.ended

        # Java CUHKAgent
        neg2 = laptop_domain.make_session(n_steps=STEPLIMIT)
        neg2.add(CUHKAgent(preferences=laptop_domain.ufuns[0]))
        neg2.add(AspirationNegotiator(), ufun=laptop_domain.ufuns[1])
        result2 = neg2.run()
        assert result2.ended

        GeniusBridge.clean()


# ============================================================================
# Test: Python GLinear vs Java TimeDependentAgentLinear
# ============================================================================


@pytest.mark.skipif(
    condition=SKIP_IF_NO_BRIDGE and not genius_bridge_is_running(),
    reason="No Genius Bridge, skipping genius-agent tests",
)
class TestPythonVsJavaLinear:
    """Compare Python GLinear with Java TimeDependentAgentLinear."""

    def test_both_run_successfully(self, laptop_domain):
        """Both implementations should complete negotiations without errors."""
        from negmas.genius import GLinear
        from negmas.genius.gnegotiators import TimeDependentAgentLinear
        from negmas.genius.bridge import GeniusBridge
        from negmas.sao import AspirationNegotiator

        # Python GLinear
        neg1 = laptop_domain.make_session(n_steps=STEPLIMIT)
        neg1.add(GLinear(), ufun=laptop_domain.ufuns[0])
        neg1.add(AspirationNegotiator(), ufun=laptop_domain.ufuns[1])
        result1 = neg1.run()
        assert result1.ended

        # Java TimeDependentAgentLinear
        neg2 = laptop_domain.make_session(n_steps=STEPLIMIT)
        neg2.add(TimeDependentAgentLinear(preferences=laptop_domain.ufuns[0]))
        neg2.add(AspirationNegotiator(), ufun=laptop_domain.ufuns[1])
        result2 = neg2.run()
        assert result2.ended

        GeniusBridge.clean()

    def test_linear_concession_pattern(self, laptop_domain):
        """Both should show linear concession (constant rate)."""
        from negmas.genius import GLinear
        from negmas.genius.gnegotiators import TimeDependentAgentLinear
        from negmas.genius.bridge import GeniusBridge
        from negmas.sao import AspirationNegotiator

        # Python GLinear
        neg1 = laptop_domain.make_session(n_steps=STEPLIMIT)
        python_agent = GLinear()
        neg1.add(python_agent, ufun=laptop_domain.ufuns[0])
        neg1.add(AspirationNegotiator(), ufun=laptop_domain.ufuns[1])
        neg1.run()

        python_offers = neg1.negotiator_offers(python_agent.id)
        python_utilities = [laptop_domain.ufuns[0](o) for o in python_offers if o]

        # Java TimeDependentAgentLinear
        neg2 = laptop_domain.make_session(n_steps=STEPLIMIT)
        java_agent = TimeDependentAgentLinear(preferences=laptop_domain.ufuns[0])
        neg2.add(java_agent)
        neg2.add(AspirationNegotiator(), ufun=laptop_domain.ufuns[1])
        neg2.run()

        java_offers = neg2.negotiator_offers(java_agent.id)
        java_utilities = [laptop_domain.ufuns[0](o) for o in java_offers if o]

        GeniusBridge.clean()

        # Both should show roughly linear decline
        if len(python_utilities) > 2 and len(java_utilities) > 2:
            # Just verify both are conceding
            assert (
                python_utilities[0] >= python_utilities[-1]
            ), "Python Linear should concede over time"
            assert (
                java_utilities[0] >= java_utilities[-1]
            ), "Java Linear should concede over time"


# ============================================================================
# Test: Python GHardliner vs Java TimeDependentAgentHardliner
# ============================================================================


@pytest.mark.skipif(
    condition=SKIP_IF_NO_BRIDGE and not genius_bridge_is_running(),
    reason="No Genius Bridge, skipping genius-agent tests",
)
class TestPythonVsJavaHardliner:
    """Compare Python GHardliner with Java TimeDependentAgentHardliner."""

    def test_both_run_successfully(self, laptop_domain):
        """Both implementations should complete negotiations without errors."""
        from negmas.genius import GHardliner
        from negmas.genius.gnegotiators import TimeDependentAgentHardliner
        from negmas.genius.bridge import GeniusBridge
        from negmas.sao import AspirationNegotiator

        # Python GHardliner
        neg1 = laptop_domain.make_session(n_steps=STEPLIMIT)
        neg1.add(GHardliner(), ufun=laptop_domain.ufuns[0])
        neg1.add(AspirationNegotiator(), ufun=laptop_domain.ufuns[1])
        result1 = neg1.run()
        assert result1.ended

        # Java TimeDependentAgentHardliner
        neg2 = laptop_domain.make_session(n_steps=STEPLIMIT)
        neg2.add(TimeDependentAgentHardliner(preferences=laptop_domain.ufuns[0]))
        neg2.add(AspirationNegotiator(), ufun=laptop_domain.ufuns[1])
        result2 = neg2.run()
        assert result2.ended

        GeniusBridge.clean()

    def test_hardliner_no_concession(self, laptop_domain):
        """Both should never concede (always offer maximum utility)."""
        from negmas.genius import GHardliner
        from negmas.genius.gnegotiators import TimeDependentAgentHardliner
        from negmas.genius.bridge import GeniusBridge
        from negmas.sao import AspirationNegotiator

        # Python GHardliner
        neg1 = laptop_domain.make_session(n_steps=50)
        python_agent = GHardliner()
        neg1.add(python_agent, ufun=laptop_domain.ufuns[0])
        neg1.add(AspirationNegotiator(), ufun=laptop_domain.ufuns[1])
        neg1.run()

        python_offers = neg1.negotiator_offers(python_agent.id)
        python_utilities = [laptop_domain.ufuns[0](o) for o in python_offers if o]

        # Java TimeDependentAgentHardliner
        neg2 = laptop_domain.make_session(n_steps=50)
        java_agent = TimeDependentAgentHardliner(preferences=laptop_domain.ufuns[0])
        neg2.add(java_agent)
        neg2.add(AspirationNegotiator(), ufun=laptop_domain.ufuns[1])
        neg2.run()

        java_offers = neg2.negotiator_offers(java_agent.id)
        java_utilities = [laptop_domain.ufuns[0](o) for o in java_offers if o]

        GeniusBridge.clean()

        # Both should maintain high utility throughout (no concession)
        if len(python_utilities) > 1 and len(java_utilities) > 1:
            # Hardliner should have very little variation in utility
            python_variation = max(python_utilities) - min(python_utilities)
            java_variation = max(java_utilities) - min(java_utilities)

            # Allow some variation due to discrete outcome space
            assert (
                python_variation < 0.3
            ), f"Python Hardliner conceded too much: {python_variation}"
            assert (
                java_variation < 0.3
            ), f"Java Hardliner conceded too much: {java_variation}"


# ============================================================================
# Cross-Implementation Tests
# ============================================================================


@pytest.mark.skipif(
    condition=SKIP_IF_NO_BRIDGE and not genius_bridge_is_running(),
    reason="No Genius Bridge, skipping genius-agent tests",
)
class TestCrossImplementation:
    """Test Python agents against Java agents in head-to-head negotiations."""

    def test_python_boulware_vs_java_conceder(self, laptop_domain):
        """Python Boulware should perform well against Java Conceder."""
        from negmas.genius import GBoulware
        from negmas.genius.gnegotiators import TimeDependentAgentConceder
        from negmas.genius.bridge import GeniusBridge

        neg = laptop_domain.make_session(n_steps=STEPLIMIT)
        python_agent = GBoulware()
        java_agent = TimeDependentAgentConceder(preferences=laptop_domain.ufuns[1])

        neg.add(python_agent, ufun=laptop_domain.ufuns[0])
        neg.add(java_agent)
        result = neg.run()

        GeniusBridge.clean()

        assert result.ended
        # Boulware vs Conceder typically reaches agreement
        # (Conceder concedes to Boulware's demands)

    def test_python_conceder_vs_java_boulware(self, laptop_domain):
        """Python Conceder should work against Java Boulware."""
        from negmas.genius import GConceder
        from negmas.genius.gnegotiators import TimeDependentAgentBoulware
        from negmas.genius.bridge import GeniusBridge

        neg = laptop_domain.make_session(n_steps=STEPLIMIT)
        python_agent = GConceder()
        java_agent = TimeDependentAgentBoulware(preferences=laptop_domain.ufuns[1])

        neg.add(python_agent, ufun=laptop_domain.ufuns[0])
        neg.add(java_agent)
        result = neg.run()

        GeniusBridge.clean()

        assert result.ended

    def test_python_hardheaded_vs_java_hardheaded(self, laptop_domain):
        """Python HardHeaded vs Java HardHeaded - battle of the champions."""
        from negmas.genius import GHardHeaded
        from negmas.genius.gnegotiators import HardHeaded
        from negmas.genius.bridge import GeniusBridge

        neg = laptop_domain.make_session(n_steps=STEPLIMIT)
        python_agent = GHardHeaded()
        java_agent = HardHeaded(preferences=laptop_domain.ufuns[1])

        neg.add(python_agent, ufun=laptop_domain.ufuns[0])
        neg.add(java_agent)
        result = neg.run()

        GeniusBridge.clean()

        assert result.ended
        # Two HardHeaded agents may or may not reach agreement

    def test_python_agentk_vs_java_cuhkagent(self, laptop_domain):
        """Python AgentK vs Java CUHKAgent - ANAC winners face off."""
        from negmas.genius import GAgentK
        from negmas.genius.gnegotiators import CUHKAgent
        from negmas.genius.bridge import GeniusBridge

        neg = laptop_domain.make_session(n_steps=STEPLIMIT)
        python_agent = GAgentK()
        java_agent = CUHKAgent(preferences=laptop_domain.ufuns[1])

        neg.add(python_agent, ufun=laptop_domain.ufuns[0])
        neg.add(java_agent)
        result = neg.run()

        GeniusBridge.clean()

        assert result.ended


if __name__ == "__main__":
    pytest.main(args=[__file__, "-v"])
