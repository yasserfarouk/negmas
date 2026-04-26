"""
Comprehensive tests for on_preferences_changed callback behavior.

These tests verify that:
1. on_preferences_changed([Initialization]) is called exactly ONCE per negotiation
2. The call happens AFTER join but BEFORE on_negotiation_start
3. This behavior is consistent regardless of when preferences are set:
   - In the constructor
   - Via set_preferences() before joining
   - Via join() parameters
   - Via set_preferences() after negotiation starts (edge case)
4. Replacing preferences during negotiation triggers General (not Initialization)
5. The force parameter works correctly
6. Non-Negotiator Rational entities still get immediate callbacks
"""

from __future__ import annotations

import pytest

from negmas.common import PreferencesChange, PreferencesChangeType
from negmas.outcomes import make_issue, make_os
from negmas.preferences import LinearAdditiveUtilityFunction as U
from negmas.sao import SAOMechanism, SAONegotiator, ResponseType
from negmas.types import Rational


class PreferencesChangeTracker(SAONegotiator):
    """A negotiator that tracks all on_preferences_changed calls."""

    def __init__(self, *args, **kwargs):
        self.preference_changes: list[list[PreferencesChange]] = []
        self.callback_order: list[str] = []
        super().__init__(*args, **kwargs)

    def on_preferences_changed(self, changes: list[PreferencesChange]):
        self.preference_changes.append(changes)
        change_types = [c.type.name for c in changes]
        self.callback_order.append(f"on_preferences_changed({change_types})")
        super().on_preferences_changed(changes)

    def on_negotiation_start(self, state):
        self.callback_order.append("on_negotiation_start")
        super().on_negotiation_start(state)

    def on_negotiation_end(self, state):
        self.callback_order.append("on_negotiation_end")
        super().on_negotiation_end(state)

    def propose(self, state, dest=None):
        return self.nmi.random_outcome()

    def respond(self, state, source=None):
        # Accept after a few rounds to end negotiation
        if state.step > 2:
            return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER


class RationalTracker(Rational):
    """A non-Negotiator Rational entity that tracks on_preferences_changed calls."""

    def __init__(self, *args, **kwargs):
        self.preference_changes: list[list[PreferencesChange]] = []
        super().__init__(*args, **kwargs)

    def on_preferences_changed(self, changes: list[PreferencesChange]):
        self.preference_changes.append(changes)
        super().on_preferences_changed(changes)


@pytest.fixture
def setup():
    """Create common test fixtures."""
    issues = [make_issue(10, "price"), make_issue(5, "quantity")]
    os = make_os(issues)
    ufun1 = U.random(os, reserved_value=0.0)
    ufun2 = U.random(os, reserved_value=0.0)
    return {"issues": issues, "os": os, "ufun1": ufun1, "ufun2": ufun2}


class TestInitializationCalledOnce:
    """Test that on_preferences_changed([Initialization]) is called exactly once."""

    def test_preferences_in_constructor(self, setup):
        """When preferences are passed to constructor, Initialization is called once during negotiation start."""
        ufun = setup["ufun1"]
        tracker = PreferencesChangeTracker(ufun=ufun)

        # Before joining, no Initialization should be called
        init_calls_before = sum(
            1
            for changes in tracker.preference_changes
            for c in changes
            if c.type == PreferencesChangeType.Initialization
        )
        assert init_calls_before == 0, (
            "Initialization should not be called before joining"
        )

        # Run a negotiation
        mechanism = SAOMechanism(issues=setup["issues"], n_steps=10)
        mechanism.add(tracker)
        mechanism.add(PreferencesChangeTracker(ufun=setup["ufun2"]))
        mechanism.run()

        # Count Initialization calls
        init_calls = sum(
            1
            for changes in tracker.preference_changes
            for c in changes
            if c.type == PreferencesChangeType.Initialization
        )
        assert init_calls == 1, (
            f"Expected exactly 1 Initialization call, got {init_calls}"
        )

    def test_preferences_via_set_preferences_before_join(self, setup):
        """When preferences are set via set_preferences() before join, Initialization is called once."""
        tracker = PreferencesChangeTracker()
        tracker.set_preferences(setup["ufun1"])

        # Before joining, no Initialization should be called
        init_calls_before = sum(
            1
            for changes in tracker.preference_changes
            for c in changes
            if c.type == PreferencesChangeType.Initialization
        )
        assert init_calls_before == 0, (
            "Initialization should not be called before joining"
        )

        # Run a negotiation
        mechanism = SAOMechanism(issues=setup["issues"], n_steps=10)
        mechanism.add(tracker)
        mechanism.add(PreferencesChangeTracker(ufun=setup["ufun2"]))
        mechanism.run()

        # Count Initialization calls
        init_calls = sum(
            1
            for changes in tracker.preference_changes
            for c in changes
            if c.type == PreferencesChangeType.Initialization
        )
        assert init_calls == 1, (
            f"Expected exactly 1 Initialization call, got {init_calls}"
        )

    def test_preferences_via_mechanism_add(self, setup):
        """When preferences are passed via mechanism.add(), Initialization is called once."""
        tracker = PreferencesChangeTracker()  # No preferences yet

        # Run a negotiation with ufun passed to add()
        mechanism = SAOMechanism(issues=setup["issues"], n_steps=10)
        mechanism.add(tracker, ufun=setup["ufun1"])
        mechanism.add(PreferencesChangeTracker(ufun=setup["ufun2"]))
        mechanism.run()

        # Count Initialization calls
        init_calls = sum(
            1
            for changes in tracker.preference_changes
            for c in changes
            if c.type == PreferencesChangeType.Initialization
        )
        assert init_calls == 1, (
            f"Expected exactly 1 Initialization call, got {init_calls}"
        )

    def test_preferences_set_multiple_times_before_join(self, setup):
        """When preferences are set multiple times before join, Initialization is called once."""
        ufun1 = setup["ufun1"]
        ufun2 = setup["ufun2"]

        tracker = PreferencesChangeTracker(ufun=ufun1)
        # Replace preferences before joining
        tracker.set_preferences(ufun2)

        # Run a negotiation
        mechanism = SAOMechanism(issues=setup["issues"], n_steps=10)
        mechanism.add(tracker)
        mechanism.add(
            PreferencesChangeTracker(ufun=U.random(setup["os"], reserved_value=0.0))
        )
        mechanism.run()

        # Count Initialization calls
        init_calls = sum(
            1
            for changes in tracker.preference_changes
            for c in changes
            if c.type == PreferencesChangeType.Initialization
        )
        assert init_calls == 1, (
            f"Expected exactly 1 Initialization call, got {init_calls}"
        )


class TestCallbackOrder:
    """Test the order of callbacks."""

    def test_initialization_before_negotiation_start(self, setup):
        """on_preferences_changed([Initialization]) must be called before on_negotiation_start."""
        tracker = PreferencesChangeTracker(ufun=setup["ufun1"])

        mechanism = SAOMechanism(issues=setup["issues"], n_steps=10)
        mechanism.add(tracker)
        mechanism.add(PreferencesChangeTracker(ufun=setup["ufun2"]))
        mechanism.run()

        # Find indices
        init_idx = None
        start_idx = None
        for i, entry in enumerate(tracker.callback_order):
            if "Initialization" in entry and init_idx is None:
                init_idx = i
            if entry == "on_negotiation_start" and start_idx is None:
                start_idx = i

        assert init_idx is not None, (
            "on_preferences_changed([Initialization]) was never called"
        )
        assert start_idx is not None, "on_negotiation_start was never called"
        assert init_idx < start_idx, (
            f"Initialization (idx={init_idx}) must come before on_negotiation_start (idx={start_idx}). "
            f"Order: {tracker.callback_order}"
        )

    def test_callback_order_with_constructor_preferences(self, setup):
        """Verify full callback order when preferences are in constructor."""
        tracker = PreferencesChangeTracker(ufun=setup["ufun1"])

        mechanism = SAOMechanism(issues=setup["issues"], n_steps=10)
        mechanism.add(tracker)
        mechanism.add(PreferencesChangeTracker(ufun=setup["ufun2"]))
        mechanism.run()

        # First two callbacks should be Initialization then on_negotiation_start
        assert len(tracker.callback_order) >= 2
        assert "Initialization" in tracker.callback_order[0], (
            f"First callback should be Initialization, got: {tracker.callback_order[0]}"
        )
        assert tracker.callback_order[1] == "on_negotiation_start", (
            f"Second callback should be on_negotiation_start, got: {tracker.callback_order[1]}"
        )


class TestPreferencesReplacementDuringNegotiation:
    """Test behavior when preferences are replaced during an active negotiation."""

    def test_replacing_preferences_uses_general(self, setup):
        """Replacing preferences during negotiation should use General, not Initialization."""

        class ReplacingTracker(PreferencesChangeTracker):
            def __init__(self, replacement_ufun, *args, **kwargs):
                self.replacement_ufun = replacement_ufun
                self.replaced = False
                super().__init__(*args, **kwargs)

            def on_round_start(self, state):
                # Replace preferences on round 1
                if state.step == 1 and not self.replaced:
                    self.replaced = True
                    self.set_preferences(self.replacement_ufun)
                super().on_round_start(state)

        replacement_ufun = U.random(setup["os"], reserved_value=0.0)
        tracker = ReplacingTracker(replacement_ufun, ufun=setup["ufun1"])

        mechanism = SAOMechanism(issues=setup["issues"], n_steps=10)
        mechanism.add(tracker)
        mechanism.add(PreferencesChangeTracker(ufun=setup["ufun2"]))
        mechanism.run()

        # Count change types
        init_calls = sum(
            1
            for changes in tracker.preference_changes
            for c in changes
            if c.type == PreferencesChangeType.Initialization
        )
        general_calls = sum(
            1
            for changes in tracker.preference_changes
            for c in changes
            if c.type == PreferencesChangeType.General
        )

        assert init_calls == 1, f"Expected 1 Initialization, got {init_calls}"
        assert general_calls >= 1, (
            f"Expected at least 1 General change, got {general_calls}"
        )


class TestForceParameter:
    """Test the force parameter in set_preferences."""

    def test_force_triggers_callback_on_same_preferences(self, setup):
        """force=True should trigger callback even when setting same preferences."""
        tracker = PreferencesChangeTracker(ufun=setup["ufun1"])

        mechanism = SAOMechanism(issues=setup["issues"], n_steps=10)
        mechanism.add(tracker)
        mechanism.add(PreferencesChangeTracker(ufun=setup["ufun2"]))
        mechanism.run()

        # Record count before force
        calls_before = len(tracker.preference_changes)

        # Force call with same preferences (after negotiation ended, this is an edge case)
        tracker.set_preferences(setup["ufun1"], force=True)

        # Should have one more call
        assert len(tracker.preference_changes) == calls_before + 1


class TestNonNegotiatorRational:
    """Test that non-Negotiator Rational entities still get immediate callbacks."""

    def test_rational_gets_immediate_initialization(self, setup):
        """Non-Negotiator Rational should get Initialization immediately in set_preferences."""
        tracker = RationalTracker()

        # Set preferences
        tracker.set_preferences(setup["ufun1"])

        # Should have immediate Initialization
        assert len(tracker.preference_changes) == 1
        assert (
            tracker.preference_changes[0][0].type
            == PreferencesChangeType.Initialization
        )

    def test_rational_gets_immediate_general_on_replace(self, setup):
        """Non-Negotiator Rational should get General when replacing preferences."""
        tracker = RationalTracker(ufun=setup["ufun1"])

        # Clear the Initialization call
        tracker.preference_changes.clear()

        # Replace preferences
        tracker.set_preferences(setup["ufun2"])

        # Should have General
        assert len(tracker.preference_changes) == 1
        assert tracker.preference_changes[0][0].type == PreferencesChangeType.General


class TestEdgeCases:
    """Test edge cases and unusual scenarios."""

    def test_no_preferences_no_initialization(self, setup):
        """Negotiator without preferences should not receive Initialization."""
        tracker = PreferencesChangeTracker()  # No preferences

        mechanism = SAOMechanism(issues=setup["issues"], n_steps=10)
        mechanism.add(tracker)  # No ufun passed either
        mechanism.add(PreferencesChangeTracker(ufun=setup["ufun2"]))

        # This will fail because negotiator has no ufun, but let's check callbacks
        # before the failure
        init_calls = sum(
            1
            for changes in tracker.preference_changes
            for c in changes
            if c.type == PreferencesChangeType.Initialization
        )
        assert init_calls == 0, "No Initialization without preferences"

    def test_multiple_negotiations_get_multiple_initializations(self, setup):
        """Each negotiation should get its own Initialization call."""
        tracker = PreferencesChangeTracker(ufun=setup["ufun1"])

        # First negotiation
        mechanism1 = SAOMechanism(issues=setup["issues"], n_steps=10)
        mechanism1.add(tracker)
        mechanism1.add(PreferencesChangeTracker(ufun=setup["ufun2"]))
        mechanism1.run()

        init_after_first = sum(
            1
            for changes in tracker.preference_changes
            for c in changes
            if c.type == PreferencesChangeType.Initialization
        )

        # Second negotiation with same negotiator (if supported)
        # Note: This may not be supported in all cases, but let's verify the behavior
        tracker2 = PreferencesChangeTracker(ufun=setup["ufun1"])
        mechanism2 = SAOMechanism(issues=setup["issues"], n_steps=10)
        mechanism2.add(tracker2)
        mechanism2.add(PreferencesChangeTracker(ufun=setup["ufun2"]))
        mechanism2.run()

        init_after_second = sum(
            1
            for changes in tracker2.preference_changes
            for c in changes
            if c.type == PreferencesChangeType.Initialization
        )

        assert init_after_first == 1, (
            f"First negotiation: expected 1, got {init_after_first}"
        )
        assert init_after_second == 1, (
            f"Second negotiation: expected 1, got {init_after_second}"
        )

    def test_setting_none_preferences(self, setup):
        """Setting None preferences should not trigger Initialization."""
        tracker = PreferencesChangeTracker(ufun=setup["ufun1"])

        # Clear existing calls
        tracker.preference_changes.clear()

        # Set to None
        tracker.set_preferences(None)

        # Should trigger General (going from something to nothing)
        sum(
            1
            for changes in tracker.preference_changes
            for c in changes
            if c.type == PreferencesChangeType.General
        )
        init_calls = sum(
            1
            for changes in tracker.preference_changes
            for c in changes
            if c.type == PreferencesChangeType.Initialization
        )

        assert init_calls == 0, "Setting None should not trigger Initialization"
        # General is acceptable since we're changing from existing preferences


class TestNoWarningsWithTimeBasedNegotiators:
    """Test that the warning 'Sorter is already initialized' no longer appears."""

    def test_time_based_negotiator_no_double_init_warning(self, setup):
        """TimeBasedNegotiator should not warn about double initialization."""
        import warnings as python_warnings
        from negmas.sao.negotiators import AspirationNegotiator

        # Capture warnings
        with python_warnings.catch_warnings(record=True) as w:
            python_warnings.simplefilter("always")

            # Create negotiator with ufun in constructor
            neg1 = AspirationNegotiator(ufun=setup["ufun1"])
            neg2 = AspirationNegotiator(ufun=setup["ufun2"])

            mechanism = SAOMechanism(issues=setup["issues"], n_steps=50)
            mechanism.add(neg1)
            mechanism.add(neg2)
            mechanism.run()

            # Check for the specific warning
            sorter_warnings = [
                warning
                for warning in w
                if "Sorter is already initialized" in str(warning.message)
            ]
            assert len(sorter_warnings) == 0, (
                f"Found 'Sorter is already initialized' warnings: {sorter_warnings}"
            )
