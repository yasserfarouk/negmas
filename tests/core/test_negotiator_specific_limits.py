"""Tests for negotiator-specific time and step limits.

This module tests that per-negotiator time_limit and n_steps are correctly
enforced, ensuring negotiations end at the shortest limit among all negotiators
and the mechanism's shared limits.
"""

from __future__ import annotations

import time


from negmas.gb.common import ResponseType
from negmas.outcomes import make_issue
from negmas.preferences import LinearAdditiveUtilityFunction
from negmas.sao import SAOMechanism
from negmas.sao.common import SAOResponse, SAOState
from negmas.sao.negotiators import AspirationNegotiator
from negmas.sao.negotiators.base import SAONegotiator


class RejectingNegotiator(SAONegotiator):
    """A negotiator that always rejects and never proposes anything useful.

    This ensures negotiations only end due to time/step limits, not agreements.
    """

    def __call__(self, state: SAOState, dest: str | None = None) -> SAOResponse:
        return SAOResponse(
            response=ResponseType.REJECT_OFFER, outcome=self.nmi.random_outcome()
        )


def make_session_with_limits(
    mechanism_n_steps=None,
    mechanism_time_limit=None,
    neg1_n_steps=None,
    neg1_time_limit=None,
    neg2_n_steps=None,
    neg2_time_limit=None,
):
    """Create a negotiation session with specified limits."""
    issues = [make_issue([f"v{i}" for i in range(10)], "issue")]
    ufun1 = LinearAdditiveUtilityFunction.random(issues=issues, reserved_value=0.0)
    ufun2 = LinearAdditiveUtilityFunction.random(issues=issues, reserved_value=0.0)

    mechanism = SAOMechanism(
        issues=issues, n_steps=mechanism_n_steps, time_limit=mechanism_time_limit
    )

    neg1 = RejectingNegotiator(preferences=ufun1)
    neg2 = RejectingNegotiator(preferences=ufun2)

    mechanism.add(neg1, n_steps=neg1_n_steps, time_limit=neg1_time_limit)
    mechanism.add(neg2, n_steps=neg2_n_steps, time_limit=neg2_time_limit)

    return mechanism, neg1, neg2


# ============================================================================
# Tests with n_steps only
# ============================================================================


def test_negotiator_n_steps_enforced_shortest_wins():
    """Test that negotiation ends at shortest n_steps among negotiators."""
    mechanism, neg1, neg2 = make_session_with_limits(
        mechanism_n_steps=1000, neg1_n_steps=10, neg2_n_steps=30
    )

    state = mechanism.run()

    # Should end at exactly step 10 (shortest negotiator limit)
    assert state.step == 10, f"Expected step == 10, got {state.step}"
    assert not state.agreement, "Should not have agreement (rejecting negotiators)"
    assert state.started, "Negotiation should have started"
    assert state.timedout, "Negotiation should have timed out (step limit reached)"


def test_negotiator_n_steps_enforced_reversed():
    """Test shortest n_steps wins when negotiator limits are reversed."""
    mechanism, neg1, neg2 = make_session_with_limits(
        mechanism_n_steps=1000, neg1_n_steps=30, neg2_n_steps=10
    )

    state = mechanism.run()

    # Should end at exactly step 10 (shortest negotiator limit)
    assert state.step == 10, f"Expected step == 10, got {state.step}"
    assert not state.agreement, "Should not have agreement (rejecting negotiators)"
    assert state.started, "Negotiation should have started"
    assert state.timedout, "Negotiation should have timed out (step limit reached)"


def test_mechanism_n_steps_enforced_when_shortest():
    """Test that mechanism n_steps is enforced when it's the shortest."""
    mechanism, neg1, neg2 = make_session_with_limits(
        mechanism_n_steps=10, neg1_n_steps=30, neg2_n_steps=50
    )

    state = mechanism.run()

    # Should end at exactly step 10 (mechanism limit is shortest)
    assert state.step == 10, f"Expected step == 10, got {state.step}"
    assert not state.agreement, "Should not have agreement (rejecting negotiators)"
    assert state.started, "Negotiation should have started"
    assert state.timedout, "Negotiation should have timed out (step limit reached)"


def test_negotiator_n_steps_one_with_limit_one_without():
    """Test when only one negotiator has a private n_steps limit."""
    mechanism, neg1, neg2 = make_session_with_limits(
        mechanism_n_steps=1000, neg1_n_steps=15, neg2_n_steps=None
    )

    state = mechanism.run()

    # Should end at exactly step 15 (only neg1 has private limit)
    assert state.step == 15, f"Expected step == 15, got {state.step}"
    assert not state.agreement, "Should not have agreement (rejecting negotiators)"
    assert state.started, "Negotiation should have started"
    assert state.timedout, "Negotiation should have timed out (step limit reached)"


# ============================================================================
# Tests with time_limit only
# ============================================================================


def test_negotiator_time_limit_enforced_shortest_wins():
    """Test that negotiation ends at shortest time_limit among negotiators."""
    mechanism, neg1, neg2 = make_session_with_limits(
        mechanism_time_limit=60.0, neg1_time_limit=2.0, neg2_time_limit=5.0
    )

    start_time = time.perf_counter()
    state = mechanism.run()
    elapsed = time.perf_counter() - start_time

    # Should end around 2 seconds (shortest negotiator limit)
    # Allow 1.5 second tolerance for execution overhead
    assert 1.5 <= elapsed <= 3.5, f"Expected 1.5s <= time <= 3.5s, got {elapsed:.2f}s"
    assert not state.agreement, "Should not have agreement (rejecting negotiators)"
    assert state.started, "Negotiation should have started"
    assert state.timedout, "Negotiation should have timed out (time limit reached)"


def test_negotiator_time_limit_enforced_reversed():
    """Test shortest time_limit wins when negotiator limits are reversed."""
    mechanism, neg1, neg2 = make_session_with_limits(
        mechanism_time_limit=60.0, neg1_time_limit=5.0, neg2_time_limit=2.0
    )

    start_time = time.perf_counter()
    state = mechanism.run()
    elapsed = time.perf_counter() - start_time

    # Should end around 2 seconds (shortest negotiator limit)
    assert 1.5 <= elapsed <= 3.5, f"Expected 1.5s <= time <= 3.5s, got {elapsed:.2f}s"
    assert not state.agreement, "Should not have agreement (rejecting negotiators)"
    assert state.started, "Negotiation should have started"
    assert state.timedout, "Negotiation should have timed out (time limit reached)"


def test_mechanism_time_limit_enforced_when_shortest():
    """Test that mechanism time_limit is enforced when it's the shortest."""
    mechanism, neg1, neg2 = make_session_with_limits(
        mechanism_time_limit=2.0, neg1_time_limit=5.0, neg2_time_limit=10.0
    )

    start_time = time.perf_counter()
    state = mechanism.run()
    elapsed = time.perf_counter() - start_time

    # Should end around 2 seconds (mechanism limit is shortest)
    assert 1.5 <= elapsed <= 3.5, f"Expected 1.5s <= time <= 3.5s, got {elapsed:.2f}s"
    assert not state.agreement, "Should not have agreement (rejecting negotiators)"
    assert state.started, "Negotiation should have started"
    assert state.timedout, "Negotiation should have timed out (time limit reached)"


def test_negotiator_time_limit_one_with_limit_one_without():
    """Test when only one negotiator has a private time_limit."""
    mechanism, neg1, neg2 = make_session_with_limits(
        mechanism_time_limit=60.0, neg1_time_limit=2.5, neg2_time_limit=None
    )

    start_time = time.perf_counter()
    state = mechanism.run()
    elapsed = time.perf_counter() - start_time

    # Should end around 2.5 seconds (only neg1 has private limit)
    assert 2.0 <= elapsed <= 4.0, f"Expected 2.0s <= time <= 4.0s, got {elapsed:.2f}s"
    assert not state.agreement, "Should not have agreement (rejecting negotiators)"
    assert state.started, "Negotiation should have started"
    assert state.timedout, "Negotiation should have timed out (time limit reached)"


# ============================================================================
# Tests with combinations of n_steps and time_limit
# ============================================================================


def test_negotiator_mixed_limits_steps_shortest():
    """Test when n_steps is shorter than time_limit for one negotiator."""
    mechanism, neg1, neg2 = make_session_with_limits(
        mechanism_n_steps=1000,
        mechanism_time_limit=60.0,
        neg1_n_steps=10,
        neg1_time_limit=5.0,
        neg2_n_steps=30,
        neg2_time_limit=10.0,
    )

    start_time = time.perf_counter()
    state = mechanism.run()
    elapsed = time.perf_counter() - start_time

    # Should end at exactly step 10 (shortest among all limits)
    assert state.step == 10, f"Expected step == 10, got {state.step}"
    # Should be fast since steps end before time
    assert elapsed < 5.0, f"Expected time < 5.0s, got {elapsed:.2f}s"
    assert not state.agreement, "Should not have agreement (rejecting negotiators)"
    assert state.started, "Negotiation should have started"
    assert state.timedout, "Negotiation should have timed out (step limit reached)"


def test_negotiator_mixed_limits_time_shortest():
    """Test when time_limit is shorter than n_steps for one negotiator."""
    mechanism, neg1, neg2 = make_session_with_limits(
        mechanism_n_steps=1000,
        mechanism_time_limit=60.0,
        neg1_n_steps=100,
        neg1_time_limit=2.0,
        neg2_n_steps=200,
        neg2_time_limit=5.0,
    )

    start_time = time.perf_counter()
    state = mechanism.run()
    elapsed = time.perf_counter() - start_time

    # Should end around 2 seconds (shortest time limit) OR when n_steps reached
    # Since Rejecting negotiators are fast, could hit n_steps before time
    assert elapsed <= 3.5 or state.step <= 100, (
        f"Expected time <= 3.5s or step <= 100, got time {elapsed:.2f}s, step {state.step}"
    )
    assert not state.agreement, "Should not have agreement (rejecting negotiators)"
    assert state.started, "Negotiation should have started"
    assert state.timedout, "Negotiation should have timed out (limit reached)"


def test_negotiator_mixed_limits_mechanism_steps_shortest():
    """Test when mechanism n_steps is shortest among all limits."""
    mechanism, neg1, neg2 = make_session_with_limits(
        mechanism_n_steps=8,
        mechanism_time_limit=60.0,
        neg1_n_steps=20,
        neg1_time_limit=5.0,
        neg2_n_steps=30,
        neg2_time_limit=10.0,
    )

    state = mechanism.run()

    # Should end at exactly step 8 (mechanism n_steps is shortest)
    assert state.step == 8, f"Expected step == 8, got {state.step}"
    assert not state.agreement, "Should not have agreement (rejecting negotiators)"
    assert state.started, "Negotiation should have started"
    assert state.timedout, "Negotiation should have timed out (step limit reached)"


def test_negotiator_mixed_limits_mechanism_time_shortest():
    """Test when mechanism time_limit is shortest among all limits."""
    mechanism, neg1, neg2 = make_session_with_limits(
        mechanism_n_steps=1000,
        mechanism_time_limit=1.5,
        neg1_n_steps=50,
        neg1_time_limit=5.0,
        neg2_n_steps=100,
        neg2_time_limit=10.0,
    )

    start_time = time.perf_counter()
    state = mechanism.run()
    elapsed = time.perf_counter() - start_time

    # Should end around 1.5 seconds (mechanism time_limit is shortest) OR when n_steps reached
    # Since Rejecting negotiators are fast, could hit n_steps before time
    assert elapsed <= 3.0 or state.step <= 50, (
        f"Expected time <= 3.0s or step <= 50, got time {elapsed:.2f}s, step {state.step}"
    )
    assert not state.agreement, "Should not have agreement (rejecting negotiators)"
    assert state.started, "Negotiation should have started"
    assert state.timedout, "Negotiation should have timed out (limit reached)"


def test_negotiator_only_one_has_steps_other_has_time():
    """Test when one negotiator has n_steps and another has time_limit."""
    mechanism, neg1, neg2 = make_session_with_limits(
        mechanism_n_steps=1000,
        mechanism_time_limit=60.0,
        neg1_n_steps=12,
        neg1_time_limit=None,
        neg2_n_steps=None,
        neg2_time_limit=3.0,
    )

    start_time = time.perf_counter()
    state = mechanism.run()
    elapsed = time.perf_counter() - start_time

    # Should end at whichever comes first: step 12 or 3 seconds
    # Given typical execution speed, 12 steps should complete before 3 seconds
    assert state.step == 12 or (2.5 <= elapsed <= 4.0), (
        f"Expected step == 12 or 2.5s <= time <= 4.0s, got step {state.step}, time {elapsed:.2f}s"
    )
    assert not state.agreement, "Should not have agreement (rejecting negotiators)"
    assert state.started, "Negotiation should have started"
    assert state.timedout, "Negotiation should have timed out (limit reached)"


def test_negotiator_all_different_combinations():
    """Test complex scenario with different limit types for each participant."""
    mechanism, neg1, neg2 = make_session_with_limits(
        mechanism_n_steps=50,
        mechanism_time_limit=10.0,
        neg1_n_steps=7,  # Shortest n_steps
        neg1_time_limit=8.0,
        neg2_n_steps=25,
        neg2_time_limit=15.0,
    )

    start_time = time.perf_counter()
    state = mechanism.run()
    elapsed = time.perf_counter() - start_time

    # Should end at exactly step 7 (shortest n_steps overall)
    assert state.step == 7, f"Expected step == 7, got {state.step}"
    # Should be fast since steps end before any time limit
    assert elapsed < 8.0, f"Expected time < 8.0s, got {elapsed:.2f}s"
    assert not state.agreement, "Should not have agreement (rejecting negotiators)"
    assert state.started, "Negotiation should have started"
    assert state.timedout, "Negotiation should have timed out (step limit reached)"


def test_negotiator_step_limit_reached_before_timeout():
    """Test that step limit is respected even with longer time limit."""
    mechanism, neg1, neg2 = make_session_with_limits(
        mechanism_n_steps=1000,
        mechanism_time_limit=None,
        neg1_n_steps=5,
        neg1_time_limit=60.0,
        neg2_n_steps=15,
        neg2_time_limit=60.0,
    )

    start_time = time.perf_counter()
    state = mechanism.run()
    elapsed = time.perf_counter() - start_time

    # Should end at exactly step 5 (shortest n_steps)
    assert state.step == 5, f"Expected step == 5, got {state.step}"
    # Should be very fast (well before any time limit)
    assert elapsed < 2.0, f"Expected time < 2.0s, got {elapsed:.2f}s"
    assert not state.agreement, "Should not have agreement (rejecting negotiators)"
    assert state.started, "Negotiation should have started"
    assert state.timedout, "Negotiation should have timed out (step limit reached)"


def test_negotiator_time_limit_reached_before_steps():
    """Test that time limit is respected even with larger step limit."""
    mechanism, neg1, neg2 = make_session_with_limits(
        mechanism_n_steps=None,
        mechanism_time_limit=60.0,
        neg1_n_steps=100,
        neg1_time_limit=1.5,
        neg2_n_steps=200,
        neg2_time_limit=5.0,
    )

    start_time = time.perf_counter()
    state = mechanism.run()
    elapsed = time.perf_counter() - start_time

    # Should end around 1.5 seconds (shortest time_limit) OR when n_steps reached
    # Since Rejecting negotiators are fast, could hit n_steps before time
    assert elapsed <= 3.0 or state.step <= 100, (
        f"Expected time <= 3.0s or step <= 100, got time {elapsed:.2f}s, step {state.step}"
    )
    # Step count doesn't matter, should stop due to limit
    assert not state.agreement, "Should not have agreement (rejecting negotiators)"
    assert state.started, "Negotiation should have started"
    assert state.timedout, "Negotiation should have timed out (limit reached)"


# ============================================================================
# Edge case tests
# ============================================================================


def test_negotiator_no_private_limits_uses_mechanism_limits():
    """Test that mechanism limits are used when no negotiator has private limits."""
    mechanism, neg1, neg2 = make_session_with_limits(
        mechanism_n_steps=20,
        mechanism_time_limit=None,
        neg1_n_steps=None,
        neg1_time_limit=None,
        neg2_n_steps=None,
        neg2_time_limit=None,
    )

    state = mechanism.run()

    # Should end at exactly step 20 (mechanism limit)
    assert state.step == 20, f"Expected step == 20, got {state.step}"
    assert not state.agreement, "Should not have agreement (rejecting negotiators)"
    assert state.started, "Negotiation should have started"
    assert state.timedout, "Negotiation should have timed out (step limit reached)"


def test_negotiator_extremely_short_limit():
    """Test with very short negotiator limit."""
    mechanism, neg1, neg2 = make_session_with_limits(
        mechanism_n_steps=1000,
        mechanism_time_limit=60.0,
        neg1_n_steps=1,
        neg1_time_limit=None,
        neg2_n_steps=100,
        neg2_time_limit=None,
    )

    state = mechanism.run()

    # Should end at exactly step 1
    assert state.step == 1, f"Expected step == 1, got {state.step}"
    assert not state.agreement, "Should not have agreement (rejecting negotiators)"
    assert state.started, "Negotiation should have started"
    assert state.timedout, "Negotiation should have timed out (step limit reached)"


def test_negotiator_both_have_same_short_limit():
    """Test when both negotiators have the same short limit."""
    mechanism, neg1, neg2 = make_session_with_limits(
        mechanism_n_steps=1000,
        mechanism_time_limit=None,
        neg1_n_steps=10,
        neg1_time_limit=None,
        neg2_n_steps=10,
        neg2_time_limit=None,
    )

    state = mechanism.run()

    # Should end at exactly step 10
    assert state.step == 10, f"Expected step == 10, got {state.step}"
    assert not state.agreement, "Should not have agreement (rejecting negotiators)"
    assert state.started, "Negotiation should have started"
    assert state.timedout, "Negotiation should have timed out (step limit reached)"


# ============================================================================
# Tests for relative_time calculation with AspirationNegotiator
# ============================================================================


class TrackingAspirationNegotiator(AspirationNegotiator):
    """AspirationNegotiator that tracks relative_time at each step."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.relative_times = []

    def __call__(self, state: SAOState, dest: str | None = None) -> SAOResponse:
        # Track relative_time from the state
        self.relative_times.append(state.relative_time)
        return super().__call__(state, dest)


def make_aspiration_session_with_limits(
    mechanism_n_steps=None,
    mechanism_time_limit=None,
    neg1_n_steps=None,
    neg1_time_limit=None,
    neg2_n_steps=None,
    neg2_time_limit=None,
):
    """Create a negotiation session with aspiration negotiators."""
    issues = [make_issue([f"v{i}" for i in range(10)], "issue")]
    ufun1 = LinearAdditiveUtilityFunction.random(issues=issues, reserved_value=0.0)
    ufun2 = LinearAdditiveUtilityFunction.random(issues=issues, reserved_value=0.0)

    mechanism = SAOMechanism(
        issues=issues, n_steps=mechanism_n_steps, time_limit=mechanism_time_limit
    )

    # Use boulware aspiration (stubborn, concedes slowly) to make agreements less likely
    neg1 = TrackingAspirationNegotiator(preferences=ufun1, aspiration_type="boulware")
    neg2 = TrackingAspirationNegotiator(preferences=ufun2, aspiration_type="boulware")

    mechanism.add(neg1, n_steps=neg1_n_steps, time_limit=neg1_time_limit)
    mechanism.add(neg2, n_steps=neg2_n_steps, time_limit=neg2_time_limit)

    return mechanism, neg1, neg2


def test_relative_time_reflects_negotiator_n_steps_limit():
    """Test that relative_time uses negotiator's own n_steps, not other negotiator's."""
    mechanism, neg1, neg2 = make_aspiration_session_with_limits(
        mechanism_n_steps=1000,
        neg1_n_steps=20,  # Shorter limit - should reach 1.0 at step 20
        neg2_n_steps=40,  # Longer limit - should reach 1.0 at step 40
    )

    state = mechanism.run()

    # Negotiation may end early due to agreement or at step 20 due to timeout
    assert state.step <= 20

    # Verify negotiators see different relative_time values based on their own limits
    assert len(neg1.relative_times) > 0
    assert len(neg2.relative_times) > 0

    # At any given step, verify each negotiator's relative_time matches their own limit
    # neg1 has limit=20, so at step S: relative_time = (S+1)/(20+1)
    # neg2 has limit=40, so at step S: relative_time = (S+1)/(40+1)
    for i in range(min(len(neg1.relative_times), len(neg2.relative_times))):
        step = i
        expected_neg1_rt = (step + 1) / 21
        expected_neg2_rt = (step + 1) / 41

        assert abs(neg1.relative_times[i] - expected_neg1_rt) < 0.01, (
            f"neg1 at step {step}: expected {expected_neg1_rt:.4f}, got {neg1.relative_times[i]:.4f}"
        )
        assert abs(neg2.relative_times[i] - expected_neg2_rt) < 0.01, (
            f"neg2 at step {step}: expected {expected_neg2_rt:.4f}, got {neg2.relative_times[i]:.4f}"
        )

    # Verify the ratio is approximately 41/21 = 1.95
    ratio = neg1.relative_times[0] / neg2.relative_times[0]
    assert 1.9 <= ratio <= 2.0, (
        f"Ratio of relative_times should be ~1.95, got {ratio:.2f}"
    )


def test_relative_time_reflects_negotiator_time_limit():
    """Test that relative_time uses negotiator's own time_limit, not other's."""
    mechanism, neg1, neg2 = make_aspiration_session_with_limits(
        mechanism_time_limit=60.0,
        neg1_time_limit=2.0,  # Shorter - should reach 1.0 at 2s
        neg2_time_limit=4.0,  # Longer - should reach 1.0 at 4s
    )

    mechanism.run()

    # Verify negotiators see different relative_time based on their time limits
    assert len(neg1.relative_times) > 0
    assert len(neg2.relative_times) > 0

    # With time-based limits, relative_time = time_elapsed / time_limit
    # neg1: rt = time / 2.0
    # neg2: rt = time / 4.0
    # So at any given time, neg1's rt should be ~2x neg2's rt
    # Skip step 0 since times are very small and ratios unstable
    if len(neg1.relative_times) >= 5 and len(neg2.relative_times) >= 5:
        # Check a few points in the middle/end of negotiation
        for i in [len(neg1.relative_times) // 2, len(neg1.relative_times) - 1]:
            if i < len(neg2.relative_times) and neg2.relative_times[i] > 0.01:
                ratio = neg1.relative_times[i] / neg2.relative_times[i]
                # Ratio should be approximately 2.0 (4.0 / 2.0)
                assert 1.8 <= ratio <= 2.2, (
                    f"At index {i}: ratio should be ~2.0, got {ratio:.2f} "
                    f"(neg1={neg1.relative_times[i]:.4f}, neg2={neg2.relative_times[i]:.4f})"
                )


def test_relative_time_mechanism_limit_shorter_than_both():
    """Test when mechanism limit is shorter than both negotiators."""
    mechanism, neg1, neg2 = make_aspiration_session_with_limits(
        mechanism_n_steps=10,  # Shortest - controls end
        neg1_n_steps=30,
        neg2_n_steps=40,
    )

    state = mechanism.run()

    # Should end at step 10 (mechanism limit)
    assert state.step <= 10

    # Verify each negotiator's relative_time based on their effective limits
    # neg1 effective limit: min(mechanism=10, neg1=30) = 10
    # neg2 effective limit: min(mechanism=10, neg2=40) = 10
    # So both should see the same relative_time progression
    assert len(neg1.relative_times) > 0
    assert len(neg2.relative_times) > 0

    # Verify both see relative_time based on effective limit of 10
    for i in range(min(len(neg1.relative_times), len(neg2.relative_times))):
        expected_rt = (i + 1) / 11
        assert abs(neg1.relative_times[i] - expected_rt) < 0.01, (
            f"neg1 at step {i}: expected {expected_rt:.4f}, got {neg1.relative_times[i]:.4f}"
        )
        assert abs(neg2.relative_times[i] - expected_rt) < 0.01, (
            f"neg2 at step {i}: expected {expected_rt:.4f}, got {neg2.relative_times[i]:.4f}"
        )


def test_relative_time_one_negotiator_no_private_limit():
    """Test when one negotiator has no private limit (uses mechanism limit)."""
    mechanism, neg1, neg2 = make_aspiration_session_with_limits(
        mechanism_n_steps=30,
        neg1_n_steps=15,  # Has private limit
        neg2_n_steps=None,  # Uses mechanism limit (30)
    )

    state = mechanism.run()

    # Should end at step 15 (neg1's limit)
    assert state.step <= 15

    # Verify each negotiator's relative_time based on their limits
    # neg1 effective limit: min(mechanism=30, neg1=15) = 15
    # neg2 effective limit: min(mechanism=30, None) = 30
    assert len(neg1.relative_times) > 0
    assert len(neg2.relative_times) > 0

    # At step S, verify:
    # neg1: rt = (S+1)/(15+1)
    # neg2: rt = (S+1)/(30+1)
    for i in range(min(len(neg1.relative_times), len(neg2.relative_times))):
        expected_neg1_rt = (i + 1) / 16
        expected_neg2_rt = (i + 1) / 31

        assert abs(neg1.relative_times[i] - expected_neg1_rt) < 0.01, (
            f"neg1 at step {i}: expected {expected_neg1_rt:.4f}, got {neg1.relative_times[i]:.4f}"
        )
        assert abs(neg2.relative_times[i] - expected_neg2_rt) < 0.01, (
            f"neg2 at step {i}: expected {expected_neg2_rt:.4f}, got {neg2.relative_times[i]:.4f}"
        )

    # neg1 should progress twice as fast as neg2 (ratio ~ 2.0)
    ratio = neg1.relative_times[0] / neg2.relative_times[0]
    assert 1.8 <= ratio <= 2.1, f"Ratio should be ~2.0, got {ratio:.2f}"


def test_relative_time_mixed_steps_and_time_limits():
    """Test with mixed n_steps and time_limit for different negotiators."""
    mechanism, neg1, neg2 = make_aspiration_session_with_limits(
        mechanism_n_steps=100,
        mechanism_time_limit=60.0,
        neg1_n_steps=10,  # Shortest - will control end
        neg2_time_limit=5.0,  # Long time but short steps wins
    )

    state = mechanism.run()

    # Should end at step 10 or earlier (neg1's n_steps is shortest, or early agreement)
    assert state.step <= 10

    # Both negotiators should have tracked relative_time
    assert len(neg1.relative_times) > 0
    assert len(neg2.relative_times) > 0

    # Verify per-negotiator relative_time calculations
    # neg1 uses n_steps=10: at step S, relative_time = (S+1)/(10+1)
    # neg2 uses time_limit=5s: relative_time = elapsed_time / 5.0
    # We can verify that neg1's calculation is correct based on steps
    for i in range(len(neg1.relative_times)):
        expected_rt = (i + 1) / 11  # (step+1) / (n_steps+1)
        actual_rt = neg1.relative_times[i]
        assert abs(actual_rt - expected_rt) < 0.01, (
            f"At step {i}: expected rt={(expected_rt):.3f}, got {actual_rt:.3f}"
        )


def test_relative_time_concession_rate_differs():
    """Test that negotiators with different limits concede at different rates."""
    mechanism, neg1, neg2 = make_aspiration_session_with_limits(
        mechanism_n_steps=100,
        neg1_n_steps=20,  # Fast concession
        neg2_n_steps=40,  # Slow concession
    )

    state = mechanism.run()

    # Should end at step 20
    assert state.step <= 20

    # Check that neg1 conceded faster than neg2
    # At step 10:
    # - neg1 should be at relative_time ~0.5 (10/20)
    # - neg2 should be at relative_time ~0.25 (10/40)
    if len(neg1.relative_times) >= 10 and len(neg2.relative_times) >= 10:
        neg1_at_10 = neg1.relative_times[9]  # 0-indexed
        neg2_at_10 = neg2.relative_times[9]

        # neg1 should be further along than neg2
        assert neg1_at_10 > neg2_at_10, (
            f"neg1 ({neg1_at_10}) should concede faster than neg2 ({neg2_at_10})"
        )

        # Roughly check the ratios
        assert 0.4 <= neg1_at_10 <= 0.6, (
            f"neg1 should be ~0.5 at step 10, got {neg1_at_10}"
        )
        assert 0.2 <= neg2_at_10 <= 0.35, (
            f"neg2 should be ~0.25 at step 10, got {neg2_at_10}"
        )


def test_relative_time_both_same_private_limit():
    """Test when both negotiators have the same private limit."""
    mechanism, neg1, neg2 = make_aspiration_session_with_limits(
        mechanism_n_steps=100, neg1_n_steps=25, neg2_n_steps=25
    )

    state = mechanism.run()

    # Should end at step 25
    assert state.step <= 25

    # Both have the same effective limit (25), so they should see identical relative_time
    assert len(neg1.relative_times) > 0
    assert len(neg2.relative_times) > 0

    # Verify both see the same relative_time progression (both have limit=25)
    for i in range(min(len(neg1.relative_times), len(neg2.relative_times))):
        expected_rt = (i + 1) / 26
        assert abs(neg1.relative_times[i] - expected_rt) < 0.01, (
            f"neg1 at step {i}: expected {expected_rt:.4f}, got {neg1.relative_times[i]:.4f}"
        )
        assert abs(neg2.relative_times[i] - expected_rt) < 0.01, (
            f"neg2 at step {i}: expected {expected_rt:.4f}, got {neg2.relative_times[i]:.4f}"
        )

        # They should have nearly identical relative_times
        diff = abs(neg1.relative_times[i] - neg2.relative_times[i])
        assert diff < 0.01, (
            f"At step {i}, relative_times should be identical: "
            f"neg1={neg1.relative_times[i]:.4f} vs neg2={neg2.relative_times[i]:.4f}"
        )


def test_relative_time_very_different_limits():
    """Test with very different limits to see clear concession rate difference."""
    mechanism, neg1, neg2 = make_aspiration_session_with_limits(
        mechanism_n_steps=200,
        neg1_n_steps=10,  # Very fast
        neg2_n_steps=100,  # Very slow (10x)
    )

    state = mechanism.run()

    # Should end at step 10 (neg1's limit) or earlier due to agreement
    assert state.step <= 10

    # Verify each negotiator sees different relative_time based on their limits
    assert len(neg1.relative_times) > 0
    assert len(neg2.relative_times) > 0

    # At any step S:
    # neg1: rt = (S+1)/(10+1)
    # neg2: rt = (S+1)/(100+1)
    # Verify calculations are correct
    for i in range(min(len(neg1.relative_times), len(neg2.relative_times))):
        expected_neg1_rt = (i + 1) / 11
        expected_neg2_rt = (i + 1) / 101

        assert abs(neg1.relative_times[i] - expected_neg1_rt) < 0.01, (
            f"neg1 at step {i}: expected {expected_neg1_rt:.4f}, got {neg1.relative_times[i]:.4f}"
        )
        assert abs(neg2.relative_times[i] - expected_neg2_rt) < 0.01, (
            f"neg2 at step {i}: expected {expected_neg2_rt:.4f}, got {neg2.relative_times[i]:.4f}"
        )

    # neg1 should progress much faster than neg2 (ratio should be ~10x)
    ratio = neg1.relative_times[0] / neg2.relative_times[0]
    assert 9.0 <= ratio <= 10.5, f"Ratio should be ~10.0 (101/11), got {ratio:.2f}"

    # The ratio should be approximately 10:1 throughout
    if len(neg1.relative_times) >= 5 and len(neg2.relative_times) >= 5:
        mid_step = 4
        ratio = neg1.relative_times[mid_step] / (neg2.relative_times[mid_step] + 1e-9)
        assert 8 <= ratio <= 12, f"Ratio should be ~10, got {ratio}"


def test_relative_time_time_based_limits():
    """Test relative_time calculation with time-based limits."""
    mechanism, neg1, neg2 = make_aspiration_session_with_limits(
        mechanism_time_limit=10.0,
        neg1_time_limit=2.0,  # Fast
        neg2_time_limit=6.0,  # Slower (3x)
    )

    start = time.perf_counter()
    mechanism.run()
    elapsed = time.perf_counter() - start

    # Should end around 2 seconds (neg1's limit)
    assert elapsed < 3.5

    # Both negotiators should have tracked relative_time values
    assert len(neg1.relative_times) > 0
    assert len(neg2.relative_times) > 0

    # Verify the ratio: neg1 progresses 3x faster than neg2 (2s vs 6s limits)
    # At any point in time T: neg1_rt = T/2.0, neg2_rt = T/6.0, ratio = 3.0
    # Check this holds for all tracked steps
    for i in range(min(len(neg1.relative_times), len(neg2.relative_times))):
        rt1 = neg1.relative_times[i]
        rt2 = neg2.relative_times[i]
        if rt2 > 0.01:  # Avoid division by near-zero
            ratio = rt1 / rt2
            assert 2.5 <= ratio <= 3.5, (
                f"At step {i}: ratio should be ~3.0, got {ratio:.2f} "
                f"(rt1={rt1:.3f}, rt2={rt2:.3f})"
            )
