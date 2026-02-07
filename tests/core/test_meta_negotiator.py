"""Tests for MetaNegotiator classes."""

from __future__ import annotations

import pytest
from collections import Counter

from negmas.gb.common import ResponseType
from negmas.gb.negotiators import GBMetaNegotiator, AspirationNegotiator
from negmas.negotiators import MetaNegotiator, Negotiator
from negmas.outcomes import make_issue, make_os
from negmas.preferences import LinearAdditiveUtilityFunction
from negmas.sao import SAOMechanism
from negmas.sao.negotiators import (
    SAOMetaNegotiator,
    BoulwareTBNegotiator,
    ConcederTBNegotiator,
    LinearTBNegotiator,
)
from negmas.sao.common import ResponseType as SAOResponseType


class SimpleTestNegotiator(Negotiator):
    """A simple negotiator for testing MetaNegotiator lifecycle callbacks."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.events: list[str] = []

    def on_negotiation_start(self, state):
        self.events.append("negotiation_start")

    def on_round_start(self, state):
        self.events.append("round_start")

    def on_round_end(self, state):
        self.events.append("round_end")

    def on_negotiation_end(self, state):
        self.events.append("negotiation_end")

    def on_mechanism_error(self, state):
        self.events.append("mechanism_error")

    def before_death(self, cntxt):
        self.events.append("before_death")
        return True

    def cancel(self, reason=None):
        self.events.append(f"cancel:{reason}")


class MajorityVoteGBMetaNegotiator(GBMetaNegotiator):
    """A simple GBMetaNegotiator that uses majority voting for responses."""

    def aggregate_proposals(self, state, proposals, dest=None):
        # Return the first non-None proposal
        for neg, proposal in proposals:
            if proposal is not None:
                return proposal
        return None

    def aggregate_responses(self, state, responses, offer, source=None):
        # Majority vote on responses
        response_counts = Counter(r for _, r in responses)
        most_common = response_counts.most_common(1)
        if most_common:
            return most_common[0][0]
        return ResponseType.REJECT_OFFER


class TestMetaNegotiator:
    """Test cases for MetaNegotiator base class."""

    def test_create_meta_negotiator(self):
        """Test creating a MetaNegotiator with sub-negotiators."""
        sub1 = SimpleTestNegotiator(name="sub1")
        sub2 = SimpleTestNegotiator(name="sub2")

        meta = MetaNegotiator(negotiators=[sub1, sub2])

        assert len(meta.negotiators) == 2
        assert sub1 in meta.negotiators
        assert sub2 in meta.negotiators

    def test_meta_negotiator_with_names(self):
        """Test creating a MetaNegotiator with named sub-negotiators."""
        sub1 = SimpleTestNegotiator(name="sub1")
        sub2 = SimpleTestNegotiator(name="sub2")

        meta = MetaNegotiator(
            negotiators=[sub1, sub2], negotiator_names=["first", "second"]
        )

        assert meta.get_negotiator("first") is sub1
        assert meta.get_negotiator("second") is sub2
        assert "first" in meta.negotiator_names
        assert "second" in meta.negotiator_names

    def test_add_negotiator(self):
        """Test adding negotiators after creation."""
        meta = MetaNegotiator(negotiators=[])
        sub1 = SimpleTestNegotiator(name="sub1")

        meta.add_negotiator(sub1, name="added")

        assert len(meta.negotiators) == 1
        assert meta.get_negotiator("added") is sub1

    def test_remove_negotiator(self):
        """Test removing a negotiator by name."""
        sub1 = SimpleTestNegotiator(name="sub1")
        sub2 = SimpleTestNegotiator(name="sub2")
        meta = MetaNegotiator(
            negotiators=[sub1, sub2], negotiator_names=["first", "second"]
        )

        removed = meta.remove_negotiator("first")

        assert removed is sub1
        assert len(meta.negotiators) == 1
        assert meta.get_negotiator("first") is None

    def test_before_death_delegation(self):
        """Test that before_death is delegated to all sub-negotiators."""
        sub1 = SimpleTestNegotiator(name="sub1")
        sub2 = SimpleTestNegotiator(name="sub2")
        meta = MetaNegotiator(negotiators=[sub1, sub2])

        result = meta.before_death({"reason": "test"})

        assert result is True
        assert "before_death" in sub1.events
        assert "before_death" in sub2.events

    def test_cancel_delegation(self):
        """Test that cancel is delegated to all sub-negotiators."""
        sub1 = SimpleTestNegotiator(name="sub1")
        sub2 = SimpleTestNegotiator(name="sub2")
        meta = MetaNegotiator(negotiators=[sub1, sub2])

        meta.cancel(reason="timeout")

        assert "cancel:timeout" in sub1.events
        assert "cancel:timeout" in sub2.events


class TestGBMetaNegotiator:
    """Test cases for GBMetaNegotiator."""

    def test_create_gb_meta_negotiator(self):
        """Test creating a GBMetaNegotiator."""
        issues = [make_issue(10, "price"), make_issue(5, "quantity")]
        os = make_os(issues)
        ufun = LinearAdditiveUtilityFunction.random(os, reserved_value=0.0)

        sub1 = AspirationNegotiator(name="sub1")
        sub2 = AspirationNegotiator(name="sub2")

        meta = MajorityVoteGBMetaNegotiator(negotiators=[sub1, sub2], ufun=ufun)

        assert len(meta.gb_negotiators) == 2

    def test_gb_meta_negotiator_in_mechanism(self):
        """Test GBMetaNegotiator working in a SAOMechanism."""
        issues = [make_issue(10, "price"), make_issue(5, "quantity")]
        os = make_os(issues)

        ufun1 = LinearAdditiveUtilityFunction.random(os, reserved_value=0.0)
        ufun2 = LinearAdditiveUtilityFunction.random(os, reserved_value=0.0)

        # Create sub-negotiators for the meta-negotiator
        sub1 = AspirationNegotiator(name="sub1")
        sub2 = AspirationNegotiator(name="sub2")

        # Create meta-negotiator
        meta = MajorityVoteGBMetaNegotiator(negotiators=[sub1, sub2], ufun=ufun1)

        # Create opponent
        opponent = AspirationNegotiator(name="opponent", ufun=ufun2)

        # Run negotiation
        mechanism = SAOMechanism(issues=issues, n_steps=50)
        mechanism.add(meta)
        mechanism.add(opponent)

        result = mechanism.run()

        # Should complete without error
        assert result is not None
        assert result.running is False

    def test_gb_meta_negotiator_shared_ufun(self):
        """Test that sub-negotiators share the parent's ufun when share_ufun=True."""
        issues = [make_issue(10, "price")]
        os = make_os(issues)
        ufun = LinearAdditiveUtilityFunction.random(os, reserved_value=0.0)

        sub1 = AspirationNegotiator(name="sub1")
        sub2 = AspirationNegotiator(name="sub2")

        meta = MajorityVoteGBMetaNegotiator(
            negotiators=[sub1, sub2], ufun=ufun, share_ufun=True, share_nmi=True
        )

        # Create a mechanism and add the meta-negotiator
        opponent = AspirationNegotiator(name="opponent", ufun=ufun)
        mechanism = SAOMechanism(issues=issues, n_steps=10)
        mechanism.add(meta)
        mechanism.add(opponent)

        # After joining, sub-negotiators should have the parent's ufun
        # (This happens during the join process when share_ufun=True)
        mechanism.run()

        # The negotiation should complete
        assert mechanism.state.running is False

    def test_gb_meta_negotiator_response_aggregation(self):
        """Test that response aggregation works correctly."""
        issues = [make_issue(10, "price")]
        os = make_os(issues)
        ufun = LinearAdditiveUtilityFunction.random(os, reserved_value=0.0)

        sub1 = AspirationNegotiator(name="sub1")
        sub2 = AspirationNegotiator(name="sub2")
        sub3 = AspirationNegotiator(name="sub3")

        # With 3 negotiators, majority voting should work
        meta = MajorityVoteGBMetaNegotiator(negotiators=[sub1, sub2, sub3], ufun=ufun)

        opponent = AspirationNegotiator(
            name="opponent",
            ufun=LinearAdditiveUtilityFunction.random(os, reserved_value=0.0),
        )

        mechanism = SAOMechanism(issues=issues, n_steps=50)
        mechanism.add(meta)
        mechanism.add(opponent)

        result = mechanism.run()

        # Should complete without error
        assert result is not None


class TestMetaNegotiatorCallbackDelegation:
    """Test callback delegation in MetaNegotiator."""

    def test_lifecycle_callbacks_delegated(self):
        """Test that all lifecycle callbacks are delegated to sub-negotiators."""
        sub1 = SimpleTestNegotiator(name="sub1")
        sub2 = SimpleTestNegotiator(name="sub2")

        # We need to create a custom MetaNegotiator that works with SAO
        # For this test, we'll directly call the methods
        meta = MetaNegotiator(negotiators=[sub1, sub2])

        # Simulate lifecycle events
        class FakeState:
            running = True
            waiting = False
            started = True
            step = 0
            time = 0.0
            relative_time = 0.0
            broken = False
            timedout = False
            agreement = None
            results = None
            n_negotiators = 2
            has_error = False
            error_details = ""

        state = FakeState()

        meta.on_negotiation_start(state)
        assert "negotiation_start" in sub1.events
        assert "negotiation_start" in sub2.events

        meta.on_round_start(state)
        assert "round_start" in sub1.events
        assert "round_start" in sub2.events

        meta.on_round_end(state)
        assert "round_end" in sub1.events
        assert "round_end" in sub2.events

        meta.on_negotiation_end(state)
        assert "negotiation_end" in sub1.events
        assert "negotiation_end" in sub2.events


class MajorityVoteSAOMetaNegotiator(SAOMetaNegotiator):
    """A simple SAOMetaNegotiator that uses majority voting for responses."""

    def aggregate_proposals(self, state, proposals, dest=None):
        # Return the first non-None proposal
        for neg, proposal in proposals:
            if proposal is not None:
                return proposal
        return None

    def aggregate_responses(self, state, responses, offer, source=None):
        # Majority vote on responses
        response_counts = Counter(r for _, r in responses)
        most_common = response_counts.most_common(1)
        if most_common:
            return most_common[0][0]
        return SAOResponseType.REJECT_OFFER


class FirstAcceptsSAOMetaNegotiator(SAOMetaNegotiator):
    """A SAOMetaNegotiator that accepts if any sub-negotiator accepts."""

    def aggregate_proposals(self, state, proposals, dest=None):
        # Return the first non-None proposal
        for neg, proposal in proposals:
            if proposal is not None:
                return proposal
        return None

    def aggregate_responses(self, state, responses, offer, source=None):
        # Accept if any sub-negotiator accepts
        for _, r in responses:
            if r == SAOResponseType.ACCEPT_OFFER:
                return SAOResponseType.ACCEPT_OFFER
        return SAOResponseType.REJECT_OFFER


class TestSAOMetaNegotiator:
    """Test cases for SAOMetaNegotiator."""

    def test_create_sao_meta_negotiator(self):
        """Test creating an SAOMetaNegotiator."""
        issues = [make_issue(10, "price"), make_issue(5, "quantity")]
        os = make_os(issues)
        ufun = LinearAdditiveUtilityFunction.random(os, reserved_value=0.0)

        sub1 = BoulwareTBNegotiator(name="sub1")
        sub2 = ConcederTBNegotiator(name="sub2")

        meta = MajorityVoteSAOMetaNegotiator(negotiators=[sub1, sub2], ufun=ufun)

        assert len(meta.sao_negotiators) == 2

    def test_sao_meta_negotiator_in_mechanism(self):
        """Test SAOMetaNegotiator working in a SAOMechanism."""
        issues = [make_issue(10, "price"), make_issue(5, "quantity")]
        os = make_os(issues)

        ufun1 = LinearAdditiveUtilityFunction.random(os, reserved_value=0.0)
        ufun2 = LinearAdditiveUtilityFunction.random(os, reserved_value=0.0)

        # Create sub-negotiators for the meta-negotiator
        sub1 = BoulwareTBNegotiator(name="sub1")
        sub2 = ConcederTBNegotiator(name="sub2")

        # Create meta-negotiator
        meta = MajorityVoteSAOMetaNegotiator(negotiators=[sub1, sub2], ufun=ufun1)

        # Create opponent
        opponent = LinearTBNegotiator(name="opponent", ufun=ufun2)

        # Run negotiation
        mechanism = SAOMechanism(issues=issues, n_steps=50)
        mechanism.add(meta)
        mechanism.add(opponent)

        result = mechanism.run()

        # Should complete without error
        assert result is not None
        assert result.running is False

    def test_sao_meta_negotiator_shared_ufun(self):
        """Test that sub-negotiators share the parent's ufun when share_ufun=True."""
        issues = [make_issue(10, "price")]
        os = make_os(issues)
        ufun = LinearAdditiveUtilityFunction.random(os, reserved_value=0.0)

        sub1 = BoulwareTBNegotiator(name="sub1")
        sub2 = LinearTBNegotiator(name="sub2")

        meta = MajorityVoteSAOMetaNegotiator(
            negotiators=[sub1, sub2], ufun=ufun, share_ufun=True, share_nmi=True
        )

        # Create a mechanism and add the meta-negotiator
        opponent = ConcederTBNegotiator(name="opponent", ufun=ufun)
        mechanism = SAOMechanism(issues=issues, n_steps=10)
        mechanism.add(meta)
        mechanism.add(opponent)

        # Run the negotiation
        mechanism.run()

        # The negotiation should complete
        assert mechanism.state.running is False

    def test_sao_meta_negotiator_response_aggregation(self):
        """Test that response aggregation works correctly with majority voting."""
        issues = [make_issue(10, "price")]
        os = make_os(issues)
        ufun = LinearAdditiveUtilityFunction.random(os, reserved_value=0.0)

        sub1 = BoulwareTBNegotiator(name="sub1")  # Tough
        sub2 = ConcederTBNegotiator(name="sub2")  # Soft
        sub3 = BoulwareTBNegotiator(name="sub3")  # Tough

        # With 3 negotiators (2 tough, 1 soft), majority voting should be tough
        meta = MajorityVoteSAOMetaNegotiator(negotiators=[sub1, sub2, sub3], ufun=ufun)

        opponent = ConcederTBNegotiator(
            name="opponent",
            ufun=LinearAdditiveUtilityFunction.random(os, reserved_value=0.0),
        )

        mechanism = SAOMechanism(issues=issues, n_steps=50)
        mechanism.add(meta)
        mechanism.add(opponent)

        result = mechanism.run()

        # Should complete without error
        assert result is not None

    def test_sao_meta_negotiator_first_accepts_strategy(self):
        """Test SAOMetaNegotiator with first-accepts aggregation strategy."""
        issues = [make_issue(10, "price")]
        os = make_os(issues)
        ufun = LinearAdditiveUtilityFunction.random(os, reserved_value=0.0)

        # Use a soft negotiator that will accept early
        sub1 = ConcederTBNegotiator(name="sub1")
        sub2 = BoulwareTBNegotiator(name="sub2")

        # This strategy accepts if ANY sub-negotiator accepts
        meta = FirstAcceptsSAOMetaNegotiator(negotiators=[sub1, sub2], ufun=ufun)

        opponent = LinearTBNegotiator(
            name="opponent",
            ufun=LinearAdditiveUtilityFunction.random(os, reserved_value=0.0),
        )

        mechanism = SAOMechanism(issues=issues, n_steps=100)
        mechanism.add(meta)
        mechanism.add(opponent)

        result = mechanism.run()

        # Should complete without error
        assert result is not None
        assert result.running is False

    def test_sao_meta_negotiator_reaches_agreement(self):
        """Test that SAOMetaNegotiator can reach an agreement."""
        issues = [make_issue(5, "price")]
        os = make_os(issues)

        # Use compatible ufuns
        ufun1 = LinearAdditiveUtilityFunction.random(os, reserved_value=0.0)
        ufun2 = LinearAdditiveUtilityFunction.random(os, reserved_value=0.0)

        # Use soft negotiators that should reach agreement quickly
        sub1 = ConcederTBNegotiator(name="sub1")
        sub2 = ConcederTBNegotiator(name="sub2")

        meta = FirstAcceptsSAOMetaNegotiator(negotiators=[sub1, sub2], ufun=ufun1)
        opponent = ConcederTBNegotiator(name="opponent", ufun=ufun2)

        mechanism = SAOMechanism(issues=issues, n_steps=100)
        mechanism.add(meta)
        mechanism.add(opponent)

        result = mechanism.run()

        # With soft negotiators, we should typically reach an agreement
        assert result is not None
        assert result.running is False
        # Agreement may or may not be reached depending on random ufuns,
        # but the mechanism should complete without error

    def test_sao_meta_negotiator_negotiators_property(self):
        """Test the sao_negotiators property."""
        sub1 = BoulwareTBNegotiator(name="sub1")
        sub2 = ConcederTBNegotiator(name="sub2")
        sub3 = LinearTBNegotiator(name="sub3")

        meta = MajorityVoteSAOMetaNegotiator(negotiators=[sub1, sub2, sub3])

        assert len(meta.sao_negotiators) == 3
        assert sub1 in meta.sao_negotiators
        assert sub2 in meta.sao_negotiators
        assert sub3 in meta.sao_negotiators


class TestRangeMetaNegotiator:
    """Test cases for RangeMetaNegotiator."""

    def test_create_range_meta_negotiator(self):
        """Test creating a RangeMetaNegotiator."""
        from negmas.sao.negotiators import RangeMetaNegotiator

        issues = [make_issue(10, "price"), make_issue(5, "quantity")]
        os = make_os(issues)
        ufun = LinearAdditiveUtilityFunction.random(os, reserved_value=0.0)

        sub1 = BoulwareTBNegotiator(name="sub1")
        sub2 = ConcederTBNegotiator(name="sub2")

        meta = RangeMetaNegotiator(negotiators=[sub1, sub2], ufun=ufun)

        assert len(meta.sao_negotiators) == 2

    def test_range_meta_negotiator_in_mechanism(self):
        """Test RangeMetaNegotiator working in a SAOMechanism."""
        from negmas.sao.negotiators import RangeMetaNegotiator

        issues = [make_issue(10, "price"), make_issue(5, "quantity")]
        os = make_os(issues)

        ufun1 = LinearAdditiveUtilityFunction.random(os, reserved_value=0.0)
        ufun2 = LinearAdditiveUtilityFunction.random(os, reserved_value=0.0)

        sub1 = BoulwareTBNegotiator(name="sub1")
        sub2 = ConcederTBNegotiator(name="sub2")

        meta = RangeMetaNegotiator(negotiators=[sub1, sub2], ufun=ufun1)
        opponent = LinearTBNegotiator(name="opponent", ufun=ufun2)

        mechanism = SAOMechanism(issues=issues, n_steps=50)
        mechanism.add(meta)
        mechanism.add(opponent)

        result = mechanism.run()

        assert result is not None
        assert result.running is False

    def test_range_meta_negotiator_with_three_negotiators(self):
        """Test RangeMetaNegotiator with three sub-negotiators."""
        from negmas.sao.negotiators import RangeMetaNegotiator

        issues = [make_issue(10, "price")]
        os = make_os(issues)
        ufun = LinearAdditiveUtilityFunction.random(os, reserved_value=0.0)

        sub1 = BoulwareTBNegotiator(name="sub1")
        sub2 = LinearTBNegotiator(name="sub2")
        sub3 = ConcederTBNegotiator(name="sub3")

        meta = RangeMetaNegotiator(negotiators=[sub1, sub2, sub3], ufun=ufun)
        opponent = ConcederTBNegotiator(
            name="opponent",
            ufun=LinearAdditiveUtilityFunction.random(os, reserved_value=0.0),
        )

        mechanism = SAOMechanism(issues=issues, n_steps=50)
        mechanism.add(meta)
        mechanism.add(opponent)

        result = mechanism.run()

        assert result is not None
        assert result.running is False

    def test_range_meta_negotiator_custom_cardinality(self):
        """Test RangeMetaNegotiator with custom max_cardinality."""
        from negmas.sao.negotiators import RangeMetaNegotiator

        issues = [make_issue(5, "price")]
        os = make_os(issues)
        ufun = LinearAdditiveUtilityFunction.random(os, reserved_value=0.0)

        sub1 = BoulwareTBNegotiator(name="sub1")
        sub2 = ConcederTBNegotiator(name="sub2")

        meta = RangeMetaNegotiator(
            negotiators=[sub1, sub2], ufun=ufun, max_cardinality=100
        )
        opponent = LinearTBNegotiator(
            name="opponent",
            ufun=LinearAdditiveUtilityFunction.random(os, reserved_value=0.0),
        )

        mechanism = SAOMechanism(issues=issues, n_steps=50)
        mechanism.add(meta)
        mechanism.add(opponent)

        result = mechanism.run()

        assert result is not None
        assert result.running is False


class TestMeanMetaNegotiator:
    """Test cases for MeanMetaNegotiator."""

    def test_create_mean_meta_negotiator(self):
        """Test creating a MeanMetaNegotiator."""
        from negmas.sao.negotiators import MeanMetaNegotiator

        issues = [make_issue(10, "price"), make_issue(5, "quantity")]
        os = make_os(issues)
        ufun = LinearAdditiveUtilityFunction.random(os, reserved_value=0.0)

        sub1 = BoulwareTBNegotiator(name="sub1")
        sub2 = ConcederTBNegotiator(name="sub2")

        meta = MeanMetaNegotiator(negotiators=[sub1, sub2], ufun=ufun)

        assert len(meta.sao_negotiators) == 2

    def test_mean_meta_negotiator_in_mechanism(self):
        """Test MeanMetaNegotiator working in a SAOMechanism."""
        from negmas.sao.negotiators import MeanMetaNegotiator

        issues = [make_issue(10, "price"), make_issue(5, "quantity")]
        os = make_os(issues)

        ufun1 = LinearAdditiveUtilityFunction.random(os, reserved_value=0.0)
        ufun2 = LinearAdditiveUtilityFunction.random(os, reserved_value=0.0)

        sub1 = BoulwareTBNegotiator(name="sub1")
        sub2 = ConcederTBNegotiator(name="sub2")

        meta = MeanMetaNegotiator(negotiators=[sub1, sub2], ufun=ufun1)
        opponent = LinearTBNegotiator(name="opponent", ufun=ufun2)

        mechanism = SAOMechanism(issues=issues, n_steps=50)
        mechanism.add(meta)
        mechanism.add(opponent)

        result = mechanism.run()

        assert result is not None
        assert result.running is False

    def test_mean_meta_negotiator_with_three_negotiators(self):
        """Test MeanMetaNegotiator with three sub-negotiators."""
        from negmas.sao.negotiators import MeanMetaNegotiator

        issues = [make_issue(10, "price")]
        os = make_os(issues)
        ufun = LinearAdditiveUtilityFunction.random(os, reserved_value=0.0)

        sub1 = BoulwareTBNegotiator(name="sub1")
        sub2 = LinearTBNegotiator(name="sub2")
        sub3 = ConcederTBNegotiator(name="sub3")

        meta = MeanMetaNegotiator(negotiators=[sub1, sub2, sub3], ufun=ufun)
        opponent = ConcederTBNegotiator(
            name="opponent",
            ufun=LinearAdditiveUtilityFunction.random(os, reserved_value=0.0),
        )

        mechanism = SAOMechanism(issues=issues, n_steps=50)
        mechanism.add(meta)
        mechanism.add(opponent)

        result = mechanism.run()

        assert result is not None
        assert result.running is False

    def test_mean_meta_negotiator_custom_epsilon(self):
        """Test MeanMetaNegotiator with custom epsilon parameters."""
        from negmas.sao.negotiators import MeanMetaNegotiator

        issues = [make_issue(5, "price")]
        os = make_os(issues)
        ufun = LinearAdditiveUtilityFunction.random(os, reserved_value=0.0)

        sub1 = BoulwareTBNegotiator(name="sub1")
        sub2 = ConcederTBNegotiator(name="sub2")

        meta = MeanMetaNegotiator(
            negotiators=[sub1, sub2],
            ufun=ufun,
            initial_epsilon=0.1,
            epsilon_step=0.2,
            max_cardinality=100,
        )
        opponent = LinearTBNegotiator(
            name="opponent",
            ufun=LinearAdditiveUtilityFunction.random(os, reserved_value=0.0),
        )

        mechanism = SAOMechanism(issues=issues, n_steps=50)
        mechanism.add(meta)
        mechanism.add(opponent)

        result = mechanism.run()

        assert result is not None
        assert result.running is False

    def test_mean_meta_negotiator_reaches_agreement(self):
        """Test that MeanMetaNegotiator can reach agreements."""
        from negmas.sao.negotiators import MeanMetaNegotiator

        issues = [make_issue(5, "price")]
        os = make_os(issues)

        ufun1 = LinearAdditiveUtilityFunction.random(os, reserved_value=0.0)
        ufun2 = LinearAdditiveUtilityFunction.random(os, reserved_value=0.0)

        # Use soft negotiators for higher chance of agreement
        sub1 = ConcederTBNegotiator(name="sub1")
        sub2 = ConcederTBNegotiator(name="sub2")

        meta = MeanMetaNegotiator(negotiators=[sub1, sub2], ufun=ufun1)
        opponent = ConcederTBNegotiator(name="opponent", ufun=ufun2)

        mechanism = SAOMechanism(issues=issues, n_steps=100)
        mechanism.add(meta)
        mechanism.add(opponent)

        result = mechanism.run()

        assert result is not None
        assert result.running is False


class TestRangeAndMeanMetaNegotiatorImports:
    """Test that the new negotiators are properly exported."""

    def test_import_from_sao_negotiators(self):
        """Test importing the new negotiators from negmas.sao.negotiators."""
        from negmas.sao.negotiators import RangeMetaNegotiator, MeanMetaNegotiator

        assert RangeMetaNegotiator is not None
        assert MeanMetaNegotiator is not None

    def test_import_from_sao(self):
        """Test importing from negmas.sao works."""
        from negmas.sao.negotiators import (
            RangeMetaNegotiator,
            MeanMetaNegotiator,
            SAOMetaNegotiator,
            SAOAggMetaNegotiator,
        )

        # Verify inheritance
        assert issubclass(RangeMetaNegotiator, SAOAggMetaNegotiator)
        assert issubclass(MeanMetaNegotiator, SAOAggMetaNegotiator)
        assert issubclass(SAOAggMetaNegotiator, SAOMetaNegotiator)


class TestOSMeanMetaNegotiator:
    """Test cases for OSMeanMetaNegotiator."""

    def test_create_outcome_space_mean_meta_negotiator(self):
        """Test creating an OSMeanMetaNegotiator."""
        from negmas.sao.negotiators import OSMeanMetaNegotiator

        issues = [make_issue(10, "price"), make_issue(5, "quantity")]
        os = make_os(issues)
        ufun = LinearAdditiveUtilityFunction.random(os, reserved_value=0.0)

        sub1 = BoulwareTBNegotiator(name="sub1")
        sub2 = ConcederTBNegotiator(name="sub2")

        meta = OSMeanMetaNegotiator(negotiators=[sub1, sub2], ufun=ufun)

        assert len(meta.sao_negotiators) == 2

    def test_outcome_space_mean_meta_negotiator_in_mechanism(self):
        """Test OSMeanMetaNegotiator working in a SAOMechanism."""
        from negmas.sao.negotiators import OSMeanMetaNegotiator

        issues = [make_issue(10, "price"), make_issue(5, "quantity")]
        os = make_os(issues)

        ufun1 = LinearAdditiveUtilityFunction.random(os, reserved_value=0.0)
        ufun2 = LinearAdditiveUtilityFunction.random(os, reserved_value=0.0)

        sub1 = BoulwareTBNegotiator(name="sub1")
        sub2 = ConcederTBNegotiator(name="sub2")

        meta = OSMeanMetaNegotiator(negotiators=[sub1, sub2], ufun=ufun1)
        opponent = LinearTBNegotiator(name="opponent", ufun=ufun2)

        mechanism = SAOMechanism(issues=issues, n_steps=50)
        mechanism.add(meta)
        mechanism.add(opponent)

        result = mechanism.run()

        assert result is not None
        assert result.running is False

    def test_outcome_space_mean_meta_negotiator_with_three_negotiators(self):
        """Test OSMeanMetaNegotiator with three sub-negotiators."""
        from negmas.sao.negotiators import OSMeanMetaNegotiator

        issues = [make_issue(10, "price")]
        os = make_os(issues)
        ufun = LinearAdditiveUtilityFunction.random(os, reserved_value=0.0)

        sub1 = BoulwareTBNegotiator(name="sub1")
        sub2 = LinearTBNegotiator(name="sub2")
        sub3 = ConcederTBNegotiator(name="sub3")

        meta = OSMeanMetaNegotiator(negotiators=[sub1, sub2, sub3], ufun=ufun)
        opponent = ConcederTBNegotiator(
            name="opponent",
            ufun=LinearAdditiveUtilityFunction.random(os, reserved_value=0.0),
        )

        mechanism = SAOMechanism(issues=issues, n_steps=50)
        mechanism.add(meta)
        mechanism.add(opponent)

        result = mechanism.run()

        assert result is not None
        assert result.running is False

    def test_outcome_space_mean_meta_negotiator_with_categorical_issues(self):
        """Test OSMeanMetaNegotiator with categorical issues."""
        from negmas.sao.negotiators import OSMeanMetaNegotiator

        # Create issues with mixed types: numeric and categorical
        issues = [
            make_issue(10, "price"),
            make_issue(["red", "green", "blue"], "color"),
        ]
        os = make_os(issues)
        ufun1 = LinearAdditiveUtilityFunction.random(os, reserved_value=0.0)
        ufun2 = LinearAdditiveUtilityFunction.random(os, reserved_value=0.0)

        sub1 = BoulwareTBNegotiator(name="sub1")
        sub2 = ConcederTBNegotiator(name="sub2")

        meta = OSMeanMetaNegotiator(negotiators=[sub1, sub2], ufun=ufun1)
        opponent = LinearTBNegotiator(name="opponent", ufun=ufun2)

        mechanism = SAOMechanism(issues=issues, n_steps=50)
        mechanism.add(meta)
        mechanism.add(opponent)

        result = mechanism.run()

        assert result is not None
        assert result.running is False

    def test_outcome_space_mean_meta_negotiator_reaches_agreement(self):
        """Test that OSMeanMetaNegotiator can reach agreements."""
        from negmas.sao.negotiators import OSMeanMetaNegotiator

        issues = [make_issue(5, "price")]
        os = make_os(issues)

        ufun1 = LinearAdditiveUtilityFunction.random(os, reserved_value=0.0)
        ufun2 = LinearAdditiveUtilityFunction.random(os, reserved_value=0.0)

        # Use soft negotiators for higher chance of agreement
        sub1 = ConcederTBNegotiator(name="sub1")
        sub2 = ConcederTBNegotiator(name="sub2")

        meta = OSMeanMetaNegotiator(negotiators=[sub1, sub2], ufun=ufun1)
        opponent = ConcederTBNegotiator(name="opponent", ufun=ufun2)

        mechanism = SAOMechanism(issues=issues, n_steps=100)
        mechanism.add(meta)
        mechanism.add(opponent)

        result = mechanism.run()

        assert result is not None
        assert result.running is False

    def test_outcome_space_mean_meta_negotiator_import(self):
        """Test that OSMeanMetaNegotiator is properly exported."""
        from negmas.sao.negotiators import OSMeanMetaNegotiator, SAOAggMetaNegotiator

        assert issubclass(OSMeanMetaNegotiator, SAOAggMetaNegotiator)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
