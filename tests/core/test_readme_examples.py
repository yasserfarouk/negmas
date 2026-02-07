"""Tests for README.rst code examples.

These tests ensure that all code examples in the README work correctly.
"""

from __future__ import annotations


def test_quick_start_basic():
    """Test the basic Quick Start example."""
    from negmas import make_issue, SAOMechanism, AspirationNegotiator
    from negmas.preferences import LinearAdditiveUtilityFunction
    from negmas.outcomes import make_os

    issues = [make_issue(10, "price")]
    os = make_os(issues)
    ufun1 = LinearAdditiveUtilityFunction.random(os, reserved_value=0.0)
    ufun2 = LinearAdditiveUtilityFunction.random(os, reserved_value=0.0)

    session = SAOMechanism(issues=issues, n_steps=100)
    session.add(AspirationNegotiator(name="a1"), ufun=ufun1)
    session.add(AspirationNegotiator(name="a2"), ufun=ufun2)
    result = session.run()

    # Should complete without error
    assert result is not None
    assert result.step is not None


def test_quick_start_multi_issue():
    """Test the multi-issue negotiation example with custom preferences."""
    from negmas import SAOMechanism, AspirationNegotiator, make_issue
    from negmas.preferences import LinearAdditiveUtilityFunction

    # Create a 2-issue negotiation domain
    issues = [
        make_issue(name="price", values=10),
        make_issue(name="quantity", values=5),
    ]

    # Define utility functions
    buyer_ufun = LinearAdditiveUtilityFunction(
        values={
            "price": lambda x: 1.0 - x / 10.0,  # lower price = better
            "quantity": lambda x: x / 5.0,  # more quantity = better
        },
        issues=issues,
    )
    seller_ufun = LinearAdditiveUtilityFunction(
        values={
            "price": lambda x: x / 10.0,  # higher price = better
            "quantity": lambda x: 1.0 - x / 5.0,  # less quantity = better
        },
        issues=issues,
    )

    # Run negotiation
    session = SAOMechanism(issues=issues, n_steps=100)
    session.add(AspirationNegotiator(name="buyer"), ufun=buyer_ufun)
    session.add(AspirationNegotiator(name="seller"), ufun=seller_ufun)
    result = session.run()

    # Should complete without error
    assert result is not None


def test_inheritance_custom_negotiator():
    """Test the inheritance-based custom negotiator example."""
    from negmas.sao import SAOMechanism, SAONegotiator, ResponseType
    from negmas.preferences import LinearAdditiveUtilityFunction as U
    from negmas.outcomes import make_issue, make_os

    class MyNegotiator(SAONegotiator):
        """A simple negotiator using inheritance."""

        def propose(self, state, dest=None):
            # Propose a random outcome above reservation value
            return self.nmi.random_outcome()

        def respond(self, state, source=None):
            offer = state.current_offer
            # Accept any offer with utility > 0.8
            if offer is not None and self.ufun(offer) > 0.8:
                return ResponseType.ACCEPT_OFFER
            return ResponseType.REJECT_OFFER

    # Use the custom negotiator
    issues = [make_issue(10, "price"), make_issue(5, "quantity")]
    os = make_os(issues)
    session = SAOMechanism(issues=issues, n_steps=100)
    session.add(MyNegotiator(name="custom"), ufun=U.random(os, reserved_value=0.0))
    session.add(MyNegotiator(name="opponent"), ufun=U.random(os, reserved_value=0.0))
    result = session.run()

    # Should complete without error
    assert result is not None


def test_composition_ensemble_meta_negotiator():
    """Test the SAOMetaNegotiator ensemble example."""
    from negmas.sao import SAOMechanism, ResponseType
    from negmas.sao.negotiators import (
        SAOMetaNegotiator,
        BoulwareTBNegotiator,
        NaiveTitForTatNegotiator,
        AspirationNegotiator,
    )
    from negmas.preferences import LinearAdditiveUtilityFunction as U
    from negmas.outcomes import make_issue, make_os

    class MajorityVoteNegotiator(SAOMetaNegotiator):
        """An ensemble negotiator that uses majority voting."""

        def aggregate_proposals(self, state, proposals, dest=None):
            # Use the proposal from the first negotiator that offers something
            for neg, proposal in proposals:
                if proposal is not None:
                    return proposal
            return None

        def aggregate_responses(self, state, responses, offer, source=None):
            # Majority vote: accept if more than half accept
            accept_count = sum(
                1 for _, r in responses if r == ResponseType.ACCEPT_OFFER
            )
            if accept_count > len(responses) / 2:
                return ResponseType.ACCEPT_OFFER
            return ResponseType.REJECT_OFFER

    # Create an ensemble of different strategies
    issues = [make_issue(10, "price"), make_issue(5, "quantity")]
    os = make_os(issues)
    ufun = U.random(os, reserved_value=0.0)

    ensemble = MajorityVoteNegotiator(
        negotiators=[
            BoulwareTBNegotiator(),  # Tough strategy
            NaiveTitForTatNegotiator(),  # Reactive strategy
            BoulwareTBNegotiator(),  # Another tough vote
        ],
        name="ensemble",
    )

    # Use in a negotiation
    session = SAOMechanism(issues=issues, n_steps=100)
    session.add(ensemble, ufun=ufun)
    session.add(
        AspirationNegotiator(name="opponent"), ufun=U.random(os, reserved_value=0.0)
    )
    result = session.run()

    # Should complete without error
    assert result is not None


def test_composition_boa_negotiator():
    """Test the BOA negotiator example."""
    from negmas.gb.negotiators.modular import BOANegotiator
    from negmas.gb.components import (
        GSmithFrequencyModel,  # Opponent modeling
        GACTime,  # Acceptance strategy
        GTimeDependentOffering,  # Offering strategy
    )
    from negmas.sao import SAOMechanism
    from negmas.preferences import LinearAdditiveUtilityFunction as U
    from negmas.outcomes import make_issue, make_os

    # Create a BOA negotiator with Genius-style components
    negotiator = BOANegotiator(
        offering=GTimeDependentOffering(e=0.2),  # Boulware-style offering
        acceptance=GACTime(t=0.95),  # Accept after 95% of time
        model=GSmithFrequencyModel(),  # Opponent frequency model
        name="my_boa_agent",
    )

    # Use in a negotiation
    issues = [make_issue(10, "price"), make_issue(5, "quantity")]
    os = make_os(issues)

    session = SAOMechanism(issues=issues, n_steps=100)
    session.add(negotiator, ufun=U.random(os, reserved_value=0.0))
    session.add(
        BOANegotiator(
            offering=GTimeDependentOffering(e=0.5),
            acceptance=GACTime(t=0.9),
            name="opponent",
        ),
        ufun=U.random(os, reserved_value=0.0),
    )
    result = session.run()

    # Should complete without error
    assert result is not None
