#!/usr/bin/env python
"""Debug script to test capabilities checking."""

from negmas import SAOMechanism
from negmas.genius import GeniusNegotiator
from negmas.outcomes import make_os, make_issue

# Create a simple mechanism with no requirements
mechanism = SAOMechanism(
    outcome_space=make_os([make_issue([1, 2, 3], "price")]), n_steps=10
)

print(f"Mechanism requirements: {mechanism.requirements}")
print(f"Mechanism _requirements: {mechanism._requirements}")

# Create a genius negotiator
negotiator = GeniusNegotiator(java_class_name="agents.anac.y2015.ParsAgent.ParsAgent")
print(f"Negotiator capabilities: {negotiator.capabilities}")

# Test is_satisfying
result = mechanism.is_satisfying(negotiator.capabilities)
print(f"is_satisfying result: {result}")

# Test can_participate
can_part = mechanism.can_participate(negotiator)
print(f"can_participate result: {can_part}")

# Test can_accept_more_negotiators
can_accept = mechanism.can_accept_more_negotiators()
print(f"can_accept_more_negotiators result: {can_accept}")

# Test can_enter
can_ent = mechanism.can_enter(negotiator)
print(f"can_enter result: {can_ent}")
