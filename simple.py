from negmas import SAOMechanism, TimeBasedConcedingNegotiator, make_issue
from negmas.preferences import LinearAdditiveUtilityFunction as LUFun

# Define what we're negotiating about
issues = [make_issue(name="price", values=100)]

# Create negotiation session (Stacked Alternating Offers)
session = SAOMechanism(issues=issues, n_steps=50)

# Add buyer (prefers low price) and seller (prefers high price)
session.add(
    TimeBasedConcedingNegotiator(name="buyer"),
    ufun=LUFun.random(issues=issues, reserved_value=0.0),
)
session.add(
    TimeBasedConcedingNegotiator(name="seller"),
    ufun=LUFun.random(issues=issues, reserved_value=0.0),
)

# Run and get result
result = session.run()
print(f"Agreement: {result.agreement}, Rounds: {result.step}")