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
session.run()

# Visualize
session.plot()