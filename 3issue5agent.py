from negmas import SAOMechanism, make_issue
from negmas.preferences import LinearAdditiveUtilityFunction
from negmas.sao.negotiators import (
    AspirationNegotiator, BoulwareTBNegotiator,
    ConcederTBNegotiator, NaiveTitForTatNegotiator,
    LinearTBNegotiator
)

# 共通の issue と ufun
issues = [
    make_issue(name="price", values=10),
    make_issue(name="quantity", values=5),
    make_issue(name="delivery", values=4),
]
buyer_ufun = LinearAdditiveUtilityFunction.random(issues=issues)
seller_ufun = LinearAdditiveUtilityFunction.random(issues=issues)

# テストするエージェントペア
agent_pairs = [
    ("Aspiration", AspirationNegotiator),
    ("Boulware", BoulwareTBNegotiator),
    ("Conceder", ConcederTBNegotiator),
    ("Linear", LinearTBNegotiator),
    ("NaiveTitForTat", NaiveTitForTatNegotiator),
]

# 各ペアで交渉を実行
for name, AgentClass in agent_pairs:
    session = SAOMechanism(issues=issues, n_steps=100)
    session.add(AgentClass(name="buyer"), ufun=buyer_ufun)
    session.add(AgentClass(name="seller"), ufun=seller_ufun)
    result = session.run()
    print(f"{name}: Agreement={result.agreement}, Rounds={result.step}")