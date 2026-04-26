package agents.anac.y2013.MetaAgent.portfolio.thenegotiatorreloaded;

public class TheNegotiatorReloaded extends BOAagent {

	/**
	 * Initializes the agent by setting the opponent model, the opponent model
	 * strategy, bidding strategy, and acceptance conditions.
	 */
	@Override
	public void agentSetup() {
		try {
			if (negotiationSession.getUtilitySpace().getDomain().getNumberOfPossibleBids() < 200000) {
				opponentModel = new IAMhagglerModel();
			} else {
				opponentModel = new NullModel();
			}
			opponentModel.init(negotiationSession, null);
			omStrategy = new NullStrategy(negotiationSession, 0.35);
			offeringStrategy = new TheNegotiatorReloaded_Offering(negotiationSession, opponentModel, omStrategy);
		} catch (Exception e) {
			e.printStackTrace();
		}
		acceptConditions = new AC_TheNegotiatorReloaded(negotiationSession, offeringStrategy, 1, 0, 1.05, 0, 0.98,
				0.99);
	}

	/**
	 * Returns the agent's name.
	 */
	@Override
	public String getName() {
		return "TheNegotiator Reloaded";
	}

	@Override
	public String getDescription() {
		return "ANAC 2013 - The Negotiator Reloaded";
	}
}