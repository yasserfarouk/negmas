package agents.anac.y2012.TheNegotiatorReloaded;

import genius.core.SupportedNegotiationSetting;
import genius.core.boaframework.BOAagentBilateral;
import genius.core.boaframework.NoModel;
import negotiator.boaframework.acceptanceconditions.anac2012.AC_TheNegotiatorReloaded;
import negotiator.boaframework.offeringstrategy.anac2012.TheNegotiatorReloaded_Offering;
import negotiator.boaframework.omstrategy.NullStrategy;
import negotiator.boaframework.opponentmodel.IAMhagglerBayesianModel;

public class TheNegotiatorReloaded extends BOAagentBilateral {

	/**
	 * Initializes the agent by setting the opponent model, the opponent model
	 * strategy, bidding strategy, and acceptance conditions.
	 */
	@Override
	public void agentSetup() {
		try {
			if (negotiationSession.getUtilitySpace().getDomain()
					.getNumberOfPossibleBids() < 200000) {
				opponentModel = new IAMhagglerBayesianModel();
			} else {
				opponentModel = new NoModel();
			}
			opponentModel.init(negotiationSession, null);
			omStrategy = new NullStrategy(negotiationSession, 0.35);
			offeringStrategy = new TheNegotiatorReloaded_Offering(
					negotiationSession, opponentModel, omStrategy);
		} catch (Exception e) {
			e.printStackTrace();
		}
		acceptConditions = new AC_TheNegotiatorReloaded(negotiationSession,
				offeringStrategy, 1, 0, 1.05, 0, 0.98, 0.99);
	}

	/**
	 * Returns the agent's name.
	 */
	@Override
	public String getName() {
		return "TheNegotiator Reloaded";
	}

	@Override
	public SupportedNegotiationSetting getSupportedNegotiationSetting() {
		return SupportedNegotiationSetting.getLinearUtilitySpaceInstance();
	}

	@Override
	public String getDescription() {
		return "ANAC2012";
	}
}