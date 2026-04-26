package agents.anac.y2013.InoxAgent;

import java.util.HashMap;
import java.util.Map;

import genius.core.SupportedNegotiationSetting;
import genius.core.boaframework.BOAagentBilateral;
import negotiator.boaframework.acceptanceconditions.anac2013.AC_InoxAgent;
import negotiator.boaframework.offeringstrategy.anac2013.InoxAgent_Offering;
import negotiator.boaframework.omstrategy.BestBid;
import negotiator.boaframework.opponentmodel.InoxAgent_OM;

public class InoxAgent extends BOAagentBilateral {

	@Override
	public void agentSetup() {
		Map<String, Double> params = new HashMap<String, Double>();
		opponentModel = new InoxAgent_OM(negotiationSession);
		opponentModel.init(negotiationSession, params);
		omStrategy = new BestBid();
		omStrategy.init(negotiationSession, opponentModel, params);
		offeringStrategy = new InoxAgent_Offering(negotiationSession,
				opponentModel, omStrategy);
		acceptConditions = new AC_InoxAgent(negotiationSession,
				offeringStrategy, opponentModel);
	}

	@Override
	public String getName() {
		return "InoxAgent";
	}

	@Override
	public SupportedNegotiationSetting getSupportedNegotiationSetting() {
		return SupportedNegotiationSetting.getLinearUtilitySpaceInstance();
	}

	@Override
	public String getDescription() {
		return "ANAC2013";
	}
}