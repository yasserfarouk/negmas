package agents.anac.y2013.TheFawkes;

import java.util.HashMap;
import java.util.Map;

import genius.core.SupportedNegotiationSetting;
import genius.core.boaframework.BOAagentBilateral;
import negotiator.boaframework.acceptanceconditions.anac2013.AC_TheFawkes;
import negotiator.boaframework.offeringstrategy.anac2013.Fawkes_Offering;
import negotiator.boaframework.omstrategy.TheFawkes_OMS;
import negotiator.boaframework.opponentmodel.TheFawkes_OM;

public class TheFawkes extends BOAagentBilateral {

	@Override
	public void agentSetup() {
		opponentModel = new TheFawkes_OM();
		Map<String, Double> params = new HashMap<>();
		opponentModel.init(negotiationSession, params);
		omStrategy = new TheFawkes_OMS();
		omStrategy.init(negotiationSession, opponentModel, params);
		offeringStrategy = new Fawkes_Offering();
		acceptConditions = new AC_TheFawkes();
		try {
			offeringStrategy.init(negotiationSession, opponentModel, omStrategy,
					null);
			acceptConditions.init(negotiationSession, offeringStrategy,
					opponentModel, null);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	@Override
	public String getName() {
		return "TheFawkes";
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