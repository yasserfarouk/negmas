package agents.rlboa;

import genius.core.boaframework.BOAagentBilateral;
import genius.core.boaframework.NoModel;
import negotiator.boaframework.acceptanceconditions.other.AC_Next;
import negotiator.boaframework.omstrategy.NullStrategy;

import java.util.HashMap;

@SuppressWarnings("deprecation")
public class AverageTitForTat1 extends BOAagentBilateral {

	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	@Override
	public void agentSetup() {

		// AverageTitForTat2 makes decisions based on its own preferences
		opponentModel = new NoModel();
		opponentModel.init(negotiationSession, new HashMap<String, Double>());

		// OMS not relevant for NoModel
		omStrategy = new NullStrategy(negotiationSession);


		offeringStrategy = new AverageTitForTatOfferingGamma1(negotiationSession, opponentModel, omStrategy);
		
		acceptConditions = new AC_Next(negotiationSession, offeringStrategy, 1, 0);	
		setDecoupledComponents(acceptConditions, offeringStrategy, opponentModel, omStrategy);
	}

	@Override
	public String getName() {
		return "AverageTitForTat with gamma 1";
	}

}
