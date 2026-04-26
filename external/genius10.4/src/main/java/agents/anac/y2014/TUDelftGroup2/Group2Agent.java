package agents.anac.y2014.TUDelftGroup2;

import java.util.HashMap;

import agents.anac.y2014.TUDelftGroup2.boaframework.acceptanceconditions.other.Group2_AS;
import agents.anac.y2014.TUDelftGroup2.boaframework.offeringstrategy.other.Group2_BS;
import agents.anac.y2014.TUDelftGroup2.boaframework.opponentmodel.Group2_OM;
import genius.core.boaframework.AcceptanceStrategy;
import genius.core.boaframework.BOAagentBilateral;
import genius.core.boaframework.OMStrategy;
import genius.core.boaframework.OfferingStrategy;
import genius.core.boaframework.OpponentModel;
import negotiator.boaframework.omstrategy.BestBid;

public class Group2Agent extends BOAagentBilateral {

	@Override
	public void agentSetup() {
		OpponentModel om = new Group2_OM();

		try {
			om.init(negotiationSession,
					new java.util.HashMap<java.lang.String, java.lang.Double>(
							1));
		} catch (Exception e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}

		OMStrategy oms = new BestBid();
		oms.init(negotiationSession, om, new HashMap<String, Double>());

		OfferingStrategy offering = new Group2_BS();
		try {
			offering.init(negotiationSession, om, oms,
					new java.util.HashMap<java.lang.String, java.lang.Double>(
							1));
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		AcceptanceStrategy ac = new Group2_AS();
		try {
			ac.init(negotiationSession, offering, om,
					new java.util.HashMap<java.lang.String, java.lang.Double>(
							1));
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		setDecoupledComponents(ac, offering, om, oms);
	}

	@Override
	public String getName() {
		return "GROUP2Agent";
	}

	@Override
	public String getDescription() {
		return "ANAC2014 compatible with non-linear utility spaces";
	}
}