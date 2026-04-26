/**
 * This file contains the main class of BraveCat agent, which should be directly imported into the GENIUS 5.1.
 * BraveCat agent is developed in order to take part in The Fourth International Automated Negotiating Agents Competition (ANAC 2014), and
 * Created by Farhad Zafari (email address: f_z_uut@yahoo.com), and Faria Nasiri Mofakham (website address: http://eng.ui.ac.ir/~fnasiri), in
 * Department of Information Technology Engineering,
 * Faculty of Engineering,
 * University of Isfahan, Isfahan, Iran (http://www.ui.ac.ir/).
 * BraveCat agent is consistent with the BOA framework devised by Baarslag et al.
 * For more information on the BOA framework you can see:
 * T. Baarslag, K. Hindriks, M. Hendrikx, A. Dirkzwager, and C. Jonker, "Decoupling negotiating agents to explore the space of negotiation strategies",
 * in Proceedings of the 5th International Workshop on Agent-based Complex Automated Negotiations, ACAN, 2012.
 */

package agents.anac.y2014.BraveCat;

import agents.anac.y2014.BraveCat.AcceptanceStrategies.AC_LAST;
import agents.anac.y2014.BraveCat.AcceptanceStrategies.AcceptanceStrategy;
import agents.anac.y2014.BraveCat.OfferingStrategies.BRTOfferingStrategy;
import agents.anac.y2014.BraveCat.OfferingStrategies.OfferingStrategy;
import agents.anac.y2014.BraveCat.OpponentModelStrategies.BestBid;
import agents.anac.y2014.BraveCat.OpponentModelStrategies.OMStrategy;
import agents.anac.y2014.BraveCat.OpponentModels.OpponentModel;
import agents.anac.y2014.BraveCat.OpponentModels.DBOMModel.DBOMModel;
import agents.anac.y2014.BraveCat.necessaryClasses.BOAagent;

public class BraveCat extends BOAagent {
	@Override
	public void agentSetup() {
		System.out.println(
				"..........................Agent Setup Started.............................");
		try {
			OpponentModel om = null;
			om = new DBOMModel();
			om.init(negotiationSession);
			OMStrategy oms = new BestBid(negotiationSession, om);
			OfferingStrategy offering = new BRTOfferingStrategy(
					this.negotiationSession, om, oms, 0.005, 0);
			AcceptanceStrategy ac = new AC_LAST();
			ac.init(this.negotiationSession, offering, om, null);
			setDecoupledComponents(ac, offering, om, oms);
		} catch (Exception ex) {
		}
		System.out.println(
				"..........................Agent Setup Ended...............................");
	}

	@Override
	public String getName() {
		return "BraveCat v0.3";
	}

	@Override
	public String getDescription() {
		return "ANAC 2014 compatible with non-linear utility spaces";
	}
}