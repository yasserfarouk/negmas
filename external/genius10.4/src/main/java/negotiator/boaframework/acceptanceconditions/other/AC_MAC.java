package negotiator.boaframework.acceptanceconditions.other;

import java.util.ArrayList;
import java.util.Map;

import genius.core.boaframework.AcceptanceStrategy;
import genius.core.boaframework.Multi_AcceptanceCondition;
import genius.core.boaframework.NegotiationSession;
import genius.core.boaframework.OfferingStrategy;
import genius.core.boaframework.OpponentModel;
import genius.core.boaframework.OutcomeTuple;

/**
 * The MAC is a tool which allows to test many acceptance strategies in the same
 * negotiation trace. Each AC generates an outcome, which is saved separately.
 * Note that while this tool allows to test a large amount of AC's in the same
 * trace, there is a computational cost. Therefore we recommend to use at most
 * 50 AC's.
 * 
 * Decoupling Negotiating Agents to Explore the Space of Negotiation Strategies
 * T. Baarslag, K. Hindriks, M. Hendrikx, A. Dirkzwager, C.M. Jonker
 * 
 * @author Alex Dirkzwager
 */
public class AC_MAC extends Multi_AcceptanceCondition {

	/**
	 * Empty constructor for the BOA framework.
	 */
	public AC_MAC() {
	}

	@Override
	public void init(NegotiationSession negoSession, OfferingStrategy strat, OpponentModel opponentModel,
			Map<String, Double> parameters) throws Exception {
		this.negotiationSession = negoSession;
		this.offeringStrategy = strat;
		outcomes = new ArrayList<OutcomeTuple>();
		ACList = new ArrayList<AcceptanceStrategy>();

		for (int a = 0; a < 2; a++) {
			for (int b = 0; b < 3; b++) {
				for (int c = 0; c < 3; c++) {
					for (int d = 0; d < 4; d++) {
						ACList.add(new AC_CombiV4(negotiationSession, offeringStrategy, 1.0 + 0.05 * a, 0.0 + 0.05 * b,
								1.0 + 0.05 * c, 0.0 + 0.05 * d, 0.95));
					}
				}
			}
		}
		for (int e = 0; e < 5; e++) {
			ACList.add(new AC_CombiMaxInWindow(negotiationSession, offeringStrategy, 0.95 + e * 0.01));
		}
	}
}