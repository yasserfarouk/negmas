package negotiator.boaframework.acceptanceconditions.other;

import java.util.Map;

import genius.core.boaframework.AcceptanceStrategy;
import genius.core.boaframework.Actions;
import genius.core.boaframework.NegotiationSession;
import genius.core.boaframework.OfferingStrategy;
import genius.core.boaframework.OpponentModel;

/**
 * This is the decoupled Acceptance Conditions Based on Tim Baarslag's paper on
 * Acceptance Conditions: "Acceptance Conditions in Automated Negotiation"
 * 
 * This Acceptance Conditions is a combination of AC_Time and AC_Next ->
 * (AC_Time OR AC_Next)
 * 
 * Decoupling Negotiating Agents to Explore the Space of Negotiation Strategies
 * T. Baarslag, K. Hindriks, M. Hendrikx, A. Dirkzwager, C.M. Jonker
 * 
 * @author Alex Dirkzwager
 */
public class AC_Combi extends AcceptanceStrategy {

	private double a;
	private double b;
	private double time;

	/**
	 * Empty constructor for the BOA framework.
	 */
	public AC_Combi() {
	}

	public AC_Combi(NegotiationSession negoSession, OfferingStrategy strat, double a, double b, double t, double c) {
		this.negotiationSession = negoSession;
		this.offeringStrategy = strat;
		this.a = a;
		this.b = b;
		this.time = t;
	}

	@Override
	public void init(NegotiationSession negoSession, OfferingStrategy strat, OpponentModel opponentModel,
			Map<String, Double> parameters) throws Exception {
		this.negotiationSession = negoSession;
		if (parameters.get("c") != null && parameters.get("t") != null) {
			a = parameters.get("a");
			b = parameters.get("b");
			time = parameters.get("t");
		} else {
			throw new Exception("Paramaters were not correctly set");
		}
	}

	@Override
	public String printParameters() {
		return "[a: " + a + " b: " + b + " t: " + time + "]";
	}

	@Override
	public Actions determineAcceptability() {

		double nextMyBidUtil = offeringStrategy.getNextBid().getMyUndiscountedUtil();
		double lastOpponentBidUtil = negotiationSession.getOpponentBidHistory().getLastBidDetails()
				.getMyUndiscountedUtil();
		if (a * lastOpponentBidUtil + b >= nextMyBidUtil || negotiationSession.getTime() >= time) {
			return Actions.Accept;
		}
		return Actions.Reject;
	}

	@Override
	public String getName() {
		return "Other - Combi";
	}
}