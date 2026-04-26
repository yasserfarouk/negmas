package negotiator.boaframework.acceptanceconditions.other;

import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import genius.core.boaframework.AcceptanceStrategy;
import genius.core.boaframework.Actions;
import genius.core.boaframework.BOAparameter;
import genius.core.boaframework.NegotiationSession;
import genius.core.boaframework.OfferingStrategy;
import genius.core.boaframework.OpponentModel;

/**
 * This is the decoupled Acceptance Conditions Based on Tim Baarslag's paper on
 * Acceptance Conditions: "Acceptance Conditions in Automated Negotiation"
 * 
 * This Acceptance Conditions is a combination of AC_Time and AC_Next
 * 
 * Decoupling Negotiating Agents to Explore the Space of Negotiation Strategies
 * T. Baarslag, K. Hindriks, M. Hendrikx, A. Dirkzwager, C.M. Jonker
 * 
 * @author Alex Dirkzwager
 */
public class AC_CombiV2 extends AcceptanceStrategy {

	private double a;
	private double b;
	private double c;
	private double d;
	private double time;

	/**
	 * Empty constructor for the BOA framework.
	 */
	public AC_CombiV2() {
	}

	public AC_CombiV2(NegotiationSession negoSession, OfferingStrategy strat, double a, double b, double t, double c,
			double d) {
		this.negotiationSession = negoSession;
		this.offeringStrategy = strat;
		this.a = a;
		this.b = b;
		this.c = c;
		this.d = d;
		this.time = t;
	}

	@Override
	public void init(NegotiationSession negoSession, OfferingStrategy strat, OpponentModel opponentModel,
			Map<String, Double> parameters) throws Exception {
		this.negotiationSession = negoSession;
		this.offeringStrategy = strat;
		if (parameters.get("a") != null || parameters.get("b") != null
				|| parameters.get("c") != null && parameters.get("t") != null) {
			a = parameters.get("a");
			b = parameters.get("b");
			c = parameters.get("c");
			d = parameters.get("d");
			time = parameters.get("t");
		} else {
			throw new Exception("Paramaters were not correctly set");
		}
	}

	@Override
	public String printParameters() {
		return "[a: " + a + " b: " + b + " t: " + time + " c: " + c + " d: " + d + "]";
	}

	@Override
	public Actions determineAcceptability() {
		double nextMyBidUtil = offeringStrategy.getNextBid().getMyUndiscountedUtil();
		double lastOpponentBidUtil = negotiationSession.getOpponentBidHistory().getLastBidDetails()
				.getMyUndiscountedUtil();

		if (lastOpponentBidUtil > d) {
			return Actions.Accept;
		}

		if (negotiationSession.getDiscountFactor() != 0.0 && c < negotiationSession.getDiscountFactor()) {
			if (a * lastOpponentBidUtil + b >= nextMyBidUtil) {
				return Actions.Accept;
			}
		} else {
			if (a * lastOpponentBidUtil + b >= nextMyBidUtil && negotiationSession.getTime() >= time) {
				return Actions.Accept;
			}
		}
		return Actions.Reject;
	}

	@Override
	public Set<BOAparameter> getParameterSpec() {
		Set<BOAparameter> set = new HashSet<BOAparameter>();
		set.add(new BOAparameter("a", 1.0, "Multiplier"));
		set.add(new BOAparameter("b", 0.0, "Constant"));
		set.add(new BOAparameter("c", 0.8, "Threshold discount"));
		set.add(new BOAparameter("d", 0.95, "Threshold"));
		set.add(new BOAparameter("t", 0.99, "Time"));

		return set;
	}

	@Override
	public String getName() {
		return "Other - CombiV2";
	}
}
