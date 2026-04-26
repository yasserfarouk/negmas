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
 * This acceptance condition uses AC_next to determine when to accept. In
 * addition, the agent also accepts when a given time has passed, and the
 * utility of the opponent's bid is higher than a given constant.
 * 
 * Decoupling Negotiating Agents to Explore the Space of Negotiation Strategies
 * T. Baarslag, K. Hindriks, M. Hendrikx, A. Dirkzwager, C.M. Jonker
 * 
 * @author Alex Dirkzwager, Mark Hendrikx
 */
public class AC_CombiV3 extends AcceptanceStrategy {

	private double a;
	private double b;
	private double c;
	private double time;

	/**
	 * Empty constructor for the BOA framework.
	 */
	public AC_CombiV3() {
	}

	public AC_CombiV3(NegotiationSession negoSession, OfferingStrategy strat, double a, double b, double t, double c) {
		this.negotiationSession = negoSession;
		this.offeringStrategy = strat;
		this.a = a;
		this.b = b;
		this.c = c;
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
			time = parameters.get("t");
		} else {
			throw new Exception("Paramaters were not correctly set");
		}
	}

	@Override
	public String printParameters() {
		return "[a: " + a + " b: " + b + " t: " + time + " c: " + c + "]";
	}

	@Override
	public Actions determineAcceptability() {
		double nextMyBidUtil = offeringStrategy.getNextBid().getMyUndiscountedUtil();
		double lastOpponentBidUtil = negotiationSession.getOpponentBidHistory().getLastBidDetails()
				.getMyUndiscountedUtil();

		double target = a * nextMyBidUtil + b;
		if (target > 1.0) {
			target = 1.0;
		}
		if (lastOpponentBidUtil >= target) {
			return Actions.Accept;
		}

		if (negotiationSession.getTime() > time && lastOpponentBidUtil > c) {
			return Actions.Accept;
		}
		return Actions.Reject;
	}

	@Override
	public Set<BOAparameter> getParameterSpec() {

		Set<BOAparameter> set = new HashSet<BOAparameter>();
		set.add(new BOAparameter("a", 1.0, "Multiplier"));
		set.add(new BOAparameter("b", 0.0, "Constant"));
		set.add(new BOAparameter("c", 0.95, "Threshold"));
		set.add(new BOAparameter("t", 0.99, "Time"));

		return set;
	}

	@Override
	public String getName() {
		return "Other - CombiV3";
	}

}
