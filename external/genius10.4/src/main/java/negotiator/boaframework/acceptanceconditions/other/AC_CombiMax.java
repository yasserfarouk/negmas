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
 * This Acceptance Condition accepts a bid if it is higher than any bid seen so
 * far
 * 
 * Decoupling Negotiating Agents to Explore the Space of Negotiation Strategies
 * T. Baarslag, K. Hindriks, M. Hendrikx, A. Dirkzwager, C.M. Jonker
 * 
 * @author Alex Dirkzwager
 */
public class AC_CombiMax extends AcceptanceStrategy {

	private double time;

	/**
	 * Empty constructor for the BOA framework.
	 */
	public AC_CombiMax() {
	}

	public AC_CombiMax(NegotiationSession negoSession, OfferingStrategy strat, double time) {
		this.negotiationSession = negoSession;
		this.offeringStrategy = strat;
		this.time = time;
	}

	@Override
	public void init(NegotiationSession negoSession, OfferingStrategy strat, OpponentModel opponentModel,
			Map<String, Double> parameters) throws Exception {
		this.negotiationSession = negoSession;
		if (parameters.get("t") != null) {
			time = parameters.get("t");
		} else {
			throw new Exception("Paramaters were not correctly set");
		}
	}

	@Override
	public String printParameters() {
		return "[t: " + time + "]";
	}

	@Override
	public Actions determineAcceptability() {
		if (negotiationSession.getOpponentBidHistory().getLastBidDetails().getMyUndiscountedUtil() >= offeringStrategy
				.getNextBid().getMyUndiscountedUtil()) {
			return Actions.Accept;
		}

		if (negotiationSession.getTime() < time) {
			return Actions.Reject;
		}

		double offeredUndiscountedUtility = negotiationSession.getOpponentBidHistory().getLastBidDetails()
				.getMyUndiscountedUtil();
		double bestUtil = negotiationSession.getOpponentBidHistory().getBestBidDetails().getMyUndiscountedUtil();

		if (offeredUndiscountedUtility >= bestUtil)
			return Actions.Accept;

		return Actions.Reject;
	}

	@Override
	public String getName() {
		return "Other - CombiMax";
	}
}
