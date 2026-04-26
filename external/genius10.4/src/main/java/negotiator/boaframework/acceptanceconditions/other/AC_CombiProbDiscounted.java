package negotiator.boaframework.acceptanceconditions.other;

import java.util.Map;

import genius.core.BidHistory;
import genius.core.bidding.BidDetails;
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
public class AC_CombiProbDiscounted extends AcceptanceStrategy {

	private double time;

	/**
	 * Empty constructor for the BOA framework.
	 */
	public AC_CombiProbDiscounted() {
	}

	public AC_CombiProbDiscounted(NegotiationSession negoSession, OfferingStrategy strat, double t) {
		this.negotiationSession = negoSession;
		this.offeringStrategy = strat;
		this.time = t;
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

		BidDetails opponentLastOffer = negotiationSession.getOpponentBidHistory().getLastBidDetails();
		double offeredDiscountedUtility = negotiationSession.getDiscountedUtility(opponentLastOffer.getBid(),
				opponentLastOffer.getTime());
		double now = negotiationSession.getTime();
		double timeLeft = 1 - now;

		// if we will still see a lot of bids
		BidHistory recentBids = negotiationSession.getOpponentBidHistory().filterBetweenTime(now - timeLeft, now);
		int remainingBids = recentBids.size();
		if (remainingBids > 10)
			return Actions.Reject;

		// v2.0
		double window = timeLeft;
		// double window = 5 * timeLeft;
		BidHistory recentBetterBids = negotiationSession.getOpponentBidHistory().discountedFilterBetween(
				offeredDiscountedUtility, 1, now - window, now, negotiationSession.getUtilitySpace());
		int n = recentBetterBids.size();
		double p = timeLeft / window;
		if (p > 1)
			p = 1;

		double pAllMiss = Math.pow(1 - p, n);
		if (n == 0)
			pAllMiss = 1;
		double pAtLeastOneHit = 1 - pAllMiss;

		double avg = recentBetterBids.getAverageDiscountedUtility(negotiationSession.getUtilitySpace());

		double expectedUtilOfWaitingForABetterBid = pAtLeastOneHit * avg;

		if (offeredDiscountedUtility > expectedUtilOfWaitingForABetterBid)
			return Actions.Accept;

		return Actions.Reject;
	}

	@Override
	public String getName() {
		return "Other - CombiProbDiscounted";
	}

}
