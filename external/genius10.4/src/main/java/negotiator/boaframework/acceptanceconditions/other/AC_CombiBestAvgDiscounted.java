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
 * This Acceptance Condition averages the opponents bids (which are better than
 * the bid that was offered) made in the previous time window. If the bid is
 * higher than the average it will accept
 * 
 * Decoupling Negotiating Agents to Explore the Space of Negotiation Strategies
 * T. Baarslag, K. Hindriks, M. Hendrikx, A. Dirkzwager, C.M. Jonker
 * 
 * @author Alex Dirkzwager
 */
public class AC_CombiBestAvgDiscounted extends AcceptanceStrategy {

	private double time;

	/**
	 * Empty constructor for the BOA framework.
	 */
	public AC_CombiBestAvgDiscounted() {
	}

	public AC_CombiBestAvgDiscounted(NegotiationSession negoSession, OfferingStrategy strat, double t) {
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

		BidDetails opponentLastOffer = negotiationSession.getOpponentBidHistory().getLastBidDetails();
		if (opponentLastOffer.getMyUndiscountedUtil() >= offeringStrategy.getNextBid().getMyUndiscountedUtil()) {
			return Actions.Accept;
		}

		if (negotiationSession.getTime() < time) {
			return Actions.Reject;
		}

		double offeredDiscountedUtility = negotiationSession.getDiscountedUtility(opponentLastOffer.getBid(),
				opponentLastOffer.getTime());
		double now = negotiationSession.getTime();
		double timeLeft = 1 - now;

		double window = timeLeft;
		BidHistory recentBetterBids = negotiationSession.getOpponentBidHistory().discountedFilterBetween(
				offeredDiscountedUtility, 1, now - window, now, negotiationSession.getUtilitySpace());

		double avgOfBetterBids = recentBetterBids.getAverageDiscountedUtility(negotiationSession.getUtilitySpace());
		double expectedUtilOfWaitingForABetterBid = avgOfBetterBids;

		if (offeredDiscountedUtility >= expectedUtilOfWaitingForABetterBid)
			return Actions.Accept;
		return Actions.Reject;
	}

	@Override
	public String getName() {
		return "Other - CombiBestAvgDiscounted";
	}

}