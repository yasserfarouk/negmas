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
 * far within the previous time window
 * 
 * Decoupling Negotiating Agents to Explore the Space of Negotiation Strategies
 * T. Baarslag, K. Hindriks, M. Hendrikx, A. Dirkzwager, C.M. Jonker
 * 
 * @author Alex Dirkzwager
 */
public class AC_CombiMaxInWindowDiscounted extends AcceptanceStrategy {

	private double time;

	/**
	 * Empty constructor for the BOA framework.
	 */
	public AC_CombiMaxInWindowDiscounted() {
	}

	public AC_CombiMaxInWindowDiscounted(NegotiationSession negoSession, OfferingStrategy strat, double t) {
		this.negotiationSession = negoSession;
		this.offeringStrategy = strat;
		this.time = t;
	}

	@Override
	public void init(NegotiationSession negoSession, OfferingStrategy strat, OpponentModel opponentModel,
			Map<String, Double> parameters) throws Exception {
		this.negotiationSession = negoSession;
		this.offeringStrategy = strat;
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
				.getNextBid().getMyUndiscountedUtil())
			return Actions.Accept;

		if (negotiationSession.getTime() < time)
			return Actions.Reject;

		BidDetails opponentLastOffer = negotiationSession.getOpponentBidHistory().getLastBidDetails();
		double offeredDiscountedUtility = negotiationSession.getDiscountedUtility(opponentLastOffer.getBid(),
				opponentLastOffer.getTime());
		double now = negotiationSession.getTime();
		double timeLeft = 1 - now;

		// v2.0
		double window = timeLeft;
		BidHistory recentBids = negotiationSession.getOpponentBidHistory().filterBetweenTime(now - window, now);

		double max;
		if (recentBids.size() > 0) {
			BidDetails maxDiscountedBidDetail = recentBids
					.getBestDiscountedBidDetails(negotiationSession.getUtilitySpace());
			max = negotiationSession.getDiscountedUtility(maxDiscountedBidDetail.getBid(),
					maxDiscountedBidDetail.getTime());
		} else
			max = 0;

		// max = 0 als n = 0
		double expectedUtilOfWaitingForABetterBid = max;

		if (offeredDiscountedUtility >= expectedUtilOfWaitingForABetterBid)
			return Actions.Accept;

		return Actions.Reject;
	}

	@Override
	public String getName() {
		return "Other - CombiMaxInWindowDiscounted";
	}

}
