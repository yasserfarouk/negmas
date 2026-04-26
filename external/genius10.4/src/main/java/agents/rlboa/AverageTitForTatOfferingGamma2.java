package agents.rlboa;

import genius.core.Bid;
import genius.core.BidHistory;
import genius.core.bidding.BidDetails;
import genius.core.boaframework.*;
import genius.core.utility.AbstractUtilitySpace;
import genius.core.utility.UtilitySpace;

import java.util.Collections;
import java.util.List;

public class AverageTitForTatOfferingGamma2 extends OfferingStrategy {

	private int gamma = 2;
	private int k = 0;
	private List<BidDetails> allBids;

	private double minUtil;
	private double maxUtil;

	public AverageTitForTatOfferingGamma2(NegotiationSession negotiationSession, OpponentModel opponentModel, OMStrategy omStrategy) {
		super.init(negotiationSession, null);

		OutcomeSpace sortedOutcomeSpace = new SortedOutcomeSpace(this.negotiationSession.getUtilitySpace());
		this.negotiationSession.setOutcomeSpace(sortedOutcomeSpace);
		allBids = sortedOutcomeSpace.getAllOutcomes();

		// get min/max utility obtainable
		try {
			AbstractUtilitySpace utilitySpace = this.negotiationSession.getUtilitySpace();
			Bid maxUtilBid = utilitySpace.getMaxUtilityBid();
			Bid minUtilBid = utilitySpace.getMinUtilityBid();
			maxUtil = utilitySpace.getUtility(maxUtilBid);
			minUtil = utilitySpace.getUtility(minUtilBid);
			minUtil = Math.max(minUtil, utilitySpace.getReservationValueUndiscounted());
		} catch (Exception e) {
			// exception is thrown by getMaxUtilityBid if there are no bids in the outcomespace
			// but I guess that's pretty rare. Default to 0.0 - 1.0 to prevent crashes.
			maxUtil = 1.0;
			minUtil = 0.0;
		}
	}
	
	@Override
	public BidDetails determineOpeningBid() {
		BidDetails openingsBid = allBids.get(k);
		k++;
		return openingsBid;
	}

	@Override
	public BidDetails determineNextBid() {
		int opponentWindowSize = this.negotiationSession.getOpponentBidHistory().size();
		BidDetails nextBid = null;

		// in the paper, the condition of applicability is t_n > 2*gamma; but because
		// we keep track of opponent/own bids seperately we have devide the threshold by 2
		// we can use averageTitForTat if the number of bids made by the opponent is larger
		// than the window we average over
		if (opponentWindowSize > gamma) {
			double targetUtility = this.averageTitForTat();
			nextBid = this.negotiationSession.getOutcomeSpace().getBidNearUtility(targetUtility);
		} else {
			nextBid = this.determineOpeningBid();
		}

		return nextBid;
	}

	private double averageTitForTat() {

		// determine relative change of opponent bid
		int tOpp = this.negotiationSession.getOpponentBidHistory().size() - 1;
		List<BidDetails> opponentHistory = this.negotiationSession.getOpponentBidHistory().getHistory();

		double opponentLastBid = opponentHistory.get(tOpp).getMyUndiscountedUtil();
		double opponentFirstBidInWindow = opponentHistory.get(tOpp - gamma).getMyUndiscountedUtil();
		double relativeChangeOpponent = opponentFirstBidInWindow / opponentLastBid;

		// target utility is the same change applied to our last bid
		double myLastBid = this.negotiationSession.getOwnBidHistory().getLastBidDetails().getMyUndiscountedUtil();
		double targetUtil = relativeChangeOpponent * myLastBid;

		return Math.min(Math.max(targetUtil, minUtil), maxUtil);
	}

	@Override
	public String getName() {
		return "AverageTitForTat2 offering";
	}

}
