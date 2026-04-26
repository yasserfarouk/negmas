package negotiator.boaframework.offeringstrategy.other;

import java.io.Serializable;
import java.util.Map;

import genius.core.Bid;
import genius.core.NegotiationResult;
import genius.core.bidding.BidDetails;
import genius.core.boaframework.NegotiationSession;
import genius.core.boaframework.OMStrategy;
import genius.core.boaframework.OfferingStrategy;
import genius.core.boaframework.OpponentModel;

/**
 * Example agent that offers a bid with a utility above the target "breakoff".
 * If the utility received at the end of a match is higher than the breakoff,
 * then the breakoff is set to this utility for the next session on this
 * preference profile. If there is no accept, then the target is decreased with
 * a predefined constant.
 * 
 * @author Mark Hendrikx.
 */
public class ANAC2013BOAExample_Offering extends OfferingStrategy {

	/** Minimum utily the opponent's bid should have. */
	private double breakoff = 0.5;

	/**
	 * Empty constructor called by BOA framework.
	 */
	public ANAC2013BOAExample_Offering() {
	}

	@Override
	public void init(NegotiationSession negotiationSession, OpponentModel opponentModel, OMStrategy omStrategy,
			Map<String, Double> parameters) throws Exception {
		this.negotiationSession = negotiationSession;
		this.opponentModel = opponentModel;
		this.omStrategy = omStrategy;
		Serializable dataFromOffering = loadData();
		if (dataFromOffering != null) {
			breakoff = (Double) dataFromOffering;
		}
	}

	@Override
	public BidDetails determineOpeningBid() {
		return determineNextBid();
	}

	/**
	 * Offer a random bid with a utility higher than the target breakoff.
	 */
	@Override
	public BidDetails determineNextBid() {

		Bid bid = null;
		try {
			do {
				bid = negotiationSession.getUtilitySpace().getDomain().getRandomBid(null);
			} while (negotiationSession.getUtilitySpace().getUtility(bid) < breakoff);
			nextBid = new BidDetails(bid, negotiationSession.getUtilitySpace().getUtility(bid));
		} catch (Exception e) {
			e.printStackTrace();
		}
		return nextBid;
	}

	/**
	 * Method called at the end of a negotiation. This is the ideal point in
	 * time to receiveMessage the target for the next negotiation.
	 */
	public void endSession(NegotiationResult result) {
		// if there was an agreement
		if (result.isAgreement()) {
			// if utility received was higher than target, increase target
			if (result.getMyDiscountedUtility() > breakoff) {
				System.out.println("Accept, my new target is: " + result.getMyDiscountedUtility());
				storeData(new Double(result.getMyDiscountedUtility()));
			}
		} else {
			// if no agreement, decrease target
			double newBreakoff = breakoff - 0.05;
			System.out.println("No accept, my new target is: " + newBreakoff);
			storeData(new Double(newBreakoff));
		}
	}

	@Override
	public String getName() {
		return "Other - ANAC2013BOA Example Offering";
	}
}