package negotiator.boaframework.acceptanceconditions.anac2013;

import java.util.List;
import java.util.Map;

import genius.core.bidding.BidDetails;
import genius.core.boaframework.AcceptanceStrategy;
import genius.core.boaframework.Actions;
import genius.core.boaframework.NegotiationSession;
import genius.core.boaframework.OfferingStrategy;
import genius.core.boaframework.OpponentModel;
import genius.core.boaframework.SortedOutcomeSpace;

/**
 * This Acceptance Condition will break when the reservation value seems to be a
 * better alternative, and accept when the opponents offer is better than the
 * median utility.
 * 
 * @author Ruben van Zessen
 */
public class AC_InoxAgent_OneIssue extends AcceptanceStrategy {
	/** Discount factor of the domain */
	private double discountFactor;
	/** Median utility in the sorted outcome space */
	private double medianutil;
	/** A check whether the median has been set yet */
	private boolean medianDecided = false;

	/**
	 * Empty constructor.
	 */
	public AC_InoxAgent_OneIssue() {
	}

	/**
	 * Regular constructor.
	 */
	public AC_InoxAgent_OneIssue(NegotiationSession negoSession, OfferingStrategy strat, OpponentModel oppModel) {
		this.negotiationSession = negoSession;
		this.offeringStrategy = strat;
		this.opponentModel = oppModel;
		discountFactor = negotiationSession.getDiscountFactor();
	}

	/**
	 * Initialization function.
	 * 
	 * Does the same as the regular constructor.
	 */
	@Override
	public void init(NegotiationSession negoSession, OfferingStrategy strat, OpponentModel oppModel,
			Map<String, Double> parameters) throws Exception {
		this.negotiationSession = negoSession;
		this.offeringStrategy = strat;
		this.opponentModel = oppModel;
		discountFactor = negotiationSession.getDiscountFactor();
		if (discountFactor == 0.0) {
			discountFactor = 1.0;
		}
	}

	/**
	 * Method which returns the action selected by this acceptance strategy.
	 */
	@Override
	public Actions determineAcceptability() {
		// Set the median if it's not set already
		if (!medianDecided) {
			SortedOutcomeSpace outcomespace = new SortedOutcomeSpace(negotiationSession.getUtilitySpace());
			int opplocation = outcomespace.getIndexOfBidNearUtility(
					negotiationSession.getOpponentBidHistory().getBestBidDetails().getMyUndiscountedUtil());
			List<BidDetails> alloutcomes = outcomespace.getAllOutcomes();
			medianutil = alloutcomes.get((int) Math.floor(((double) opplocation) / 2)).getMyUndiscountedUtil();
			medianDecided = true;
		}

		// Read time
		double time = negotiationSession.getTime();
		// Determine utility of the opponent bid to be evaluated
		double lastOpponentBidUtil = negotiationSession
				.getDiscountedUtility(negotiationSession.getOpponentBidHistory().getLastBid(), time);

		// Accept if the opponents offer is better than the median offer
		if (lastOpponentBidUtil >= medianutil * Math.pow(discountFactor, time)) {
			return Actions.Accept;
			// Break if the reservation value is looking attractive
		} else if (negotiationSession.getUtilitySpace().getReservationValueUndiscounted() >= medianutil
				* Math.pow(discountFactor, time)) {
			return Actions.Break;
		}
		// If none of the above conditions are met, send our own generated bid.
		return Actions.Reject;
	}

	@Override
	public String getName() {
		return "2013 - InoxAgent_OneIssue";
	}
}