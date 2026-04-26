package negotiator.boaframework.offeringstrategy.anac2013;

import java.util.Map;

import genius.core.bidding.BidDetails;
import genius.core.boaframework.NegotiationSession;
import genius.core.boaframework.OMStrategy;
import genius.core.boaframework.OfferingStrategy;
import genius.core.boaframework.OpponentModel;
import genius.core.boaframework.SortedOutcomeSpace;

/**
 * This class implements a simple strategy for a single issue domain.
 * 
 * The utility of offers that are being generated slowly decline to the utility
 * of the median bid between our best bid and the opponent's first bid.
 * 
 * @author Ruben van Zessen, Mariana Branco
 */
public class InoxAgent_Offering_OneIssue extends OfferingStrategy {

	/** Discount factor of the current domain */
	private double discountFactor;
	/** Median utility in the sorted outcome space */
	private double medianutil;
	/** A check whether the median has been set yet */
	private boolean medianDecided = false;

	/** Outcome space */
	private SortedOutcomeSpace outcomespace;
	/** Best possible bid */
	private BidDetails bestBid;

	/**
	 * Empty constructor.
	 */
	public InoxAgent_Offering_OneIssue() {
	}

	/**
	 * Regular constructor.
	 */
	public InoxAgent_Offering_OneIssue(NegotiationSession negoSession, OpponentModel model, OMStrategy oms) {
		this.negotiationSession = negoSession;
		discountFactor = negotiationSession.getDiscountFactor();
		outcomespace = new SortedOutcomeSpace(negotiationSession.getUtilitySpace());
		negotiationSession.setOutcomeSpace(outcomespace);
		this.opponentModel = model;
		this.omStrategy = oms;
	}

	/**
	 * Initialization function.
	 * 
	 * Does the same as the regular constructor.
	 */
	@Override
	public void init(NegotiationSession negoSession, OpponentModel model, OMStrategy oms,
			Map<String, Double> parameters) throws Exception {
		this.negotiationSession = negoSession;
		discountFactor = negotiationSession.getDiscountFactor();
		outcomespace = new SortedOutcomeSpace(negotiationSession.getUtilitySpace());
		negotiationSession.setOutcomeSpace(outcomespace);
		this.opponentModel = model;
		this.omStrategy = oms;
	}

	/**
	 * Initially offer the best possible bid.
	 */
	@Override
	public BidDetails determineOpeningBid() {
		bestBid = negotiationSession.getOutcomeSpace().getMaxBidPossible();
		return bestBid;
	}

	/**
	 * Bidding strategy based on slowly conceding towards the median utility.
	 */
	@Override
	public BidDetails determineNextBid() {
		// Read time
		double time = negotiationSession.getTime();

		// Set median if it's not set already
		if (!medianDecided) {
			double opputil = negotiationSession.getOpponentBidHistory().getFirstBidDetails().getMyUndiscountedUtil();
			// Set the "median" to be slightly in our favor
			medianutil = (1.0 - opputil) / 1.2 + opputil;
			medianDecided = true;
		}

		// Calculate bid to be sent
		BidDetails sendBid = calculateBid(time);
		return sendBid;

	}

	/**
	 * Function that generates a bid to send based on time and discount factor.
	 */
	private BidDetails calculateBid(double t) {
		// If dealing with a discounted domain, transform this function such
		// that
		// we reach the median when the discount is 0.8
		double timeshift = 1;
		if (discountFactor < 0.8) {
			timeshift = Math.min((Math.log(0.8) / Math.log(discountFactor)), 1.0);
		}
		double tarUtil = Math.max(1 - (1 - medianutil) * Math.pow((t / timeshift), 27), medianutil);
		return negotiationSession.getOutcomeSpace().getBidNearUtility(tarUtil);
	}

	@Override
	public String getName() {
		return "2013 - INOX_OneIssue";
	}

}