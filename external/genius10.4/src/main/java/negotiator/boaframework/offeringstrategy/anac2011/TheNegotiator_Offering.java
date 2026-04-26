package negotiator.boaframework.offeringstrategy.anac2011;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Map;
import java.util.Random;

import genius.core.Bid;
import genius.core.bidding.BidDetails;
import genius.core.bidding.BidDetailsStrictSorterUtility;
import genius.core.boaframework.NegotiationSession;
import genius.core.boaframework.NoModel;
import genius.core.boaframework.OMStrategy;
import genius.core.boaframework.OfferingStrategy;
import genius.core.boaframework.OpponentModel;
import negotiator.boaframework.opponentmodel.DefaultModel;
import negotiator.boaframework.sharedagentstate.anac2011.TheNegotiatorSAS;

/**
 * This is the decoupled Offering Strategy for TheNegotiator (ANAC2011). The
 * code was taken from the ANAC2011 TheNegotiator and adapted to work within the
 * BOA framework.
 * 
 * Adapted to be compatible with an opponent model.
 * 
 * DEFAULT OM: None
 * 
 * @author Alex Dirkzwager, Mark Hendrikx
 * @version 27-12-11
 */
public class TheNegotiator_Offering extends OfferingStrategy {

	private final double RANDOM_MOVE = 0.3f;
	private Random random100;
	private Random random200;
	private final boolean TEST_EQUIVALENCE = false;

	/**
	 * Empty constructor for the BOA framework.
	 */
	public TheNegotiator_Offering() {
	}

	/**
	 * Init required for the Decoupled Framework.
	 */
	@Override
	public void init(NegotiationSession negoSession, OpponentModel model, OMStrategy oms,
			Map<String, Double> parameters) throws Exception {
		if (model instanceof DefaultModel) {
			model = new NoModel();
		}
		this.negotiationSession = negoSession;
		this.opponentModel = model;
		this.omStrategy = oms;
		if (TEST_EQUIVALENCE) {
			random100 = new Random(100);
			random200 = new Random(200);
		} else {
			random100 = new Random();
			random200 = new Random();
		}
		helper = new TheNegotiatorSAS(negoSession);
	}

	/**
	 * Determine the next bid.
	 */
	@Override
	public BidDetails determineNextBid() {
		double time = negotiationSession.getTime();
		double threshold = ((TheNegotiatorSAS) helper).calculateThreshold(time);
		((TheNegotiatorSAS) helper).getPhase();
		nextBid = determineOffer(((TheNegotiatorSAS) helper).getPhase(), threshold);
		return nextBid;
	}

	/**
	 * Determine what (counter)offer should be made in a given phase with a
	 * minimum threshold.
	 * 
	 * @param agentID
	 *            of our agent
	 * @param phase
	 *            of the negotation
	 * @param threshold
	 *            minimum threshold
	 * @return (counter)offer
	 */
	public BidDetails determineOffer(int phase, double threshold) {
		Bid bid = null;
		double upperThreshold = getUpperThreshold(threshold, 0.20);

		if (phase == 1) {
			// if the random value is above the random factor, do a normal move

			if (random100.nextDouble() > RANDOM_MOVE) {
				// do a normal move, which is a move between an threshold
				// interval
				bid = getOwnBidBetween(threshold, upperThreshold);
			} else {
				// do a move which can be oppertunistic (ignore upperbound)
				bid = getOwnBidBetween(upperThreshold - 0.00001, 1.1);
			}
		} else { // phase 2 or 3

			// play best moves of opponent if above threshold

			bid = getBestPartnerBids(threshold);

			// could be that there is no opponent bid above the threshold
			if (bid == null) {
				if (random100.nextDouble() > RANDOM_MOVE) {
					bid = getOwnBidBetween(threshold, upperThreshold);
				} else {
					bid = getOwnBidBetween(upperThreshold - 0.00001, 1.1);
				}
			}
		}
		BidDetails nextBid;
		try {
			nextBid = new BidDetails(bid, negotiationSession.getUtilitySpace().getUtility(bid),
					negotiationSession.getTime());
			return nextBid;
		} catch (Exception e) {
			e.printStackTrace();
		}

		return null;
	}

	/**
	 * Calculate the upperthreshold based on the lowerthreshold and a given
	 * percentage.
	 * 
	 * @param threshold
	 * @param percentage
	 * @return upperbound on the threshold
	 */
	public double getUpperThreshold(double threshold, double percentage) {
		int boundary = 0;
		while (boundary < ((TheNegotiatorSAS) helper).getPossibleBids().size()
				&& ((TheNegotiatorSAS) helper).getPossibleBids().get(boundary).getMyUndiscountedUtil() >= threshold) {
			boundary++;
		}
		if (boundary > 0)
			boundary--;
		int index = boundary - (int) Math.ceil(percentage * boundary);

		double utility = ((TheNegotiatorSAS) helper).getPossibleBids().get(index).getMyUndiscountedUtil();
		return utility;
	}

	public Bid getOwnBidBetween(double lowerThres, double upperThres) {
		return getOwnBidBetween(lowerThres, upperThres, 0);
	}

	/**
	 * Get a random bid between two given thresholds.
	 * 
	 * @param lowerThres
	 *            lowerbound threshold
	 * @param upperThres
	 *            upperbound threshold
	 * @return random bid between thresholds
	 */
	public Bid getOwnBidBetween(double lowerThres, double upperThres, int counter) {
		int lB = 0;
		int uB = 0;
		Bid bid = null;

		// determine upperbound and lowerbound by visiting all points
		for (int i = 0; i < ((TheNegotiatorSAS) helper).getPossibleBids().size(); i++) {
			double util = ((TheNegotiatorSAS) helper).getPossibleBids().get(i).getMyUndiscountedUtil();
			if (util > upperThres) {
				uB++;
			}
			if (util >= lowerThres) {
				lB++;
			}
		}
		// if there are no points between the bounds
		if (lB == uB) {
			if (counter == 1) {
				return ((TheNegotiatorSAS) helper).getPossibleBids().get(0).getBid(); // safe
																						// fallback
																						// value
			}
			// ignore upper threshold
			bid = getOwnBidBetween(lowerThres, 1.1, 1);
		} else {
			// decrement upper- and lowerbound to get the correct index
			// (count counts from 1, while arrays are indexed from 0)
			if (lB > 0) {
				lB--;
			}
			if ((uB + 1) <= lB) {
				uB++;
			}

			// in this case we ignore the upperbound, as it doesn't matter if we
			// take the opponent in to account
			if (!(opponentModel instanceof NoModel)) {
				ArrayList<BidDetails> temp = ((TheNegotiatorSAS) helper).getPossibleBids();
				temp = new ArrayList<BidDetails>(temp.subList(uB, lB + 1));
				BidDetails bidToOffer = omStrategy.getBid(temp);
				bid = bidToOffer.getBid();
			} else {
				// calculate a random bid index
				int result = uB + (int) (random200.nextDouble() * (lB - uB) + 0.5);

				bid = ((TheNegotiatorSAS) helper).getPossibleBids().get(result).getBid();
			}
		}
		try {
			nextBid = new BidDetails(bid, negotiationSession.getUtilitySpace().getUtility(bid),
					negotiationSession.getTime());
		} catch (Exception e) {
			e.printStackTrace();
		}
		return bid;
	}

	/**
	 * Get a partner bid which has a utility of at least a certain value. Null
	 * is returned if no such bid exists.
	 * 
	 * @param threshold
	 * @return bid with utility > threshold if exists
	 */
	public Bid getBestPartnerBids(double threshold) {
		ArrayList<BidDetails> temp = (ArrayList<BidDetails>) new ArrayList<BidDetails>(
				negotiationSession.getOpponentBidHistory().getHistory()).clone();
		if (TEST_EQUIVALENCE) {
			Collections.sort(temp, new BidDetailsStrictSorterUtility());
		} else {
			Collections.sort(temp);
		}
		Bid bid = null;

		int count = 0;
		while (count < temp.size() && temp.get(count).getMyUndiscountedUtil() >= threshold) {
			count++;
		}

		if (count > 0) {
			if (!(opponentModel instanceof NoModel)) {
				bid = omStrategy.getBid(temp).getBid();
			} else {
				bid = temp.get(random200.nextInt(count)).getBid();
			}
		}
		return bid;
	}

	@Override
	public BidDetails determineOpeningBid() {
		nextBid = determineNextBid();
		return nextBid;
	}

	@Override
	public String getName() {
		return "2011 - TheNegotiator";
	}
}