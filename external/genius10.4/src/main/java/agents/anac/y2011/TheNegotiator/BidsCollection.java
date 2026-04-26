package agents.anac.y2011.TheNegotiator;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;

import genius.core.Bid;
import genius.core.bidding.BidDetails;
import genius.core.bidding.BidDetailsStrictSorterUtility;

/**
 * The BidsCollection class stores the bids of the partner and all possible bids.
 * 
 * @author Alex Dirkzwager, Mark Hendrikx, Julian de Ruiter
 */
public class BidsCollection {

	// bids done by partner
	private ArrayList<BidDetails> partnerBids;
	// all possible bids (for us) which do not violate the constraints
	private ArrayList<BidDetails> possibleBids;
	private Random random200;
	private final boolean TEST_EQUIVALENCE = false;

	/**
	 * Creates a BidsCollection-object which stores the partner bids and all possible
	 * bids.
	 */
	public BidsCollection() {
		partnerBids = new ArrayList<BidDetails>();
		possibleBids  = new ArrayList<BidDetails>();
		if (TEST_EQUIVALENCE) {
			random200 = new Random(200);
		} else {
			random200 = new Random(200);
		}
	}

	/**
	 * @return the partnerBids
	 */
	public ArrayList<BidDetails> getPartnerBids() {
		return partnerBids;
	}

	/**
	 * @return the possibleBids
	 */
	public ArrayList<BidDetails> getPossibleBids() {
		return possibleBids;
	}

	/**
	 * Add a partner bid to the history. Bids are stored at the front
	 * to preserve the timeline.
	 * 
	 * @param bid made by partner in the current turn
	 * @param utility of the bid
	 */
	public void addPartnerBid(Bid bid, double utility, double time) {
		BidDetails utbid = new BidDetails(bid, utility, time);
		partnerBids.add(0, utbid);
	}
	
	/**
	 * Add a possible bid to the list of possible bids. The given bid 
	 * should not violate the constraints of the negotiation.
	 * 
	 * @param bid which is possible
	 * @param utility of the bid
	 */
	public void addPossibleBid(Bid bid, double utility) {
		BidDetails utbid = new BidDetails(bid, utility, -1.0);
		possibleBids.add(utbid);
	}
	
	/**
	 * Sorts all possible bids in reverse natural order.
	 */
	public void sortPossibleBids() {
		if (TEST_EQUIVALENCE) {
			Collections.sort(possibleBids, new BidDetailsStrictSorterUtility());
		} else {
			Collections.sort(possibleBids);
		}
	}
		
	/**
	 * Get a partner bid.
	 * 
	 * @param i 
	 * @return the i'th bid in the timeline
	 */
	public Bid getPartnerBid(int i) {
		Bid bid = null;
		
		if (i < partnerBids.size()) {
			bid = partnerBids.get(i).getBid();
		} else {
			ErrorLogger.log("BIDSCOLLECTION: Out of bounds");
		}
		return bid;
	}

	/**
	 * Get a partner bid which has a utility of at least a certain
	 * value. Null is returned if no such bid exists.
	 * 
	 * @param threshold
	 * @return bid with utility > threshold if exists
	 */
	public Bid getBestPartnerBids(double threshold) {
		ArrayList<BidDetails> temp = partnerBids;
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
			bid = temp.get(random200.nextInt(count)).getBid();
		}
		return bid;
	}

	public Bid getOwnBidBetween(double lowerThres, double upperThres) {
		return getOwnBidBetween(lowerThres, upperThres, 0);
	}
	
	/**
	 * Get a random bid between two given thresholds.
	 * 
	 * @param lowerThres lowerbound threshold
	 * @param upperThres upperbound threshold
	 * @return random bid between thresholds
	 */
	public Bid getOwnBidBetween(double lowerThres, double upperThres, int counter) {
		int lB = 0;
		int uB = 0;
		Bid bid = null;

		// determine upperbound and lowerbound by visiting all points
		for (int i = 0; i < possibleBids.size(); i++) {
			double util = possibleBids.get(i).getMyUndiscountedUtil();
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
				return possibleBids.get(0).getBid(); // safe fallback value
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
			// calculate a random bid index
			int result = uB + (int) ( random200.nextDouble() * (lB - uB) + 0.5);
			bid = possibleBids.get(result).getBid();
		}
		return bid;
	}

	/**
	 * Calculate the upperthreshold based on the lowerthreshold and a given percentage.
	 * @param threshold
	 * @param percentage
	 * @return
	 */
	public double getUpperThreshold(double threshold, double percentage) {
		int boundary = 0;
		while (boundary < possibleBids.size() && possibleBids.get(boundary).getMyUndiscountedUtil() >= threshold) {
			boundary++;
		}
		if (boundary > 0)
			boundary--;
		int index = boundary - (int) Math.ceil(percentage * boundary);
	
		double utility = possibleBids.get(index).getMyUndiscountedUtil();
		return utility;
	}
}