package genius.core.boaframework;

import java.util.ArrayList;
import java.util.List;

import genius.core.Bid;
import genius.core.BidIterator;
import genius.core.bidding.BidDetails;
import genius.core.misc.Range;
import genius.core.utility.AbstractUtilitySpace;

/**
 * This class generates the complete outcome space and is therefore useful if
 * someone wants to quickly implement an agent. Note that while this
 * outcomespace is faster upon initialization, the sorted outcomespace class is
 * faster during the negotiation.
 * 
 * @author Alex Dirkzwager, Mark Hendrikx
 */
public class OutcomeSpace {

	/** Reference to the utility space */
	protected AbstractUtilitySpace utilitySpace;
	/** List of all possible bids in the domain */
	protected List<BidDetails> allBids = new ArrayList<BidDetails>();

	/**
	 * Creates an unsorted outcome space. Warning: this call iterates over ALL
	 * possible bids.
	 * 
	 * @param utilSpace
	 */
	public OutcomeSpace(AbstractUtilitySpace utilSpace) {
		this.utilitySpace = utilSpace;
		generateAllBids(utilSpace);
	}

	/**
	 * Generates all the possible bids in the domain
	 * 
	 * @param utilSpace
	 */
	public void generateAllBids(AbstractUtilitySpace utilSpace) {

		BidIterator iter = new BidIterator(utilSpace.getDomain());
		while (iter.hasNext()) {
			Bid bid = iter.next();
			try {
				BidDetails bidDetails = new BidDetails(bid, utilSpace.getUtility(bid), -1);
				allBids.add(bidDetails);
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
	}

	/**
	 * @return list of all possible bids
	 */
	public List<BidDetails> getAllOutcomes() {
		return allBids;
	}
	
	/**
	 * @return list of all possible bids without their utilities
	 */
	public List<Bid> getAllBidsWithoutUtilities() {

		List<Bid> bidsList = new ArrayList<Bid>();
		for (BidDetails bid : this.allBids) {
			bidsList.add(bid.getBid());
		} 
		return bidsList;
	}
	
	/**
	 * Returns a list of bids (from possibleBids) that have a utility within the
	 * given range.
	 * 
	 * @param range
	 *            in which the bids must be found.
	 * @return list of bids which a utility in the given range.
	 */
	public List<BidDetails> getBidsinRange(Range range) {
		ArrayList<BidDetails> result = new ArrayList<BidDetails>();
		double upperbound = range.getUpperbound();
		double lowerbound = range.getLowerbound();

		for (BidDetails bid : allBids) {
			if (bid.getMyUndiscountedUtil() > lowerbound && bid.getMyUndiscountedUtil() < upperbound) {
				result.add(bid);
			}
		}
		return result;
	}

	/**
	 * gets a BidDetails which is closest to the given utility
	 * 
	 * @param utility
	 *            to which the found bid must be closest.
	 * @return BidDetails
	 */
	public BidDetails getBidNearUtility(double utility) {
		return allBids.get(getIndexOfBidNearUtility(utility));
	}

	/**
	 * @return best bid in the domain.
	 */
	public BidDetails getMaxBidPossible() {
		BidDetails maxBid = allBids.get(0);
		for (BidDetails bid : allBids) {
			if (bid.getMyUndiscountedUtil() > maxBid.getMyUndiscountedUtil()) {
				maxBid = bid;
			}
		}
		return maxBid;
	}

	/**
	 * @return worst bid in the domain.
	 */
	public BidDetails getMinBidPossible() {
		BidDetails minBid = allBids.get(0);
		for (BidDetails bid : allBids) {
			if (bid.getMyUndiscountedUtil() < minBid.getMyUndiscountedUtil()) {
				minBid = bid;
			}
		}
		return minBid;
	}

	/**
	 * @param utility
	 *            to which the found bid must be closest.
	 * @return index of the bid with the utility closest to the given utilty.
	 */
	public int getIndexOfBidNearUtility(double utility) {
		double closesDistance = 1;
		int best = 0;
		for (int i = 0; i < allBids.size(); i++) {
			if (Math.abs(allBids.get(i).getMyUndiscountedUtil() - utility) < closesDistance) {
				closesDistance = Math.abs(allBids.get(i).getMyUndiscountedUtil() - utility);
				best = i;
			}
		}
		return best;
	}

	@Override
	public String toString() {
		String all = "";
		for (BidDetails b : allBids) {
			all += b.toString() + "\n,";
		}
		return "OutcomeSpace[" + all + "]";
	}
}