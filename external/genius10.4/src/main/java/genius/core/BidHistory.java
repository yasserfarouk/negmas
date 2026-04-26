package genius.core;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import genius.core.bidding.BidDetails;
import genius.core.bidding.BidDetailsSorterTime;
import genius.core.bidding.BidDetailsSorterUtility;
import genius.core.bidding.BidDetailsStrictSorterUtility;
import genius.core.utility.AbstractUtilitySpace;

/**
 * This class contains the bidding history of a negotiation agent.
 * 
 * @author Alex Dirkzwager, Mark Hendrikx, Tim Baarslag
 */
public class BidHistory implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1663962498632353562L;
	// list used to store the bids in the order in which they are received
	private List<BidDetails> bidList;
	/**
	 * Set this boolean to true if you want to verify that two agents are
	 * exactly equal
	 */
	private final boolean TEST_EQUIVALENCE = false;

	/**
	 * Creates a bid history given an array of bids offered by the negotiation
	 * agent.
	 * 
	 * @param bids
	 */
	public BidHistory(List<BidDetails> bids) {
		bidList = bids;
	}

	/**
	 * Creates an empty bid history.
	 */
	public BidHistory() {
		bidList = new ArrayList<BidDetails>();
	}

	/**
	 * Returns the set of bids offered between time instances t1 and t2: (t1,
	 * t2].
	 * 
	 * @param t1
	 * @param t2
	 * @return bids done in (t1, t2]
	 */
	public BidHistory filterBetweenTime(double t1, double t2) {
		return filterBetween(0, 1.1, t1, t2);
	}

	/**
	 * Returns the set of bids with a utility of at least u1 and at most u2:
	 * (u1, u2]. If u1 = u2, then it returns all bids with utility u1.
	 * 
	 * @param minU
	 *            minimum utility.
	 * @param maxU
	 *            maximum utility.
	 * @return bids with a utility in (u1, u2]
	 */
	public BidHistory filterBetweenUtility(double minU, double maxU) {
		if (minU == maxU)
			return filterUtility(minU);

		BidHistory bidHistory = new BidHistory();
		for (BidDetails b : bidList) {
			if (minU < b.getMyUndiscountedUtil()
					&& b.getMyUndiscountedUtil() <= maxU)
				bidHistory.add(b);
		}
		return bidHistory;
	}

	/**
	 * Returns the set of bids offered between time instances t1 and t2: (t1,
	 * t2] and with a utility in (u1, u2].
	 * 
	 * @param minU
	 *            minimum utility.
	 * @param maxU
	 *            maximum utility.
	 * @param minT
	 *            minimum time.
	 * @param maxT
	 *            maximum time.
	 * @return bids with utility (minU, maxU] made in the time (minT, maxT].
	 */
	public BidHistory filterBetween(double minU, double maxU, double minT,
			double maxT) {
		BidHistory bidHistory = new BidHistory();
		for (BidDetails b : bidList) {
			if (minU < b.getMyUndiscountedUtil()
					&& b.getMyUndiscountedUtil() <= maxU && minT < b.getTime()
					&& b.getTime() <= maxT)
				bidHistory.add(b);
		}
		return bidHistory;
	}

	/**
	 * Returns the set of bids with utility u.
	 * 
	 * @param u
	 *            utility.
	 * @return set of bids with utility u.
	 */
	public BidHistory filterUtility(double u) {
		BidHistory bidHistory = new BidHistory();
		for (BidDetails b : bidList)
			if (b.getMyUndiscountedUtil() == u)
				bidHistory.add(b);
		return bidHistory;
	}

	/**
	 * Returns the set of bids offered between time instances t1 and t2: (t1,
	 * t2] and with a utility in (u1, u2].
	 * 
	 * @param minU
	 *            minimum discounted utility.
	 * @param maxU
	 *            maximum discounted utility.
	 * @param minT
	 *            minimum time.
	 * @param maxT
	 *            maximum time.
	 * @param utilSpace
	 *            preference profile used to find the discounted utility.
	 * @return bids with discounted utility (minU, maxU] made in the time (minT,
	 *         maxT].
	 */
	public BidHistory discountedFilterBetween(double minU, double maxU,
			double minT, double maxT, AbstractUtilitySpace utilSpace) {
		BidHistory bidHistory = new BidHistory();
		for (BidDetails b : bidList) {
			if (minU < utilSpace.getUtilityWithDiscount(b.getBid(), b.getTime())
					&& utilSpace.getUtilityWithDiscount(b.getBid(),
							b.getTime()) <= maxU
					&& minT < b.getTime() && b.getTime() <= maxT)
				bidHistory.add(b);
		}
		return bidHistory;
	}

	/**
	 * Add an offered bid o the history.
	 * 
	 * @param bid
	 *            offered bid.
	 */
	public void add(BidDetails bid) {
		bidList.add(bid);
	}

	/**
	 * Returns the full history.
	 * 
	 * @return history
	 */
	public List<BidDetails> getHistory() {
		return bidList;
	}

	/**
	 * Returns the last bid details added to the history.
	 * 
	 * @return last added bid details
	 */
	public BidDetails getLastBidDetails() {
		BidDetails bid = null;
		if (bidList.size() > 0) {
			bid = bidList.get(bidList.size() - 1);
		}
		return bid;
	}

	/**
	 * Returns the last bid added to the history.
	 * 
	 * @return last added bid, or null if no such bid.
	 */
	public Bid getLastBid() {
		BidDetails lastBidDetails = getLastBidDetails();
		if (lastBidDetails == null)
			return null;
		return lastBidDetails.getBid();
	}

	/**
	 * Returns the first bid stored in the history
	 * 
	 * @return first bid of history
	 */
	public BidDetails getFirstBidDetails() {
		return bidList.get(0);
	}

	/**
	 * Returns the bid with the highest utility stored in the history.
	 * 
	 * @return bid with highest utility
	 */
	public BidDetails getBestBidDetails() {
		double max = Double.NEGATIVE_INFINITY;
		BidDetails bestBid = null;
		for (BidDetails b : bidList) {
			double utility = b.getMyUndiscountedUtil();
			if (utility >= max) {
				max = utility;
				bestBid = b;
			}
		}
		return bestBid;
	}

	/**
	 * Returns the bid with the highest discounted utility stored in the
	 * history.
	 * 
	 * @param util
	 *            preference profile used to determine the discounted utility of
	 *            a bid.
	 * @return bid with highest utility
	 */
	public BidDetails getBestDiscountedBidDetails(AbstractUtilitySpace util) {
		double max = Double.NEGATIVE_INFINITY;
		BidDetails bestBid = null;
		for (BidDetails b : bidList) {
			double discountedUtility = util.getUtilityWithDiscount(b.getBid(),
					b.getTime());
			if (discountedUtility >= max) {
				max = discountedUtility;
				bestBid = b;
			}
		}
		return bestBid;
	}

	/**
	 * Returns the bid with the lowest utility stored in the history.
	 * 
	 * @return bid with lowest utility
	 */
	public BidDetails getWorstBidDetails() {
		double min = Double.POSITIVE_INFINITY;
		BidDetails worstBid = null;
		for (BidDetails b : bidList) {
			double utility = b.getMyUndiscountedUtil();
			if (utility < min) {
				min = utility;
				worstBid = b;
			}
		}
		return worstBid;
	}

	/**
	 * Returns a list of the top N bids which the opponent has offered.
	 * 
	 * @param count
	 *            amount of N best bids.
	 * @return a list of bids.
	 */
	public List<BidDetails> getNBestBids(int count) {
		List<BidDetails> result = new ArrayList<BidDetails>();
		List<BidDetails> sortedOpponentBids = new ArrayList<BidDetails>(
				bidList);
		if (TEST_EQUIVALENCE) {
			Collections.sort(sortedOpponentBids,
					new BidDetailsStrictSorterUtility());
		} else {
			Collections.sort(sortedOpponentBids, new BidDetailsSorterUtility());
		}

		for (int i = 0; i < count && i < sortedOpponentBids.size(); i++) {
			result.add(sortedOpponentBids.get(i));
		}

		return result;
	}

	/**
	 * @return amount of bids stored.
	 */
	public int size() {
		return bidList.size();
	}

	/**
	 * @return average utility of bids stored.
	 */
	public double getAverageUtility() {
		int size = size();
		if (size == 0)
			return 0;
		double totalUtil = 0;
		for (BidDetails bid : bidList) {
			totalUtil += bid.getMyUndiscountedUtil();
		}
		return totalUtil / size;
	}

	/**
	 * @param utilSpace
	 *            preference profile used to determine the discounted utility of
	 *            a bid.
	 * @return average discounted utility of bids stored.
	 */
	public double getAverageDiscountedUtility(AbstractUtilitySpace utilSpace) {
		int size = size();
		if (size == 0)
			return 0;
		double totalUtil = 0;
		for (BidDetails bid : bidList) {
			totalUtil += utilSpace.getUtilityWithDiscount(bid.getBid(),
					bid.getTime());
		}
		return totalUtil / size;
	}

	/**
	 * Sorts the bids contained in this BidHistory object on utility.
	 * 
	 * @return sorted BidHistory.
	 */
	public BidHistory sortToUtility() {
		BidHistory sortedHistory = this;
		Collections.sort(sortedHistory.getHistory(),
				new BidDetailsSorterUtility());
		return sortedHistory;
	}

	/**
	 * Sorts the bids contained in this BidHistory object on time.
	 * 
	 * @return sorted BidHistory.
	 */
	public BidHistory sortToTime() {
		BidHistory sortedHistory = this;
		Collections.sort(sortedHistory.getHistory(),
				new BidDetailsSorterTime());
		return sortedHistory;
	}

	/**
	 * @return random bid from this BidHistory.
	 */
	public BidDetails getRandom() {
		return getRandom(new Random());
	}

	/**
	 * @param rand
	 *            random generator.
	 * @return random bid from this BidHistory using the given random generator.
	 */
	public BidDetails getRandom(Random rand) {
		int size = size();
		if (size == 0)
			return null;
		int index = rand.nextInt(size);
		return bidList.get(index);
	}

	/**
	 * Checks if BidHistory (array) is empty or not.
	 * 
	 * @return true if no bids are stored.
	 */
	public boolean isEmpty() {
		return bidList.isEmpty();
	}

	public BidDetails getMedianUtilityBid() {
		BidHistory sortedHistory = this.sortToUtility();

		BidDetails medianBid = sortedHistory.getHistory()
				.get(sortedHistory.getHistory().size() / 2);

		return medianBid;
	}
}
