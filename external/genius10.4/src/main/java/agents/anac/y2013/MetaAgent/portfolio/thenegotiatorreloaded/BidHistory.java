package agents.anac.y2013.MetaAgent.portfolio.thenegotiatorreloaded;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import genius.core.Bid;
import genius.core.utility.AdditiveUtilitySpace;

/**
 * This class contains the bidding history of a negotiation agent.
 * 
 * @author Alex Dirkzwager, Mark Hendrikx, Tim Baarslag
 */
public class BidHistory {
	
	// list used to store the bids in the order in which they are received
	private List<BidDetails> bidList;
	
	/**
	 * Creates a bid history given an array of bids offered
	 * by the negotiation agent.
	 * @param bids
	 */
	public BidHistory(ArrayList<BidDetails> bids) {
		bidList =  bids;
	}
	
	public BidHistory(AdditiveUtilitySpace utilSpace) {
		BidIterator bidsIter = new BidIterator(utilSpace.getDomain());
		bidList = new ArrayList<BidDetails>();
		while (bidsIter.hasNext()) {
			Bid bid = bidsIter.next();
			double util = 0;
			try {
				util = utilSpace.getUtility(bid);
			} catch (Exception e) {
				e.printStackTrace();
			}
			bidList.add(new BidDetails(bid, util));
		}
	}
	
	/**
	 * Creates an empty bid history.
	 */
	public BidHistory() {
		bidList = new ArrayList<BidDetails>();
	}
	
	/**
	 * Returns the set of bids offered between time instances t1 and t2: (t1, t2].
	 * @param t1
	 * @param t2
	 * @return bids done in (t1, t2]
	 */
	public BidHistory filterBetweenTime(double t1, double t2)
	{
		return filterBetween(0, 1.1, t1, t2);		
	}
	
	/**
	 * Returns the set of bids with a utility of at least u1 and at most u2: (u1, u2].
	 * @param u1
	 * @param u2
	 * @return bids with a utility in (u1, u2]
	 */
	public BidHistory filterBetweenUtility(double u1, double u2)
	{
		return filterBetween(u1, u2, 0, 1.1);		
	}

	/**
	 * Returns the set of bids offered between time instances t1 and t2: (t1, t2] and
	 * with a utility in (u1, u2].
	 * @param minU
	 * @param maxU
	 * @param minT
	 * @param maxT
	 * @return
	 */
	public BidHistory filterBetween(double minU, double maxU, double minT, double maxT)
	{
		BidHistory bidHistory = new BidHistory();
		for (BidDetails b : bidList)
		{
			if (minU < b.getMyUndiscountedUtil() &&
					b.getMyUndiscountedUtil() <= maxU &&
					minT < b.getTime() &&
					b.getTime() <= maxT)
				bidHistory.add(b);
		}
		return bidHistory;			
	}
	
	/**
	 * Add an offered bid o the history.
	 * @param offered bid
	 */
	public void add(BidDetails bid){
		bidList.add(bid);
	}

	/**
	 * Returns the full history.
	 * @return history
	 */
	public List<BidDetails> getHistory() {
		return bidList;
	}
	
	/**
	 * Returns the last bid added to the history.
	 * @return last added bid
	 */
	public BidDetails getLastBidDetails(){
		BidDetails bid = null;
		if (bidList.size() > 0) {
			bid = bidList.get(bidList.size() - 1);
		}
		return bid;
	}
	
	/**
	 * Returns the first bid stored in the history
	 * @return first bid of history
	 */
	public BidDetails getFirstBidDetails() {
		return bidList.get(0);
	}
	
	/**
	 * Returns the bid with the highest utility stored in the history.
	 * @return bid with highest utility
	 */
	public BidDetails getBestBidDetails(){
		double max = Double.NEGATIVE_INFINITY;
		BidDetails bestBid = null;
		for (BidDetails b : bidList)
		{
			double utility = b.getMyUndiscountedUtil();
			if (utility >= max)
			{
				max = utility;
				bestBid = b;
			}
		}
		return bestBid;
	}
	
	/**
	 * Returns the bid with the lowest utility stored in the history.
	 * @return bid with lowest utility
	 */
	public BidDetails getWorstBidDetails(){
		double min = Double.POSITIVE_INFINITY;
		BidDetails worstBid = null;
		for (BidDetails b : bidList)
		{
			double utility = b.getMyUndiscountedUtil();
			if (utility < min)
			{
				min = utility;
				worstBid = b;
			}
		}
		return worstBid;
	}
	
	/**
	 * Returns a list of the top N bids which the opponent has offered.
	 * @param count
	 * @return a list of UTBids
	 */
	public List<BidDetails> getNBestBids(int count) {
		List<BidDetails> result = new ArrayList<BidDetails>();
		List<BidDetails> sortedOpponentBids = new ArrayList<BidDetails>(bidList);
		Collections.sort(sortedOpponentBids, new BidDetailsSorterUtility());
		for (int i = 0; i < count && i < sortedOpponentBids.size(); i++) {
			result.add(sortedOpponentBids.get(i));
		}

		return result;
	}

	public int size() {
		return bidList.size();
	}
	
	public double getAverageUtility() {
		int size = size();
		if (size == 0)
			return 0;
		double totalUtil = 0;
		for(BidDetails bid : bidList){
			totalUtil =+bid.getMyUndiscountedUtil();
		}
		return totalUtil / size;
	}
	
	/**
	 * Get the {@link BidDetails} of the {@link Bid} with utility closest to u.
	 */
	public BidDetails getBidDetailsOfUtility(double u) {
		double minDistance = -1;
		BidDetails closestBid = null;
		for (BidDetails b : bidList)
		{
			double utility = b.getMyUndiscountedUtil();
			if (Math.abs(utility - u) <= minDistance || minDistance == -1)
			{
				minDistance = Math.abs(utility - u);
				closestBid = b;
			}
		}
		return closestBid;
	}
	
	public BidHistory sortToUtility() {
		BidHistory sortedHistory = this;
		Collections.sort(sortedHistory.getHistory(), new BidDetailsSorterUtility());
		return sortedHistory;
	}
	
	public BidHistory sortToTime() {
		BidHistory sortedHistory = this;
		Collections.sort(sortedHistory.getHistory(), new BidDetailsSorterTime());
		return sortedHistory;
	}
	
	public BidDetails getRandom() {
		return getRandom(new Random());
	}
	
	public BidDetails getRandom(Random rand)
	{
		int size = size();
		if (size == 0)
			return null;
		int index = rand.nextInt(size);
		return bidList.get(index);
	}
}
