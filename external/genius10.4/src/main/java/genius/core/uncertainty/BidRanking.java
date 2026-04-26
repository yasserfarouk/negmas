package genius.core.uncertainty;

import java.util.List;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.Random;

import agents.org.apache.commons.lang.StringUtils;
import genius.core.Bid;
import genius.core.BidIterator;
import genius.core.Domain;
import genius.core.issue.Issue;


/**
 * Provides a (total) ranking of bids: b1 <= b2 <= ... <= bn
 */
public class BidRanking implements Iterable<Bid> {
	/** Ordered from low to high */
	private final List<Bid> bidOrder;
	private final double lowUtility;
	private final double highUtility;
	protected Random random = new Random();

	/**
	 * 
	 * @param bidOrder
	 *            bids, Ordered from low to high utility for me. Must not be
	 *            empty.
	 * @param lowUtil
	 *            A suggestion for the utility of the first (worst) bid in the
	 *            list. Must be in [0,1]
	 * @param highUtil
	 *            Suggested utility for me of the last (best) bid in the list.
	 *            Must be in [lowUtil,1]
	 */
	public BidRanking(List<Bid> bidOrder, double lowUtil, double highUtil) {
		if (bidOrder == null || bidOrder.isEmpty()) {
			throw new IllegalArgumentException(
					"bid order must contain at least one value.");
		}
		if (lowUtil < 0 || lowUtil > 1) {
			throw new IllegalArgumentException("low utility must be in [0,1]");
		}
		if (highUtil < lowUtil || lowUtil > 1) {
			throw new IllegalArgumentException(
					"low utility must be in [" + lowUtil + ",1]");
		}
		this.bidOrder = bidOrder;
		this.lowUtility = lowUtil;
		this.highUtility = highUtil;
	}

	public Bid getMinimalBid() {
		return bidOrder.get(0);
	}

	/**
	 * 
	 * @return The utility of the first (worst) bid in the list. 
	 * In [0,1]
	 */
	public Double getLowUtility() {
		return lowUtility;
	}

	/**
	 * 
	 * @return The utility of the last (best) bid in the list.
	 *         In [lowUtil,1].
	 */
	public Double getHighUtility() {
		return highUtility;
	}

	public Bid getMaximalBid() {
		return bidOrder.get(bidOrder.size() - 1);
	}

	public int indexOf(Bid b) {
		return bidOrder.indexOf(b);
	}

	public List<OutcomeComparison> getPairwiseComparisons() {
		ArrayList<OutcomeComparison> comparisons = new ArrayList<OutcomeComparison>();
		for (int i = 0; i < bidOrder.size() - 1; i++)
			comparisons.add(new OutcomeComparison(bidOrder.get(i),
					bidOrder.get(i + 1), -1));
		return comparisons;

	}
	
	/**
	 * This function computes the sum of pairwise distances between bids in the bidRanking.
	 * @return double 
	 */
	public double getTotalVarDistance() {
		double totalVarDist = 0;
		for (int i = 0; i<bidOrder.size(); i++) {
			for (int j = i+1; j<bidOrder.size(); j++) {
				totalVarDist += bidOrder.get(i).getDistance(bidOrder.get(j));
				System.out.println(bidOrder.get(i).getDistance(bidOrder.get(j)));
			}
		}
		double correcting_factor = (double)2/((double)(Math.pow(bidOrder.size(),2) - bidOrder.size()));
		return totalVarDist*correcting_factor;
	}
	
	/**
	 * Computes the added TV distance brought by the bid in input.
	 * @param Bid b
	 * @return double
	 */
	
	public double addedTV(Bid b) {
		double addedDist = 0;
		if (bidOrder.contains(b))
			return 0;
		for (int i=0; i<bidOrder.size(); i++) {
			addedDist += b.getDistance(bidOrder.get(i));
		}
		//double correcting_factor = (double)2/((double)(Math.pow(bidOrder.size(),2) - (double)bidOrder.size()));
		return addedDist;
	}
	
	/**
	 * This function returns the bid in the domain which is the farthest apart from the bids in the bidRanking.
	 * Note: There can be possibly many such bids, the function just return the first one it finds.
	 * @return Bid 
	 */
	public Bid getTVMaximizer(){
		Domain domain = this.getMaximalBid().getDomain();
		double max = 0;
		BidIterator bidIterator = new BidIterator(domain);
		Bid maximizer = bidIterator.next();
		while(bidIterator.hasNext()) {
			Bid nextBid = bidIterator.next();
			if(this.addedTV(nextBid) > max) {
				max = this.addedTV(nextBid);
				maximizer = nextBid;
			}
		}
		return maximizer;
	}
	
	/**
	 * This function returns the bid in the domain which is the closest from the bids in the bidRanking.
	 * Note: There can be possibly many such bids, the function just return the first one it finds.
	 * @return Bid 
	 */
	public Bid getTVMinimizer(){
		Domain domain = this.getMaximalBid().getDomain();
		double min = domain.getNumberOfPossibleBids();
		BidIterator bidIterator = new BidIterator(domain);
		Bid maximizer = bidIterator.next();
		while(bidIterator.hasNext()) {
			Bid nextBid = bidIterator.next();
			if(this.addedTV(nextBid) < min) {
				min = this.addedTV(nextBid);
				maximizer = nextBid;
			}
		}
		return maximizer;
	}
	/**
	 * Returns a random bid not in the bid rankin
	 * @return rdm bid
	 */
	public Bid getRandomBid(){
		Domain domain = this.getMaximalBid().getDomain();
		Bid rdm = domain.getRandomBid(random);
		while (bidOrder.contains(rdm)){
			rdm = domain.getRandomBid(random);
		}
		return rdm;
	}
	
	
	public List<Bid> getBidOrder() 
	{
		return bidOrder;
	}
	
	/**
	 * Gets all the issues from the first bid. 
	 * If all bids are from the same domain then this is a list of all issues in the domain.
	 */
	public List<Issue> getBidIssues()
	{
		return bidOrder.get(0).getIssues();
	}	
	
	/**
	 * The size equals 1 + the number of comparisons
	 * 
	 * @return
	 */
	public int getSize() {
		return bidOrder.size();
	}

	public int getAmountOfComparisons() {
		return getSize() - 1;
	}

	@Override
	public String toString() {
		return StringUtils.join(bidOrder.iterator(), " <= ");
	}

	/**
	 * Iterates the bids from low to high
	 */
	@Override
	public Iterator<Bid> iterator() {
		return bidOrder.iterator();
	}
}
