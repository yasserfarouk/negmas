package agents.anac.y2011.Nice_Tit_for_Tat;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

import genius.core.Bid;
import genius.core.BidIterator;
import genius.core.Domain;
import genius.core.bidding.BidDetails;
import genius.core.bidding.BidDetailsSorterUtility;
import genius.core.bidding.BidDetailsStrictSorterUtility;
import genius.core.utility.AdditiveUtilitySpace;

public class BidHistory implements Iterable<BidDetails>
{
	List<BidDetails> history;
	private final boolean TEST_EQUIVALENCE = false;
	
	public BidHistory()
	{
		this.history = new ArrayList<BidDetails>();
	}
	
	public BidHistory(BidHistory b)
	{
		this.history = new ArrayList<BidDetails>(b.getHistory());
	}
	
	/**
	 * Pretends all bids in the domain were made on t = 0.
	 */
	public BidHistory(AdditiveUtilitySpace u)
	{
		this();
		Domain domain = u.getDomain();
		BidIterator myBidIterator = new BidIterator(domain);
		while (myBidIterator.hasNext()) 
		{
			Bid b = myBidIterator.next();
			double utility = 0;
			try
			{
				utility = u.getUtility(b);
			} catch (Exception e)
			{
				e.printStackTrace();
			}
			BidDetails bidDetails = new BidDetails(b, utility, 0);
			add(bidDetails);
		}
	}
	
	public BidHistory filterBetweenTime(double minT, double maxT)
	{
		return filterBetween(0, 1, minT, maxT);		
	}

	public BidHistory filterBetween(double minU, double maxU, double minT, double maxT)
	{
		BidHistory bidHistory = new BidHistory();
		for (BidDetails b : history)
		{
			if (minU < b.getMyUndiscountedUtil() &&
					b.getMyUndiscountedUtil() <= maxU &&
					minT < b.getTime() &&
					b.getTime() <= maxT)
				bidHistory.add(b);
		}
		return bidHistory;			
	}
	
	public void add(BidDetails b)
	{
		history.add(b);
	}
	
	public BidDetails getLastBidDetails()
	{
		if (history.isEmpty())
			return null;

		BidDetails bidDetails = history.get(size() - 1);
		return bidDetails;
	}
	
	public BidDetails getFirstBidDetails()
	{
		if (history.isEmpty())
			return null;

		BidDetails bidDetails = history.get(0);
		return bidDetails;
	}
	
	public Bid getLastBid()
	{
		BidDetails lastBidDetails = getLastBidDetails();
		if (lastBidDetails == null)
			return null;
		return lastBidDetails.getBid();
	}
	
	public Bid getSecondLastBid()
	{
		if (history.size() < 2)
			return null;
		
		return history.get(history.size() - 2).getBid();
	}
	
	public int size()
	{
		return history.size();
	}
	
	public List<BidDetails> getHistory()
	{
		return history;
	}
	
	/**
	 * Get the {@link BidDetails} of the {@link Bid} with utility closest to u.
	 */
	public BidDetails getBidDetailsOfUtility(double u)
	{
		double minDistance = -1;
		BidDetails closestBid = null;
		for (BidDetails b : history)
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
	
	public double getMaximumUtility()
	{
		double max = -1;
		for (BidDetails b : history)
		{
			double utility = b.getMyUndiscountedUtil();
			if (utility >= max || max == -1)
				max = utility;
		}
		return max;		
	}

	public double getMinimumUtility()
	{
		double min = -1;
		for (BidDetails b : history)
		{
			double utility = b.getMyUndiscountedUtil();
			if (utility <= min || min == -1)
				min = utility;
		}
		return min;		
	}
	
	/**
	 * Gets the best bid for me.
	 */
	public Bid getBestBid()
	{
		BidDetails bestBidDetails = getBestBidDetails();
		if (bestBidDetails == null)
			return null;
		return bestBidDetails.getBid();
	}
	
	/**
	 * Gets the details of the best bid for me.
	 */
	public BidDetails getBestBidDetails()
	{
		double max = -1;
		BidDetails bestBid = null;
		for (BidDetails b : history)
		{
			double utility = b.getMyUndiscountedUtil();
			if (utility >= max || max == -1)
			{
				max = utility;
				bestBid = b;
			}
		}
		return bestBid;
	}
	
	/**
	 * Gets the history part of the best n (or less) bids for me.
	 */
	public BidHistory getBestBidHistory(int n)
	{
		BidHistory copySortedToUtility = getCopySortedToUtility();
		BidHistory best = new BidHistory();
		int i = 0;
		for (BidDetails b : copySortedToUtility)
		{
			best.add(b);
			i++;
			if (i >= n)
				break;
		}
		return best;
	}
	
	public BidDetails getRandom()
	{
		int size = size();
		if (size == 0)
			return null;
		int index = (new Random()).nextInt(size);
		return history.get(index);
	}
	
	public BidDetails getRandom(Random r)
	{
		int size = size();
		if (size == 0)
			return null;
		int index = r.nextInt(size);
		return history.get(index);
	}
	
	public double getAverageUtility()
	{
		int size = size();
		if (size == 0)
			return 0;
		double totalUtil = 0;
		for (BidDetails b : history)
			totalUtil += b.getMyUndiscountedUtil();
		return totalUtil / size;
	}
	
	public void sortToUtility()
	{
		if (TEST_EQUIVALENCE) {
			Collections.sort(history, new BidDetailsStrictSorterUtility());
		} else {
			Collections.sort(history, new BidDetailsSorterUtility());
		}
	}
	
	public BidHistory getCopySortedToUtility()
	{
		BidHistory bidHistoryCopy = new BidHistory(this);
		bidHistoryCopy.sortToUtility();
		return bidHistoryCopy;
	}

	@Override
	public String toString()
	{
		return "" + history;	
	}

	public Iterator<BidDetails> iterator()
	{
		return history.iterator();
	}
}
