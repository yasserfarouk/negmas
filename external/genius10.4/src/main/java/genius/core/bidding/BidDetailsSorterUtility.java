package genius.core.bidding;

import java.util.Comparator;

/**
 * Comparator which sorts a set of BidDetails based on their utility.
 * The bid with the highest utility is on the front of the list.
 */
public class BidDetailsSorterUtility implements Comparator<BidDetails>
{
	/**
	 * Comperator. If util b1 > b2 then -1, else if < then 1, else
	 * compare hashcodes.
	 */
	public int compare(BidDetails b1, BidDetails b2)
	{
		if (b1 == null || b2 == null)
			throw new NullPointerException();
		if (b1.equals(b2))
			return 0;
		if (b1.getMyUndiscountedUtil() > b2.getMyUndiscountedUtil())
			return -1;
		else if (b1.getMyUndiscountedUtil() < b2.getMyUndiscountedUtil())
	        return 1;
	    else
	        return ((Integer) b1.hashCode()).compareTo(b2.hashCode());
	}
}