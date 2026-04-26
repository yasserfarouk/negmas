package genius.core.bidding;

import java.util.Comparator;

import genius.core.bidding.BidDetails;

/**
 * Comparator which sorts a set of BidDetails based on their utility.
 * The bid with the highest utility is on the front of the list.
 * In addition, the ordering is unique: bids with exactly the same utility
 * are always ordered the same. Use this class ONLY when comparing if two
 * strategies are equivalent.
 * 
 * @author Mark Hendrikx
 */
public class BidDetailsStrictSorterUtility implements Comparator<BidDetails>
{
	/**
	 * Comperator. If util b1 > b2 then -1, else if < then 1, else
	 * compare hashcodes.
	 */
	public int compare(BidDetails b1, BidDetails b2)
	{
		if (b1 == null || b2 == null)
			throw new NullPointerException();
		if (b1.getMyUndiscountedUtil() == b2.getMyUndiscountedUtil()) {
			return String.CASE_INSENSITIVE_ORDER.compare(b1.getBid().toString(), b2.getBid().toString());
		}
		if (b1.getMyUndiscountedUtil() > b2.getMyUndiscountedUtil())
			return -1;
		else if (b1.getMyUndiscountedUtil() < b2.getMyUndiscountedUtil())
	        return 1;
	    else
	        return ((Integer) b1.hashCode()).compareTo(b2.hashCode());
	}
}