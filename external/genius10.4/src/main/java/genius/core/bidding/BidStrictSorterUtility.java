package genius.core.bidding;

import java.util.Comparator;

import genius.core.Bid;
import genius.core.utility.AdditiveUtilitySpace;

/**
 * Comparator which sorts a set of Bids based on their utility.
 * The bid with the highest utility is on the front of the list.
 * In addition, the ordering is unique: bids with exactly the same utility
 * are always ordered the same. Use this class ONLY when comparing if two
 * strategies are equivalent.
 * 
 * @author Mark Hendrikx
 */
public class BidStrictSorterUtility implements Comparator<Bid>{
	
	private AdditiveUtilitySpace utilitySpace;
	
	/**
	 * Initializes the comperator by setting the bidding space.
	 * The utility space is necessary to evaluate the utility of the bids.
	 * @param utilitySpace used to evaluate the utility of the bids.
	 */
	public BidStrictSorterUtility(AdditiveUtilitySpace utilitySpace) {
		super();
		this.utilitySpace = utilitySpace;
	}

	/**
	 * Comperator. If util b1 > b2 then -1, else if < then 1, else
	 * compare hashcodes.
	 */
	public int compare(Bid b1, Bid b2)
	{
		try{
			if (b1 == null || b2 == null)
				throw new NullPointerException();
			if (utilitySpace.getUtility(b1) == utilitySpace.getUtility(b2)) {
				return String.CASE_INSENSITIVE_ORDER.compare(b1.toString(), b2.toString());
			}
			if (utilitySpace.getUtility(b1) > utilitySpace.getUtility(b2))
				return -1;
			else if (utilitySpace.getUtility(b1)< utilitySpace.getUtility(b2))
		        return 1;
		    else
		        return ((Integer) b1.hashCode()).compareTo(b2.hashCode());
		} catch (Exception e) {
			e.printStackTrace();
		}
		return -1;
	}
}