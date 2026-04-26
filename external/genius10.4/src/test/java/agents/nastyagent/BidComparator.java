package agents.nastyagent;

import java.util.Comparator;

import genius.core.Bid;
import genius.core.utility.UtilitySpace;

/**
 * {@link Comparator} for {@link Bid}s. Used for sorting a set of bids.
 * 
 * Use example:
 * <code>		Collections.sort(bids, new BidComparator(utilitySpace));
</code>
 * 
 * @author W.Pasman
 *
 */
public class BidComparator implements java.util.Comparator<Bid> {
	UtilitySpace utilspace;

	public BidComparator(UtilitySpace us) {
		if (us == null)
			throw new NullPointerException("null utility space");
		utilspace = us;
	}

	public int compare(Bid b1, Bid b2) throws ClassCastException {
		double d1 = 0, d2 = 0;
		try {
			d1 = utilspace.getUtility(b1);
			d2 = utilspace.getUtility(b2);
		} catch (Exception e) {
			e.printStackTrace();
		}

		if (d1 < d2)
			return 1;
		if (d1 > d2)
			return -1;
		return 0;
	}
}
