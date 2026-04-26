package genius.core.utility;

import java.io.Serializable;

import genius.core.Bid;
import genius.core.BidIterator;
import genius.core.Domain;

/**
 * Companion class to {@link UtilitySpace}. Some utility functions that work on
 * a utility space. Works as an add-on on a given {@link UtilitySpace}. Does not
 * extend it, so that receivers of an abstract {@link UtilitySpace} can connect
 * it with these tools too.
 * <p>
 * This is a class, not a set of static functions, to allow caching of results
 * (not yet implemented). Serializable so that eg AbstractuUtilitySpace can use
 * this as inner class and store cached results.
 *
 */
public class UtilitySpaceTools implements Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = -1727366033194010640L;
	private UtilitySpace utilSpace;

	public UtilitySpaceTools(UtilitySpace space) {
		utilSpace = space;
	}

	/**
	 * Returns the maximum bid in the utility space. This is only supported for
	 * linear utility spaces. Expensive: this does a brute-force search through
	 * bidspace.
	 * 
	 * @return a bid with the maximum utility value attainable in this util
	 *         space
	 * @throws IllegalStateException
	 *             if there is no bid at all in this util space.
	 */
	public final Bid getMaxUtilityBid() {
		Bid maxBid = null;
		double maxutil = 0.;
		BidIterator bidit = new BidIterator(utilSpace.getDomain());

		if (!bidit.hasNext())
			throw new IllegalStateException(
					"The domain does not contain any bids!");
		while (bidit.hasNext()) {
			Bid thisBid = bidit.next();
			double thisutil = utilSpace.getUtility(thisBid);
			if (thisutil > maxutil) {
				maxutil = thisutil;
				maxBid = thisBid;
			}
		}
		return maxBid;
	}

	/**
	 * Returns the worst bid in the utility space. This is only supported for
	 * linear utility spaces.
	 * 
	 * @return a bid with the lowest possible utility
	 * @throws IllegalStateException
	 *             if there is no bid at all in the util space
	 */
	public Bid getMinUtilityBid() {
		Bid minBid = null;
		double minUtil = 1.2;
		BidIterator bidit = new BidIterator(utilSpace.getDomain());

		if (!bidit.hasNext())
			throw new IllegalStateException(
					"The domain does not contain any bids!");
		while (bidit.hasNext()) {
			Bid thisBid = bidit.next();
			double thisutil = utilSpace.getUtility(thisBid);
			if (thisutil < minUtil) {
				minUtil = thisutil;
				minBid = thisBid;
			}
		}
		return minBid;
	}
	
	
	/**
	 * Check if this utility space is ready for negotiation. To be so, the
	 * domain must match the given domain and the space must be complete.
	 * 
	 * @param dom
	 *            is the domain in which nego is taking place
	 * @throws IllegalStateException
	 *             if there is somethign wrong with this domain
	 */
	public void checkReadyForNegotiation(Domain dom) {
		// check if utility spaces are instance of the domain
		// following checks normally succeed, as the domain of the domain space
		// is enforced in the loader.
		if (!(dom.equals(utilSpace.getDomain())))
			throw new IllegalStateException(
					"domain does not match the negotiation domain");
		String err = utilSpace.isComplete();
		if (err != null)
			throw new IllegalStateException(
					"utility space is incomplete:" + err);
	}
}
