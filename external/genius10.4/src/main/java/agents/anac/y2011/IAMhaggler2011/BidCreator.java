package agents.anac.y2011.IAMhaggler2011;

import genius.core.Bid;
import genius.core.utility.AdditiveUtilitySpace;

public interface BidCreator {

	/**
	 * Get a bid.
	 * 
	 * @param utilitySpace
	 *            The utility space to generate the bid from.
	 * @return a bid.
	 */
	// public Bid getBid(UtilitySpace utilitySpace);

	/**
	 * Get a bid (above a minimum utility value if possible).
	 * 
	 * @param utilitySpace
	 *            The utility space to generate the bid from.
	 * @param min
	 *            The minimum utility value.
	 * @return a bid (above a minimum utility value if possible).
	 */
	// public Bid getBid(UtilitySpace utilitySpace, double min);

	/**
	 * Get a bid (within a utility range if possible).
	 * 
	 * @param utilitySpace
	 *            The utility space to generate the bid from.
	 * @param min
	 *            The minimum utility value.
	 * @param max
	 *            The maximum utility value.
	 * @return a bid (within a utility range if possible).
	 */
	public Bid getBid(AdditiveUtilitySpace utilitySpace, double min, double max);
}
