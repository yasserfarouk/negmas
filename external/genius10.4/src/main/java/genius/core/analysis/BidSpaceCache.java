package genius.core.analysis;

import java.util.Arrays;

import genius.core.utility.AbstractUtilitySpace;

/**
 * Caches the BidSpace such that we don't have to recompute it each time. Only
 * one BidSpace can be stored at a time to limit the memory costs.
 * 
 * @author Mark Hendrikx
 */
public class BidSpaceCache {

	/** String representation of the stored domains. */
	private static String[] identifier;
	/** Reference to the cached BidSpace. */
	private static BidSpace cachedBidSpace;

	/**
	 * Method used to load a BidSpace. If the BidSpace is already cached, then
	 * the cached BidSpace is used. Otherwise, the BidSpace is constructed.
	 * 
	 * @param spaces
	 *            from which a BidSpace must be constructed.
	 * @return BidSpace belonging to the given UtilitySpace's.
	 */
	public static BidSpace getBidSpace(AbstractUtilitySpace... spaces) {
		// determine the unique identifier of the given utilityspaces.
		String[] ident = new String[spaces.length];
		for (int i = 0; i < spaces.length; i++) {
			ident[i] = spaces[i].getFileName();
		}

		// check if the space is already cached. If not, createFrom bidspace.
		if (!Arrays.equals(ident, identifier)) {
			try {
				cachedBidSpace = new BidSpace(spaces);
				identifier = ident;
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
		return cachedBidSpace;
	}
}