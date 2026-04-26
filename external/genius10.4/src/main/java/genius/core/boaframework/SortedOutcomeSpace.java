package genius.core.boaframework;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import genius.core.bidding.BidDetails;
import genius.core.bidding.BidDetailsSorterUtility;
import genius.core.misc.Range;
import genius.core.utility.AbstractUtilitySpace;

/**
 * This class is an OutcomeSpace but with a sorted list of BidDetails based on
 * the utility Methods have been optimized to work with a sorted list. Useful if
 * someone wants to quickly implement an agent.
 * 
 * @author Alex Dirkzwager, Mark Hendrikx
 */
public class SortedOutcomeSpace extends OutcomeSpace {

	/**
	 * Instantiates a SortedOutcomeSpace: an enumeration of all possible bids in
	 * the domain which can be efficiently searched using the provided methods.
	 * Warning: this call iterates over ALL possible bids.
	 * 
	 * @param utilSpace
	 *            utilityspace of the agent.
	 */
	public SortedOutcomeSpace(AbstractUtilitySpace utilSpace) {
		super(utilSpace);
		Collections.sort(allBids, new BidDetailsSorterUtility());
	}
	
	/**
	 * Gives back all the bids ordered from high to low utility.
	 * @return List<BidDetails
	 */
	public List<BidDetails> getOrderedList() {
		return allBids;
	}
	
	/**
	 * gets a list of BidDetails that have a utility within the range
	 * 
	 * @param r
	 * @return A list of BidDetails
	 */
	@Override
	public List<BidDetails> getBidsinRange(Range r) {
		// get upperbound index
		int upperboundIndex = searchIndexWith(r.getUpperbound());
		// get lowerbound index
		int lowerboundIndex = searchIndexWith(r.getLowerbound());

		// test upperbound index element is under upperbound
		if (allBids.get(upperboundIndex).getMyUndiscountedUtil() <= r.getUpperbound() && upperboundIndex > 0) {
			upperboundIndex--;
		}
		// test lowerbound index element is under lowerbound
		if (allBids.get(lowerboundIndex).getMyUndiscountedUtil() >= r.getLowerbound()
				&& lowerboundIndex < allBids.size()) {
			lowerboundIndex++;
		}

		// Sublist return empty list if upper and lower bounds are equal, thus
		// this side case
		if (lowerboundIndex == upperboundIndex) {
			ArrayList<BidDetails> list = new ArrayList<BidDetails>();
			list.add(allBids.get(lowerboundIndex));
			return list;
		}

		ArrayList<BidDetails> subList = new ArrayList<BidDetails>(allBids.subList(upperboundIndex, lowerboundIndex));
		return subList;

	}

	/**
	 * Gets a BidDetails which is close to the utility
	 * 
	 * @param utility
	 * @return BidDetails
	 */
	@Override
	public BidDetails getBidNearUtility(double utility) {
		return allBids.get(getIndexOfBidNearUtility(utility));

	}

	/**
	 * Gets a BidDetails which is close to the utility
	 * 
	 * @param utility
	 * @return BidDetails
	 */
	@Override
	public int getIndexOfBidNearUtility(double utility) {
		int index = searchIndexWith(utility);
		int newIndex = -1;
		double closestDistance = Math.abs(allBids.get(index).getMyUndiscountedUtil() - utility);

		// checks if the BidDetails above the selected is closer to the
		// targetUtility
		if (index > 0 && Math.abs(allBids.get(index - 1).getMyUndiscountedUtil() - utility) < closestDistance) {
			newIndex = index - 1;
			closestDistance = Math.abs(allBids.get(index - 1).getMyUndiscountedUtil() - utility);
		}

		// checks if the BidDetails below the selected is closer to the
		// targetUtility
		if (index + 1 < allBids.size()
				&& Math.abs(allBids.get(index + 1).getMyUndiscountedUtil() - utility) < closestDistance) {
			newIndex = index + 1;
			closestDistance = Math.abs(allBids.get(index + 1).getMyUndiscountedUtil() - utility);
		} else
			newIndex = index;
		return newIndex;

	}

	/**
	 * Binary search of a BidDetails with a particular value if there is no
	 * BidDetails with the exact value gives the last index because this is the
	 * closest BidDetails to the value
	 * 
	 * @param value
	 * @return index
	 */
	public int searchIndexWith(double value) {
		int middle = -1;
		int low = 0;
		int high = allBids.size() - 1;
		int lastMiddle = 0;
		while (lastMiddle != middle) {
			lastMiddle = middle;
			middle = (low + high) / 2;
			if (allBids.get(middle).getMyUndiscountedUtil() == value) {
				return middle;
			}
			if (allBids.get(middle).getMyUndiscountedUtil() < value) {
				high = middle;
			}
			if (allBids.get(middle).getMyUndiscountedUtil() > value) {
				low = middle;
			}
		}
		return middle;
	}

	/**
	 * @return best bid in the domain.
	 */
	@Override
	public BidDetails getMaxBidPossible() {
		return allBids.get(0);
	}

	/**
	 * @return worst bid in the domain.
	 */
	public BidDetails getMinBidPossible() {
		return allBids.get(allBids.size() - 1);
	}
	
	
}