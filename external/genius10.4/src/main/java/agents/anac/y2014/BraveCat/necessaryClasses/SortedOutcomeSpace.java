package agents.anac.y2014.BraveCat.necessaryClasses;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import genius.core.bidding.BidDetails;
import genius.core.bidding.BidDetailsSorterUtility;
import genius.core.misc.Range;
import genius.core.utility.AbstractUtilitySpace;

public class SortedOutcomeSpace extends OutcomeSpace {
	public SortedOutcomeSpace(AbstractUtilitySpace utilSpace) {
		super(utilSpace);
		Collections.sort(this.allBids, new BidDetailsSorterUtility());
	}

	public List<BidDetails> getBidsinRange(Range r) {
		int upperboundIndex = searchIndexWith(r.getUpperbound());

		int lowerboundIndex = searchIndexWith(r.getLowerbound());

		if ((((BidDetails) this.allBids.get(upperboundIndex))
				.getMyUndiscountedUtil() <= r.getUpperbound())
				&& (upperboundIndex > 0)) {
			upperboundIndex--;
		}

		if ((((BidDetails) this.allBids.get(lowerboundIndex))
				.getMyUndiscountedUtil() >= r.getLowerbound())
				&& (lowerboundIndex < this.allBids.size())) {
			lowerboundIndex++;
		}

		if (lowerboundIndex == upperboundIndex) {
			ArrayList list = new ArrayList();
			list.add((BidDetails) this.allBids.get(lowerboundIndex));
			return list;
		}

		ArrayList subList = new ArrayList(this.allBids.subList(upperboundIndex,
				lowerboundIndex));
		return subList;
	}

	public BidDetails getBidNearUtility(double utility) {
		return (BidDetails) this.allBids.get(getIndexOfBidNearUtility(utility));
	}

	public int getIndexOfBidNearUtility(double utility) {
		int index = searchIndexWith(utility);
		int newIndex = -1;
		double closestDistance = Math
				.abs(((BidDetails) this.allBids.get(index))
						.getMyUndiscountedUtil() - utility);

		if ((index > 0)
				&& (Math.abs(((BidDetails) this.allBids.get(index - 1))
						.getMyUndiscountedUtil() - utility) < closestDistance)) {
			newIndex = index - 1;
			closestDistance = Math.abs(((BidDetails) this.allBids
					.get(index - 1)).getMyUndiscountedUtil() - utility);
		}

		if ((index + 1 < this.allBids.size())
				&& (Math.abs(((BidDetails) this.allBids.get(index + 1))
						.getMyUndiscountedUtil() - utility) < closestDistance)) {
			newIndex = index + 1;
			closestDistance = Math.abs(((BidDetails) this.allBids
					.get(index + 1)).getMyUndiscountedUtil() - utility);
		} else {
			newIndex = index;
		}
		return newIndex;
	}

	public int searchIndexWith(double value) {
		int middle = -1;
		int low = 0;
		int high = this.allBids.size() - 1;
		int lastMiddle = 0;
		while (lastMiddle != middle) {
			lastMiddle = middle;
			middle = (low + high) / 2;
			if (((BidDetails) this.allBids.get(middle)).getMyUndiscountedUtil() == value) {
				return middle;
			}
			if (((BidDetails) this.allBids.get(middle)).getMyUndiscountedUtil() < value) {
				high = middle;
			}
			if (((BidDetails) this.allBids.get(middle)).getMyUndiscountedUtil() > value) {
				low = middle;
			}
		}
		return middle;
	}

	public BidDetails getMaxBidPossible() {
		return (BidDetails) this.allBids.get(0);
	}

	public BidDetails getMinBidPossible() {
		return (BidDetails) this.allBids.get(this.allBids.size() - 1);
	}
}