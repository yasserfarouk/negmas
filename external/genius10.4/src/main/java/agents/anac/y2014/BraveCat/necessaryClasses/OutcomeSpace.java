package agents.anac.y2014.BraveCat.necessaryClasses;

import java.util.ArrayList;
import java.util.List;

import genius.core.Bid;
import genius.core.BidIterator;
import genius.core.bidding.BidDetails;
import genius.core.misc.Range;
import genius.core.utility.AbstractUtilitySpace;

public class OutcomeSpace {
	protected AbstractUtilitySpace utilitySpace;
	protected List<BidDetails> allBids = new ArrayList();

	public OutcomeSpace(AbstractUtilitySpace utilSpace) {
		this.utilitySpace = utilSpace;
		System.out.println("Generating All Bids...");
		generateAllBids(utilSpace);
		System.out.println("All Bids Generated!");
	}

	public void generateAllBids(AbstractUtilitySpace utilSpace) {
		BidIterator iter = new BidIterator(utilSpace.getDomain());
		while (iter.hasNext()) {
			Bid bid = iter.next();
			try {
				BidDetails BidDetails = new BidDetails(bid,
						utilSpace.getUtility(bid), -1.0D);
				this.allBids.add(BidDetails);
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
	}

	public List<BidDetails> getAllOutcomes() {
		return this.allBids;
	}

	public List<BidDetails> getBidsinRange(Range range) {
		ArrayList result = new ArrayList();
		double upperbound = range.getUpperbound();
		double lowerbound = range.getLowerbound();

		for (BidDetails bid : this.allBids) {
			if ((bid.getMyUndiscountedUtil() > lowerbound)
					&& (bid.getMyUndiscountedUtil() < upperbound)) {
				result.add(bid);
			}
		}
		return result;
	}

	public BidDetails getBidNearUtility(double utility) {
		return (BidDetails) this.allBids.get(getIndexOfBidNearUtility(utility));
	}

	public BidDetails getMaxBidPossible() {
		BidDetails maxBid = (BidDetails) this.allBids.get(0);
		for (BidDetails bid : this.allBids) {
			if (bid.getMyUndiscountedUtil() > maxBid.getMyUndiscountedUtil()) {
				maxBid = bid;
			}
		}
		return maxBid;
	}

	public BidDetails getMinBidPossible() {
		BidDetails minBid = (BidDetails) this.allBids.get(0);
		for (BidDetails bid : this.allBids) {
			if (bid.getMyUndiscountedUtil() < minBid.getMyUndiscountedUtil()) {
				minBid = bid;
			}
		}
		return minBid;
	}

	public int getIndexOfBidNearUtility(double utility) {
		double closesDistance = 1.0D;
		int best = 0;
		for (int i = 0; i < this.allBids.size(); i++) {
			if (Math.abs(((BidDetails) this.allBids.get(i))
					.getMyUndiscountedUtil() - utility) < closesDistance) {
				closesDistance = Math.abs(((BidDetails) this.allBids.get(i))
						.getMyUndiscountedUtil() - utility);
				best = i;
			}
		}
		return best;
	}
}