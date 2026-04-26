package agents.anac.y2011.ValueModelAgent;

import java.util.Comparator;

import genius.core.Bid;
import genius.core.utility.AbstractUtilitySpace;

public class BidWrapper {
	public boolean sentByUs;
	public boolean sentByThem;
	public int lastSentBid;
	public double ourUtility;
	public double theirUtility;
	public double theirUtilityReliability;
	public Bid bid;

	public BidWrapper(Bid bid, AbstractUtilitySpace space, double maxUtil) {
		this.bid = bid;
		sentByUs = false;
		sentByThem = false;
		try {
			ourUtility = space.getUtility(bid) / maxUtil;
		} catch (Exception e) {
			ourUtility = 0;
		}
		theirUtility = 0;
	}

	public void update(ValueModeler model) {
		try {
			ValueDecrease val = model.utilityLoss(bid);
			theirUtility = 1 - val.getDecrease();
			theirUtilityReliability = val.getReliabilty();
		} catch (Exception ex) {

		}
	}

	public class OpponentUtilityComperator implements Comparator<BidWrapper> {

		public int compare(BidWrapper o1, BidWrapper o2) {
			if (o1.theirUtility < o2.theirUtility) {
				return 1;
			}
			if (o1.theirUtility > o2.theirUtility) {
				return -1;
			}
			return 0;
		}

	}

	public class OurUtilityComperator implements Comparator<BidWrapper> {

		public int compare(BidWrapper o1, BidWrapper o2) {
			if (o1.ourUtility < o2.ourUtility) {
				return 1;
			}
			if (o1.ourUtility > o2.ourUtility) {
				return -1;
			}
			return 0;
		}

	}

	public boolean equals(BidWrapper o) {
		return bid.equals(o.bid);
	}

	@Override
	public boolean equals(Object obj) {
		if (obj instanceof BidWrapper)
			return equals((BidWrapper) obj);
		return false;
	}

}
