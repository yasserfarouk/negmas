package agents.anac.y2011.ValueModelAgent;

import genius.core.timeline.TimeLineInfo;
import genius.core.utility.AdditiveUtilitySpace;

public class OpponentModeler {
	AdditiveUtilitySpace utilitySpace;
	double lastTime;
	public double delta;
	double discount;
	TimeLineInfo timeline;
	BidList ourPastBids;
	BidList theirPastBids;
	BidList allBids;
	ValueModeler vmodel;
	ValueModelAgent agent;
	// reasonable to assume that this is the ratio between how much
	// they gave us and how much they lost.
	// i.e. if they gave us 9% than its reasonable to assume they lost at least
	// 3%.
	// this is used for fail safe estimation
	double paretoRatioEstimation = 5;

	public OpponentModeler(int bidCount, AdditiveUtilitySpace space, TimeLineInfo timeline, BidList our, BidList their,
			ValueModeler vmodeler, BidList allBids, ValueModelAgent agent) {
		ourPastBids = our;
		theirPastBids = their;
		utilitySpace = space;
		lastTime = timeline.getTime() * timeline.getTotalTime() * 1000;
		discount = utilitySpace.getDiscountFactor();
		if (discount >= 1) {
			discount = 0; // compatiblity with old discount mode
		}
		this.timeline = timeline;
		delta = 1.0 / (timeline.getTotalTime() * 1000);
		vmodel = vmodeler;
		this.allBids = allBids;
		this.agent = agent;
	}

	public void tick() {
		double newTime = timeline.getTime() * timeline.getTotalTime() * 1000;
		delta = 0.8 * delta + (newTime - lastTime) / 5;
	}

	private int expectedBidsToConvergence() {
		return 10;
	}

	public int expectedBidsToTimeout() {
		if (delta > 0)
			return (int) ((1 - timeline.getTime()) / delta);
		else
			return (int) (1 - timeline.getTime()) * 1000;
	}

	public double expectedDiscountRatioToConvergence() {
		double expectedPart = (double) (expectedBidsToConvergence() * delta);
		if (timeline.getTime() + expectedPart > 1) {
			return 1.1;
		} else {
			double div = 1 - (discount * expectedPart);
			if (div > 0) {
				return 1 / div;
			} else
				return 1.1;
		}
	}

	public double guessCurrentBidUtil() {
		int s2 = theirPastBids.bids.size();
		if (s2 == 0) {
			return 1;
		}
		double sum = 0;
		double count = 0;
		double symetricLowerBound = allBids.bids.get(s2).ourUtility;
		// trying to learn the average of the current bids
		for (int i = s2 - 2; i >= 0 && i > s2 - 50; i--) {
			theirPastBids.bids.get(i).update(vmodel);
			if (theirPastBids.bids.get(i).theirUtilityReliability > 0.7) {
				sum += theirPastBids.bids.get(i).theirUtility;
				count++;
			}
		}
		double shield = timeline.getTime() * 0.6;
		if (shield < 0.03)
			shield = 0.03;
		double minBound = symetricLowerBound;
		if (count >= 5 && sum / count < minBound) {
			minBound = sum / count;
		}
		if (minBound > (1 - shield))
			return minBound;
		// it is very unsafe to assume our opponent conceded more than 15...
		else
			return (1 - shield);
	}
}
