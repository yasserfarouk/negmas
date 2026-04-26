package agents.anac.y2010.Southampton.similarity;

import java.util.ArrayList;

import agents.anac.y2010.Southampton.SouthamptonAgent;
import agents.anac.y2010.Southampton.utils.OpponentModel;
import agents.anac.y2010.Southampton.utils.Pair;
import genius.core.Bid;
import genius.core.utility.AdditiveUtilitySpace;

public abstract class SimilarityAgent extends SouthamptonAgent {

	/**
	 * The best bids (in terms of our utility) that we have seen from the
	 * opponent.
	 */
	protected ArrayList<Pair<Double, Double>> bestOpponentBidUtilityHistory;
	/**
	 * The best bid (in terms of our utility) that we have seen from the
	 * opponent.
	 */
	private Bid bestOpponentBid;

	/**
	 * The utility (to us) of the best bid (in terms of our utility) that we
	 * have seen from the opponent.
	 */
	private double bestOpponentUtility;

	/**
	 * The utility (to us) of the first bid made by the opponent.
	 */
	protected double utility0 = 0;

	/**
	 * The expected utility (to us) of the final bid made by the opponent.
	 */
	protected final double utility1 = 0.95;

	public SimilarityAgent() {
		super();
		bestOpponentBidUtilityHistory = new ArrayList<Pair<Double, Double>>();
		// CONCESSIONFACTOR = 0.01;
	}

	public void init() {
		super.init();
		prepareOpponentModel();
	}

	@Override
	public String getVersion() {
		return "1.0";
	}

	protected Bid proposeInitialBid() {
		Bid bid = null;

		try {
			bid = bidSpace.getMaxUtilityBid();
		} catch (Exception e) {
			e.printStackTrace();
		}
		return bid;
	}

	protected Bid proposeNextBid(Bid opponentBid) {
		try {
			performUpdating(opponentBid);
		} catch (Exception e) {
			e.printStackTrace();
		}

		double myUtility = 0, opponentUtility = 0, targetUtility;
		// Both parties have made an initial bid. Compute associated utilities
		// from my point of view.
		try {
			myUtility = utilitySpace.getUtility(myLastBid);
			opponentUtility = utilitySpace.getUtility(opponentBid);
			if (opponentPreviousBid == null)
				utility0 = opponentUtility;
		} catch (Exception e) {
			e.printStackTrace();
		}
		targetUtility = getTargetUtility(myUtility, opponentUtility);
		Bid nextBid = getTradeOffExhaustive(targetUtility, opponentBid, 1000);
		return nextBid;
	}

	protected abstract double getTargetUtility(double myUtility,
			double opponentUtility);

	/*
	 * (non-Javadoc)
	 * 
	 * @see agents.southampton.SouthamptonAgent#getRandomBidInRange(double,
	 * double)
	 */
	@Override
	protected Bid getRandomBidInRange(double lowerBound, double upperBound)
			throws Exception {
		throw new Exception(
				"Method 'getRandomBidInRange' is not implemented in this agent.");
	}

	private Bid getTradeOffExhaustive(double ourUtility, Bid opponentBid,
			int count) {
		// Project a point onto the bidspace...
		bestOpponentBid = getBestBid(opponentBid);

		if (bestOpponentUtility * acceptMultiplier >= ourUtility) {
			return bestOpponentBid;
		}

		ArrayList<Bid> bids = bidSpace.Project(
				bidSpace.getPoint(bestOpponentBid), ourUtility, count,
				(AdditiveUtilitySpace) this.utilitySpace, this.opponentModel);
		if (bids.size() == 0) {
			return getTradeOffExhaustive(ourUtility, opponentBid, count + 10000);
		}
		double maxOpponentUtility = 0;
		Bid bestBid = null;

		for (Bid bid : bids) {
			try {
				double opponentUtility = opponentModel
						.getNormalizedUtility(bid);
				if (opponentUtility > maxOpponentUtility) {
					maxOpponentUtility = opponentUtility;
					bestBid = bid;
				}
			} catch (Exception e) {
				e.printStackTrace();
			}
		}

		return bestBid;
	}

	private Bid getBestBid(Bid opponentBid) {
		double utility;
		try {
			utility = utilitySpace.getUtility(opponentBid);
			if (utility >= bestOpponentUtility) {
				bestOpponentUtility = utility;
				bestOpponentBid = opponentBid;
			}
			storeDataPoint(bestOpponentUtility);
		} catch (Exception e) {
			e.printStackTrace();
		}
		return bestOpponentBid;
	}

	private void storeDataPoint(double utility) {
		double time = timeline.getTime();
		// bestOpponentBidUtilityHistory.add(new Pair<Double,
		// Double>(-Math.log(1 - ((utility - utility0) / (utility1 -
		// utility0))), time));
		bestOpponentBidUtilityHistory.add(new Pair<Double, Double>(utility,
				time));
	}

	private void performUpdating(Bid opponentBid) throws Exception {
		double currentTime = timeline.getTime() * timeline.getTotalTime()
				* 1000;
		double totalTime = timeline.getTotalTime() * 1000;
		opponentModel.updateBeliefs(opponentBid, Math.round(currentTime),
				totalTime);
	}

	private void prepareOpponentModel() {
		opponentModel = new OpponentModel((AdditiveUtilitySpace) utilitySpace);
	}
}
