package agents.anac.y2014.TUDelftGroup2.boaframework.offeringstrategy.other;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import agents.anac.y2014.TUDelftGroup2.boaframework.opponentmodel.Group2_OM;
import genius.core.Bid;
import genius.core.bidding.BidDetails;
import genius.core.boaframework.NegotiationSession;
import genius.core.boaframework.OMStrategy;
import genius.core.boaframework.OfferingStrategy;
import genius.core.boaframework.OpponentModel;
import genius.core.boaframework.OutcomeSpace;
import genius.core.boaframework.SortedOutcomeSpace;

/**
 * This is the actual {@link OfferingStrategy} used. It's called
 * {@code Group2_BS} because in turn it calls different strategies, namely:
 * {@link OpeningGameStrategy}, {@link MiddleGameStrategy},
 * {@link EndGameStrategy}. They are as there names suggest called in that
 * particular order. The first switch is made when the {@code foreplayCount} is
 * reached (editable at the top of the {@code Group2_BS.java} source file) and
 * the second switch is made when the Kalai-Somorodisnky point is reached, or
 * more accurately: When the Kalai-Smorodinsky point plus hysteresis and offset
 * are reached. The values for the {@code hysteresis} and {@code offset} can
 * also be found in this {@code Group2_BS.java} source file. Note that these
 * should have the same values as the ones in {@link EndGameStrategy}.
 */
public class Group2_BS extends OfferingStrategy {
	// NONLINEAR modification
	/** DISPLAY RAW BIDSPACE EXPLORATION OR PLAY AS CONCEDER GENT */
	public static boolean disp_exploration = false;

	/** {@link OutcomeSpace} */
	SortedOutcomeSpace outcomeSpace;
	/**
	 * Either {@link OpeningGameStrategy}, {@link MiddleGameStrategy} or
	 * {@link EndGameStrategy}
	 */
	AbstractBiddingStrategy biddingStrategyState;
	/** When to start the endgame */
	double normalizedEndgameStartingTime = 0.90; // = 18.0 seconds @ 180 seconds
	/** How long to keep the openinggame */
	public static int foreplayCount = 1200; // 25 250
	/** Offset (should equal {@code EndGameStrategy.offset} (private var) */
	double offset = 0.05;
	/**
	 * Hysteresis (should equal {@code EndGameStrategy.hystersis} (private var)
	 */
	double hysteresis = 0.05;

	// Used for MatLab link
	private int switchover_toMiddle;
	private int switchover_toEnd;

	// all used to get the timing of the failsafe right (used in the AC)
	int counter = 1;
	List<Double> timer = new ArrayList<Double>(10000);
	int thisNumBids;
	int thatNumBids;
	public int timing_numbids;
	public double timing_currentTime;
	public double timing_diffTime;
	public double timing_mean;
	public double timing_sd;
	public double timing_10avg;

	private boolean switchedToMid;

	/**
	 * Empty constructor, guess BOA framework needs this.
	 */
	public Group2_BS() {
	}

	/**
	 * Default contructor, gets some imputs from enviroment
	 * 
	 * @param negotiationSession
	 *            This is the Domain
	 * @param opponentModel
	 *            This is the model that we've implemented (in case of prototype
	 *            it is the {code {@link Group2_OM}).
	 * @param opponentModelStrategy
	 *            We won't be using this for the prototype.
	 * @param concessionFactor
	 *            This determines how easily the agent will make a lower bid.
	 * @param k
	 * @param maxTargetUtil
	 *            Maximum utility of the bid that will be made.
	 * @param minTargetUtil
	 *            Minimum utility of the bid that will be made.
	 */
	public Group2_BS(NegotiationSession negotiationSession, OpponentModel opponentModel,
			OMStrategy opponentModelStrategy, double concessionFactor, double k, double maxTargetUtil,
			double minTargetUtil) {
		this.negotiationSession = negotiationSession;
		this.outcomeSpace = new SortedOutcomeSpace(negotiationSession.getUtilitySpace());
		this.negotiationSession.setOutcomeSpace(outcomeSpace);
		this.opponentModel = opponentModel;
		this.omStrategy = opponentModelStrategy;
		this.biddingStrategyState = new OpeningGameStrategy(negotiationSession, opponentModel);
	}

	/**
	 * Method which initializes the agent by setting all parameters.
	 */
	@Override
	public void init(NegotiationSession negotiationSession, OpponentModel opponentModel,
			OMStrategy opponentModelStrategy, Map<String, Double> parameters) throws Exception {
		// setup
		this.negotiationSession = negotiationSession;

		// DONT DO THIS IS NONLINEAR
		// this.outcomeSpace = new
		// SortedOutcomeSpace(negotiationSession.getUtilitySpace());
		// this.negotiationSession.setOutcomeSpace(outcomeSpace);
		this.opponentModel = opponentModel;
		this.omStrategy = opponentModelStrategy;
		this.biddingStrategyState = new OpeningGameStrategy(negotiationSession, opponentModel);

		this.switchedToMid = false;

	}

	/**
	 * I think this gets called on the first run. Just use the next bid as
	 * opening bid.
	 */
	@Override
	public BidDetails determineOpeningBid() {
		return determineNextBid();
	}

	/**
	 * Instead of directly determining next bid, this method only checks with
	 * it's {@code biddingStrategyState} what bid it should do. This function
	 * does determine the actual strategy used (hence the "Meta" in it's
	 * classname).
	 */
	@Override
	public BidDetails determineNextBid() {
		try {
			// Timing stuff used for determening when to kick in the failsafe
			int thisNumBids_now = negotiationSession.getOwnBidHistory().size();
			int thatNumBids_now = negotiationSession.getOpponentBidHistory().size();
			if (thisNumBids != thisNumBids_now || thatNumBids != thatNumBids_now) {
				timing_currentTime = negotiationSession.getTimeline().getCurrentTime();
				timer.add(timing_currentTime);
				timing_numbids = thisNumBids_now;

				// timing_diffTime =
				// (timer.get(thisNumBids_now)-timer.get(thisNumBids));
				timing_mean = SetMath.mean(timer);
				timing_sd = Math.sqrt(SetMath.variance(timer));
				timing_10avg = SetMath.avg(timer, 10);
				// System.out.println(String.format("bd(%4d) t=%.4f td=%.1fms
				// (mean=%.1fms, sd=%.1fms, 10avg=%.1fms)",
				// timing_numbids,timing_currentTime,timing_diffTime,timing_mean,timing_sd,timing_10avg));
				thisNumBids = thisNumBids_now;
				thatNumBids = thatNumBids_now;
			}

			/* ~~~ meta strategy (Start of actual metastrategy)~~~ */
			// NONLINEAR modification
			// only do this if not in Displaying raw Exploration

			if ((!switchedToMid)
					&& (negotiationSession.getTime() > (Math.pow(negotiationSession.getDiscountFactor(), 2) * 0.25))) {
				biddingStrategyState = new MiddleGameStrategy(negotiationSession, opponentModel);
				switchedToMid = true;

			}

			// Get bid from the strategy currently in use

			Bid bid = biddingStrategyState.getBid();
			double actualUtil = this.negotiationSession.getUtilitySpace().getUtility(bid);
			BidDetails bidDetails = new BidDetails(bid, actualUtil);

			/* ~~~ /meta strategy (End of actual metastrategy)~~~ */

			// // FIlter by Opponent model
			// double om_score = opponentModel.getBidEvaluation(bid);
			// //
			// actualUtil<om_score
			//

			//

			// return the bid found by the current strategy
			return bidDetails;
		} catch (Exception e) {
			e.printStackTrace();
			return null;
		}
	}

	/**
	 * Helper class with functions needed for timing. These are straightforward
	 * statistics calculations.
	 */
	static class SetMath {

		public static double sum(List<Double> input) {
			double sum = 0;
			for (int i = 1; i < input.size(); i++)
				sum += input.get(i) - input.get(i - 1);
			return sum;
		}

		public static double mean(List<Double> input) {
			return sum(input) / input.size();
		}

		public static double variance(List<Double> input) {
			double mean = mean(input);
			// [1/(n-1)] * sum(yi -mean)^2
			double sum = 0;
			double prefix = 1D / (input.size() - 1D);
			for (int i = 1; i < input.size(); i++)
				sum += Math.pow((input.get(i) - input.get(i - 1) - mean), 2);
			return prefix * sum;
		}

		public static double avg(List<Double> input, int n) {
			if (input.size() <= n)
				return 0;
			double sum = 0;
			for (int i = input.size() - n; i < input.size(); i++) {
				sum += input.get(i) - input.get(i - 1);
			}
			return sum / ((double) n);
		}
	}

	@Override
	public String getName() {
		return "2014 - Group2";
	}
}
