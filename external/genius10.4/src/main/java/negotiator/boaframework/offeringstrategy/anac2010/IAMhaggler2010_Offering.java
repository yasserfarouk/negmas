package negotiator.boaframework.offeringstrategy.anac2010;

import java.util.ArrayList;
import java.util.Map;
import java.util.Random;

import genius.core.Bid;
import genius.core.bidding.BidDetails;
import genius.core.boaframework.NegotiationSession;
import genius.core.boaframework.NoModel;
import genius.core.boaframework.OMStrategy;
import genius.core.boaframework.OfferingStrategy;
import genius.core.boaframework.OpponentModel;
import genius.core.utility.AdditiveUtilitySpace;
import negotiator.boaframework.offeringstrategy.anac2010.IAMhaggler2010.BidSpace;
import negotiator.boaframework.offeringstrategy.anac2010.IAMhaggler2010.ConcessionFunction;
import negotiator.boaframework.offeringstrategy.anac2010.IAMhaggler2010.Pair;
import negotiator.boaframework.offeringstrategy.anac2010.IAMhaggler2010.SpecialTimeConcessionFunction;
import negotiator.boaframework.offeringstrategy.anac2010.IAMhaggler2010.TimeConcessionFunction;
import negotiator.boaframework.opponentmodel.DefaultModel;
import negotiator.boaframework.opponentmodel.IAMhagglerBayesianModel;

/**
 * This is the decoupled Offering Strategy for IAMhaggler2010 (ANAC2010). The
 * code was taken from the ANAC2010 IAMhaggler2010 and adapted to work within
 * the BOA framework.
 * 
 * The default opponent model implementation selects the best bid for the
 * opponent.
 * 
 * Decoupling Negotiating Agents to Explore the Space of Negotiation Strategies
 * T. Baarslag, K. Hindriks, M. Hendrikx, A. Dirkzwager, C.M. Jonker
 */
public class IAMhaggler2010_Offering extends OfferingStrategy {

	/**
	 * The lowest target we have tried to use (initialised to 1).
	 */
	private double lowestTarget = 1.0;

	/**
	 * The minimum discounting factor.
	 */
	private final double minDiscounting = 0.1;

	/**
	 * The minimum beta value, which stops the agent from becoming too tough.
	 */
	private final double minBeta = 0.01;

	/**
	 * The maximum beta value, which stops the agent from becoming too weak.
	 */
	private double maxBeta = 2.0;

	/**
	 * The default value for beta, used when there is not enough information to
	 * perform the linear regression.
	 */
	private double defaultBeta = 1;

	/**
	 * The minimum target value when the opponent is considered to be a
	 * hardhaed.
	 */
	private final double hardHeadTarget = 0.8;

	protected ConcessionFunction cf;
	protected ArrayList<Pair<Double, Double>> bestOpponentBidUtilityHistory;
	private Bid bestOpponentBid;
	private double bestOpponentUtility;
	protected double utility0 = 0;
	protected final double utility1 = 0.95;
	protected static double MAXIMUM_ASPIRATION = 0.9;
	protected BidSpace bidSpace;
	protected BidDetails myLastBid = null;
	protected Bid opponentPreviousBid = null;
	protected final double acceptMultiplier = 1.02;
	protected boolean opponentIsHardHead;

	@Override
	public void init(NegotiationSession negoSession, OpponentModel model, OMStrategy oms,
			Map<String, Double> parameters) throws Exception {
		if (model instanceof DefaultModel) {
			model = new IAMhagglerBayesianModel();
			model.init(negoSession, null);
			oms.setOpponentModel(model);
		}

		this.opponentModel = model;

		this.omStrategy = oms;
		this.negotiationSession = negoSession;
		bestOpponentBidUtilityHistory = new ArrayList<Pair<Double, Double>>();
		cf = new TimeConcessionFunction(TimeConcessionFunction.Beta.LINEAR, TimeConcessionFunction.BREAKOFF);
		myLastBid = null;

		try {
			bidSpace = new BidSpace((AdditiveUtilitySpace) this.negotiationSession.getUtilitySpace());
		} catch (Exception e) {
			e.printStackTrace();
		}

		opponentIsHardHead = true;
		int discreteCombinationCount = this.bidSpace.getDiscreteCombinationsCount();
		if (this.bidSpace.isContinuousWeightsZero()) {
			defaultBeta = Math.min(0.5, Math.max(0.0625, discreteCombinationCount * 0.001));
			if (negotiationSession.getUtilitySpace().isDiscounted()) {
				maxBeta = (2.0 + (4.0 * Math.min(0.5, negotiationSession.getUtilitySpace().getDiscountFactor())))
						* defaultBeta;
			} else {
				maxBeta = 2.0 + 4.0 * defaultBeta;
			}
		}
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see agents.southampton.similarity.VariableConcessionSimilarityAgent#
	 * getTargetUtility(double, double)
	 */
	protected double getTargetUtility(double myUtility, double oppntUtility) {
		double currentTime = negotiationSession.getTime() * negotiationSession.getTimeline().getTotalTime() * 1000;
		double beta = bidSpace.getBeta(bestOpponentBidUtilityHistory, negotiationSession.getTime(), utility0, utility1,
				minDiscounting, minBeta, maxBeta, defaultBeta, currentTime, currentTime);

		cf = new SpecialTimeConcessionFunction(beta, defaultBeta, TimeConcessionFunction.DEFAULT_BREAKOFF);

		double target = 0;
		try {
			target = getConcession(negotiationSession.getUtilitySpace()
					.getUtility(negotiationSession.getOwnBidHistory().getFirstBidDetails().getBid()));
		} catch (Exception e) {
			e.printStackTrace();
		}

		lowestTarget = Math.min(target, lowestTarget);
		if (opponentIsHardHead) {
			return Math.max(hardHeadTarget, lowestTarget);
		}
		return lowestTarget;
	}

	private void storeDataPoint(double utility) {
		double time = negotiationSession.getTime();
		// bestOpponentBidUtilityHistory.add(new Pair<Double,
		// Double>(-Math.log(1 - ((utility - utility0) / (utility1 -
		// utility0))), time));
		bestOpponentBidUtilityHistory.add(new Pair<Double, Double>(utility, time));
	}

	/**
	 * Handle an opponent's offer.
	 */
	private BidDetails handleOffer(Bid opponentBid) throws Exception {
		Bid chosenAction = null;

		if (myLastBid == null) {
			// Special case to handle first action
			Bid b = proposeInitialBid();
			myLastBid = new BidDetails(b, negotiationSession.getUtilitySpace().getUtility(b),
					negotiationSession.getTime());
			chosenAction = b;
		} else {
			Bid plannedBid = proposeNextBid(opponentBid);
			chosenAction = plannedBid;

			opponentPreviousBid = opponentBid;
		}

		return new BidDetails(chosenAction, negotiationSession.getUtilitySpace().getUtility(chosenAction),
				negotiationSession.getTime());
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

	@Override
	protected void finalize() throws Throwable {
		// displayFrame.dispose();
		super.finalize();
	}

	protected Bid proposeNextBid(Bid opponentBid) {
		BidDetails opponentHisBid = negotiationSession.getOpponentBidHistory().getLastBidDetails();
		if (opponentHisBid != null) {
			try {
				if (opponentIsHardHead && negotiationSession.getOpponentBidHistory().size() > 0
						&& Math.abs(negotiationSession.getUtilitySpace()
								.getUtility(negotiationSession.getOpponentBidHistory().getFirstBidDetails().getBid())
								- opponentHisBid.getMyUndiscountedUtil()) > 0.02) {
					opponentIsHardHead = false;
				}
			} catch (Exception e) {
				e.printStackTrace();
			}
		}

		double myUtility = 0, opponentUtility = 0, targetUtility;
		// Both parties have made an initial bid. Compute associated utilities
		// from my point of view.
		try {
			myUtility = negotiationSession.getUtilitySpace().getUtility(myLastBid.getBid());
			opponentUtility = negotiationSession.getUtilitySpace().getUtility(opponentBid);
			if (opponentPreviousBid == null)
				utility0 = opponentUtility;
		} catch (Exception e) {
			e.printStackTrace();
		}
		targetUtility = getTargetUtility(myUtility, opponentUtility);
		Bid nextBid = getTradeOffExhaustive(targetUtility, opponentBid, 1000);

		return nextBid;
	}

	private double getConcession(double startUtility) {
		double currentTime = negotiationSession.getTimeline().getCurrentTime() * 1000;
		double totalTime = negotiationSession.getTimeline().getTotalTime() * 1000;
		return cf.getConcession(startUtility, Math.round(currentTime), totalTime);
	}

	private Bid getBestBid(Bid opponentBid) {
		double utility;
		try {
			utility = negotiationSession.getUtilitySpace().getUtility(opponentBid);
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

	private Bid getTradeOffExhaustive(double ourUtility, Bid opponentBid, int count) {
		// Project a point onto the bidspace...
		Bid bestBidOpp = getBestBid(opponentBid);

		if (bestOpponentUtility * acceptMultiplier >= ourUtility) {
			return bestBidOpp;
		}

		// SHOULD FIX OM USAGE HERE!
		ArrayList<BidDetails> bids = bidSpace.Project(bidSpace.getPoint(bestOpponentBid), ourUtility, count,
				(AdditiveUtilitySpace) negotiationSession.getUtilitySpace(), opponentModel);
		if (bids.size() == 0) {
			return getTradeOffExhaustive(ourUtility, opponentBid, count + 10000);
		}

		Bid bestBid = null;

		// double maxOpponentUtility = 0;
		// for (Bid bid : bids) {
		// try {
		// double opponentUtility = opponentModel.getBidEvaluation(bid);
		// if (opponentUtility > maxOpponentUtility) {
		// maxOpponentUtility = opponentUtility;
		// bestBid = bid;
		// }
		// } catch (Exception e) {
		// e.printStackTrace();
		// }
		// }
		if (opponentModel instanceof NoModel) {
			Random random = new Random();
			bestBid = bids.get(random.nextInt(bids.size())).getBid();
		} else {
			bestBid = omStrategy.getBid(bids).getBid();
		}

		return bestBid;
	}

	public BidDetails determineNextBid() {
		BidDetails chosenAction = null;

		if (this.myLastBid == null) {
			Bid bid = proposeInitialBid();
			try {
				chosenAction = new BidDetails(bid, negotiationSession.getUtilitySpace().getUtility(bid),
						negotiationSession.getTime());
			} catch (Exception e) {
				e.printStackTrace();
			}
		} else {
			try {
				chosenAction = handleOffer(negotiationSession.getOpponentBidHistory().getLastBidDetails().getBid());
			} catch (Exception e) {
				e.printStackTrace();
			}
		}

		myLastBid = chosenAction;

		return chosenAction;
	}

	@Override
	public BidDetails determineOpeningBid() {
		return determineNextBid();
	}

	@Override
	public String getName() {
		return "2010 - IAMhaggler";
	}
}
