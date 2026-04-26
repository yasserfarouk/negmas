package negotiator.boaframework.offeringstrategy.anac2011;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;

import genius.core.Bid;
import genius.core.BidHistory;
import genius.core.BidIterator;
import genius.core.analysis.BidPoint;
import genius.core.analysis.BidSpace;
import genius.core.bidding.BidDetails;
import genius.core.boaframework.NegotiationSession;
import genius.core.boaframework.NoModel;
import genius.core.boaframework.OMStrategy;
import genius.core.boaframework.OfferingStrategy;
import genius.core.boaframework.OpponentModel;
import negotiator.boaframework.opponentmodel.DefaultModel;
import negotiator.boaframework.opponentmodel.ScalableBayesianModel;
import negotiator.boaframework.sharedagentstate.anac2011.NiceTitForTatSAS;

/**
 * This is the decoupled Offering Strategy for NiceTitForTat (ANAC2011). The
 * code was taken from the ANAC2011 NiceTitForTat and adapted to work within the
 * BOA framework.
 * 
 * >>Extension of opponent model>> As default the agent implements a behavior
 * when an opponent model is used and when not. The only extension we made is to
 * allow to switch the opponent model.
 * 
 * DEFAULT OM: ScalableBayesianModel
 * 
 * @author Alex Dirkzwager, Mark Hendrikx
 */
public class NiceTitForTat_Offering extends OfferingStrategy {

	private static final double TIME_USED_TO_DETERMINE_OPPONENT_STARTING_POINT = 0.01;
	private double myNashUtility;
	private double initialGap;
	private long DOMAINSIZE;
	private final boolean TEST_EQUIVALENCE = false;
	private Random random100;

	/**
	 * Empty constructor for the BOA framework.
	 */
	public NiceTitForTat_Offering() {
	}

	@Override
	public void init(NegotiationSession negotiationSession, OpponentModel om, OMStrategy oms,
			Map<String, Double> parameters) throws Exception {
		if (om instanceof DefaultModel) {
			om = new ScalableBayesianModel();
			om.init(negotiationSession, null);
			oms.setOpponentModel(om);
		}
		initializeAgent(negotiationSession, om, oms);
	}

	public void initializeAgent(NegotiationSession negoSession, OpponentModel om, OMStrategy oms) {
		this.negotiationSession = negoSession;
		opponentModel = om;

		omStrategy = oms;
		DOMAINSIZE = negoSession.getUtilitySpace().getDomain().getNumberOfPossibleBids();
		helper = new NiceTitForTatSAS(negoSession);
		if (TEST_EQUIVALENCE) {
			random100 = new Random(100);
		} else {
			random100 = new Random();
		}
	}

	@Override
	public BidDetails determineOpeningBid() {
		try {
			BidDetails maxBid = negotiationSession.getMaxBidinDomain();
			return maxBid;
		} catch (Exception e) {
			e.printStackTrace();
		}
		return null;
	}

	@Override
	public BidDetails determineNextBid() {
		Bid opponentLastBid = negotiationSession.getOpponentBidHistory().getLastBidDetails().getBid();

		if (!(opponentModel instanceof NoModel)) {
			// If we have time, we receiveMessage the opponent model
			if (omStrategy.canUpdateOM()) {
				// opponentModel.updateModel(opponentLastBid.getBid());
				updateMyNashUtility();
			}
		}

		double myUtilityOfOpponentLastBid = getUtility(opponentLastBid);
		double maximumOfferedUtilityByOpponent = negotiationSession.getOpponentBidHistory().getBestBidDetails()
				.getMyUndiscountedUtil();
		double minUtilityOfOpponentFirstBids = getMinimumUtilityOfOpponentFirstBids(myUtilityOfOpponentLastBid);

		double opponentConcession = maximumOfferedUtilityByOpponent - minUtilityOfOpponentFirstBids;
		/**
		 * Measures how far away the opponent is from myNashUtility. 0 =
		 * farthest, 1 = closest
		 */
		double opponentConcedeFactor = Math.min(1,
				opponentConcession / (myNashUtility - minUtilityOfOpponentFirstBids));
		double myConcession = opponentConcedeFactor * (1 - myNashUtility);

		double myCurrentTargetUtility = 1 - myConcession;

		initialGap = 1 - minUtilityOfOpponentFirstBids;

		double gapToNash = Math.max(0, myCurrentTargetUtility - myNashUtility);

		double bonus = getBonus();
		double tit = bonus * gapToNash;

		myCurrentTargetUtility -= tit;

		ArrayList<BidDetails> myBids = new ArrayList<BidDetails>(getBidsOfUtility(myCurrentTargetUtility));

		Bid myBid;
		Bid bestBid = null;
		if (!(opponentModel instanceof NoModel)) {
			bestBid = getBestBidForOpponent(myBids);
			myBid = makeAppropriate(bestBid);
		} else {
			myBid = getBestBidForSelf(myBids);
		}

		try {
			nextBid = new BidDetails(myBid, negotiationSession.getUtilitySpace().getUtility(myBid),
					negotiationSession.getTime());
			return nextBid;
		} catch (Exception e) {
			e.printStackTrace();
		}

		return null;
	}

	/**
	 * Gives us our utility of the starting point of our opponent. This is used
	 * as a reference whether the opponent concedes or not.
	 */
	private double getMinimumUtilityOfOpponentFirstBids(double myUtilityOfOpponentLastBid) {
		ArrayList<BidDetails> firstBidList = ((NiceTitForTatSAS) helper).filterBetweenTime(0,
				TIME_USED_TO_DETERMINE_OPPONENT_STARTING_POINT);
		BidHistory firstBids = new BidHistory(firstBidList);
		double firstBidsMinUtility;
		if (firstBids.size() == 0)
			firstBidsMinUtility = negotiationSession.getOpponentBidHistory().getFirstBidDetails()
					.getMyUndiscountedUtil();
		else
			firstBidsMinUtility = firstBids.getWorstBidDetails().getMyUndiscountedUtil();
		return firstBidsMinUtility;
	}

	/**
	 * Returns a small bonus multiplier to our target utility if we have to be
	 * fast.
	 * 
	 * discount = 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 bonus = 0.5 0.3
	 * 0.1
	 * 
	 * Normal domains: time = 0.91 to 0.96 bonus = 0.0 to 1.0
	 * 
	 * Big domains: time = 0.85 to 0.9 bonus = 0.0 to 1.0
	 */
	private double getBonus() {
		double discountFactor = negotiationSession.getDiscountFactor();
		if (discountFactor <= 0 || discountFactor >= 1)
			discountFactor = 1;

		double discountBonus = 0.5 - 0.4 * discountFactor;
		boolean isBigDomain = DOMAINSIZE > 3000;

		double timeBonus = 0;
		double time = negotiationSession.getTime();
		double minTime = 0.91;
		if (isBigDomain)
			minTime = 0.85;
		if (time > minTime) {
			timeBonus = Math.min(1, 20 * (time - minTime));
		}

		double bonus = Math.max(discountBonus, timeBonus);
		if (bonus < 0)
			bonus = 0;
		if (bonus > 1)
			bonus = 1;
		return bonus;
	}

	private void updateMyNashUtility() {
		BidSpace bs = null;
		myNashUtility = 0.7;
		try {
			double nashMultiplier = getNashMultiplier(initialGap);
			if (DOMAINSIZE < 200000) {
				bs = new BidSpace(negotiationSession.getUtilitySpace(), opponentModel.getOpponentUtilitySpace(), true,
						true);

				BidPoint nash = bs.getNash();
				if (nash != null && nash.getUtilityA() != null)
					myNashUtility = nash.getUtilityA();
			}
			myNashUtility *= nashMultiplier;

			// don't ask for too much or too little
			if (myNashUtility > 1)
				myNashUtility = 1;
			if (myNashUtility < 0.5)
				myNashUtility = 0.5;

		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	/**
	 * If the gap is big (e.g. 0.8) the multiplier is smaller than 1 (e.g. 0.92)
	 * If the gap is small (e.g. 0.2) the multiplier is bigger than 1 (e.g.
	 * 1.28)
	 */
	private double getNashMultiplier(double gap) {
		double mult = 1.4 - 0.6 * gap;
		if (mult < 0)
			mult = 0;
		return mult;
	}

	/**
	 * Get all bids in a utility range.
	 */
	private List<BidDetails> getBidsOfUtility(double target) {
		if (target > 1)
			target = 1;

		double min = target * 0.98;
		double max = target + 0.04;
		do {
			max += 0.01;
			List<BidDetails> bids = getBidsOfUtility(min, max);
			int size = bids.size();
			// We need at least 2 bids. Or if max = 1, then we settle for 1 bid.
			if (size > 1 || (max >= 1 && size > 0)) {
				return bids;
			}
		} while (max <= 1);

		// Weird if this happens
		ArrayList<BidDetails> best = new ArrayList<BidDetails>();
		Bid maxBid = negotiationSession.getMaxBidinDomain().getBid();
		try {
			best.add(new BidDetails(maxBid, negotiationSession.getUtilitySpace().getUtility(maxBid)));
		} catch (Exception e) {
			e.printStackTrace();
		}
		return best;
	}

	private List<BidDetails> getBidsOfUtility(double lowerBound, double upperBound) {
		// In big domains, we need to find only this amount of bids around the
		// target. Should be at least 2.
		final int limit = 2;

		List<BidDetails> bidsInRange = new ArrayList<BidDetails>();
		BidIterator myBidIterator = new BidIterator(negotiationSession.getUtilitySpace().getDomain());
		while (myBidIterator.hasNext()) {
			Bid b = myBidIterator.next();
			try {
				double util = negotiationSession.getUtilitySpace().getUtility(b);
				if (util >= lowerBound && util <= upperBound)
					bidsInRange.add(new BidDetails(b, util));

				// In big domains, break early
				if (((NiceTitForTatSAS) helper).isDomainBig() && bidsInRange.size() >= limit)
					return bidsInRange;

			} catch (Exception e) {
				e.printStackTrace();
			}
		}

		return bidsInRange;
	}

	private Bid getBestBidForOpponent(ArrayList<BidDetails> bids) {
		if (TEST_EQUIVALENCE) {
			// We first make a bid history for the opponent, then pick the best
			// one.
			BidHistory possibleBidHistory = new BidHistory();
			for (BidDetails bid : bids) {
				Bid b = bid.getBid();
				double utility;
				try {
					utility = opponentModel.getBidEvaluation(b);
					BidDetails bidDetails = new BidDetails(b, utility, 0.0);
					possibleBidHistory.add(bidDetails);
				} catch (Exception e) {
					e.printStackTrace();
				}
			}

			// Pick the top 3 to 20 bids, depending on the domain size
			int n = (int) Math.round(bids.size() / 10.0);
			if (n < 3)
				n = 3;
			if (n > 20)
				n = 20;

			ArrayList<BidDetails> list = (ArrayList<BidDetails>) possibleBidHistory.getNBestBids(n);
			BidHistory bestN = new BidHistory(list);
			BidDetails randomBestN = bestN.getRandom(random100);

			return randomBestN.getBid();
		}
		return omStrategy.getBid(bids).getBid();
	}

	private Bid getBestBidForSelf(ArrayList<BidDetails> bids) {
		Bid maxBid = bids.get(0).getBid();
		for (int i = 0; i < bids.size(); i++) {
			try {
				if (negotiationSession.getUtilitySpace().getUtility(bids.get(i).getBid()) > negotiationSession
						.getUtilitySpace().getUtility(maxBid)) {
					maxBid = bids.get(i).getBid();
				}
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
		return maxBid;
	}

	/**
	 * Prevents the agent from overshooting. Replaces the planned {@link Bid} by
	 * something more appropriate if there was a bid by the opponent that is
	 * better than our planned bid.
	 */
	private Bid makeAppropriate(Bid myPlannedBid) {
		BidDetails bestBidByOpponent = negotiationSession.getOpponentBidHistory().getBestBidDetails();
		// Bid by opponent is better than our planned bid
		double bestUtilityByOpponent = getUtility(bestBidByOpponent.getBid());
		double myPlannedUtility = getUtility(myPlannedBid);

		if (bestUtilityByOpponent >= myPlannedUtility) {
			return bestBidByOpponent.getBid();
		}
		return myPlannedBid;
	}

	public double getUtility(Bid bid) {
		return negotiationSession.getUtilitySpace().getUtilityWithDiscount(bid,
				negotiationSession.getTimeline().getTime());
	}

	@Override
	public String getName() {
		return "2011 - NiceTitForTat";
	}
}
