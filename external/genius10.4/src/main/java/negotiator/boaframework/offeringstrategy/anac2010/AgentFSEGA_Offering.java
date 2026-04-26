package negotiator.boaframework.offeringstrategy.anac2010;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Map;

import genius.core.Bid;
import genius.core.BidIterator;
import genius.core.bidding.BidDetails;
import genius.core.boaframework.NegotiationSession;
import genius.core.boaframework.NoModel;
import genius.core.boaframework.OMStrategy;
import genius.core.boaframework.OfferingStrategy;
import genius.core.boaframework.OpponentModel;
import genius.core.boaframework.SortedOutcomeSpace;
import genius.core.utility.AbstractUtilitySpace;
import negotiator.boaframework.opponentmodel.DefaultModel;
import negotiator.boaframework.opponentmodel.FSEGABayesianModel;

/**
 * This is the decoupled Offering Strategy for AgentFSEGA (ANAC2010). The code
 * was taken from the ANAC2010 AgentFSEGA and adapted to work within the BOA
 * framework.
 * 
 * DEFAULT OM: FSEGABayesianModel (model is broken)
 * 
 * Decoupling Negotiating Agents to Explore the Space of Negotiation Strategies
 * T. Baarslag, K. Hindriks, M. Hendrikx, A. Dirkzwager, C.M. Jonker
 * 
 * @author Alex Dirkzwager, Mark Hendrikx
 */
public class AgentFSEGA_Offering extends OfferingStrategy {

	private static final double MIN_ALLOWED_UTILITY = 0.5;
	private static final double SIGMA = 0.01;
	private static final double SIGMA_MAX = 0.5;
	private final boolean TEST_EQUIVALENCE = false;

	private double elapsedTimeNorm;
	private ArrayList<Bid> leftBids;
	private SortedOutcomeSpace outcomeSpace;

	/**
	 * Empty constructor for the BOA framework.
	 */
	public AgentFSEGA_Offering() {
	}

	@Override
	public void init(NegotiationSession negotiationSession, OpponentModel model, OMStrategy oms,
			Map<String, Double> parameters) throws Exception {
		if (model instanceof DefaultModel) {
			model = new FSEGABayesianModel();
			model.init(negotiationSession, null);
			oms.setOpponentModel(model);
		}
		initializeAgent(negotiationSession, model, oms);
	}

	private void initializeAgent(NegotiationSession negoSession, OpponentModel model, OMStrategy oms) {
		this.negotiationSession = negoSession;
		this.omStrategy = oms;
		this.opponentModel = model;
		if (!(model instanceof NoModel || model instanceof FSEGABayesianModel)) {
			outcomeSpace = new SortedOutcomeSpace(negoSession.getUtilitySpace());
		}
		// initialize left bids
		BidIterator bIterCount = new BidIterator(negotiationSession.getUtilitySpace().getDomain());
		// proposedBids = new ArrayList<Bid>();

		double[] nrBids = { 0, 0, 0, 0, 0 };

		while (bIterCount.hasNext()) {
			Bid tmpBid = bIterCount.next();
			double utility;
			try {
				utility = negotiationSession.getUtilitySpace().getUtility(tmpBid);
				// exclude bids with utility < MIN_ALLOWED_UTILITY
				if ((utility > 1.0) && (utility > 0.9))
					nrBids[0]++;
				else if ((utility <= 0.9) && (utility > 0.8))
					nrBids[1]++;
				else if ((utility <= 0.8) && (utility > 0.7))
					nrBids[2]++;
				else if ((utility <= 0.7) && (utility > 0.6))
					nrBids[3]++;
				else if (utility >= MIN_ALLOWED_UTILITY)
					nrBids[4]++;
			} catch (Exception e) {
			}
		}
		try {
			Thread.sleep(1000);
		} catch (Exception e) {
		}

		//
		double arrayBidCount = 0;
		int iMin = 0;
		do {
			arrayBidCount = nrBids[iMin];
			iMin++;
		} while ((arrayBidCount == 0) && (iMin < 5));

		// end Test

		// initialize left bids
		BidIterator bIter = new BidIterator(negotiationSession.getUtilitySpace().getDomain());
		// proposedBids = new ArrayList<Bid>();
		leftBids = new ArrayList<Bid>();

		while (bIter.hasNext()) {
			Bid tmpBid = bIter.next();
			try {
				// exclude bids with utility < MIN_ALLOWED_UTILITY
				if (negotiationSession.getUtilitySpace().getUtility(tmpBid) >= 0.9 - 0.1 * iMin)
					leftBids.add(tmpBid);
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
		Collections.sort(leftBids, new ReverseBidComparator(negotiationSession.getUtilitySpace()));
	}

	@Override
	public BidDetails determineOpeningBid() {
		try {
			return initialOffer();
		} catch (Exception e) {
			e.printStackTrace();
		}
		return null;
	}

	@Override
	public BidDetails determineNextBid() {
		int timeCase; // the curent time interval id for cameleonic behaviour

		elapsedTimeNorm = negotiationSession.getTime();

		if (elapsedTimeNorm < 0.85)
			timeCase = 0;
		else if (elapsedTimeNorm < 0.95)
			timeCase = 1;
		else
			timeCase = 2;

		if (negotiationSession.getOwnBidHistory().getLastBidDetails() != null) {
			try {
				Bid bid = getNextBid(timeCase);
				double bidUtil = negotiationSession.getUtilitySpace().getUtility(bid);
				nextBid = new BidDetails(bid, bidUtil, negotiationSession.getTime());
			} catch (Exception e) {
				e.printStackTrace();
			}
		}

		if (!(opponentModel instanceof NoModel || opponentModel instanceof FSEGABayesianModel)) {
			try {
				nextBid = omStrategy.getBid(outcomeSpace,
						negotiationSession.getUtilitySpace().getUtility(nextBid.getBid()));
			} catch (Exception e) {
				e.printStackTrace();
			}
		}

		return nextBid;
	}

	private BidDetails initialOffer() throws Exception {
		nextBid = negotiationSession.getMaxBidinDomain();
		return negotiationSession.getMaxBidinDomain();
	}

	private Bid getNextBid(int pTimeCase) throws Exception {
		switch (pTimeCase) {
		case 0:
			return getSmartBid(pTimeCase);
		case 1:
			return getSmartBid(1);
		default:
			return getSmartBid(2);
		}
	}

	private Bid getSmartBid(int pTimeCase) {
		// log("Nr. candidates: " + leftBids.size());

		double currentOpponentUtility = 0.0;
		double lastOpponentUtility = 0.0;
		;

		double myBestUtil; // utility for remained best bid
		double myNextUtility; // utility for next bid
		// java.util.Iterator<Bid> lbi = leftBids.iterator();

		Bid nextBest = null; // best for me and opponent
		Bid theBest = null; // the best for me, top of the ordered list

		// get first entry
		if (leftBids.size() > 0) {
			theBest = leftBids.get(0);
			try {
			} catch (Exception e1) {
				e1.printStackTrace();
			}

			try {
				myBestUtil = negotiationSession.getUtilitySpace().getUtility(theBest);
			} catch (Exception e) {
				myBestUtil = 1;
			}

			nextBest = theBest;
		} else {
			// log("no other bid remained");
			return negotiationSession.getOwnBidHistory().getLastBidDetails().getBid();
		}

		Bid lNext = null;

		double minUtilAllowed = Math.max(0.98 * Math.exp(Math.log(0.52) * elapsedTimeNorm), 0.5);
		double minArrayUtility = 0;
		try {
			minArrayUtility = negotiationSession.getUtilitySpace().getUtility(leftBids.get(leftBids.size() - 1));
		} catch (Exception e) {
		}

		// log("Minimum array utility: " + minArrayUtility);
		// log("Minimum utility allowed: " + minUtilAllowed);
		// log("Left bids no: " + leftBids.size());
		// !!! left bids receiveMessage
		if (((minArrayUtility > minUtilAllowed + SIGMA) || (minArrayUtility > (myBestUtil - SIGMA)))) {
			BidIterator bIter = new BidIterator(negotiationSession.getUtilitySpace().getDomain());

			for (Bid tmpBid = bIter.next(); bIter.hasNext(); tmpBid = bIter.next()) {
				try {
					double tmpBidUtil = negotiationSession.getUtilitySpace().getUtility(tmpBid);
					if ((tmpBidUtil > MIN_ALLOWED_UTILITY) && (tmpBidUtil < minArrayUtility)
							&& (tmpBidUtil > Math.min(minUtilAllowed, myBestUtil) - 0.1))
						leftBids.add(tmpBid);
				} catch (Exception e) {
				}
			}
			Collections.sort(leftBids, new ReverseBidComparator(negotiationSession.getUtilitySpace()));
		}

		// get second entry in last bids
		if (leftBids.size() > 1) {
			lNext = leftBids.get(1);
		} else {
			// proposedBids.add(nextBest);
			leftBids.remove(nextBest);
			return nextBest;
		}

		try {
			myNextUtility = negotiationSession.getUtilitySpace().getUtility(lNext);
		} catch (Exception e) {
			myNextUtility = 0;
		}

		double lowerAcceptableUtilLimit;

		// if time case is 2 -> make concession
		if (pTimeCase == 2) {
			lowerAcceptableUtilLimit = myBestUtil
					- (Math.exp(Math.log(SIGMA_MAX) / (0.05 * negotiationSession.getTimeline().getTotalTime())));
		} else {
			// minimum allowed utility for me at this time
			if (pTimeCase == 0)
				lowerAcceptableUtilLimit = Math.max(myBestUtil - SIGMA, minUtilAllowed);
			else
				lowerAcceptableUtilLimit = Math.min(myBestUtil - SIGMA, minUtilAllowed);
		}

		// eliminate first bid + next bid
		java.util.Iterator<Bid> lbi = leftBids.iterator();
		if (leftBids.size() > 1) {
			lbi.next(); // first bid
			lbi.next();
		} else {
			return negotiationSession.getOwnBidHistory().getLastBidDetails().getBid();
		}

		// get a bid in interval (max_util - SIGMA, max_util]
		while ((myNextUtility > lowerAcceptableUtilLimit) && (myNextUtility <= minUtilAllowed)) {
			// check (my next util) < (last opponent bid's utility for me)
			// in this case, offer previous bid
			// Cristi Litan 19.03.2010
			if (pTimeCase == 0) // use behaviour 0
			{
				try // catch getUtility exceptions
				{
					// do not offer bids with utility smaller than that gived by
					// opponent in time case 0
					if (myNextUtility < negotiationSession.getUtilitySpace()
							.getUtility(negotiationSession.getOpponentBidHistory().getLastBidDetails().getBid())) {
						nextBest = negotiationSession.getOwnBidHistory().getLastBidDetails().getBid();
						break;
					}
				} catch (Exception e) { /* do nothing - ignore */
				}
			}

			try {
				// log("opponent util");
				// TODO: Dan modified because it's not necessary to normalize
				// these utilities
				currentOpponentUtility = opponentModel.getBidEvaluation(lNext);
			} catch (Exception e) {
				currentOpponentUtility = 0;
			}

			// log("exit from cycle");

			if (currentOpponentUtility > lastOpponentUtility) {
				lastOpponentUtility = currentOpponentUtility;
				nextBest = lNext;
			}

			if (lbi.hasNext()) {
				lNext = lbi.next();

				// get my utility for next possible bid
				try {
					myNextUtility = negotiationSession.getUtilitySpace().getUtility(lNext);
				} catch (Exception e) {
					myNextUtility = 0;
				}
			} else
				// log("no other in possible bids");
				break;
		}

		try {
			// test under limit case
			if (negotiationSession.getUtilitySpace().getUtility(nextBest) <= minUtilAllowed) {
				// TODO: place here code for under limit case
				// log(myNextUtility + " < " + minUtilAllowed + " under limit");
				return negotiationSession.getOwnBidHistory().getLastBidDetails().getBid();
			}
		} catch (Exception e) {
		}

		// log("iteration count: " + iteration_count);

		// proposedBids.add(nextBest);
		leftBids.remove(nextBest);

		// log("return a bid in interval");

		return nextBest;
	}

	private class ReverseBidComparator implements Comparator<Bid> {
		private AbstractUtilitySpace usp;

		public ReverseBidComparator(AbstractUtilitySpace pUsp) {
			usp = pUsp;
		}

		public int compare(Bid b1, Bid b2) {
			try {
				double u1 = usp.getUtility(b1);
				double u2 = usp.getUtility(b2);
				if (TEST_EQUIVALENCE) {
					if (u1 == u2) {
						return String.CASE_INSENSITIVE_ORDER.compare(b1.toString(), b2.toString());
					}
				}
				if (u1 > u2)
					return -1; // ! is reversed
				else if (u1 < u2)
					return 1;
				else
					return 0;
			} catch (Exception e) {
				return -1;
			}
		}
	}

	@Override
	public String getName() {
		return "2010 - AgentFSEGA";
	}
}
