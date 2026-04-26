package agents.anac.y2010.AgentFSEGA;

import java.util.ArrayList;
import java.util.Collections;

import genius.core.Agent;
import genius.core.Bid;
import genius.core.BidIterator;
import genius.core.SupportedNegotiationSetting;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.EndNegotiation;
import genius.core.actions.Offer;
import genius.core.utility.AdditiveUtilitySpace;

/**
 * ANAC2010 competitor AgentFSEGA.
 */
public class AgentFSEGA extends Agent {
	enum ACTIONTYPE {
		START, OFFER, ACCEPT, BREAKOFF
	};

	enum STRATEGY {
		SMART, SERIAL, RESPONSIVE, RANDOM, TIT_FOR_TAT
	};

	private Action lastOponentAction;
	private Action myLastAction;
	private OpponentModel opponentModel;
	private ArrayList<Bid> leftBids;
	private double elapsedTimeNorm;
	private static final double SIGMA = 0.01;
	private static final double MIN_ALLOWED_UTILITY = 0.5;
	private static final double SIGMA_MAX = 0.5;

	/** Returns the version of the agent. */
	@Override
	public String getVersion() {
		return "1.0";
	}

	/** Initializes the agent. */
	@Override
	public void init() {
		lastOponentAction = null;
		myLastAction = null;

		// initialize opponent model
		opponentModel = new MyBayesianOpponentModel(
				(AdditiveUtilitySpace) utilitySpace);

		// initialize left bids
		BidIterator bIterCount = new BidIterator(utilitySpace.getDomain());
		// proposedBids = new ArrayList<Bid>();
		leftBids = new ArrayList<Bid>();

		double[] nrBids = { 0, 0, 0, 0, 0 };

		while (bIterCount.hasNext()) {
			Bid tmpBid = bIterCount.next();
			double utility;
			try {
				utility = utilitySpace.getUtility(tmpBid);
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
				e.printStackTrace();
			}
		}
		try {
			Thread.sleep(1000);
		} catch (Exception e) {
		}

		double arrayBidCount = 0;
		int iMin = 0;

		do {
			arrayBidCount = nrBids[iMin];
			iMin++;
		} while ((arrayBidCount == 0) && (iMin < 5));

		// initialize left bids
		BidIterator bIter = new BidIterator(utilitySpace.getDomain());
		leftBids = new ArrayList<Bid>();

		while (bIter.hasNext()) {
			Bid tmpBid = bIter.next();
			try {
				// exclude bids with utility < MIN_ALLOWED_UTILITY
				if (utilitySpace.getUtility(tmpBid) >= 0.9 - 0.1 * iMin)
					leftBids.add(tmpBid);
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
		Collections.sort(leftBids,
				new ReverseBidComparator((AdditiveUtilitySpace) utilitySpace));
	}

	/** Receive the opponent's bid. */
	@Override
	public void ReceiveMessage(Action opponentAction) {
		lastOponentAction = opponentAction;
	}

	/** Choose action to present to the opponent. */
	@Override
	public Action chooseAction() {
		Action action = null;
		Bid opponentBid = null;
		try {
			switch (getActionType(lastOponentAction)) {
			case OFFER:
				int timeCase; // the curent time interval id for cameleonic
								// behaviour
				elapsedTimeNorm = timeline.getTime(); // between 0 .. 1

				if (elapsedTimeNorm < 0.85)
					timeCase = 0;
				else if (elapsedTimeNorm < 0.95)
					timeCase = 1;
				else
					timeCase = 2;

				opponentBid = ((Offer) lastOponentAction).getBid();

				// receiveMessage beliefs
				opponentModel.updateBeliefs(opponentBid);

				if (myLastAction != null) {
					// check if the opponent offer is acceptable
					if (utilitySpace.getUtility(opponentBid)
							* 1.03 >= utilitySpace.getUtility(
									((Offer) myLastAction).getBid())) {
						action = new Accept(getAgentID(), opponentBid);
					} else {
						Bid nextBid = getNextBid(timeCase);
						action = new Offer(getAgentID(), nextBid);

						// check if the opponent offer utility is greater than
						// next bid
						if (utilitySpace.getUtility(opponentBid) > utilitySpace
								.getUtility(nextBid)) {
							action = new Accept(getAgentID(), opponentBid);
						}
					}
				} else {
					// opponent begins and this is my first action check if
					// opponent's offer is acceptable
					if (utilitySpace.getUtility(opponentBid) == utilitySpace
							.getUtility(utilitySpace.getMaxUtilityBid())) {
						action = new Accept(getAgentID(), opponentBid);
						break;
					}
					// this is the first offer of this node this bid have
					// maximum utility
					action = initialOffer();
				}
				break;
			case ACCEPT:
			case BREAKOFF:
				// Nothing to do in this case
				break;
			default: // normally occurs when action type is START
				// verify if is agent's first offer
				if (myLastAction == null) {
					// is the first offer of this node this bid have maximum
					// utility
					action = initialOffer();
				} else {
					action = myLastAction;
				}
			}
		} catch (Exception e) {
			e.printStackTrace();
		}

		myLastAction = action;

		return action;
	}

	private Action initialOffer() throws Exception {
		// proposes the offer with maximum possible utility
		return new Offer(getAgentID(), utilitySpace.getMaxUtilityBid());
	}

	private ACTIONTYPE getActionType(Action lAction) {
		ACTIONTYPE lActionType = ACTIONTYPE.START;
		if (lAction instanceof Offer)
			lActionType = ACTIONTYPE.OFFER;
		else if (lAction instanceof Accept)
			lActionType = ACTIONTYPE.ACCEPT;
		else if (lAction instanceof EndNegotiation)
			lActionType = ACTIONTYPE.BREAKOFF;
		return lActionType;
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

	@Override
	public String getName() {
		return "AgentFSEGA";
	}

	private Bid getSmartBid(int pTimeCase) {
		double currentOpponentUtility = 0.0;
		double lastOpponentUtility = 0.0;

		double myBestUtil; // utility for remained best bid
		double myNextUtility; // utility for next bid

		Bid nextBest = null; // best for me and opponent
		Bid theBest = null; // the best for me, top of the ordered list

		// get first entry
		if (leftBids.size() > 0) {
			theBest = leftBids.get(0);

			try {
				myBestUtil = utilitySpace.getUtility(theBest);
			} catch (Exception e) {
				myBestUtil = 1;
			}

			nextBest = theBest;
		} else {
			return ((Offer) myLastAction).getBid();
		}

		Bid lNext = null;

		double minUtilAllowed = Math
				.max(0.98 * Math.exp(Math.log(0.52) * elapsedTimeNorm), 0.5);
		double minArrayUtility = 0;

		try {
			minArrayUtility = utilitySpace
					.getUtility(leftBids.get(leftBids.size() - 1));
		} catch (Exception e) {
			e.printStackTrace();
		}

		// !!! left bids receiveMessage
		if (((minArrayUtility > minUtilAllowed + SIGMA)
				|| (minArrayUtility > (myBestUtil - SIGMA)))) {
			BidIterator bIter = new BidIterator(utilitySpace.getDomain());

			for (Bid tmpBid = bIter.next(); bIter
					.hasNext(); tmpBid = bIter.next()) {
				try {
					double tmpBidUtil = utilitySpace.getUtility(tmpBid);
					if ((tmpBidUtil > MIN_ALLOWED_UTILITY)
							&& (tmpBidUtil < minArrayUtility)
							&& (tmpBidUtil > Math.min(minUtilAllowed,
									myBestUtil) - 0.1))
						leftBids.add(tmpBid);
				} catch (Exception e) {
				}
			}
			Collections.sort(leftBids, new ReverseBidComparator(
					(AdditiveUtilitySpace) utilitySpace));
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
			myNextUtility = utilitySpace.getUtility(lNext);
		} catch (Exception e) {
			myNextUtility = 0;
		}

		double lowerAcceptableUtilLimit;

		// if time case is 2 -> make concession
		if (pTimeCase == 2) {
			lowerAcceptableUtilLimit = myBestUtil - (Math
					.exp(Math.log(SIGMA_MAX) / (0.05 * timeline.getTotalTime()))
					* elapsedTimeNorm);
		} else {
			// minimum allowed utility for me at this time
			if (pTimeCase == 0)
				lowerAcceptableUtilLimit = Math.max(myBestUtil - SIGMA,
						minUtilAllowed);
			else
				lowerAcceptableUtilLimit = Math.min(myBestUtil - SIGMA,
						minUtilAllowed);
		}

		// eliminate first bid + next bid
		java.util.Iterator<Bid> lbi = leftBids.iterator();
		if (leftBids.size() > 1) {
			lbi.next(); // first bid
			lbi.next();
		} else {
			return ((Offer) myLastAction).getBid();
		}

		// get a bid in interval (max_util - SIGMA, max_util]
		while ((myNextUtility > lowerAcceptableUtilLimit)
				&& (myNextUtility <= minUtilAllowed)) {
			// check (my next util) < (last opponent bid's utility for me)
			// in this case, offer previous bid
			// Cristi Litan 19.03.2010
			if (pTimeCase == 0) // use behaviour 0
			{
				try // catch getUtility exceptions
				{
					// do not offer bids with utility smaller than that gived by
					// opponent in time case 0
					if (myNextUtility < utilitySpace
							.getUtility(((Offer) lastOponentAction).getBid())) {
						nextBest = ((Offer) myLastAction).getBid();
						break;
					}
				} catch (Exception e) { /* do nothing - ignore */
				}
			}

			try {
				currentOpponentUtility = opponentModel
						.getExpectedUtility(lNext);
			} catch (Exception e) {
				currentOpponentUtility = 0;
			}

			if (currentOpponentUtility > lastOpponentUtility) {
				lastOpponentUtility = currentOpponentUtility;
				nextBest = lNext;
			}

			if (lbi.hasNext()) {
				lNext = lbi.next();

				// get my utility for next possible bid
				try {
					myNextUtility = utilitySpace.getUtility(lNext);
				} catch (Exception e) {
					myNextUtility = 0;
				}
			} else
				// log("no other in possible bids");
				break;
		}

		try {
			if (utilitySpace.getUtility(nextBest) <= minUtilAllowed) {
				return ((Offer) myLastAction).getBid();
			}
		} catch (Exception e) {
		}

		leftBids.remove(nextBest);

		return nextBest;
	}

	@Override
	public SupportedNegotiationSetting getSupportedNegotiationSetting() {
		return SupportedNegotiationSetting.getLinearUtilitySpaceInstance();
	}

	@Override
	public String getDescription() {
		return "ANAC2010";
	}
}