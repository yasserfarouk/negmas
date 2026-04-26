package agents.anac.y2019.fsega2019.agent;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import agents.anac.y2019.fsega2019.fsegaoppmodel.MyBayesianOpponentModel;
import agents.anac.y2019.fsega2019.fsegaoppmodel.OpponentModel;
import agents.anac.y2019.fsega2019.fsegaoppmodel.ReverseBidComparator;
import genius.core.AgentID;
import genius.core.Bid;
import genius.core.BidIterator;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.EndNegotiation;
import genius.core.actions.Offer;
import genius.core.parties.AbstractNegotiationParty;
import genius.core.parties.NegotiationInfo;
import genius.core.uncertainty.ExperimentalUserModel;
import genius.core.utility.AbstractUtilitySpace;

public class FSEGA2019 extends AbstractNegotiationParty {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

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
	private int count = 0;
	private NegotiationInfo info;
	private double initialTime;
	private static final double SIGMA_MAX = 0.5;
	private AbstractUtilitySpace realUSpace;

	public static String getVersion() {
		return "1.0";
	}

	@Override
	public void init(NegotiationInfo info) {

		initialTime = info.getTimeline().getTime();
		super.init(info);

		this.info = info;

		lastOponentAction = null;
		myLastAction = null;

		// initialize opponent model
		opponentModel = new MyBayesianOpponentModel(utilitySpace);

		// initialize left bids

		leftBids = new ArrayList<Bid>();

		// initialize left bids

		double[] nrBids = { 0, 0, 0, 0, 0 };
		BidIterator bIterCount = new BidIterator(utilitySpace.getDomain());

		while (bIterCount.hasNext()) {
			Bid tmpBid = bIterCount.next();
			double utility;
			try {
				utility = utilitySpace.getUtility(tmpBid);
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
		BidIterator bIter = new BidIterator(utilitySpace.getDomain());
		// proposedBids = new ArrayList<Bid>();
		leftBids = new ArrayList<Bid>();

		if (userModel instanceof ExperimentalUserModel) {
			ExperimentalUserModel e = (ExperimentalUserModel) userModel;
			realUSpace = e.getRealUtilitySpace();
		}
		Bid bestBid = null;
		Bid secondBestBid = null;
		double bestBidUt = -1, secondBestBidUt = -1;
		int i = 0;
		while (bIter.hasNext()) {
			i++;
			Bid tempBid = bIter.next();
			if (getUtilitySpace().getUtility(tempBid) >= 0.9 - 0.1 * iMin) {
				if (getUtilitySpace().getUtility(tempBid) > bestBidUt) {
					bestBidUt = getUtilitySpace().getUtility(tempBid);
					secondBestBid = bestBid;
					bestBid = tempBid;
				} else if (getUtilitySpace().getUtility(tempBid) > secondBestBidUt) {
					secondBestBidUt = getUtilitySpace().getUtility(tempBid);
					secondBestBid = tempBid;
				}
			}
		}
		leftBids.add(bestBid);
		leftBids.add(secondBestBid);

	}

	@Override
	public void receiveMessage(AgentID sender, Action act) {
		super.receiveMessage(sender, act);
		if (act instanceof Offer) {
			lastOponentAction = (Offer) act;
		} else if (act instanceof Accept) {
			lastOponentAction = (Accept) act;
		} else if (act instanceof EndNegotiation) {
			lastOponentAction = (EndNegotiation) act;
		}

	}

	@Override
	public Action chooseAction(List<Class<? extends Action>> possibleActions) {

		Action action = null;
		Bid opponentBid = null;
		try {
			switch (getActionType(lastOponentAction)) {
			case OFFER:

				int timeCase; // the curent time interval id for cameleonic behaviour

				elapsedTimeNorm = getTimeLine().getTime();

				if (elapsedTimeNorm < 0.85)
					timeCase = 0;
				else if (elapsedTimeNorm < 0.95)
					timeCase = 1;
				else
					timeCase = 2;

				opponentBid = ((Offer) lastOponentAction).getBid();

				// update beliefs
				opponentModel.updateBeliefs(opponentBid);

				if (myLastAction != null) {
					// check if the opponent offer is acceptable
					if (getUtilitySpace().getUtility(opponentBid) * 1.03 >= getUtilitySpace()
							.getUtility(((Offer) myLastAction).getBid())) {
						action = new Accept(getPartyId(), opponentBid);
					} else {
						Bid nextBid = getNextBid(timeCase);

						if (nextBid != null)
						action = new Offer(getPartyId(), nextBid);

						// check if the opponent offer utility is greater than next bid
						if (getUtilitySpace().getUtility(opponentBid) > getUtilitySpace().getUtility(nextBid)) {
							action = new Accept(getPartyId(), opponentBid);
						}

					}
				} else {
					// opponent begins and this is my first action
					// check if opponent's offer is acceptable
					if (getUtilitySpace().getUtility(opponentBid) == getUtilitySpace()
							.getUtility(utilitySpace.getMaxUtilityBid())) {
						action = new Accept(getPartyId(), opponentBid);
						break;
					}
					// this is the first offer of this node
					// this bid have maximum utility
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
					// is the first offer of this node
					action = initialOffer();
				} else {
					action = myLastAction;
				}
			}
		} catch (Exception e) {
			System.out.println("EXCEPTION description: " + e.getMessage() + " stack ");
			e.printStackTrace();
		}

		myLastAction = action;

		return action;
	}

	@Override
	public String getDescription() {
		return "FSEGA Agent with preference uncertainty";
	}

	@Override
	public AbstractUtilitySpace estimateUtilitySpace() {
		return super.estimateUtilitySpace();
	}

	private Action initialOffer() throws Exception {
		return new Offer(getPartyId(), getUtilitySpace().getMaxUtilityBid());
	}

	private Bid getNextBid(int pTimeCase) throws Exception {
		switch (pTimeCase) {
		case 0:
			Bid b = getSmartBid(pTimeCase);
			return b;
		case 1:
			b = getSmartBid(1);
			return b;
		default:
			b = getSmartBid(2);
			return b;
		}
	}

	private Bid getSmartBid(int pTimeCase) {

		double currentOpponentUtility = 0.0;
		double lastOpponentUtility;

		double myBestUtil; // utility for remained best bid
		double myNextUtility; // utility for next bid

		Bid nextBest = null; // best for me and opponent
		Bid theBest = null; // the best for me, top of the ordered list

		count++;

		// get first entry
		if (leftBids.size() > 0 && leftBids.get(0) != null) {
			theBest = leftBids.get(0);
			try {
				lastOpponentUtility = opponentModel.getExpectedUtility(nextBest);
			} catch (Exception e) {
				lastOpponentUtility = 0;
			}

			try {
				myBestUtil = getUtilitySpace().getUtility(theBest);
			} catch (Exception e) {
				myBestUtil = 1;
			}

			nextBest = theBest;

		} else {
			return ((Offer) myLastAction).getBid();
		}

		Bid lNext = null;

		double minUtilAllowed = Math.max(0.98 * Math.exp(Math.log(0.52) * elapsedTimeNorm), 0.5);

		double minArrayUtility = 0;
		try {
			minArrayUtility = getUtilitySpace().getUtility(leftBids.get(leftBids.size() - 1));
		} catch (Exception e) {
		}

		// !!! left bids update
		Bid bestBid = null;
		Bid secondBestBid = null;
		double bestBidUt = -1, secondBestBidUt = -1;
		if (((minArrayUtility > minUtilAllowed + SIGMA) || (minArrayUtility > (myBestUtil - SIGMA)))) {
			BidIterator bIter = new BidIterator(utilitySpace.getDomain());

			for (Bid tmpBid = bIter.next(); bIter.hasNext(); tmpBid = bIter.next()) {
				double tmpBidUtil = getUtilitySpace().getUtility(tmpBid);
				if ((tmpBidUtil > MIN_ALLOWED_UTILITY) && (tmpBidUtil < minArrayUtility)
						&& (tmpBidUtil > Math.min(minUtilAllowed, myBestUtil) - 0.1)) {
					if (getUtilitySpace().getUtility(tmpBid) > bestBidUt) {
						bestBidUt = getUtilitySpace().getUtility(tmpBid);
						secondBestBid = bestBid;
						bestBid = tmpBid;
					} else if (getUtilitySpace().getUtility(tmpBid) > secondBestBidUt) {
						secondBestBidUt = getUtilitySpace().getUtility(tmpBid);
						secondBestBid = tmpBid;
					}
				}
			}
			leftBids.add(bestBid);
			leftBids.add(secondBestBid);

			Collections.sort(leftBids, new ReverseBidComparator(utilitySpace));

		}

		// get second entry in last bids
		if (leftBids.size() > 1) {
			lNext = leftBids.get(1);
		} else {
			leftBids.remove(nextBest);
			return nextBest;
		}

		try {
			myNextUtility = getUtilitySpace().getUtility(lNext);
		} catch (Exception e) {
			myNextUtility = 0;
		}

		double lowerAcceptableUtilLimit;

		// if time case is 2 -> make concession
		if (pTimeCase == 2) {
			lowerAcceptableUtilLimit = myBestUtil
					- (Math.exp(Math.log(SIGMA_MAX) / (0.05 * getTimeLine().getTotalTime())) * elapsedTimeNorm);

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
			return ((Offer) myLastAction).getBid();
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
					// do not offer bids with utility smaller than that given by opponent in time
					// case 0
					if (myNextUtility < getUtilitySpace().getUtility(((Offer) lastOponentAction).getBid())) {
						nextBest = ((Offer) myLastAction).getBid();
						break;
					}
				} catch (Exception e) {
					/* do nothing - ignore */ }
			}

			try {

				currentOpponentUtility = opponentModel.getExpectedUtility(lNext);
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
					myNextUtility = getUtilitySpace().getUtility(lNext);
				} catch (Exception e) {
					myNextUtility = 0;
				}
			} else
				// log("no other in possible bids");
				break;
		}

		try {
			// test under limit case
			if (getUtilitySpace().getUtility(nextBest) <= minUtilAllowed) {

				return ((Offer) myLastAction).getBid();
			}
		} catch (Exception e) {
		}

		leftBids.remove(nextBest);

		return nextBest;
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
}
