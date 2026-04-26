package negotiator.boaframework.offeringstrategy.anac2011;

import java.util.Map;
import java.util.Random;

import genius.core.Bid;
import genius.core.BidIterator;
import genius.core.bidding.BidDetails;
import genius.core.boaframework.NegotiationSession;
import genius.core.boaframework.OMStrategy;
import genius.core.boaframework.OfferingStrategy;
import genius.core.boaframework.OpponentModel;
import genius.core.utility.AdditiveUtilitySpace;
import negotiator.boaframework.offeringstrategy.anac2011.valuemodelagent.BidList;
import negotiator.boaframework.offeringstrategy.anac2011.valuemodelagent.BidWrapper;
import negotiator.boaframework.offeringstrategy.anac2011.valuemodelagent.OpponentModeler;
import negotiator.boaframework.offeringstrategy.anac2011.valuemodelagent.ValueDecrease;
import negotiator.boaframework.offeringstrategy.anac2011.valuemodelagent.ValueModeler;
import negotiator.boaframework.offeringstrategy.anac2011.valuemodelagent.ValueSeperatedBids;
import negotiator.boaframework.sharedagentstate.anac2011.ValueModelAgentSAS;

/**
 * This is the decoupled Offering Strategy for ValueModelAgent (ANAC2011). The
 * code was taken from the ANAC2011 ValueModelAgent and adapted to work within
 * the BOA framework.
 * 
 * This agent has no OM implementation.
 * 
 * @author Mark Hendrikx
 */
public class ValueModelAgent_Offering extends OfferingStrategy {

	private ValueModeler opponentUtilModel = null;
	private BidList allBids = null;
	private BidList approvedBids = null;
	private BidList iteratedBids = null;
	private Bid myLastBid = null;
	private int bidCount;
	private BidList opponentBids;
	private BidList ourBids;
	private OpponentModeler opponent;
	private double lowestAcceptable;
	private double lowestApproved;
	private double opponentStartbidUtil;
	private Bid opponentMaxBid;
	private double myMaximumUtility;
	private int amountOfApproved;
	private double noChangeCounter;
	private boolean retreatMode;
	private double concessionInOurUtility;
	private double concessionInTheirUtility;
	private ValueSeperatedBids seperatedBids;
	private final boolean TEST_EQUIVALENCE = false;
	Random random100;
	Random random200;

	@Override
	public void init(NegotiationSession negotiationSession, OpponentModel model, OMStrategy oms,
			Map<String, Double> parameters) throws Exception {
		this.negotiationSession = negotiationSession;
		this.opponentModel = model;
		this.omStrategy = oms;
		helper = new ValueModelAgentSAS();
		opponentUtilModel = null;
		allBids = null;
		approvedBids = null;
		bidCount = 0;
		opponentBids = new BidList();
		ourBids = new BidList();
		iteratedBids = new BidList();
		seperatedBids = new ValueSeperatedBids();
		lowestAcceptable = 0.7;
		lowestApproved = 1;
		amountOfApproved = 0;
		myMaximumUtility = 1;
		noChangeCounter = 0;
		retreatMode = false;
		concessionInOurUtility = 0;
		concessionInTheirUtility = 0;

		if (TEST_EQUIVALENCE) {
			random100 = new Random(100);
			random200 = new Random(200);
		} else {
			random100 = new Random();
			random200 = new Random();
		}
	}

	// remember our new bid
	private void bidSelected(BidWrapper bid) {
		bid.sentByUs = true;
		ourBids.addIfNew(bid);
		myLastBid = bid.bid;
		if (opponentUtilModel != null)
			seperatedBids.bidden(bid.bid, bidCount);
		// bidCount*2,bid.theirUtility);
	}

	// remember our new bid in all needed data structures
	private void bidSelectedByOpponent(Bid bid) {
		BidWrapper opponentBid = new BidWrapper(bid, (AdditiveUtilitySpace) negotiationSession.getUtilitySpace(),
				myMaximumUtility);
		opponentBid.lastSentBid = bidCount;
		opponentBid.sentByThem = true;
		if (opponentBids.addIfNew(opponentBid)) {
			noChangeCounter = 0;
			// opponentBids.bids.size());
		} else {
			noChangeCounter++;
		}
		try {
			double opponentUtil = negotiationSession.getUtilitySpace().getUtility(bid) / myMaximumUtility;
			if (((ValueModelAgentSAS) helper).getOpponentMaxBidUtil() < opponentUtil) {
				((ValueModelAgentSAS) helper).setOpponentMaxBidUtil(opponentUtil);
				opponentMaxBid = bid;
			}
			if (opponentUtilModel.initialized) {
				double concession = opponentUtil - opponentStartbidUtil;
				if (concession > concessionInOurUtility)
					concessionInOurUtility = concession;
				// concession,opponentUtil,opponentStartbidUtil);
				// assumed utility he lost (should be lower
				// if our opponent is smart, but lets be honest
				// it is quite possible we missevaluated so lets
				// use our opponent utility instead
				ValueDecrease val = opponentUtilModel.utilityLoss(opponentBid.bid);
				concession = val.getDecrease();
				if (concession > concessionInTheirUtility)
					concessionInTheirUtility = concession;
				seperatedBids.bidden(bid, bidCount);
				// concession,concessionInTheirUtility);

			}
		} catch (Exception ex) {
			ex.printStackTrace();
		}
	}

	private boolean setApprovedThreshold(double threshold, boolean clear) {
		if (clear) {
			approvedBids.bids.clear();
			seperatedBids.clear();
		}
		int i;
		if (clear)
			i = 0;
		else
			i = amountOfApproved;
		for (; i < allBids.bids.size(); i++) {
			if (allBids.bids.get(i).ourUtility < threshold) {
				break;
			}
			approvedBids.bids.add(allBids.bids.get(i));
			seperatedBids.addApproved(allBids.bids.get(i));
		}
		lowestApproved = threshold;
		((ValueModelAgentSAS) helper).setLowestApprovedInitial(threshold);
		boolean added = amountOfApproved != i;
		amountOfApproved = i;
		return added;
	}

	public BidDetails determineNextBid() {
		Bid opponentBid = null;
		BidDetails offer = null;
		try {
			// our first bid, initializing
			if (allBids == null) {
				allBids = new BidList();
				approvedBids = new BidList();
				opponentUtilModel = new ValueModeler();
				seperatedBids.init((AdditiveUtilitySpace) negotiationSession.getUtilitySpace(), opponentUtilModel);
				myMaximumUtility = negotiationSession.getUtilitySpace()
						.getUtility(negotiationSession.getUtilitySpace().getMaxUtilityBid());
				BidIterator iter = new BidIterator(negotiationSession.getUtilitySpace().getDomain());
				while (iter.hasNext()) {
					Bid tmpBid = iter.next();
					try {
						// if(utilitySpace.getCost(tmpBid)<=1200){
						BidWrapper wrap = new BidWrapper(tmpBid,
								(AdditiveUtilitySpace) negotiationSession.getUtilitySpace(), myMaximumUtility);
						allBids.bids.add(wrap);
						// }
					} catch (Exception ex) {

						BidWrapper wrap = new BidWrapper(tmpBid,
								(AdditiveUtilitySpace) negotiationSession.getUtilitySpace(), myMaximumUtility);
						allBids.bids.add(wrap);
					}
				} // while
				allBids.sortByOurUtil();
				// allBids.bids.size());

				// amountOfApproved = (int) (0.05*allBids.bids.size());
				// if(amountOfApproved<2 && allBids.bids.size()>=2)
				// amountOfApproved=2;
				// if(amountOfApproved>20) amountOfApproved=20;
				setApprovedThreshold(0.98, false);

				// if(opponentUtilModel!=null)
				// approvedBids.sortByOpponentUtil(opponentUtilModel);
				iteratedBids.bids.add(allBids.bids.get(0));
			}
			// first bid is the highest bid
			if (bidCount == 0) {
				offer = new BidDetails(allBids.bids.get(0).bid,
						negotiationSession.getUtilitySpace().getUtility(allBids.bids.get(0).bid));
				bidSelected(allBids.bids.get(0));
			}

			// treat opponent's offer

			if (negotiationSession.getOpponentBidHistory().size() > 0) {
				BidDetails lastBidByOpp = negotiationSession.getOpponentBidHistory().getLastBidDetails();
				opponentBid = lastBidByOpp.getBid();
				double opponentUtil = negotiationSession.getUtilitySpace().getUtility(opponentBid) / myMaximumUtility;

				((ValueModelAgentSAS) helper).setOpponentUtil(opponentUtil);

				bidSelectedByOpponent(opponentBid);
				if (opponent == null) {

					opponentStartbidUtil = opponentUtil;
					opponent = new OpponentModeler(bidCount,
							(AdditiveUtilitySpace) negotiationSession.getUtilitySpace(),
							negotiationSession.getTimeline(), ourBids, opponentBids, opponentUtilModel, allBids);
					opponentUtilModel.initialize((AdditiveUtilitySpace) negotiationSession.getUtilitySpace(),
							opponentBid);
					approvedBids.sortByOpponentUtil(opponentUtilModel);
				} else {

					opponent.tick();
					if (noChangeCounter == 0) {
						double opponentExpectedBidValue = opponent.guessCurrentBidUtil();
						opponentUtilModel.assumeBidWorth(opponentBid, 1 - opponentExpectedBidValue, 0.02);
					}
				}
				// it seems like I should try to accept
				// his best bid (assuming its still on the table)
				// if its a good enough bid to be reasonable
				// (we don't accept 0.3,0.92 around here...)
				// currently by reasonable we mean that we will
				// get most of what would be a "fair" distribution
				// of the utility

				if (negotiationSession.getTime() > 0.9 && negotiationSession.getTime() <= 0.96) {
					offer = chickenGame(0.039, 0, 0.7);
				}
				if (negotiationSession.getTime() > 0.96 && negotiationSession.getTime() <= 0.97) {
					offer = chickenGame(0.019, 0.5, 0.65);

				}
				if (negotiationSession.getTime() > 0.98 && negotiationSession.getTime() <= 0.99) {
					((ValueModelAgentSAS) helper).setLowestApprovedInitial(lowestApproved);

					if (bidCount % 5 == 0) {
						offer = exploreScan();
					} else
						offer = bestScan();
				}
				if (negotiationSession.getTime() > 0.99 && negotiationSession.getTime() <= 0.995) {
					offer = chickenGame(0.004, 0.8, 0.6);

				}

				if (negotiationSession.getTime() > 0.995 && ((ValueModelAgentSAS) helper).getOpponentMaxBidUtil() > 0.55
						&& opponentUtil < ((ValueModelAgentSAS) helper).getOpponentMaxBidUtil() * 0.99) {
					offer = new BidDetails(opponentMaxBid,
							negotiationSession.getUtilitySpace().getUtility(opponentMaxBid));

					// this will probably not work but what can we
					// do? he dosn't even give us 50%!
				}

				if (negotiationSession.getTime() > 0.995) {
					offer = bestScan();
				}

				if (offer != null) {
					bidCount++;
					return offer;
				}

				((ValueModelAgentSAS) helper).setLowestApproved(lowestApproved);

				// otherwise we try to stretch it out
				if (opponentUtil > lowestApproved * 0.99 && !negotiationSession.getUtilitySpace().isDiscounted()) {
					lowestApproved += 0.01;
					setApprovedThreshold(lowestApproved, true);
					retreatMode = true;
				}
				if (bidCount > 0 && bidCount < 4) {
					offer = new BidDetails(allBids.bids.get(0).bid,
							negotiationSession.getUtilitySpace().getUtility(allBids.bids.get(0).bid));
					if (bidCount < allBids.bids.size()) {
						bidSelected(allBids.bids.get(bidCount));
					}
				}
				if (bidCount >= 4) {
					// utility he gave us

					double concession = opponentUtil - opponentStartbidUtil;
					// assumed utility he lost (should be lower
					// if our opponent is smart, but lets be honest
					// it is quite possible we missevaluated so lets
					// use our opponent utility instead
					// double concession2 =
					// opponentUtilModel.utilityLoss(opponentBid).getDecrease();
					double concession2 = 1 - opponent.guessCurrentBidUtil();
					double minConcession = concession < concession2 ? concession : concession2;
					minConcession = minConcessionMaker(minConcession,
							1 - ((ValueModelAgentSAS) helper).getOpponentMaxBidUtil());

					if (minConcession > (1 - lowestApproved)) {
						if (lowestAcceptable > (1 - minConcession)) {
							lowestApproved = lowestAcceptable;
						} else {
							lowestApproved = 1 - minConcession;
						}
						if (lowestApproved < ((ValueModelAgentSAS) helper).getOpponentMaxBidUtil())
							lowestApproved = ((ValueModelAgentSAS) helper).getOpponentMaxBidUtil() + 0.001;

						if (setApprovedThreshold(lowestApproved, false)) {
							approvedBids.sortByOpponentUtil(opponentUtilModel);
						}

					}

					if (bidCount % 5 == 0) {
						offer = exploreScan();
					} else
						offer = bestScan();
				}

			}

			if (offer == null) {
				offer = negotiationSession.getOwnBidHistory().getLastBidDetails();
			}
		} catch (Exception e) {
			e.printStackTrace();
			((ValueModelAgentSAS) helper).triggerSkipAcceptDueToCrash();

			if (myLastBid == null) {
				try {
					Bid maxBid = negotiationSession.getUtilitySpace().getMaxUtilityBid();
					return new BidDetails(maxBid, negotiationSession.getUtilitySpace().getUtility(maxBid));
				} catch (Exception e2) {
					e2.printStackTrace();
				}
			}

			try {
				return new BidDetails(myLastBid, negotiationSession.getUtilitySpace().getUtility(myLastBid));
			} catch (Exception e1) {
				e1.printStackTrace();
			}

		}
		bidCount++;
		return offer;
	}

	private BidDetails bestScan() {
		approvedBids.sortByOpponentUtil(opponentUtilModel);
		BidDetails toOffer = null;

		// find the "best" bid for opponent
		// and choose it if we didn't send it to opponent
		for (int i = 0; i < approvedBids.bids.size(); i++) {
			BidWrapper tempBid = approvedBids.bids.get(i);
			if (!tempBid.sentByUs) {
				try {
					toOffer = new BidDetails(tempBid.bid, negotiationSession.getUtilitySpace().getUtility(tempBid.bid));
				} catch (Exception e) {
					e.printStackTrace();
				}
				bidSelected(tempBid);
				break;
			}
		}
		if (toOffer == null) {
			int maxIndex = approvedBids.bids.size() / 4;
			BidWrapper tempBid = approvedBids.bids.get((int) (random200.nextDouble() * maxIndex));
			try {
				toOffer = new BidDetails(tempBid.bid, negotiationSession.getUtilitySpace().getUtility(tempBid.bid));
			} catch (Exception e) {
				e.printStackTrace();
			}
			bidSelected(tempBid);
		}
		return toOffer;
	}

	private BidDetails exploreScan() {
		BidDetails toOffer = null;
		BidWrapper tempBid = seperatedBids.explore(bidCount);
		if (tempBid != null) {
			try {
				toOffer = new BidDetails(tempBid.bid, negotiationSession.getUtilitySpace().getUtility(tempBid.bid));
			} catch (Exception e) {
				e.printStackTrace();
			}
			bidSelected(tempBid);
		} else
			toOffer = bestScan();
		return toOffer;
	}

	private BidDetails chickenGame(double timeToGive, double concessionPortion, double acceptableThresh) {
		BidDetails toOffer = null;

		// set timeToGive to be 0.005 unless each turn is very
		// large
		// double timeToGive =
		// 0.005>(opponent.delta*4)?0.005:(opponent.delta*4);
		double concessionLeft = lowestApproved - ((ValueModelAgentSAS) helper).getOpponentMaxBidUtil();
		double planedThresh = lowestApproved - concessionPortion * concessionLeft;
		if (acceptableThresh > planedThresh)
			planedThresh = acceptableThresh;
		setApprovedThreshold(planedThresh, false);
		approvedBids.sortByOpponentUtil(opponentUtilModel);

		((ValueModelAgentSAS) helper).setPlanedThreshold(planedThresh);

		if (1.0 - negotiationSession.getTime() - timeToGive > 0) {
			if (!TEST_EQUIVALENCE) {
				double time = (1.0 - negotiationSession.getTime() - timeToGive) * 180000;
				try {
					Thread.sleep(Math.round(time));
				} catch (InterruptedException e) {
					e.printStackTrace();
				}
			}
		}
		if (retreatMode || ((ValueModelAgentSAS) helper).getOpponentMaxBidUtil() >= planedThresh - 0.01) {
			try {
				toOffer = new BidDetails(opponentMaxBid,
						negotiationSession.getUtilitySpace().getUtility(opponentMaxBid));
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
		// offer him the best bid for him amongst the
		// bids that are above our limit
		else {
			// BUG: nothing is one with this bid in the original code.
			approvedBids.bids.get(0);
		}
		return toOffer;
	}

	double theirMaxUtilities[] = new double[21];
	double ourMinUtilities[] = new double[21];

	private double minConcessionMaker(double minConcession, double concessionLeft) {
		theirMaxUtilities[0] = opponentStartbidUtil;
		ourMinUtilities[0] = 1;
		double t = negotiationSession.getTime();
		int tind = (int) (negotiationSession.getTime() * 20) + 1;
		double segPortion = (t - (tind - 1) * 0.05) / 0.05;
		if (ourMinUtilities[tind] == 0)
			ourMinUtilities[tind] = 1;
		if (ourMinUtilities[tind - 1] == 0)
			ourMinUtilities[tind - 1] = lowestApproved;
		if (lowestApproved < ourMinUtilities[tind])
			ourMinUtilities[tind] = lowestApproved;
		if (((ValueModelAgentSAS) helper).getOpponentMaxBidUtil() > theirMaxUtilities[tind])
			theirMaxUtilities[tind] = ((ValueModelAgentSAS) helper).getOpponentMaxBidUtil();
		double d = negotiationSession.getDiscountFactor();
		double defaultVal = 1 - ourMinUtilities[tind - 1];
		if (tind == 1 || tind >= 19)
			return defaultVal;
		if (ourMinUtilities[tind - 2] == 0)
			ourMinUtilities[tind - 2] = lowestApproved;

		// if(defaultVal>minConcession) return minConcession;
		boolean theyMoved = theirMaxUtilities[tind] - theirMaxUtilities[tind - 2] > 0.01;
		boolean weMoved = ourMinUtilities[tind - 2] - ourMinUtilities[tind - 1] > 0;
		double returnVal = defaultVal;

		if (!negotiationSession.getUtilitySpace().isDiscounted()) {
			// first 10% is reserved for 0.98...
			if (tind > 2) {
				// if we havn't compromised in the last session
				if (!weMoved) {
					// we didn't move, they did
					if (theyMoved) {
						// return defaultVal;
					} else if (tind <= 16) {
						returnVal = defaultVal + 0.02;
						// give concession only if we think they conceded more!
						if (returnVal > minConcession * 2 / 3)
							returnVal = minConcession * 2 / 3;

					}
				}
			}

			// the negotiation is ending and they are not moving its time for
			// compromize
			if (tind > 16 && !theyMoved) {
				// Compromise another portion every time...
				returnVal = (concessionLeft - defaultVal) / (21 - tind) + defaultVal;
				if (returnVal > minConcession + 0.05)
					returnVal = minConcession + 0.05;
			}
			// return defaultVal;

		} else {
			double discountEstimate = d * 0.05;
			double expectedRoundGain = theirMaxUtilities[tind - 1] - theirMaxUtilities[tind - 2] - discountEstimate;

			if (tind <= 16) {
				returnVal = defaultVal + 0.02;
				if (defaultVal - expectedRoundGain > returnVal)
					returnVal = defaultVal - expectedRoundGain;
				if (returnVal > minConcession)
					returnVal = minConcession;

			} else {
				// Compromise another portion every time...
				returnVal = (concessionLeft - defaultVal) / (21 - tind) + defaultVal;
			}
		}
		// making a concession in steps. its slower but safer
		returnVal = defaultVal + (returnVal - defaultVal) * segPortion;
		return returnVal;
	}

	@Override
	public BidDetails determineOpeningBid() {
		return determineNextBid();
	}

	@Override
	public String getName() {
		return "2011 - ValueModelAgent";
	}
}