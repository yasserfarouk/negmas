package agents.anac.y2011.ValueModelAgent;

import java.util.Random;

import genius.core.Agent;
import genius.core.Bid;
import genius.core.BidIterator;
import genius.core.SupportedNegotiationSetting;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.ActionWithBid;
import genius.core.actions.Offer;
import genius.core.utility.AdditiveUtilitySpace;

//This agent uses a very complex form of temporal difference reinforcement
//learning to learn opponent's utility space.
//The learning is focused on finding the amount of utility lost by
//opponent for each value.
//However, as the bid (expected) utilities represent the decrease in all
//issues, we needed away to decide which values should change the most.
//We use estimations of standard deviation and reliability to decide how
//to make the split.
//The reliability is also used to decide the learning factor of the individual
//learning.
//The agent then tries to slow down the compromise rate, until we have to be fair.

//This is the agent of Asaf, Dror and Gal.
public class ValueModelAgent extends Agent {
	private ValueModeler opponentUtilModel = null;
	private BidList allBids = null;
	private BidList approvedBids = null;
	private BidList iteratedBids = null;
	private Action actionOfPartner = null;
	private Action myLastAction = null;
	private Bid myLastBid = null;
	private int bidCount;
	public BidList opponentBids;
	public BidList ourBids;
	private OpponentModeler opponent;
	private double lowestAcceptable;
	private double lowestApproved;
	public double opponentStartbidUtil;
	public double opponentMaxBidUtil;
	private Bid opponentMaxBid;
	public double myMaximumUtility;
	private int amountOfApproved;
	public double noChangeCounter;
	private boolean retreatMode;
	private double concessionInOurUtility;
	private double concessionInTheirUtility;
	private ValueSeperatedBids seperatedBids;
	private final boolean TEST_EQUIVALENCE = false;
	Random random100;
	Random random200;
	int round = 0;

	private Action lAction = null;

	private double opponentUtil;

	@Override
	public void init() {
		@SuppressWarnings("unused")
		// it's not unused, check space type.
		AdditiveUtilitySpace a = (AdditiveUtilitySpace) utilitySpace;
		opponentUtilModel = null;
		allBids = null;
		approvedBids = null;
		actionOfPartner = null;
		bidCount = 0;
		opponentBids = new BidList();
		ourBids = new BidList();
		iteratedBids = new BidList();
		seperatedBids = new ValueSeperatedBids();
		lowestAcceptable = 0.7;
		lowestApproved = 1;
		amountOfApproved = 0;
		opponentMaxBidUtil = 0;
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

	@Override
	public void ReceiveMessage(Action opponentAction) {
		round++;
		actionOfPartner = opponentAction;
	}

	// remember our new bid
	private void bidSelected(BidWrapper bid) {
		bid.sentByUs = true;
		ourBids.addIfNew(bid);
		myLastBid = bid.bid;
		if (opponentUtilModel != null)
			seperatedBids.bidden(bid.bid, bidCount);
	}

	// remember our new bid in all needed data structures
	private void bidSelectedByOpponent(Bid bid) {
		BidWrapper opponentBid = new BidWrapper(bid, utilitySpace,
				myMaximumUtility);
		opponentBid.lastSentBid = bidCount;
		opponentBid.sentByThem = true;
		if (opponentBids.addIfNew(opponentBid)) {
			noChangeCounter = 0;
		} else {
			noChangeCounter++;
		}
		try {
			double opponentUtil = utilitySpace.getUtility(bid)
					/ myMaximumUtility;
			if (opponentMaxBidUtil < opponentUtil) {
				opponentMaxBidUtil = opponentUtil;
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
				ValueDecrease val = opponentUtilModel
						.utilityLoss(opponentBid.bid);
				concession = val.getDecrease();
				if (concession > concessionInTheirUtility)
					concessionInTheirUtility = concession;
				seperatedBids.bidden(bid, bidCount);

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
		boolean added = amountOfApproved != i;
		amountOfApproved = i;
		return added;
	}

	@Override
	public Action chooseAction() {
		Bid opponentBid = null;
		lAction = null;
		try {
			// our first bid, initializing

			if (allBids == null) {
				allBids = new BidList();
				approvedBids = new BidList();
				opponentUtilModel = new ValueModeler();
				seperatedBids.init((AdditiveUtilitySpace) utilitySpace,
						opponentUtilModel);
				myMaximumUtility = utilitySpace
						.getUtility(utilitySpace.getMaxUtilityBid());
				BidIterator iter = new BidIterator(utilitySpace.getDomain());
				while (iter.hasNext()) {
					Bid tmpBid = iter.next();
					try {
						BidWrapper wrap = new BidWrapper(tmpBid, utilitySpace,
								myMaximumUtility);
						allBids.bids.add(wrap);
						// }
					} catch (Exception ex) {
						ex.printStackTrace();
						BidWrapper wrap = new BidWrapper(tmpBid, utilitySpace,
								myMaximumUtility);
						allBids.bids.add(wrap);
					}
				} // while
				allBids.sortByOurUtil();

				setApprovedThreshold(0.98, false);

				iteratedBids.bids.add(allBids.bids.get(0));
			}
			// first bid is the highest bid
			if (bidCount == 0) {
				lAction = new Offer(getAgentID(), allBids.bids.get(0).bid);
				bidSelected(allBids.bids.get(0));
			}

			// treat opponent's offer
			if (actionOfPartner instanceof Offer) {
				opponentBid = ((Offer) actionOfPartner).getBid();
				opponentUtil = utilitySpace.getUtility(opponentBid)
						/ myMaximumUtility;
				bidSelectedByOpponent(opponentBid);

				if (opponent == null) {
					opponentStartbidUtil = opponentUtil;
					opponent = new OpponentModeler(bidCount,
							(AdditiveUtilitySpace) utilitySpace, timeline,
							ourBids, opponentBids, opponentUtilModel, allBids,
							this);
					opponentUtilModel.initialize(
							(AdditiveUtilitySpace) utilitySpace, opponentBid);
					approvedBids.sortByOpponentUtil(opponentUtilModel);
				} else {
					opponent.tick();
					if (noChangeCounter == 0) {
						double opponentExpectedBidValue = opponent
								.guessCurrentBidUtil();
						opponentUtilModel.assumeBidWorth(opponentBid,
								1 - opponentExpectedBidValue, 0.02);
					}

				}
				// it seems like I should try to accept
				// his best bid (assuming its still on the table)
				// if its a good enough bid to be reasonable
				// (we don't accept 0.3,0.92 around here...)
				// currently by reasonable we mean that we will
				// get most of what would be a "fair" distribution
				// of the utility

				if (timeline.getTime() > 0.9 && timeline.getTime() <= 0.96) {
					chickenGame(0.039, 0, 0.7);
				}
				if (timeline.getTime() > 0.96 && timeline.getTime() <= 0.97) {
					chickenGame(0.019, 0.5, 0.65);

				}
				if (timeline.getTime() > 0.98 && timeline.getTime() <= 0.99) {

					if (opponentUtil >= lowestApproved - 0.01) {
						return new Accept(getAgentID(), opponentBid);
					}
					if (bidCount % 5 == 0) {
						exploreScan();
					} else
						bestScan();
				}
				if (timeline.getTime() > 0.99 && timeline.getTime() <= 0.995) {
					chickenGame(0.004, 0.8, 0.6);

				}

				if (timeline.getTime() > 0.995) {

					if (opponentMaxBidUtil > 0.55) {

						// they might have a bug and not accept, so
						// if their offer is close enough accept
						if (opponentUtil >= opponentMaxBidUtil * 0.99) {
							return new Accept(getAgentID(), opponentBid);
						} else
							return new Offer(getAgentID(), opponentMaxBid);
					}
					bestScan();
					// this will probably not work but what can we
					// do? he dosn't even give us 50%!
				}

				if (lAction != null) {
					myLastAction = lAction;
					bidCount++;
					return lAction;
				}

				// if our opponent settled enough for us we accept, and there is
				// a discount factor we accept
				// if(opponent.expectedDiscountRatioToConvergence()*opponentUtil
				// > lowestApproved){
				if ((opponentUtil > lowestApproved)// || opponentUtil>0.98)
						&& (utilitySpace.isDiscounted()
								|| opponentUtil > 0.975)) {
					return new Accept(getAgentID(), opponentBid);
				}

				// otherwise we try to stretch it out
				if (opponentUtil > lowestApproved * 0.99
						&& !utilitySpace.isDiscounted()) {
					lowestApproved += 0.01;
					setApprovedThreshold(lowestApproved, true);
					retreatMode = true;
				}
				if (bidCount > 0 && bidCount < 4) {

					lAction = new Offer(getAgentID(), allBids.bids.get(0).bid);
					bidSelected(allBids.bids.get(bidCount));
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
					double minConcession = concession < concession2 ? concession
							: concession2;
					minConcession = minConcessionMaker(minConcession,
							1 - opponentMaxBidUtil);

					if (minConcession > (1 - lowestApproved)) {
						if (lowestAcceptable > (1 - minConcession)) {
							lowestApproved = lowestAcceptable;
						} else {
							lowestApproved = 1 - minConcession;
						}
						if (lowestApproved < opponentMaxBidUtil)
							lowestApproved = opponentMaxBidUtil + 0.001;

						if (setApprovedThreshold(lowestApproved, false)) {
							approvedBids.sortByOpponentUtil(opponentUtilModel);
						}
					}

					if (bidCount % 5 == 0) {
						exploreScan();
					} else
						bestScan();
				}

			}

			if (lAction == null) {
				lAction = myLastAction;
			}
		} catch (Exception e) {
			if (myLastBid == null) {
				try {
					return new Offer(getAgentID(),
							utilitySpace.getMaxUtilityBid());
				} catch (Exception e2) {
					return new Accept(getAgentID(), opponentBid);
				}
			}
			lAction = new Offer(getAgentID(), myLastBid);
		}
		myLastAction = lAction;
		bidCount++;
		return lAction;
	}

	public void bestScan() {
		approvedBids.sortByOpponentUtil(opponentUtilModel);

		// find the "best" bid for opponent
		// and choose it if we didn't send it to opponent
		for (int i = 0; i < approvedBids.bids.size(); i++) {
			BidWrapper tempBid = approvedBids.bids.get(i);
			if (!tempBid.sentByUs) {
				lAction = new Offer(getAgentID(), tempBid.bid);
				bidSelected(tempBid);
				break;
			}
		}
		if (lAction == null) {
			int maxIndex = approvedBids.bids.size() / 4;
			BidWrapper tempBid = approvedBids.bids
					.get((int) (random200.nextDouble() * maxIndex));
			lAction = new Offer(getAgentID(), tempBid.bid);
			bidSelected(tempBid);
		}
	}

	public void exploreScan() {
		BidWrapper tempBid = seperatedBids.explore(bidCount);
		if (tempBid != null) {
			lAction = new Offer(getAgentID(), tempBid.bid);
			bidSelected(tempBid);
		} else
			bestScan();
	}

	public void chickenGame(double timeToGive, double concessionPortion,
			double acceptableThresh) {
		// set timeToGive to be 0.005 unless each turn is very
		// large
		// double timeToGive =
		// 0.005>(opponent.delta*4)?0.005:(opponent.delta*4);
		double concessionLeft = lowestApproved - opponentMaxBidUtil;
		double planedThresh = lowestApproved
				- concessionPortion * concessionLeft;
		if (acceptableThresh > planedThresh)
			planedThresh = acceptableThresh;
		setApprovedThreshold(planedThresh, false);
		approvedBids.sortByOpponentUtil(opponentUtilModel);

		if (opponentUtil >= planedThresh - 0.01) {
			lAction = new Accept(getAgentID(),
					((ActionWithBid) actionOfPartner).getBid());
			return;
		}
		if (1.0 - timeline.getTime() - timeToGive > 0) {
			if (!TEST_EQUIVALENCE) {
				sleep(1.0 - timeline.getTime() - timeToGive);
			}
		}
		if (retreatMode || opponentMaxBidUtil >= planedThresh - 0.01) {
			lAction = new Offer(getAgentID(), opponentMaxBid);
		}
		// offer him the best bid for him amongst the
		// bids that are above our limit
		else {
			approvedBids.bids.get(0);
			// return new Offer(getAgentID(), approvedBids.bids.get(0).bid);
		}
	}

	double theirMaxUtilities[] = new double[21];
	double ourMinUtilities[] = new double[21];

	private double minConcessionMaker(double minConcession,
			double concessionLeft) {
		theirMaxUtilities[0] = opponentStartbidUtil;
		ourMinUtilities[0] = 1;
		double t = timeline.getTime();
		int tind = (int) (timeline.getTime() * 20) + 1;
		double segPortion = (t - (tind - 1) * 0.05) / 0.05;
		if (ourMinUtilities[tind] == 0)
			ourMinUtilities[tind] = 1;
		if (ourMinUtilities[tind - 1] == 0)
			ourMinUtilities[tind - 1] = lowestApproved;
		if (lowestApproved < ourMinUtilities[tind])
			ourMinUtilities[tind] = lowestApproved;
		if (opponentMaxBidUtil > theirMaxUtilities[tind])
			theirMaxUtilities[tind] = opponentMaxBidUtil;
		double d = utilitySpace.getDiscountFactor();
		double defaultVal = 1 - ourMinUtilities[tind - 1];
		if (tind == 1 || tind >= 19)
			return defaultVal;
		if (ourMinUtilities[tind - 2] == 0)
			ourMinUtilities[tind - 2] = lowestApproved;
		// if(defaultVal>minConcession) return minConcession;
		boolean theyMoved = theirMaxUtilities[tind]
				- theirMaxUtilities[tind - 2] > 0.01;
		boolean weMoved = ourMinUtilities[tind - 2]
				- ourMinUtilities[tind - 1] > 0;
		double returnVal = defaultVal;

		if (!utilitySpace.isDiscounted()) {
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
				returnVal = (concessionLeft - defaultVal) / (21 - tind)
						+ defaultVal;
				if (returnVal > minConcession + 0.05)
					returnVal = minConcession + 0.05;

			}
			// return defaultVal;

		} else {
			double discountEstimate = d * 0.05;
			double expectedRoundGain = theirMaxUtilities[tind - 1]
					- theirMaxUtilities[tind - 2] - discountEstimate;

			if (tind <= 16) {
				returnVal = defaultVal + 0.02;
				if (defaultVal - expectedRoundGain > returnVal)
					returnVal = defaultVal - expectedRoundGain;
				if (returnVal > minConcession)
					returnVal = minConcession;

			} else {
				// Compromise another portion every time...
				returnVal = (concessionLeft - defaultVal) / (21 - tind)
						+ defaultVal;
			}
		}
		// making a concession in steps. its slower but safer
		returnVal = defaultVal + (returnVal - defaultVal) * segPortion;
		return returnVal;
	}

	@Override
	public String getName() {
		return "Value Model Agent";
	}

	@Override
	public SupportedNegotiationSetting getSupportedNegotiationSetting() {
		return SupportedNegotiationSetting.getLinearUtilitySpaceInstance();
	}

	@Override
	public String getDescription() {
		return "ANAC2011";
	}
}