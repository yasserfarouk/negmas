package agents.anac.y2014.DoNA;

import java.util.ArrayList;
import java.util.List;

import genius.core.AgentID;
import genius.core.Bid;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.EndNegotiation;
import genius.core.actions.Offer;
import genius.core.bidding.BidDetails;
import genius.core.boaframework.SortedOutcomeSpace;
import genius.core.misc.Range;
import genius.core.timeline.TimeLineInfo;
import genius.core.utility.AbstractUtilitySpace;

//1.  Check Utility is not 1. Multiple with MaxUtility
//2.  CHECK DISTANCE BETWEEN AVERAGE RESPONSE AND MINIMUM RESPONSE

//  a. How many are opponent sequence steps made?  
//  b. How many rounds, which the opponent stay, are made?
//  c. When did the opponent go?
//  d  how many bids, from the opponent space-bids are made?

/**
 * An agent that waits for the last moment and then offers something.
 *
 * @author Eden Erez, Erel Segal haLevi
 * @since 2013-01-13
 */
public class ClearDefaultStrategy {
	public static AbstractUtilitySpace utilitySpace;
	public static TimeLineInfo timeline;

	private Bid firstBidOfOpponent, lastBidOfOpponent;
	private Bid maxUtilityBid;
	private Bid minUtilityBid;
	private double maxUtility;
	private double minUtility;
	private double UtilityRange;
	private OpponentBidHistory opponentBidHistory;
	private double averageResponseTime;
	private SortedOutcomeSpace outcomeSpace;
	int numOfRounds;
	boolean NotFirstTime;
	List<BidDetails> SortedBid;

	private double theTimeIAmReadyToInvestInNegotiation;
	private double theFastUtilityInvestInNegotiation;

	private int countBid;
	private double MyOfferedUtility;
	boolean IConcedeLast;
	boolean IsPrintForDebug;
	int numberOfStepsDoneByOpponent;
	private AgentID agentID;

	public void init(AgentID id) {
		this.agentID = id;
		try {
			if (utilitySpace.getReservationValue() > 0 && utilitySpace.getReservationValue() < 1
					&& utilitySpace.getDiscountFactor() < 1)
				theTimeIAmReadyToInvestInNegotiation = Math.sqrt(Math.sqrt(1 - utilitySpace.getReservationValue())
						* Math.sqrt(utilitySpace.getDiscountFactor()));
			/*
			 * (1-utilitySpace.getReservationValue())
			 * 
			 * Math.pow(utilitySpace.getDiscountFactor(),2)
			 * 
			 * 2;
			 */
			else
				theTimeIAmReadyToInvestInNegotiation = 1.0;
			theFastUtilityInvestInNegotiation = (1 - Math.sqrt(
					Math.sqrt(1 - utilitySpace.getDiscountFactor()) * Math.sqrt(utilitySpace.getReservationValue())))
					* 0.33 + 0.62;

			System.out.println("theTimeIAmReadyToInvestInRiskyNegotiation=" + theTimeIAmReadyToInvestInNegotiation);

			numOfRounds = 0;
			maxUtilityBid = utilitySpace.getMaxUtilityBid();
			minUtilityBid = utilitySpace.getMinUtilityBid();
			maxUtility = utilitySpace.getUtility(maxUtilityBid);
			minUtility = utilitySpace.getUtility(minUtilityBid);
			UtilityRange = maxUtility - minUtility;
			firstBidOfOpponent = lastBidOfOpponent = null;
			opponentBidHistory = new OpponentBidHistory();
			opponentBidHistory.initializeDataStructures(utilitySpace.getDomain());
			outcomeSpace = new SortedOutcomeSpace(utilitySpace);
			NotFirstTime = false;
			SortedBid = outcomeSpace.getAllOutcomes();
			/*
			 * for (int i = 0; i < SortedBid.size(); i++) { BidDetails
			 * bidDetails= SortedBid.get(i);
			 * System.out.println("getMyUndiscountedUtil:" +
			 * bidDetails.getMyUndiscountedUtil()); }
			 */
			countBid = 0;
			MyOfferedUtility = 2;
			IConcedeLast = false;
			IsPrintForDebug = false;

		} catch (Exception e) {
			System.out.println("initialization error" + e.getMessage());
		}
	}

	public void ReceiveMessage(Action opponentAction) {
		if (opponentAction instanceof Offer) {
			lastBidOfOpponent = ((Offer) opponentAction).getBid();
			if (firstBidOfOpponent == null)
				firstBidOfOpponent = lastBidOfOpponent;

			double remainingTime = theTimeIAmReadyToInvestInNegotiation - timeline.getTime();
			double weight = remainingTime * 100;
			opponentBidHistory.updateOpponentModel(lastBidOfOpponent, weight, utilitySpace);

			numOfRounds++;
			averageResponseTime = timeline.getTime() / numOfRounds;
			numberOfStepsDoneByOpponent = opponentBidHistory.getNumberOfDistinctBids();

			// System.out.println("BIU: time="+timeline.getTime()+"
			// #bids="+numOfRounds+" averageResponseTime="+averageResponseTime);
		}
	}

	/**
	 * @param RoundLeft
	 *            - estimation of the number of rounds left until the time we
	 *            decided to quit. Logarithmic scale (1/2, 1/4, 1/8..).
	 * @param MinUtil
	 *            the minimum utility we want to get from this nego
	 * @return
	 */
	public Bid middleBid(int RoundLeft, double MinUtil) throws Exception {
		final double MAX_ROUNDS = 12.0; // log scale
		/*
		 * double explorationSurface = SortedBid.size()*(RoundLeft/10);
		 * System.out
		 * .println(Integer.parseInt(Double.toString(explorationSurface)));
		 * return
		 * SortedBid.get(Integer.parseInt(Double.toString(explorationSurface
		 * ))).getBid();
		 */
		// System.out.println(utilitySpace.getDiscountFactor() + ", " +
		// utilitySpace.getReservationValue());
		double utilOfMyBestBid = maxUtility;
		double utilOfBestOpponentBid = utilitySpace.getUtility(this.opponentBidHistory.getBestBidInHistory());
		// double constFactor =
		// ((1-0.6+utilitySpace.getDiscountFactor())*RoundLeft)/10 +
		// 0.6*(utilitySpace.getDiscountFactor());

		double weightOfOurBestBid = UtilityRange * (RoundLeft / MAX_ROUNDS) + MinUtil; // larger
																						// constFactor
																						// -
																						// larger
																						// weight
																						// to
																						// our
																						// best
																						// bid.
		// When RoundLeft=10, the weight is 1 - we take ONLY our best bid.
		// When RoundLeft=0, the weight is MinUtil
		// When RoundLeft=5, it is the average between 1 and MinUtil.
		// double averageUtility = 0.8*utilOfMyBid + 0.2*utilOfOpponentBid;
		double averageUtility = weightOfOurBestBid * utilOfMyBestBid + (1 - weightOfOurBestBid) * utilOfBestOpponentBid;

		/*
		 * Range r = new Range(averageUtility,1); List<BidDetails> RangeBid =
		 * outcomeSpace.getBidsinRange(r);
		 * 
		 * int RndInx =
		 * Integer.parseInt(Double.toString((Math.random()*RangeBid.size())));
		 * return RangeBid.get(RndInx).getBid();
		 */

		List<BidDetails> bidDetailsWeAreWillingToAccept = outcomeSpace.getBidsinRange(new Range(averageUtility, 1.0));
		List<Bid> bidsWeAreWillingToAccept = new ArrayList<Bid>();
		for (BidDetails bd : bidDetailsWeAreWillingToAccept)
			bidsWeAreWillingToAccept.add(bd.getBid());
		Bid bid = opponentBidHistory.ChooseBid(bidsWeAreWillingToAccept, utilitySpace.getDomain());
		if (bid == null) {

			try {
				Bid tmp = outcomeSpace.getBidNearUtility(averageUtility).getBid();
				return tmp;
			} catch (Exception e) {
				System.out.println("***********************Exception");
				return SortedBid.get(0).getBid();
			}
		} else {
			return bid;
		}
	}

	/**
	 * This will be returned when there is exception in other strategies
	 */
	public Action ExceptionStrategy() {
		return this.FastLastMomentStrategyWithoutEndNegotiation(1, 0, timeline.getTime());
	}

	/**
	 * If the utility of the given bid is better than the reservation value -
	 * offer it. Otherwise - return EndNegotiation.
	 * 
	 * @throws Exception
	 */
	private Action EndNegotiationOrAcceptOrNewOfferr(Bid myOfferedBid) throws Exception {
		double lastBidOfOpponentUtility = (lastBidOfOpponent == null ? 0 : utilitySpace.getUtility(lastBidOfOpponent));

		double MyOfferedUtility = utilitySpace.getUtility(myOfferedBid);

		Bid bestOpponentBid = this.opponentBidHistory.getBestBidInHistory();

		double bestOpponentBidUtility = (bestOpponentBid == null) ? utilitySpace.getUtility(this.minUtilityBid)
				: utilitySpace.getUtility(bestOpponentBid);

		double endNegotiationUtility = utilitySpace.getReservationValue();

		// 1st priority - EndNegotiation is the best choice
		if (endNegotiationUtility >= MyOfferedUtility && endNegotiationUtility >= bestOpponentBidUtility
				&& endNegotiationUtility >= lastBidOfOpponentUtility) {
			return new EndNegotiation(agentID);
		}

		// 2nd priority - Accept (lastBidOfOpponentUtil) is the best choice
		else if (lastBidOfOpponentUtility >= bestOpponentBidUtility && lastBidOfOpponentUtility >= MyOfferedUtility
				&& lastBidOfOpponentUtility >= endNegotiationUtility) {
			return new Accept(agentID, lastBidOfOpponent);
		}

		// 3rd priority - bestOpponentBidUtility is the best choice
		else if (bestOpponentBidUtility >= lastBidOfOpponentUtility && bestOpponentBidUtility >= MyOfferedUtility
				&& bestOpponentBidUtility >= endNegotiationUtility) {
			return new Offer(agentID, this.opponentBidHistory.getBestBidInHistory());
		}

		// 4th priority - myOfferedBid
		else
			return new Offer(agentID, myOfferedBid);
	}

	/**
	 * This strategy counts the number of steps the opponent has done towards us
	 * (= distinct bids), and does the same number of steps towards him.
	 */
	public Action OneStepStrategy() {
		try {
			Bid bid = SortedBid.get(numberOfStepsDoneByOpponent).getBid();
			return EndNegotiationOrAcceptOrNewOfferr(bid);
		} catch (Exception e) {
			System.out.println("OneStepStrategy: Exception in ChooseAction:" + e.getMessage());
			return ExceptionStrategy(); // terminate if anything goes wrong.
		}
	}

	public Action RandomStepStrategy() {
		try {
			Bid bid = SortedBid.get(numberOfStepsDoneByOpponent).getBid();
			double MyOfferedUtility = utilitySpace.getUtility(bid);
			List<BidDetails> bidDetailsWeAreWillingToAccept = outcomeSpace
					.getBidsinRange(new Range(MyOfferedUtility, 1.0));
			List<Bid> bidsWeAreWillingToAccept = new ArrayList<Bid>();
			for (BidDetails bd : bidDetailsWeAreWillingToAccept)
				bidsWeAreWillingToAccept.add(bd.getBid());
			Bid bid2 = opponentBidHistory.ChooseBid(bidsWeAreWillingToAccept, utilitySpace.getDomain());
			if (bid2 == null) {

				try {
					Bid tmp = outcomeSpace.getBidNearUtility(MyOfferedUtility).getBid();
					return EndNegotiationOrAcceptOrNewOfferr(tmp);
				} catch (Exception e) {
					System.out.println("***********************Exception");
					return EndNegotiationOrAcceptOrNewOfferr(SortedBid.get(0).getBid());
				}
			} else {
				return EndNegotiationOrAcceptOrNewOfferr(bid2);
			}

		} catch (Exception e) {
			System.out.println("OneStepStrategy: Exception in ChooseAction:" + e.getMessage());
			return ExceptionStrategy(); // terminate if anything goes wrong.
		}
	}

	public Action OneStepStrategy(double ratio) {
		try {
			int bidNumber = (int) (ratio * numberOfStepsDoneByOpponent);
			Bid bid = SortedBid.get(bidNumber + 1).getBid();
			// System.out.println("bid: " + bid.toString());
			return EndNegotiationOrAcceptOrNewOfferr(bid);
		} catch (Exception e) {
			System.out.println("OneStepStrategy: Exception in ChooseAction:" + e.getMessage());
			return ExceptionStrategy(); // terminate if anything goes wrong.
		}
	}

	/**
	 * This strategy tries to achieve an agreement as fast as possible.
	 */
	public Action FastStrategy() {
		try {
			return EndNegotiationOrAcceptOrNewOfferr(SortedBid.get(countBid++).getBid());
		} catch (Exception e) {
			System.out.println("Exception in ChooseAction:" + e.getMessage());
			return new EndNegotiation(agentID);
		}
	}

	public Action FastStrategyWithoutEndNegotiation(double VirtulReservation) {
		try {
			double lastBidOfOpponentUtil = lastBidOfOpponent == null ? 0 : utilitySpace.getUtility(lastBidOfOpponent);
			MyOfferedUtility = utilitySpace.getUtility(SortedBid.get(countBid).getBid());
			double bestOpponentBidUtility = utilitySpace.getUtility(this.opponentBidHistory.getBestBidInHistory());
			// lastBidOfOpponentUtil is the best choice
			if (lastBidOfOpponentUtil >= bestOpponentBidUtility && lastBidOfOpponentUtil >= MyOfferedUtility
					&& lastBidOfOpponentUtil >= VirtulReservation) {
				return new Accept(agentID, lastBidOfOpponent);
			}
			// bestOpponentBidUtility is the best choice
			else if (bestOpponentBidUtility >= lastBidOfOpponentUtil && bestOpponentBidUtility >= MyOfferedUtility
					&& bestOpponentBidUtility >= VirtulReservation) {
				return new Offer(agentID, this.opponentBidHistory.getBestBidInHistory());
			}
			// Reservation is the best choice
			else if (VirtulReservation >= MyOfferedUtility && VirtulReservation >= bestOpponentBidUtility
					&& VirtulReservation >= lastBidOfOpponentUtil) {
				Bid bid = SortedBid.get(countBid).getBid();
				double MyOfferedUtility = utilitySpace.getUtility(bid);
				List<BidDetails> bidDetailsWeAreWillingToAccept = outcomeSpace
						.getBidsinRange(new Range(MyOfferedUtility, 1.0));
				List<Bid> bidsWeAreWillingToAccept = new ArrayList<Bid>();
				for (BidDetails bd : bidDetailsWeAreWillingToAccept)
					bidsWeAreWillingToAccept.add(bd.getBid());
				Bid bid2 = opponentBidHistory.ChooseBid(bidsWeAreWillingToAccept, utilitySpace.getDomain());
				if (bid2 == null) {

					try {
						return EndNegotiationOrAcceptOrNewOfferr(bid);
					} catch (Exception e) {
						System.out.println("***********************Exception");
						return EndNegotiationOrAcceptOrNewOfferr(SortedBid.get(0).getBid());
					}
				} else {
					return new Offer(agentID, bid2);
				}
				// return new Offer(SortedBid.get(countBid).getBid());
				// return new EndNegotiation();
			} else
				return new Offer(agentID, SortedBid.get(countBid++).getBid());

		} catch (Exception e) {
			System.out.println("Exception in ChooseAction:" + e.getMessage());
			return FastStrategy();
		}
	}

	public Action FastLastMomentStrategyWithoutEndNegotiation(double StartMinUtil, double MinUtil,
			double dblMyRoundTime) {
		double Delta = (StartMinUtil - MinUtil) / 11;
		// double MinUtil = 0.75;
		// double dblMyRoundTime =
		// timeline.getTime()/(1.1*(1-utilitySpace.getReservationValue()));
		// double dblMyRoundTime = timeline.getTime();
		int FactorOfAverageResponseTime = 1;
		if (dblMyRoundTime >= 1 - Math.pow(2, 2) * FactorOfAverageResponseTime * averageResponseTime) { // Other
																										// before-last
																										// moments
																										// -
																										// logarithmic
																										// scale
			return FastStrategyWithoutEndNegotiation(MinUtil);
		} else if (dblMyRoundTime >= 1 - Math.pow(2, 3) * FactorOfAverageResponseTime * averageResponseTime) {
			return FastStrategyWithoutEndNegotiation(MinUtil + Delta);
		} else if (dblMyRoundTime >= 1 - Math.pow(2, 4) * FactorOfAverageResponseTime * averageResponseTime) {
			return FastStrategyWithoutEndNegotiation(MinUtil + 2 * Delta);
		} else if (dblMyRoundTime >= 1 - Math.pow(2, 5) * FactorOfAverageResponseTime * averageResponseTime) {
			return FastStrategyWithoutEndNegotiation(MinUtil + 3 * Delta);
		} else if (dblMyRoundTime >= 1 - Math.pow(2, 6) * FactorOfAverageResponseTime * averageResponseTime) {
			return FastStrategyWithoutEndNegotiation(MinUtil + 4 * Delta);
		} else if (dblMyRoundTime >= 1 - Math.pow(2, 7) * FactorOfAverageResponseTime * averageResponseTime) {
			return FastStrategyWithoutEndNegotiation(MinUtil + 5 * Delta);
		} else if (dblMyRoundTime >= 1 - Math.pow(2, 8) * FactorOfAverageResponseTime * averageResponseTime) {
			return FastStrategyWithoutEndNegotiation(MinUtil + 6 * Delta);
		} else if (dblMyRoundTime >= 1 - Math.pow(2, 9) * FactorOfAverageResponseTime * averageResponseTime) {
			return FastStrategyWithoutEndNegotiation(MinUtil + 7 * Delta);
		} else if (dblMyRoundTime >= 1 - Math.pow(2, 10) * FactorOfAverageResponseTime * averageResponseTime) {
			return FastStrategyWithoutEndNegotiation(MinUtil + 8 * Delta);
		} else if (dblMyRoundTime >= 1 - Math.pow(2, 11) * FactorOfAverageResponseTime * averageResponseTime) {
			return FastStrategyWithoutEndNegotiation(MinUtil + 9 * Delta);
		} else if (dblMyRoundTime >= 1 - Math.pow(2, 12) * FactorOfAverageResponseTime * averageResponseTime) {
			return FastStrategyWithoutEndNegotiation(MinUtil + 10 * Delta);
		} else {
			return FastStrategyWithoutEndNegotiation(StartMinUtil);
		}
	}

	public Action LastMomentStrategy(double MinUtil, double dblMyRoundTime) {
		try {
			int FactorOfAverageResponseTime = 2;
			Bid bid;
			if (dblMyRoundTime >= 1 - FactorOfAverageResponseTime * averageResponseTime) { // last
																							// last
																							// moment
				double utilOfBid = utilitySpace.getUtility(lastBidOfOpponent);
				if (utilitySpace.getReservationValue() > utilOfBid) {
					if (IsPrintForDebug)
						System.out.println("***********EndNegotiation");
					return new EndNegotiation(agentID);
				} else
					return new Accept(agentID, lastBidOfOpponent);

			} else if (dblMyRoundTime >= 1 - Math.pow(2, 1) * FactorOfAverageResponseTime * averageResponseTime) { // one
																													// before
																													// last
																													// moment
				double utilOflastBidOfOpponent = utilitySpace.getUtility(lastBidOfOpponent);
				double utilOfBestBidOfOppenent = utilitySpace.getUtility(this.opponentBidHistory.getBestBidInHistory());
				if (utilitySpace.getReservationValue() > utilOfBestBidOfOppenent
						&& utilitySpace.getReservationValue() > utilOflastBidOfOpponent) {
					if (IsPrintForDebug)
						System.out.println("***********EndNegotiation1");
					return new EndNegotiation(agentID);
				} else if (utilOflastBidOfOpponent >= utilOfBestBidOfOppenent)
					return new Accept(agentID, lastBidOfOpponent);
				else
					return new Offer(agentID, this.opponentBidHistory.getBestBidInHistory());
			} else if (dblMyRoundTime >= 1 - Math.pow(2, 2) * FactorOfAverageResponseTime * averageResponseTime) { // Other
																													// before-last
																													// moments
																													// -
																													// logarithmic
																													// scale
				return FastStrategy();
			} else if (dblMyRoundTime >= 1 - Math.pow(2, 3) * FactorOfAverageResponseTime * averageResponseTime) {
				return FastStrategy();
			} else if (dblMyRoundTime >= 1 - Math.pow(2, 4) * FactorOfAverageResponseTime * averageResponseTime) {
				bid = middleBid(3, MinUtil);
			} else if (dblMyRoundTime >= 1 - Math.pow(2, 5) * FactorOfAverageResponseTime * averageResponseTime) {
				bid = middleBid(4, MinUtil);
			} else if (dblMyRoundTime >= 1 - Math.pow(2, 6) * FactorOfAverageResponseTime * averageResponseTime) {
				bid = middleBid(5, MinUtil);
			} else if (dblMyRoundTime >= 1 - Math.pow(2, 7) * FactorOfAverageResponseTime * averageResponseTime) {
				bid = middleBid(6, MinUtil);
			} else if (dblMyRoundTime >= 1 - Math.pow(2, 8) * FactorOfAverageResponseTime * averageResponseTime) {
				bid = middleBid(7, MinUtil);
			} else if (dblMyRoundTime >= 1 - Math.pow(2, 9) * FactorOfAverageResponseTime * averageResponseTime) {
				bid = middleBid(8, MinUtil);
			} else if (dblMyRoundTime >= 1 - Math.pow(2, 10) * FactorOfAverageResponseTime * averageResponseTime) {
				bid = middleBid(9, MinUtil);
			} else if (dblMyRoundTime >= 1 - Math.pow(2, 11) * FactorOfAverageResponseTime * averageResponseTime) {
				bid = middleBid(10, MinUtil);
			} else if (dblMyRoundTime >= 1 - Math.pow(2, 12) * FactorOfAverageResponseTime * averageResponseTime) {
				bid = middleBid(11, MinUtil);
			} else {
				double utilOfOurBestBid = maxUtility;
				if (utilitySpace.getReservationValue() > utilOfOurBestBid) {
					if (IsPrintForDebug)
						System.out.println("***********EndNegotiation2");
					return new EndNegotiation(agentID);
				} else {
					return new Offer(agentID, this.maxUtilityBid);
				}
			}
			double utilOflastBidOfOpponent = utilitySpace.getUtility(lastBidOfOpponent);
			double utilOfBid = utilitySpace.getUtility(bid);
			if (utilitySpace.getReservationValue() > utilOfBid
					&& utilitySpace.getReservationValue() > utilOflastBidOfOpponent) {
				if (IsPrintForDebug)
					System.out.println("***********EndNegotiation3");
				return new EndNegotiation(agentID);
			} else if (utilOflastBidOfOpponent > utilOfBid) {
				return new Accept(agentID, lastBidOfOpponent);
			} else {
				return new Offer(agentID, bid);
			}
		} catch (Exception e) {
			System.out.println("LastMomentStrategy: Exception in ChooseAction:" + e.getMessage());
			e.printStackTrace();
			return ExceptionStrategy();
		}
	}

	public Action chooseAction() {
		try {

			System.out.println("Domain size: " + utilitySpace.getDomain().getNumberOfPossibleBids());
			boolean highReservation = (utilitySpace.getReservationValue() >= 0.75);// (minUtility+0.75*UtilityRange));
			boolean midReservation = (utilitySpace.getReservationValue() < 0.75
					&& utilitySpace.getReservationValue() > 0.25);// (minUtility+0.75*UtilityRange)
																	// &&
																	// utilitySpace.getReservationValue()
																	// >=
																	// (minUtility+0.375*UtilityRange));
			boolean lowReservation = (utilitySpace.getReservationValue() <= 0.25);// (minUtility+0.375*UtilityRange));

			boolean highDiscount = (utilitySpace.getDiscountFactor() >= 0.75);
			boolean midDiscount = (utilitySpace.getDiscountFactor() < 0.75 && utilitySpace.getDiscountFactor() > 0.25);
			boolean lowDiscount = (utilitySpace.getDiscountFactor() <= 0.25);
			try {
				double currentTime = timeline.getTime() / (theTimeIAmReadyToInvestInNegotiation / 1.2);
				// System.out.println("currentTime: " + currentTime);
				double MyNextOfferedUtility = utilitySpace.getUtility(SortedBid.get(countBid).getBid());

				if (highDiscount) {
					return LastMomentStrategy(0.75, timeline.getTime());
				} else if (highReservation) {
					return new EndNegotiation(agentID);
				} else if (lowReservation) {
					return FastStrategy();
				} else if (midDiscount && lowReservation) {
					return OneStepStrategy();
				} else {
					if (currentTime <= 0.25 && MyNextOfferedUtility >= 0.95) {
						return FastStrategyWithoutEndNegotiation(0.95);
					} else if (currentTime <= 0.5 && MyNextOfferedUtility >= 0.85) {
						return FastLastMomentStrategyWithoutEndNegotiation(1, 0.85, currentTime * 2);
					} else if (currentTime > 0.75) {
						return LastMomentStrategy(0.65, currentTime);
					} else {
						return OneStepStrategy();
					}
				}

			} catch (Exception e) {
				System.out.println("Exception in ChooseAction:" + e.getMessage());
				return ExceptionStrategy(); // terminate if anything goes wrong.
			}
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			return ExceptionStrategy();
		}
	}

	public String getName() {
		return getClass().getSimpleName();
	}
}
