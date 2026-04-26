package agents.anac.y2014.DoNA;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Random;

import genius.core.Agent;
import genius.core.Bid;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.DefaultAction;
import genius.core.actions.EndNegotiation;
import genius.core.actions.Offer;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.IssueInteger;
import genius.core.issue.IssueReal;
import genius.core.issue.Value;
import genius.core.issue.ValueInteger;
import genius.core.issue.ValueReal;

/**
 * @author Eden Erez
 * @since 2014-03-21
 */
public class DoNA extends Agent {

	boolean isPrinting = false;
	ClearDefaultStrategy defaultStrategy;
	private Bid firstBidOfOpponent, lastBidOfOpponent;
	private Bid maxUtilityBid;
	private Bid minUtilityBid;
	private double maxUtility;
	private double minUtility;
	private double UtilityRange;
	private OpponentBidHistory opponentBidHistory;
	private double averageResponseTime;
	private double stiaResponseTime;
	int numOfRounds;
	boolean NotFirstTime;

	private double theTimeIAmReadyToInvestInNegotiation;
	private double theMotivationIHaveToInvestInNegotiation;

	private int countBid;
	private double MyOfferedUtility;
	boolean IConcedeLast;
	boolean IsPrintForDebug;
	int numberOfStepsDoneByOpponent;

	private Action actionOfPartner = null;
	private double stia = 0;
	HashMap<Double, Bid> hashBids;
	ArrayList<Double> arrSamples;

	HashMap<Double, Bid> otherHashBids;
	ArrayList<Double> otherArrSamples;

	HashMap<Double, Bid> alternativeHashBids;
	ArrayList<Double> alternativeArrSamples;

	double[] timeArr;
	int currentTimeIndex = 0;

	boolean highReservation;// = (utilitySpace.getReservationValue() >=
							// 0.8);//(minUtility+0.75*UtilityRange));
	boolean midReservation;// = (utilitySpace.getReservationValue() < 0.8 &&
							// utilitySpace.getReservationValue() >
							// 0.2);//(minUtility+0.75*UtilityRange) &&
							// utilitySpace.getReservationValue() >=
							// (minUtility+0.375*UtilityRange));
	boolean lowReservation;// = (utilitySpace.getReservationValue() <=
							// 0.2);//(minUtility+0.375*UtilityRange));

	boolean highDiscount;// = (utilitySpace.getDiscountFactor() >= 0.8);
	boolean midDiscount;// = (utilitySpace.getDiscountFactor() < 0.8 &&
						// utilitySpace.getDiscountFactor() > 0.2);
	boolean lowDiscount;// = (utilitySpace.getDiscountFactor() <= 0.2);
	double R, D;
	double UtilityFactor;
	double EndFactor;
	double TimeFactor;

	int numOfSamplesForAvgTime;
	double sumUtil = 0;
	int numOfUtil = 0;
	Action lastAction;
	double dblMinUtil = 0.0;
	double myMinUtil = 1;
	boolean flagStopExplor;
	int countOffer;
	long DomainSize;

	@Override
	public void init() {

		DomainSize = utilitySpace.getDomain().getNumberOfPossibleBids();

		if (DomainSize > 1 && DomainSize < 1000000) {
			defaultStrategy = new ClearDefaultStrategy();
			defaultStrategy.utilitySpace = utilitySpace;
			defaultStrategy.timeline = timeline;

			defaultStrategy.init(getAgentID());
			return;
		}

		highReservation = (utilitySpace.getReservationValue() >= 0.8);// (minUtility+0.75*UtilityRange));
		midReservation = (utilitySpace.getReservationValue() < 0.8
				&& utilitySpace.getReservationValue() > 0.2);// (minUtility+0.75*UtilityRange)
																// &&
																// utilitySpace.getReservationValue()
																// >=
																// (minUtility+0.375*UtilityRange));
		lowReservation = (utilitySpace.getReservationValue() <= 0.2);// (minUtility+0.375*UtilityRange));

		highDiscount = (utilitySpace.getDiscountFactor() >= 0.8);
		midDiscount = (utilitySpace.getDiscountFactor() < 0.8
				&& utilitySpace.getDiscountFactor() > 0.2);
		lowDiscount = (utilitySpace.getDiscountFactor() <= 0.2);
		R = utilitySpace.getReservationValue();
		D = utilitySpace.getDiscountFactor();

		int numOfSamples = 100000;
		numOfSamplesForAvgTime = 12;

		if ((highReservation) || (midReservation && lowDiscount)) {
			return;
		}

		try {
			if (utilitySpace.getReservationValue() > 0
					&& utilitySpace.getReservationValue() < 1
					&& utilitySpace.getDiscountFactor() < 1)
				theTimeIAmReadyToInvestInNegotiation = Math
						.sqrt(Math.sqrt(1 - utilitySpace.getReservationValue())
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

			numOfRounds = 0;
			firstBidOfOpponent = lastBidOfOpponent = null;
			opponentBidHistory = new OpponentBidHistory();
			opponentBidHistory
					.initializeDataStructures(utilitySpace.getDomain());
			NotFirstTime = false;
			countBid = 0;
			MyOfferedUtility = 2;
			IConcedeLast = false;
			IsPrintForDebug = false;

		} catch (Exception e) {
			System.out.println("initialization error" + e.getMessage());
		}

		numOfUtil = 0;
		sumUtil = 0;
		arrSamples = new ArrayList<Double>();
		hashBids = new HashMap<Double, Bid>();
		otherHashBids = new HashMap<Double, Bid>();
		otherArrSamples = new ArrayList<Double>();
		alternativeHashBids = new HashMap<Double, Bid>();
		alternativeArrSamples = new ArrayList<Double>();

		for (int i = 0; i < numOfSamples; i++) {
			lastAction = chooseRandomBidAction(0);
			Bid myBid = ((Offer) lastAction).getBid();
			// myBid = getBestNigberBid(myBid,0);
			double myOfferedUtil = getUtility(myBid);
			numOfUtil++;
			// System.out.println("numOfUtil: " + numOfUtil);
			sumUtil += myOfferedUtil;
			hashBids.put(myOfferedUtil, myBid);
			arrSamples.add(myOfferedUtil);
		}
		double avg = sumUtil / numOfUtil;
		if (isPrinting)
			System.out.println("m=" + (avg));

		double sumAll = 0;
		for (Double double1 : arrSamples) {
			sumAll += Math.pow(double1 - avg, 2);
		}
		stia = Math.pow(sumAll / numOfSamples, 0.5);
		if (isPrinting)
			System.out.println("S=" + stia);
		dblMinUtil = 2.4 * stia + avg * 1.2;

		opponentBidHistory = new OpponentBidHistory();
		opponentBidHistory.initializeDataStructures(utilitySpace.getDomain());

		Collections.sort(arrSamples);

		maxUtilityBid = hashBids.get(arrSamples.get(arrSamples.size() - 10));
		maxUtility = getUtility(maxUtilityBid);

		if (dblMinUtil > maxUtility)
			dblMinUtil = maxUtility;

		minUtilityBid = hashBids.get(arrSamples.get(arrSamples.size() - 100));
		minUtility = getUtility(minUtilityBid);

		if (dblMinUtil < minUtility)
			dblMinUtil = minUtility;

		maxUtilityBid = hashBids.get(arrSamples.get(arrSamples.size() - 1));
		minUtilityBid = hashBids.get(arrSamples.get(0));
		myMinUtil = 1;
		try {
			maxUtility = utilitySpace.getUtility(maxUtilityBid);
			minUtility = utilitySpace.getUtility(minUtilityBid);
			myMinUtil = maxUtility;

		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		lastAction = new Offer(getAgentID(), maxUtilityBid);
		countOffer = arrSamples.size() - 1;

		timeArr = new double[numOfSamplesForAvgTime];
		int currentTimeIndex = 0;

		theMotivationIHaveToInvestInNegotiation = dblMinUtil
				- utilitySpace.getReservationValue();
		UtilityFactor = (1 - (D - 0.2) * 5 / 3) * 0.1 + 0.05;
		EndFactor = (1 - R * 5 / 4) / (1 / (dblMinUtil - R));
		TimeFactor = 1 / (((D - 0.2) / 0.6) * (1 - EndFactor) + EndFactor);
		flagStopExplor = false;
	}

	@Override
	public void ReceiveMessage(Action opponentAction) {

		if (DomainSize > 1 && DomainSize < 1000000) {
			defaultStrategy.ReceiveMessage(opponentAction);
			return;
		}

		actionOfPartner = opponentAction;
		if (opponentAction instanceof Offer) {
			lastBidOfOpponent = ((Offer) opponentAction).getBid();
			if (firstBidOfOpponent == null)
				firstBidOfOpponent = lastBidOfOpponent;

			double remainingTime = timeline.getTime();
			double weight = remainingTime * 100;
			opponentBidHistory.updateOpponentModel(lastBidOfOpponent, weight,
					utilitySpace);

			numOfRounds++;

			double dbltime = timeline.getTime();
			timeArr[currentTimeIndex] = dbltime;
			currentTimeIndex++;
			if (currentTimeIndex == numOfSamplesForAvgTime) {
				currentTimeIndex = 0;
				averageResponseTime = (timeArr[numOfSamplesForAvgTime - 1]
						- timeArr[0]) / (numOfSamplesForAvgTime - 1);
				double sumAll = 0;
				for (int i = 1; i < numOfSamplesForAvgTime; i++) {
					sumAll += Math.pow(
							timeArr[i] - timeArr[i - 1] - averageResponseTime,
							2);
				}
				stiaResponseTime = Math
						.pow(sumAll / (numOfSamplesForAvgTime - 1), 0.5);

				NotFirstTime = true;
				// averageResponseTime = timeline.getTime() / numOfRounds;
			}

			numberOfStepsDoneByOpponent = opponentBidHistory
					.getNumberOfDistinctBids();
		}

	}

	int countRRR = 0;

	@Override
	public Action chooseAction() {

		// if(NotFirstTime==false)
		// return cleanEndNegotiation();

		if (DomainSize > 1 && DomainSize < 1000000) {
			return defaultStrategy.chooseAction();
		}

		if (highReservation || (midReservation && lowDiscount)
				|| theMotivationIHaveToInvestInNegotiation <= 0.1) {
			return cleanEndNegotiation();
		}

		double lastBidOfOpponentUtil = 0;
		try {
			lastBidOfOpponentUtil = lastBidOfOpponent == null ? 0
					: utilitySpace.getUtility(lastBidOfOpponent);
		} catch (Exception e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}

		if ((lastBidOfOpponentUtil > dblMinUtil
				|| lastBidOfOpponentUtil > myMinUtil)
				&& lastBidOfOpponentUtil > R) {
			if (isPrinting)
				System.out.println("Accept1 Util: " + lastBidOfOpponentUtil);
			if (R > utilitySpace.getUtilityWithDiscount(lastBidOfOpponent,
					timeline.getTime()))
				return cleanEndNegotiation();
			return new Accept(getAgentID(), lastBidOfOpponent);
		}

		double currentTime = timeline.getTime();

		// check BEST neighbor with highest utility
		if (actionOfPartner != null) {
			Bid Nbid = getBestNigberBid(
					DefaultAction.getBidFromAction(actionOfPartner), 0);
			double dblNbid = getUtility(Nbid);
			if (dblNbid > dblMinUtil && currentTime < 0.9) {
				alternativeHashBids.put(dblNbid, Nbid);
				alternativeArrSamples.add(dblNbid);
				Collections.sort(alternativeArrSamples);
			}
		}

		if (isPrinting)
			System.out.println("111111111111111111111");
		if (highDiscount) {
			if (NotFirstTime == false) {
				Bid curBid = hashBids.get(arrSamples.get(countOffer));
				lastAction = cleanOffer(curBid, lastBidOfOpponentUtil);
				return lastAction;
			}

			double utilOfBid = 0;
			double utilOflastBidOfOpponent = 0;
			double utilOfBestBidOfOppenent = 0;

			try {
				utilOfBid = utilitySpace.getUtility(lastBidOfOpponent);
				utilOflastBidOfOpponent = utilOfBid;
				utilOfBestBidOfOppenent = utilitySpace.getUtility(
						this.opponentBidHistory.getBestBidInHistory());
			} catch (Exception e) {
				System.out.println(
						"Exception in ChooseAction1:" + e.getMessage());

				lastAction = cleanOffer(
						hashBids.get(arrSamples.get(countOffer)),
						lastBidOfOpponentUtil); // best
												// guess
												// if
												// things
												// go
												// wrong.

			}

			int FactorOfAverageResponseTime = 1;
			Bid bid;
			if (currentTime >= 1 - FactorOfAverageResponseTime
					* (averageResponseTime + 2 * stiaResponseTime)) { // last
																		// last
																		// moment
				if (utilitySpace.getReservationValue() > utilOfBid) {
					if (isPrinting)
						System.out.println("Enddddddddddddd 1111111111111");
					return cleanEndNegotiation();
				} else {
					if (isPrinting)
						System.out.println("Accept2 Util: " + utilOfBid);
					if (R > utilitySpace.getUtilityWithDiscount(
							lastBidOfOpponent, timeline.getTime()))
						return cleanEndNegotiation();
					return new Accept(getAgentID(), lastBidOfOpponent);
				}
			} else if (currentTime >= 1 - Math.pow(2, 1)
					* FactorOfAverageResponseTime * averageResponseTime) { // one
																			// before
																			// last
																			// moment
				if (isPrinting)
					System.out.println(
							"averageResponseTime: " + averageResponseTime);
				if (isPrinting)
					System.out.println("Math.pow(2,1): " + Math.pow(2, 1));
				if (isPrinting)
					System.out.println("currentTime: " + currentTime + " / "
							+ (1 - Math.pow(2, 1) * FactorOfAverageResponseTime
									* averageResponseTime));
				if (utilitySpace.getReservationValue() > utilOfBestBidOfOppenent
						&& utilitySpace
								.getReservationValue() > utilOflastBidOfOpponent) {
					if (isPrinting)
						System.out.println("Enddddddddddddd  2222222222222");

					return cleanEndNegotiation();
				} else if (utilOflastBidOfOpponent >= utilOfBestBidOfOppenent) {
					if (isPrinting)
						System.out.println(
								"Accept3 Util: " + utilOflastBidOfOpponent);
					if (R > utilitySpace.getUtilityWithDiscount(
							lastBidOfOpponent, timeline.getTime()))
						return cleanEndNegotiation();

					return new Accept(getAgentID(), lastBidOfOpponent);
				} else {
					double dblutility = getUtility(
							this.opponentBidHistory.getBestBidInHistory());
					if (isPrinting)
						System.out.println("Offer best history: " + dblutility);
					return new Offer(getAgentID(),
							this.opponentBidHistory.getBestBidInHistory());
				}
			}
			if (isPrinting)
				System.out.println("222222222222222222222");

			return FastLastMomentStrategyWithoutEndNegotiation(
					dblMinUtil + 0.25 * stia, dblMinUtil, currentTime);
		}
		if (lowDiscount) {
			return FastStrategy();
		}

		if (NotFirstTime == false) {
			Bid curBid = hashBids.get(arrSamples.get(countOffer));
			lastAction = cleanOffer(curBid, lastBidOfOpponentUtil);
			return lastAction;
		}

		// System.out.println("TimeFactor: " + TimeFactor);
		currentTime = timeline.getTime() * TimeFactor;

		double utilOfBid = 0;
		double utilOflastBidOfOpponent = 0;
		double utilOfBestBidOfOppenent = 0;

		try {
			utilOfBid = utilitySpace.getUtility(lastBidOfOpponent);
			utilOflastBidOfOpponent = utilOfBid;
			utilOfBestBidOfOppenent = utilitySpace
					.getUtility(this.opponentBidHistory.getBestBidInHistory());
		} catch (Exception e) {
			System.out.println("Exception in ChooseAction2:" + e.getMessage());
			lastAction = cleanEndNegotiation(); // best guess if things go
												// wrong.
		}

		int FactorOfAverageResponseTime = 1;
		Bid bid;
		if (currentTime >= 1 - FactorOfAverageResponseTime
				* (averageResponseTime + 2 * stiaResponseTime)) { // last
																	// last
																	// moment
			if (utilitySpace.getReservationValue() > utilOfBid) {
				if (isPrinting)
					System.out.println("Enddddddddddddd  3333333333333");

				return cleanEndNegotiation();
			} else {
				if (isPrinting)
					System.out.println("Accept4 Util: " + utilOfBid);
				if (R > utilitySpace.getUtilityWithDiscount(lastBidOfOpponent,
						timeline.getTime()))
					return cleanEndNegotiation();

				return new Accept(getAgentID(), lastBidOfOpponent);
			}
		} else if (currentTime >= 1 - Math.pow(2, 1)
				* FactorOfAverageResponseTime * averageResponseTime) { // one
																		// before
																		// last
																		// moment
			if (utilitySpace.getReservationValue() > utilOfBestBidOfOppenent
					&& utilitySpace
							.getReservationValue() > utilOflastBidOfOpponent) {
				if (isPrinting)
					System.out.println("Enddddddddddddd  444444444444");

				return cleanEndNegotiation();
			} else if (utilOflastBidOfOpponent >= utilOfBestBidOfOppenent) {
				if (isPrinting)
					System.out.println(
							"Accept5 Util: " + utilOflastBidOfOpponent);
				if (R > utilitySpace.getUtilityWithDiscount(lastBidOfOpponent,
						timeline.getTime()))
					return cleanEndNegotiation();

				return new Accept(getAgentID(), lastBidOfOpponent);
			} else
				return new Offer(getAgentID(),
						this.opponentBidHistory.getBestBidInHistory());
		}

		return FastLastMomentStrategyWithoutEndNegotiation(
				dblMinUtil - UtilityFactor + 0.25 * stia,
				dblMinUtil - UtilityFactor, currentTime);

	}

	/**
	 * If the utility of the given bid is better than the reservation value -
	 * offer it. Otherwise - return EndNegotiation.
	 * 
	 * @throws Exception
	 */
	private Action EndNegotiationOrAcceptOrNewOfferr(Bid myOfferedBid)
			throws Exception {
		double lastBidOfOpponentUtility = (lastBidOfOpponent == null ? 0
				: utilitySpace.getUtility(lastBidOfOpponent));

		double MyOfferedUtility = utilitySpace.getUtility(myOfferedBid);

		Bid bestOpponentBid = this.opponentBidHistory.getBestBidInHistory();

		double bestOpponentBidUtility = (bestOpponentBid == null)
				? utilitySpace.getUtility(this.minUtilityBid)
				: utilitySpace.getUtility(bestOpponentBid);

		double endNegotiationUtility = utilitySpace.getReservationValue();

		// 1st priority - EndNegotiation is the best choice
		if (endNegotiationUtility >= MyOfferedUtility
				&& endNegotiationUtility >= bestOpponentBidUtility
				&& endNegotiationUtility >= lastBidOfOpponentUtility) {
			if (isPrinting)
				System.out.println("Enddddddddddddd  5555555555555555");

			return cleanEndNegotiation();
		}

		// 2nd priority - Accept (lastBidOfOpponentUtil) is the best choice
		else if (lastBidOfOpponentUtility >= bestOpponentBidUtility
				&& lastBidOfOpponentUtility >= MyOfferedUtility
				&& lastBidOfOpponentUtility >= endNegotiationUtility) {
			if (isPrinting)
				System.out.println("Accept6 Util: " + lastBidOfOpponentUtility);
			if (R > utilitySpace.getUtilityWithDiscount(lastBidOfOpponent,
					timeline.getTime()))
				return cleanEndNegotiation();

			return new Accept(getAgentID(), lastBidOfOpponent);
		}

		// 3rd priority - bestOpponentBidUtility is the best choice
		else if (bestOpponentBidUtility >= lastBidOfOpponentUtility
				&& bestOpponentBidUtility >= MyOfferedUtility
				&& bestOpponentBidUtility >= endNegotiationUtility) {
			if (isPrinting)
				System.out.println(
						"new Offer(this.opponentBidHistory.getBestBidInHistory())");
			return new Offer(getAgentID(),
					this.opponentBidHistory.getBestBidInHistory());
		}

		// 4th priority - myOfferedBid
		else {
			if (isPrinting)
				System.out.println("else");
			lastAction = cleanOffer(myOfferedBid, lastBidOfOpponentUtility);
			MyOfferedUtility = utilitySpace
					.getUtility(DefaultAction.getBidFromAction(lastAction));
			if (isPrinting)
				System.out.println(
						"else MyOfferedUtility 333: " + MyOfferedUtility);
			return lastAction;
		}
	}

	/**
	 * This strategy tries to achieve an agreement as fast as possible.
	 */
	public Action FastStrategy() {
		Bid curBid = hashBids.get(arrSamples.get(countOffer--));
		lastAction = new Offer(getAgentID(), curBid);
		try {
			return EndNegotiationOrAcceptOrNewOfferr(curBid);
		} catch (Exception e) {
			System.out.println("Exception in ChooseAction3:" + e.getMessage());
			return cleanEndNegotiation();
		}
	}

	@Override
	public String getName() {
		return "DoNA";
	}

	private Bid getRandomBid(double minUtil) throws Exception {
		HashMap<Integer, Value> values = new HashMap<Integer, Value>(); // pairs
																		// <issuenumber,chosen
																		// value
																		// string>
		List<Issue> issues = utilitySpace.getDomain().getIssues();
		Random randomnr = new Random();

		double tmpMinUtil = 0, currentUtil = 0;
		// create a random bid with utility>MINIMUM_BID_UTIL.
		// note that this may never succeed if you set MINIMUM too high!!!
		// in that case we will search for a bid till the time is up (3 minutes)
		// but this is just a simple agent.
		Bid bid = null;
		int countTimes = 0;
		do {
			countTimes++;
			tmpMinUtil = UpdateMinUtil(minUtil);
			for (Issue lIssue : issues) {
				switch (lIssue.getType()) {
				case DISCRETE:
					IssueDiscrete lIssueDiscrete = (IssueDiscrete) lIssue;
					int optionIndex = randomnr
							.nextInt(lIssueDiscrete.getNumberOfValues());
					values.put(lIssue.getNumber(),
							lIssueDiscrete.getValue(optionIndex));
					break;
				case REAL:
					IssueReal lIssueReal = (IssueReal) lIssue;
					int optionInd = randomnr.nextInt(
							lIssueReal.getNumberOfDiscretizationSteps() - 1);
					values.put(lIssueReal.getNumber(), new ValueReal(lIssueReal
							.getLowerBound()
							+ (lIssueReal.getUpperBound()
									- lIssueReal.getLowerBound()) * (optionInd)
									/ (lIssueReal
											.getNumberOfDiscretizationSteps())));
					break;
				case INTEGER:
					IssueInteger lIssueInteger = (IssueInteger) lIssue;
					int optionIndex2 = lIssueInteger.getLowerBound()
							+ randomnr.nextInt(lIssueInteger.getUpperBound()
									- lIssueInteger.getLowerBound());
					values.put(lIssueInteger.getNumber(),
							new ValueInteger(optionIndex2));
					break;
				default:
					throw new Exception("issue type " + lIssue.getType()
							+ " not supported by SimpleAgent2");
				}
			}
			bid = new Bid(utilitySpace.getDomain(), values);
			// System.out.println("sumUtil/numOfUtil: " + (sumUtil/numOfUtil));
			currentUtil = getUtility(bid);
			// System.out.println("currentUtil: " + currentUtil + " / " +
			// tmpMinUtil);
			if (currentUtil >= dblMinUtil) {
				otherHashBids.put(currentUtil, bid);
				otherArrSamples.add(currentUtil);
			}
			if (countTimes >= 150000)
				flagStopExplor = true;
		} while (currentUtil < tmpMinUtil && otherArrSamples.size() < 500
				&& (flagStopExplor == false));

		if (otherArrSamples.size() > 0) {
			Collections.sort(otherArrSamples);
			double tmpUtil = otherArrSamples.get(otherArrSamples.size() - 1);
			bid = otherHashBids.get(tmpUtil);
			otherArrSamples.remove(tmpUtil);
			otherHashBids.remove(tmpUtil);
		} else {
			if (currentUtil < tmpMinUtil) {
				bid = hashBids.get(arrSamples.get(arrSamples.size() - 1));
			}
		}
		return bid;
	}

	private double UpdateMinUtil(double minUtil) {
		double k = 0.01;
		double time = timeline.getTime();
		return minUtil - (time / k) * 0.01 * stia * 0.5;
	}

	private Action chooseRandomBidAction(double minUtil) {
		double lastBidOfOpponentUtil = lastBidOfOpponent == null ? 0
				: getUtility(lastBidOfOpponent);
		Bid nextBid = null;
		try {
			nextBid = getRandomBid(minUtil);
		} catch (Exception e) {
			System.out.println("Problem with received bid:" + e.getMessage()
					+ ". cancelling bidding");
		}
		if (nextBid == null)
			return cleanOffer(
					hashBids.get(arrSamples.get(arrSamples.size() - 1)),
					lastBidOfOpponentUtil);
		if (lastBidOfOpponent == null)
			return new Offer(getAgentID(), nextBid);
		else {
			if (isPrinting)
				System.out
						.println("chooseRandomBidAction-lastBidOfOpponentUtil: "
								+ lastBidOfOpponentUtil);
			return cleanOffer(nextBid, lastBidOfOpponentUtil);
		}
	}

	private Bid getBestNigberBid(Bid currentBid, int deep) {
		HashMap<Integer, Value> tmpvalues;
		List<Issue> issues = utilitySpace.getDomain().getIssues();

		double secondMaxUtil = 0;
		double maxUtil = 0;
		Bid secondBestBid = null;
		Bid bestBid = null;
		Bid newBid;
		double newUtil;
		int numOfValues;
		for (Issue lIssue : issues) {
			switch (lIssue.getType()) {
			case DISCRETE:
				IssueDiscrete lIssueDiscrete = (IssueDiscrete) lIssue;
				numOfValues = lIssueDiscrete.getNumberOfValues();
				for (int optionIndex = 0; optionIndex < numOfValues; optionIndex++) {

					newBid = currentBid;
					newBid = newBid.putValue(lIssue.getNumber(),
							lIssueDiscrete.getValue(optionIndex));
					if (deep > 0) {
						newBid = getBestNigberBid(newBid, deep - 1);
					}
					newUtil = getUtility(newBid);
					if (newUtil >= maxUtil) {
						secondMaxUtil = maxUtil;
						maxUtil = newUtil;
						secondBestBid = bestBid;
						bestBid = newBid;
					} else if (newUtil >= secondMaxUtil) {
						secondMaxUtil = newUtil;
						secondBestBid = newBid;
					}
				}
				break;
			case REAL:
				IssueReal lIssueReal = (IssueReal) lIssue;
				numOfValues = lIssueReal.getNumberOfDiscretizationSteps() - 1;
				for (int optionInd = 0; optionInd < numOfValues; optionInd++) {
					newBid = currentBid;
					newBid = newBid.putValue(lIssue.getNumber(), new ValueReal(
							lIssueReal.getLowerBound() + (lIssueReal
									.getUpperBound()
									- lIssueReal.getLowerBound()) * (optionInd)
									/ (lIssueReal
											.getNumberOfDiscretizationSteps())));
					if (deep > 0) {
						newBid = getBestNigberBid(newBid, deep - 1);
					}
					newUtil = getUtility(newBid);
					if (newUtil >= maxUtil) {
						secondMaxUtil = maxUtil;
						maxUtil = newUtil;
						secondBestBid = bestBid;
						bestBid = newBid;
					} else if (newUtil >= secondMaxUtil) {
						secondMaxUtil = newUtil;
						secondBestBid = newBid;
					}
				}
				break;
			case INTEGER:
				IssueInteger lIssueInteger = (IssueInteger) lIssue;

				numOfValues = lIssueInteger.getUpperBound()
						- lIssueInteger.getLowerBound();
				for (int optionIndex2 = lIssueInteger
						.getLowerBound(); optionIndex2 < numOfValues; optionIndex2++) {
					newBid = currentBid;
					newBid = newBid.putValue(lIssue.getNumber(),
							new ValueInteger(optionIndex2));
					if (deep > 0) {
						newBid = getBestNigberBid(newBid, deep - 1);
					}
					newUtil = getUtility(newBid);
					if (newUtil >= maxUtil) {
						secondMaxUtil = maxUtil;
						maxUtil = newUtil;
						secondBestBid = bestBid;
						bestBid = newBid;
					} else if (newUtil >= secondMaxUtil) {
						secondMaxUtil = newUtil;
						secondBestBid = newBid;
					}
				}
				break;

			}
		}
		if (bestBid.equals(currentBid))
			return secondBestBid;
		else
			return bestBid;
	}

	public Action FastStrategyWithoutEndNegotiation(double VirtulReservation) {
		if (myMinUtil > VirtulReservation)
			myMinUtil = VirtulReservation;

		try {
			double lastBidOfOpponentUtil = lastBidOfOpponent == null ? 0
					: utilitySpace.getUtility(lastBidOfOpponent);
			Bid curBid = hashBids.get(arrSamples.get(countOffer));
			lastAction = cleanOffer(curBid, lastBidOfOpponentUtil);

			double bestOpponentBidUtility = utilitySpace
					.getUtility(this.opponentBidHistory.getBestBidInHistory());
			// lastBidOfOpponentUtil is the best choice
			if ((lastBidOfOpponentUtil > MyOfferedUtility
					|| lastBidOfOpponentUtil >= VirtulReservation)
					&& lastBidOfOpponentUtil > R) {
				if (isPrinting)
					System.out
							.println("Accept7 Util: " + lastBidOfOpponentUtil);
				if (R > utilitySpace.getUtilityWithDiscount(lastBidOfOpponent,
						timeline.getTime()))
					return cleanEndNegotiation();

				return new Accept(getAgentID(), lastBidOfOpponent);
			}
			// bestOpponentBidUtility is the best choice
			else if (bestOpponentBidUtility > MyOfferedUtility
					|| bestOpponentBidUtility >= VirtulReservation) {
				if (isPrinting)
					System.out.println("bestOpponentBidUtility: "
							+ bestOpponentBidUtility + " MyOfferedUtility: "
							+ MyOfferedUtility + " VirtulReservation: "
							+ VirtulReservation);
				return new Offer(getAgentID(),
						this.opponentBidHistory.getBestBidInHistory());
			}
			// Reservation is the best choice
			else if (VirtulReservation > MyOfferedUtility) {
				if (isPrinting)
					System.out.println("MyOfferedUtility 111: ");
				double dbltime = timeline.getTime();

				if (flagStopExplor == false && dbltime < 0.90) {
					if (isPrinting)
						System.out.println("flagStopExplor==false");
					lastAction = chooseRandomBidAction(VirtulReservation);
				} else {

					if (otherArrSamples.size() > 0) {
						if (isPrinting)
							System.out.println("otherArrSamples.size()>0");

						Collections.sort(otherArrSamples);
						double tmpUtil = otherArrSamples
								.get(otherArrSamples.size() - 1);
						curBid = otherHashBids.get(tmpUtil);
						otherArrSamples.remove(tmpUtil);
						otherHashBids.remove(tmpUtil);

						if (curBid == null) {
							if (isPrinting)
								System.out.println("curBid==null");

							int tmpDif = arrSamples.size() - countOffer;
							double randNum = Math.random();
							if (isPrinting)
								System.out.println("countOffer: " + countOffer
										+ " tmpDif: " + tmpDif + " randNum: "
										+ randNum);
							int randIndex = ((int) (randNum * tmpDif))
									+ countOffer;
							curBid = hashBids.get(arrSamples.get(randIndex));
							if (isPrinting)
								System.out.println("randIndex: " + randIndex);
							if (isPrinting)
								System.out.println("curBid: " + curBid);
						}
					} else {
						if (isPrinting)
							System.out.println("otherArrSamples.size()>0 else");

						int tmpDif = arrSamples.size() - countOffer;
						double randNum = Math.random();
						if (isPrinting)
							System.out.println(
									"countOffer: " + countOffer + " tmpDif: "
											+ tmpDif + " randNum: " + randNum);
						int randIndex = ((int) (randNum * tmpDif)) + countOffer;
						curBid = hashBids.get(arrSamples.get(randIndex));
						if (isPrinting)
							System.out.println("randIndex: " + randIndex);
						if (isPrinting)
							System.out.println("curBid: " + curBid);

					}
					lastAction = cleanOffer(curBid, lastBidOfOpponentUtil);
				}
				MyOfferedUtility = utilitySpace
						.getUtility(DefaultAction.getBidFromAction(lastAction));
				if (isPrinting)
					System.out.println(
							"MyOfferedUtility 111: " + MyOfferedUtility);

				return lastAction;
			} else {
				if (isPrinting)
					System.out.println("MyOfferedUtility 222: ");

				countOffer--;
				lastAction = cleanOffer(curBid, lastBidOfOpponentUtil);
				MyOfferedUtility = utilitySpace
						.getUtility(DefaultAction.getBidFromAction(lastAction));
				if (isPrinting)
					System.out.println(
							"MyOfferedUtility 222: " + MyOfferedUtility);

				return lastAction;
			}
		} catch (Exception e) {
			System.out.println("Exception in ChooseAction4:" + e.getMessage());
			return FastStrategy();
		}
	}

	public Action FastLastMomentStrategyWithoutEndNegotiation(
			double StartMinUtil, double MinUtil, double dblMyRoundTime) {

		int constForNumOfRounds = 100;
		double Delta = (StartMinUtil - MinUtil) / 11;
		// double MinUtil = 0.75;
		// double dblMyRoundTime =
		// timeline.getTime()/(1.1*(1-utilitySpace.getReservationValue()));
		// double dblMyRoundTime = timeline.getTime();
		int FactorOfAverageResponseTime = 1;
		if (dblMyRoundTime >= 1 - constForNumOfRounds
				* FactorOfAverageResponseTime * averageResponseTime) { // Other
																		// before-last
																		// moments
																		// -
																		// logarithmic
																		// scale
			if (isPrinting)
				System.out.println("dblMyRoundTime 1111");
			return FastStrategyWithoutEndNegotiation(MinUtil);
		} else if (dblMyRoundTime >= 1 - constForNumOfRounds
				* FactorOfAverageResponseTime * averageResponseTime) {
			if (isPrinting)
				System.out.println("dblMyRoundTime 2222");
			return FastStrategyWithoutEndNegotiation(MinUtil + Delta);
		} else if (dblMyRoundTime >= 1 - constForNumOfRounds
				* FactorOfAverageResponseTime * averageResponseTime) {
			if (isPrinting)
				System.out.println("dblMyRoundTime 33333");
			return FastStrategyWithoutEndNegotiation(MinUtil + 2 * Delta);
		} else if (dblMyRoundTime >= 1 - constForNumOfRounds
				* FactorOfAverageResponseTime * averageResponseTime) {
			if (isPrinting)
				System.out.println("dblMyRoundTime 44444");
			return FastStrategyWithoutEndNegotiation(MinUtil + 3 * Delta);
		} else if (dblMyRoundTime >= 1 - constForNumOfRounds
				* FactorOfAverageResponseTime * averageResponseTime) {
			if (isPrinting)
				System.out.println("dblMyRoundTime 55555");
			return FastStrategyWithoutEndNegotiation(MinUtil + 4 * Delta);
		} else if (dblMyRoundTime >= 1 - constForNumOfRounds
				* FactorOfAverageResponseTime * averageResponseTime) {
			if (isPrinting)
				System.out.println("dblMyRoundTime 6666");
			return FastStrategyWithoutEndNegotiation(MinUtil + 5 * Delta);
		} else if (dblMyRoundTime >= 1 - constForNumOfRounds
				* FactorOfAverageResponseTime * averageResponseTime) {
			if (isPrinting)
				System.out.println("dblMyRoundTime 7777");
			return FastStrategyWithoutEndNegotiation(MinUtil + 6 * Delta);
		} else if (dblMyRoundTime >= 1 - constForNumOfRounds
				* FactorOfAverageResponseTime * averageResponseTime) {
			if (isPrinting)
				System.out.println("dblMyRoundTime 88888");
			return FastStrategyWithoutEndNegotiation(MinUtil + 7 * Delta);
		} else if (dblMyRoundTime >= 1 - constForNumOfRounds
				* FactorOfAverageResponseTime * averageResponseTime) {
			if (isPrinting)
				System.out.println("dblMyRoundTime 9999");
			return FastStrategyWithoutEndNegotiation(MinUtil + 8 * Delta);
		} else if (dblMyRoundTime >= 1 - constForNumOfRounds
				* FactorOfAverageResponseTime * averageResponseTime) {
			if (isPrinting)
				System.out.println("dblMyRoundTime 111 000");
			return FastStrategyWithoutEndNegotiation(MinUtil + 9 * Delta);
		} else if (dblMyRoundTime >= 1 - constForNumOfRounds
				* FactorOfAverageResponseTime * averageResponseTime) {
			if (isPrinting)
				System.out.println("dblMyRoundTime 111 111");
			return FastStrategyWithoutEndNegotiation(MinUtil + 10 * Delta);
		} else {
			if (isPrinting)
				System.out.println("dblMyRoundTime 111 222");
			return FastStrategyWithoutEndNegotiation(StartMinUtil);
		}
	}

	public Action cleanOffer(Bid curBid, double lastBidOfOpponentUtil) {
		try {
			MyOfferedUtility = utilitySpace.getUtility(curBid);
			boolean flagViol = false;
			while (MyOfferedUtility <= lastBidOfOpponentUtil) {
				flagViol = true;
				countOffer--;
				if (countOffer == -1) {
					System.out.println("countOffer==-1");
					break;
				}
				curBid = hashBids.get(arrSamples.get(countOffer));
				lastAction = new Offer(getAgentID(), curBid);
				MyOfferedUtility = utilitySpace.getUtility(curBid);
			}
			if (flagViol) {
				int tmpSize = arrSamples.size() - 1 - countOffer;
				arrSamples = new ArrayList<Double>();
				for (double key : hashBids.keySet()) {
					arrSamples.add(key);
				}
				if (countOffer == -1) {
					System.out.println("countOffer==-1");
					hashBids.get(arrSamples.get(arrSamples.size() - 1));
				}
				countOffer = arrSamples.size() - 1 - tmpSize;
				Collections.sort(arrSamples);
				curBid = hashBids.get(arrSamples.get(countOffer));
				lastAction = new Offer(getAgentID(), curBid);
				MyOfferedUtility = utilitySpace.getUtility(curBid);
			}
			if (MyOfferedUtility < dblMinUtil) {
				countOffer--;
				curBid = hashBids.get(arrSamples.get(countOffer));
				lastAction = new Offer(getAgentID(), curBid);
				MyOfferedUtility = utilitySpace.getUtility(curBid);
			}
			if (MyOfferedUtility <= lastBidOfOpponentUtil
					|| MyOfferedUtility < R) {
				if (isPrinting)
					System.out.println("Enddddddddddddd  7777777777777777");

				return cleanEndNegotiation();
			}
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			System.out.println(e.getMessage());
			// System.out.println("nnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn");
			return bestFromOpponent();

		}

		try {
			MyOfferedUtility = utilitySpace.getUtility(curBid);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return new Offer(getAgentID(), curBid);
	}

	private Action cleanEndNegotiation() {
		// TODO Auto-generated method stub
		java.util.Random generateRandom = new java.util.Random();
		int randIndex = generateRandom.nextInt(countOffer - arrSamples.size());
		if (randIndex == countOffer)
			return new Offer(getAgentID(),
					hashBids.get(arrSamples.get(randIndex)));
		else
			return new EndNegotiation(getAgentID());
	}

	private Action bestFromOpponent() {
		double dblUtilVal = alternativeArrSamples
				.get(alternativeArrSamples.size() - 1);
		Bid lastBid = alternativeHashBids.get(dblUtilVal);

		alternativeHashBids.remove(dblUtilVal);
		alternativeArrSamples.remove(dblUtilVal);

		if (R > utilitySpace.getUtilityWithDiscount(lastBid,
				timeline.getTime()))
			return cleanEndNegotiation();
		else
			return new Offer(getAgentID(), lastBid);
	}

	@Override
	public String getDescription() {
		return "ANAC2014 compatible with non-linear utility spaces";
	}
}
