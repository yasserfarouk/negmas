package agents.anac.y2015.Phoenix;

/**
 * @author Max Lam @ CUHK
 *
 */

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Random;

import agents.Jama.Matrix;
import agents.anac.y2015.Phoenix.GP.CovLINard;
import agents.anac.y2015.Phoenix.GP.CovNoise;
import agents.anac.y2015.Phoenix.GP.CovSum;
import agents.anac.y2015.Phoenix.GP.CovarianceFunction;
import agents.anac.y2015.Phoenix.GP.GaussianProcess;
import genius.core.AgentID;
import genius.core.Bid;
import genius.core.BidIterator;
import genius.core.Domain;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.EndNegotiation;
import genius.core.actions.NoAction;
import genius.core.actions.Offer;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.Value;
import genius.core.parties.AbstractNegotiationParty;
import genius.core.parties.NegotiationInfo;

public class PhoenixParty extends AbstractNegotiationParty {
	// ///////////////////////////////////////////////////////////////////
	// //////////////////// Define Needed Variables //////////////////////
	// ///////////////////////////////////////////////////////////////////

	PrintStream logTxt, OriOut;
	private boolean debug = false;
	private boolean cctime = true;
	private double startTime = 0;

	private Action opponentAction = null;
	private Random rand = null;
	private Bid myLastBid = null;
	private double myLastBidUtility;
	private double myLastBidTime;
	private Bid opponentPreviousBid1 = null;
	private Bid opponentPreviousBid = null;
	private Bid opponentBestOfferBid1 = null;
	private Bid opponentFirstBid1 = null;
	private double opponentPreviousBidUtility1;
	private double opponentPreviousBidUtility;
	private double opponentPreviousBidTime1;
	private Bid bestReceivedBid1 = null;
	private double bestReceivedBidUtility1;
	private Bid opponentPreviousBid2 = null;
	private Bid opponentBestOfferBid2 = null;
	private Bid opponentFirstBid2 = null;
	private double opponentPreviousBidUtility2;
	private double opponentPreviousBidTime2;
	private Bid bestReceivedBid2 = null;
	private double bestReceivedBidUtility2;
	private Bid maxBid = null;
	private double maxBidUtility;
	private Bid minBid = null;
	private double minBidUtility;
	private ArrayList<Double> allBids = null;
	private ArrayList<Bid> opponentBids = null;
	private ArrayList<Bid> myPreviousBids = null;
	private double opponentNoOfferTime = 10;
	private double compaction, compaction2;
	private double trainingTimeSeperator = 0.05;
	private int timeDivisionSize = 400;
	private int utilityDivisionSize = 100;
	private double a = 3, b = 12;
	private double meanOfOpponentUtilities;
	private double varianceOfOpponentUtilities;
	private double estimatedMean;
	private double selfVar;
	private double sumOfOpponentBidUtilities = 0;
	private int randomSeeCreate;
	private double firstAcceptTime = 1.0;
	Domain domain;
	double[][] freqFunc;
	int countOpp = 0, lastCountOpp = 0;
	Offer[] roundOffers = null;
	int TimeToPrint;
	String fileName;
	int markNo = 0;

	// ///////////////////////////////////////////////////////////////////
	// /////////////// Public Functions Overriding Agent /////////////////
	// ///////////////////////////////////////////////////////////////////

	@Override
	public void init(NegotiationInfo info) {
		super.init(info);
		init();
	}

	public void init() {
		try {

			if (getNumberOfParties() > 0) {
				roundOffers = new Offer[getNumberOfParties() - 1];
			}
			startTime = timeline.getCurrentTime();
			domain = utilitySpace.getDomain();
			issues = domain.getIssues();
			markNo = 0;
			countOpp = 0;
			firstAcceptTime = 1.0;
			fileName = getName();
			lastCountOpp = 0;
			currentTimeIndex = 0;
			alpha = 3.0; // impact of delta over time
			beta = 1.5; // concession factor
			omega = 1.25; // weight reflects
			epsilon = 0.7; // controls influence of lambda
			xi = 0.95; // maximum concession
			gamma = 1.0;
			opponentNoOfferTime = 30;
			trainingTimeSeperator = 0.01;
			timeDivisionSize = 200;
			utilityDivisionSize = 45;
			ppp = 0;
			int maxSIze = 0;
			for (Integer i = 0; i < issues.size(); i++) {
				IssueDiscrete issue = (IssueDiscrete) issues.get(i);
				if (issue.getNumberOfValues() > maxSIze) {
					maxSIze = issue.getNumberOfValues();
				}
			}
			freqFunc = new double[issues.size()][maxSIze];
			for (Integer i = 0; i < issues.size(); i++) {
				for (int j = 0; j < maxSIze; j++) {
					freqFunc[i][j] = 1.0;
				}
			}
			a = 3;
			b = 12;
			rand = new Random(randomSeeCreate);
			lastTrainingIndex = 0;
			sumOfOpponentBidUtilities = 0;
			randomSeeCreate = 123;
			compareX = 0;
			fixedInputSize = 40;
			rangeofPrediction = 10;
			timeDivision = 5;
			tryNewBidTime = 0;
			lastTrainingTime = 0;
			phiAcceptanceBound = 0.5;
			allBids = new ArrayList<Double>();
			maxUtilitiesAsInputData = new ArrayList<Double>();
			timePointsAsOutputData = new ArrayList<Double>();
			delta = utilitySpace.getDiscountFactor();
			theta = utilitySpace.getReservationValue();
			System.out.println("theta=" + theta);
			System.out.println("delta=" + (delta));
			currentTimeIndex = 0;
			opponentAction = null;
			fileName = getName() + hashCode() + ".txt";
			logTxt = new PrintStream(new FileOutputStream(fileName, false));
			logTxt.println("  ");
			myLastBid = null;
			opponentPreviousBid1 = null;
			bestReceivedBid1 = null;
			opponentFirstBid1 = null;
			opponentPreviousBid2 = null;
			bestReceivedBid2 = null;
			opponentFirstBid2 = null;
			maxBid = utilitySpace.getMaxUtilityBid();
			minBid = utilitySpace.getMinUtilityBid();
			maxBidUtility = utilitySpace.getUtility(maxBid);
			minBidUtility = utilitySpace.getUtility(minBid);
			myLastBidUtility = maxBidUtility;
			uTargetForAcceptance = maxBidUtility;
			uTargetForBidding = maxBidUtility;
			opponentBids = new ArrayList<Bid>();
			myPreviousBids = new ArrayList<Bid>();
			compaction = 1 / (maxBidUtility - minBidUtility);
			a += compaction - varianceOfOpponentUtilities;
			generateTimeSamples(timeDivisionSize);
			saveInTermsOfTime = new timeList[timeDivisionSize];
			for (int i = 0; i < timeDivisionSize; i++) {
				saveInTermsOfTime[i] = new timeList();
			}
			fixedInputSize = (int) Math.min(
					utilitySpace.getDomain().getNumberOfPossibleBids(),
					fixedInputSize);
			if (utilitySpace.getDomain().getNumberOfPossibleBids() < 50) {
				if (utilitySpace.getDomain().getNumberOfPossibleBids() <= 5) {
					utilityDivisionSize = (int) utilitySpace.getDomain()
							.getNumberOfPossibleBids() / 2;
				} else if (utilitySpace.getDomain()
						.getNumberOfPossibleBids() <= 11) {
					utilityDivisionSize = (int) utilitySpace.getDomain()
							.getNumberOfPossibleBids() / 2;
				} else {
					utilityDivisionSize = (int) utilitySpace.getDomain()
							.getNumberOfPossibleBids() / 3;
				}
			} else {
				utilityDivisionSize = Math.min(
						(int) (utilitySpace.getDomain()
								.getNumberOfPossibleBids() / 15),
						utilityDivisionSize);
			}
			generateUtilitySamples(utilityDivisionSize);
			saveInTermsOfUtility = new BidList[utilityDivisionSize];
			for (int i = 0; i < utilityDivisionSize; i++) {
				saveInTermsOfUtility[i] = new BidList();
			}
			covFunc = new CovSum(1, new CovLINard(1), new CovNoise());
			// covFunc = new CovSum(1,new CovNNone(), new CovNoise());
			// CovSEiso cf = new CovSEiso();
			gp = new GaussianProcess(covFunc);
			double[][] logtheta0 = new double[][] { { 0.1 }, { 0.2 }
					// {Math.log(0.1)}
			};
			params0 = new Matrix(logtheta0);
			// setPartyId(getPartyId());
			estimatedMean = minBidUtility
					+ (minBidUtility + maxBidUtility) / 3.0;
			if (domain.getNumberOfPossibleBids() <= 6400) {
				allBids.clear();
				BidIterator bidItr = new BidIterator(domain);
				while (bidItr.hasNext()) {
					recordBidInUtilities(bidItr.next());
				}
				estimatedMean = estimateMean();
			}
			Edelta = estimatedMean;
		} catch (Exception ex) {
			System.out.println("init:" + ex);
			ex.printStackTrace();
		}
	}

	public double estimateMean() {
		double sum = 0;
		for (int i = 0; i < allBids.size(); i++) {
			sum += allBids.get(i);
		}
		sum /= (allBids.size() * 1.0);
		return sum;
	}

	public String getName() {
		return "PhoenixParty";
	}

	public double getTime() {
		if (cctime) {
			return (timeline.getCurrentTime() - startTime)
					* (Math.pow(2 - delta, 0.5)) / (180.0);
		}
		return timeline.getTime() * (Math.pow(2 - delta, 0.5));

	}

	public double getTrueTime() {
		if (cctime) {
			return (timeline.getCurrentTime() - startTime) / 180.0;
		}
		return timeline.getTime();
	}

	public double getCurrentTime() {
		if (cctime) {
			return (timeline.getCurrentTime() - startTime);
		}
		return timeline.getCurrentTime();
	}

	@Override
	public void receiveMessage(AgentID sender, Action arguments) {
		super.receiveMessage(sender, arguments);
		countOpp++;
		if (arguments instanceof Offer) {
			try {
				// next opponent
				if (countOpp == lastCountOpp + 1) {
					opponentPreviousBid1 = ((Offer) arguments).getBid();
					opponentPreviousBidUtility1 = getUtility(
							opponentPreviousBid1);
				}
				// next and next opponent (last opponent)
				else if (countOpp == lastCountOpp + 2) {
					opponentPreviousBid2 = ((Offer) arguments).getBid();
					opponentPreviousBidUtility2 = getUtility(
							opponentPreviousBid2);
				}
				opponentPreviousBid = ((Offer) arguments).getBid();
				opponentPreviousBidUtility = getUtility(opponentPreviousBid);
				recordOffer((Offer) arguments);
			} catch (Exception e) {
				System.out.println(e.getMessage());
			}
		} else if (arguments instanceof Accept) {

			if (countOpp == lastCountOpp + 1) {
				controlFreqFunc(myLastBid, -100 * Math.pow(getTime(), 0.5));
				if (getTrueTime() < firstAcceptTime) {
					firstAcceptTime = getTrueTime();
				}
			} else {
				controlFreqFunc(opponentPreviousBid,
						-100 * Math.pow(getTime(), 0.5));
			}

		}
	}

	@Override
	public Action chooseAction(List<Class<? extends Action>> possibleActions) {

		Action currentAction = new NoAction(getPartyId());
		double currentTime = getTime();
		double compareU = 1;
		try {
			if (possibleActions.contains(Accept.class)
					|| possibleActions.contains(Offer.class)) {
				double bound = (uTargetForBidding + uTargetForAcceptance) / 2;
				if (currentTimeIndex > TimeToPrint) {
					TimeToPrint = currentTimeIndex;

				}
				lastCountOpp = countOpp;
				if (getNumberOfParties() > 0) {
					roundOffers = new Offer[getNumberOfParties() - 1];
				}
				theta = utilitySpace.getReservationValueUndiscounted();
				Domain domain = utilitySpace.getDomain();
				if ((myLastBid == null) || (opponentPreviousBid1 == null)
						|| (opponentPreviousBid2 == null)) {
					myLastBid = maxBid;
					uTargetForAcceptance = maxBidUtility;
					uTargetForBidding = maxBidUtility;
					currentAction = new Offer(getPartyId(), myLastBid);
				} else {
					if (timeToTrain()) {
						updateParas();
						getTargetUtility();
					}
					if (debug && getTime() > 0.02) {
						try {
							OriOut = System.out;
							logTxt = new PrintStream(
									new FileOutputStream(fileName, true));

							System.setOut(logTxt);
						} catch (FileNotFoundException ex) {
						}
						System.out.println(Double.toString(getTrueTime()) + ","
								+ Double.toString(uTargetForAcceptance) + ","
								+ Double.toString(bound) + "," + Edelta + ","
								+ Predict(getTime()));
						System.setOut(OriOut);
					}
					if (opponentPreviousBidUtility >= uTargetForBidding
							|| (getTime() > (delta * 0.6 + theta * 0.3)
									&& opponentPreviousBidUtility >= uTargetForAcceptance)) {
						currentAction = new Accept(getPartyId(),
								opponentPreviousBid);
						System.out.println(
								"Accccccccccccccccccccccccccccccccccc");
					} else {
						double reservationValue = theta
								* Math.pow(delta, currentTime);
						if (theta > 0 && ((currentTime - myLastBidTime)
								* timeline.getTotalTime() > opponentNoOfferTime
								|| ((domain.getNumberOfPossibleBids() < 100
										&& currentTime > delta * (1.0 - theta
												+ bestReceivedBidUtility2))
										&& (theta >= bestReceivedBidUtility2))
										&& theta >= 0.4
												* (maxBidUtility - utilitySpace
														.getUtility(
																opponentFirstBid2))
												+ utilitySpace.getUtility(
														opponentFirstBid2))) {

							currentAction = new EndNegotiation(getPartyId());
						} else {
							Bid bidToOffer = BiddingMethod();
							currentAction = new Offer(getPartyId(), bidToOffer);
							if (reservationValue > utilitySpace
									.getUtility(bidToOffer)) {
								currentAction = new EndNegotiation(
										getPartyId());
							}
							compareU = utilitySpace.getUtility(bidToOffer);
						}
					}
				}

				if ((currentAction instanceof Offer)) {
					myLastBid = ((Offer) currentAction).getBid();
					myPreviousBids.add(myLastBid);
					myLastBidTime = currentTime;
					myLastBidUtility = utilitySpace.getUtility(myLastBid);
				}
				double endTime = 0.998;
				if (currentTime >= endTime
						&& domain.getNumberOfPossibleBids() > 5) {
					if (opponentPreviousBidUtility > theta) {
						if (theta == 0 || opponentPreviousBidUtility > theta * 2
								|| myLastBidUtility
										- opponentPreviousBidUtility < (maxBidUtility
												- minBidUtility) / 4) {
							currentAction = new Accept(getPartyId(),
									opponentPreviousBid);
							System.out.println(
									"Accccccccccccccccccccccccccccccccccc");
						}
					}
				}
				if (theta >= uTargetForBidding) {
					currentAction = new EndNegotiation(getPartyId());
					// System.out.println("compareU="+compareU+"
					// uTargetForBidding="+uTargetForBidding);
				}
			} else {
				currentAction = new NoAction(getPartyId());
			}

		} catch (Exception ex) {
			// System.out.println(ex.getMessage());
		}

		return currentAction;
	}

	// ///////////////////////////////////////////////////////////////////
	// //////////////// Implement Our Designed Function //////////////////
	// ///////////////////////////////////////////////////////////////////

	class timeList extends ArrayList<Double> {
	}

	timeList[] saveInTermsOfTime;

	class BidList extends ArrayList<Bid> {
	}

	BidList[] saveInTermsOfUtility;
	private double[] timeSamples;
	private double[] utilitySamples;

	private ArrayList<Double> maxUtilitiesAsInputData = new ArrayList<Double>();
	private ArrayList<Double> timePointsAsOutputData = new ArrayList<Double>();

	private Matrix inputDataMatrix;
	private Matrix outputDataMatrix;
	private Matrix params0;
	private HashMap lastBidValues;
	private CovarianceFunction covFunc;
	private GaussianProcess gp;

	//

	// Parameters
	//
	double delta;
	double theta;
	double alpha = 2.0; // impact of delta over time
	double beta = 1.5; // concession factor
	double omega = 1.3; // weight reflects
	double epsilon = 0.7; // controls influence of lambda
	double xi = 0.95; // acceptance tolerance for pessimistic forecast
	double lambda; // maximum concession
	double gamma = 1.0; // ratio of new to all counter-offers over past ten
						// intervals
	double rho; // compromise point
	double phi; // probability of accepting the most possible bid
	double Elow; // lowest expectation to negotiation
	double Edelta;
	double[] EdeltaPast;
	double R;
	double ppp;
	double tV;
	double uHat;
	double uTarget;
	double uTargetForAcceptance;
	double uTargetForBidding;
	double compareX = 0;
	double opponentBestOfferUtility1;
	double opponentBestOfferUtility2;
	List<Issue> issues;
	int fixedInputSize = 50;
	int rangeofPrediction = 20;
	int timeDivision = 10;
	double phiAcceptanceBound = 0.5;

	private void generateUtilitySamples(int m) {
		utilitySamples = new double[m];
		for (int i = 0; i < m; i++) {
			utilitySamples[i] = minBidUtility
					+ i * (maxBidUtility - minBidUtility) / (m * 1.0);
		}
	}

	private void generateTimeSamples(int n) {
		timeSamples = new double[n];
		EdeltaPast = new double[n];
		for (int i = 0; i < n; i++) {
			timeSamples[i] = (i * 1.0 / n);
		}
	}

	double tryNewBidTime = 0;

	public Bid BiddingMethod() throws Exception {
		try {
			Bid bid = utilitySpace.getMaxUtilityBid();
			Domain domain = utilitySpace.getDomain();
			int limits = 0;
			if (domain.getNumberOfPossibleBids() > 6400) {
				allBids.clear();
				while (limits <= Math.max(domain.getNumberOfPossibleBids() / 10,
						888)) {
					recordBidInUtilities(domain.getRandomBid(rand));
					limits++;
				}
				double maybeEstimatedMean = estimateMean();
				if (maybeEstimatedMean > estimatedMean) {
					estimatedMean = (maybeEstimatedMean + estimatedMean) / 2.0;
				}
			}
			int indexToFindBids = findBidUtilityIndex(uTargetForBidding);
			if (utilityDivisionSize == 3) {
				bid = (saveInTermsOfUtility[1 + rand.nextInt(2)]).get(0);
				return bid;
			}

			double mean = 0;
			int countb = 0;
			for (Bid bb : myPreviousBids) {
				countb++;
				mean += utilitySpace.getUtility(bb);
				if (countb > 100) {
					break;
				}
			}
			mean /= countb;
			countb = 0;
			for (Bid bb : myPreviousBids) {
				countb++;
				selfVar += Math.pow(mean - utilitySpace.getUtility(bb), 2);
				if (countb > 100) {
					break;
				}
			}
			selfVar /= countb;

			if (utilityDivisionSize > 30) {
				if (rand.nextInt(12) > 10) {
					if (indexToFindBids > 0)
						indexToFindBids--;
				}
				int indexToRandBids = findBidUtilityIndex(
						(maxBidUtility + uTargetForAcceptance) / 2);
				if (domain.getNumberOfPossibleBids() < 1000) {
					if (rand.nextInt(10) >= 3
							+ delta * Math.pow(6, 1 - getTime())) {
						indexToFindBids += rand.nextInt(Math.min(
								utilityDivisionSize - 1 - indexToFindBids,
								Math.abs(indexToRandBids - indexToFindBids)));
					}
				} else {
					if (rand.nextInt(11) >= 3
							+ delta * Math.pow(6, 1 - getTime())) {
						indexToFindBids += rand.nextInt(Math.min(
								utilityDivisionSize - 1 - indexToFindBids,
								Math.abs(indexToRandBids - indexToFindBids)));
					}
				}
			} else if (utilityDivisionSize == 3) {
				if (rand.nextInt(20) > 18) {
					if (indexToFindBids > 0)
						indexToFindBids--;
				}
			}

			BidList tempBL = saveInTermsOfUtility[indexToFindBids];
			while (tempBL.size() == 0) {
				indexToFindBids++;
				tempBL = saveInTermsOfUtility[indexToFindBids];
				if (indexToFindBids == utilityDivisionSize - 1)
					break;
			}

			double rank = heuristicRank(bid);
			limits = 0;
			/* time : bad bids -> good bids */
			double changingTime = 0.4 * delta / opponentConcession();
			for (Bid tempbid : tempBL) {
				if (changingTime < getTime()) {
					if (heuristicRank(tempbid)
							* (0.95 + 0.05 * rand.nextDouble()) >= rank) {
						bid = tempbid;
						rank = heuristicRank(bid);
					}
				} else {
					if (heuristicRank(tempbid)
							* (0.95 + 0.05 * rand.nextDouble()) <= rank) {
						bid = tempbid;
						rank = heuristicRank(bid);
					}
				}
				if (limits++ > 100) {
					break;
				}
			}
			if (rand.nextDouble() > delta * Math.min(
					(Math.pow(Math.min(getTime() + 0.1, 0.8), 0.3)), 0.5)) {
				bid = tempBL.get(rand.nextInt(tempBL.size()));
			}
			if (bid.equals(opponentFirstBid1)
					|| bid.equals(opponentFirstBid2)) {
				bid = myLastBid;
			}
			controlFreqFunc(bid, 0.5);
			double timeToPlay = (delta * 1.0 - theta) * rand.nextDouble()
					+ 0.1 * delta;
			double durationToPlay = trainingTimeSeperator * delta;
			if (getTime() >= timeToPlay
					&& getTime() <= timeToPlay + durationToPlay) {
				bid = myLastBid;
			}
			return bid;
		} catch (Exception e) {
			System.out.println(e.getMessage());
			return utilitySpace.getMaxUtilityBid();
		}
	}

	private int findBidUtilityIndex(double utility) {
		// System.out.println("utilityDivisionSize="+utilityDivisionSize);
		for (int i = 1; i < utilityDivisionSize; i++) {
			if (utilitySamples[i] > utility) {
				return i - 1;
			}
		}
		return utilityDivisionSize - 1;
	}

	private void controlFreqFunc(Bid bid, double weight) {
		if (bid != null) {
			for (Integer in = 0; in < issues.size(); in++) {
				Value bv = (bid.getValues()).get(issues.get(in).getNumber());
				IssueDiscrete issue = (IssueDiscrete) issues.get(in);
				for (Integer j = 0; j < issue.getNumberOfValues(); j++) {
					Value v = issue.getValue(j);
					if (debug) {
						// System.out.println("f["+in+","+j+"]="+freqFunc[in][j]);
					}
					if (v.equals(bv)) {
						freqFunc[in][j] *= 1.0 + 0.01 * weight;
					}
				}
			}
		}
	}

	private double heuristicRank(Bid bid) {
		double distance = 0;
		HashMap<Integer, Value> bidValues = bid.getValues();
		HashMap<Integer, Value> compareBidValues = opponentPreviousBid1
				.getValues();
		HashMap<Integer, Value> compareBidValues2 = opponentBestOfferBid1
				.getValues();
		HashMap<Integer, Value> compareBidValues3 = opponentFirstBid1
				.getValues();

		double w1 = 2, w2 = 4, w3 = 10;
		if (selfVar < 0.05 || varianceOfOpponentUtilities < 0.05) {
			w3 *= 1 + 0.5 * rand.nextDouble();
		}
		if (bidValues.equals(opponentFirstBid1.getValues())) {
			return 1e3;
		}
		if (bidValues.equals(compareBidValues)) {
			return 0;
		}
		int numberOfIssues = issues.size();
		for (Integer in = 0; in < numberOfIssues; in++) {
			double a = 0, b = 0;
			Value av = bidValues.get(issues.get(in).getNumber());
			Value bv = compareBidValues.get(issues.get(in).getNumber());
			Issue issue = issues.get(in);
			switch (issue.getType()) {
			case DISCRETE:
				IssueDiscrete lIssueDiscrete = (IssueDiscrete) issue;
				for (int j = 0; j < lIssueDiscrete.getNumberOfValues(); j++) {
					Value v = lIssueDiscrete.getValue(j);
					if (v.equals(av)) {
						a = j;
					}
					if (v.equals(bv)) {
						b = j;
						w1 *= Math.min(freqFunc[in][j], w1);
					}
				}
				break;
			}
			distance += w1 * Math.abs(b - a);
		}
		for (Integer in = 0; in < numberOfIssues; in++) {
			double a = 0, b = 0;
			Value av = bidValues.get(issues.get(in).getNumber());
			Value bv = compareBidValues2.get(issues.get(in).getNumber());
			Issue issue = issues.get(in);

			switch (issue.getType()) {
			case DISCRETE:
				IssueDiscrete lIssueDiscrete = (IssueDiscrete) issue;
				for (int j = 0; j < lIssueDiscrete.getNumberOfValues(); j++) {
					Value v = lIssueDiscrete.getValue(j);
					if (v.equals(av)) {
						a = j;
					}
					if (v.equals(bv)) {
						b = j;
						w2 *= Math.min(freqFunc[in][j], w2);
					}
				}
				break;
			}
			distance += w2 * Math.abs(b - a);
		}
		for (Integer in = 0; in < numberOfIssues; in++) {
			double a = 0, b = 0;
			Value av = bidValues.get(issues.get(in).getNumber());
			Value bv = compareBidValues3.get(issues.get(in).getNumber());
			Issue issue = issues.get(in);
			switch (issue.getType()) {
			case DISCRETE:
				IssueDiscrete lIssueDiscrete = (IssueDiscrete) issue;
				for (int j = 0; j < lIssueDiscrete.getNumberOfValues(); j++) {
					Value v = lIssueDiscrete.getValue(j);
					if (v.equals(av)) {
						a = j;
					}
					if (v.equals(bv)) {
						b = j;
						w3 *= Math.min(freqFunc[in][j], w3);
					}
				}
				break;
			}
			distance += w3 * Math.abs(b - a);
		}
		double w12 = 1, w22 = 3, w32 = 10;
		compareBidValues = opponentPreviousBid2.getValues();
		compareBidValues2 = opponentBestOfferBid2.getValues();
		compareBidValues3 = opponentFirstBid2.getValues();
		if (selfVar < 0.05 || varianceOfOpponentUtilities < 0.05) {
			w32 *= 1 + 0.2 * rand.nextDouble();
		}
		if (bidValues.equals(opponentFirstBid2.getValues())) {
			return 1e3;
		}
		if (bidValues.equals(compareBidValues)) {
			return 0;
		}
		numberOfIssues = issues.size();
		for (Integer in = 0; in < numberOfIssues; in++) {
			double a = 0, b = 0;
			Value av = bidValues.get(issues.get(in).getNumber());
			Value bv = compareBidValues.get(issues.get(in).getNumber());
			Issue issue = issues.get(in);
			switch (issue.getType()) {
			case DISCRETE:
				IssueDiscrete lIssueDiscrete = (IssueDiscrete) issue;
				for (int j = 0; j < lIssueDiscrete.getNumberOfValues(); j++) {
					Value v = lIssueDiscrete.getValue(j);
					if (v.equals(av)) {
						a = j;
					}
					if (v.equals(bv)) {
						b = j;
						w12 *= Math.min(freqFunc[in][j], w12);
					}
				}
				break;
			}
			distance += w12 * Math.abs(b - a);
		}
		for (Integer in = 0; in < numberOfIssues; in++) {
			double a = 0, b = 0;
			Value av = bidValues.get(issues.get(in).getNumber());
			Value bv = compareBidValues2.get(issues.get(in).getNumber());
			Issue issue = issues.get(in);

			switch (issue.getType()) {
			case DISCRETE:
				IssueDiscrete lIssueDiscrete = (IssueDiscrete) issue;
				for (int j = 0; j < lIssueDiscrete.getNumberOfValues(); j++) {
					Value v = lIssueDiscrete.getValue(j);
					if (v.equals(av)) {
						a = j;
					}
					if (v.equals(bv)) {
						b = j;
						w22 *= Math.min(freqFunc[in][j], w22);
					}
				}
				break;
			}
			distance += w22 * Math.abs(b - a);
		}
		for (Integer in = 0; in < numberOfIssues; in++) {
			double a = 0, b = 0;
			Value av = bidValues.get(issues.get(in).getNumber());
			Value bv = compareBidValues3.get(issues.get(in).getNumber());
			Issue issue = issues.get(in);
			switch (issue.getType()) {
			case DISCRETE:
				IssueDiscrete lIssueDiscrete = (IssueDiscrete) issue;
				for (int j = 0; j < lIssueDiscrete.getNumberOfValues(); j++) {
					Value v = lIssueDiscrete.getValue(j);
					if (v.equals(av)) {
						a = j;
					}
					if (v.equals(bv)) {
						b = j;
						w32 *= Math.min(freqFunc[in][j], w32);
					}
				}
				break;
			}
			distance += w32 * Math.abs(b - a);
		}
		return Math.sqrt(distance / ((w1 + w2 + w3 + w12 + w22 + w32)));
	}

	private void recordBidInUtilities(Bid bid) throws Exception {
		double utilityToRecord = utilitySpace.getUtility(bid);
		allBids.add(utilityToRecord);
		int index = findBidUtilityIndex(utilityToRecord);
		if (!(saveInTermsOfUtility[index]).contains(bid)) {
			(saveInTermsOfUtility[index]).add(bid);
		}
	}

	double opponentConcession() {
		double modelPast = 0;
		double modelConcess = Edelta;
		int count = 0;
		int startTimeIndex = Math.max(0, currentTimeIndex - 5);
		for (int i = startTimeIndex; i < currentTimeIndex; i++) {
			modelPast += EdeltaPast[i];
			count++;
		}
		modelPast /= (1.0 * count);
		modelConcess /= modelPast;
		/*
		 * 
		 * modelConcess > 1 : opponent is conceding modelConcess < 1 : opponent
		 * is exploiting
		 */
		return modelConcess;
	}

	double uncertainty;

	private void getTargetUtility() throws Exception {
		double currentTime = getTime();
		R = RFunction(currentTime);

		Edelta = discountedExpectation(currentTime + 0.02);
		uncertainty = PredictUncertainty(currentTime + 0.02);

		double rate = 1.5 + 0.3 * rand.nextDouble();
		rate *= (0.3 + uTargetForAcceptance - Edelta + (1 - Math.pow(1.0, 1.3))
				- Math.pow(theta, 1.5));
		rate /= opponentConcession();
		double opponentBidDistance = Math.abs(Edelta - uTargetForAcceptance)
				* compaction2;
		uTargetForBidding = (R * Math.pow(3, -getTime() * rate)
				+ Edelta * Math.abs(1 - opponentBidDistance) / (2.0))
				/ (Math.abs(1 - opponentBidDistance) / (2.0)
						+ Math.pow(3, -getTime() * rate));

		double baseValue = (Math.max(Elow, Edelta) + Edelta / uncertainty
				+ uncertainty * Elow) / (1.0 + 1 / uncertainty + uncertainty);
		double diffToElow = 1.2 * Math.abs(uTargetForBidding - baseValue);
		if (delta <= 1) {
			double x = 2 * (1 - getTime()) * delta, y = delta,
					z = (2 * getTime()) / delta;
			double uTargetForBiddingF1 = diffToElow * Math.pow(5,
					-(Math.pow(getTime(), 0.6) * (2.3 - firstAcceptTime))
							* Math.pow(2 - delta, getTime()));
			double uTargetForBiddingF2 = diffToElow
					* (1 / ((Math.pow((2 * getTime()) * (2.3 - firstAcceptTime),
							delta + theta / 3.0) + 0.05) * 20) + 0.4 * delta);
			double uTargetForBiddingF3 = diffToElow * (1
					/ ((Math.log((20.0 / (delta + theta / 3.0)) * getTime()
							* (2.3 - firstAcceptTime) + 1.1)) * 10)
					+ 0.3 * delta);
			uTargetForBidding = baseValue
					+ (z * uTargetForBiddingF3 + y * uTargetForBiddingF2
							+ x * uTargetForBiddingF1) / (x + y + z);
		}
		if ((selfVar < 0.05 || varianceOfOpponentUtilities < 0.05)
				&& timeline.getCurrentTime() % 2 == 1) {
			uTargetForBidding *= 0.95 + 0.1 * rand.nextDouble();
		}
		if (uTargetForBidding <= uTargetForAcceptance) {
			uTargetForAcceptance = (uTargetForBidding + uTargetForAcceptance)
					/ 2.0;
		}
	}

	private void updateParas() {
		try {
			selfVar = 0;
			lastTrainingTime = getTime();
			getTrainingData();
			trainGP();
			meanOfOpponentUtilities = 0;
			varianceOfOpponentUtilities = 0;
			for (double u : maxUtilitiesAsInputData) {
				meanOfOpponentUtilities += u;
			}
			meanOfOpponentUtilities /= maxUtilitiesAsInputData.size();
			for (double u : maxUtilitiesAsInputData) {
				varianceOfOpponentUtilities += Math
						.pow(u - meanOfOpponentUtilities, 2);
			}
			varianceOfOpponentUtilities /= maxUtilitiesAsInputData.size();
			double currentTime = getTime();

		} catch (Exception ex) {
			// System.out.println("updateParas: "+ex.getMessage());
		}
	}

	private double Predict(double queryTimePt) {
		try {
			double[][] q = { { queryTimePt } };
			Matrix queryInput = new Matrix(q);
			Matrix[] res = gp.predict(queryInput);
			return Math.max(Math.min(res[0].get(0, 0), maxBidUtility),
					minBidUtility);
		} catch (Exception ex) {
			// System.out.println("Predict error: "+ex.getMessage());
			return minBidUtility;
		}
	}

	private double PredictUncertainty(double queryTimePt) {
		try {
			double[][] q = { { queryTimePt } };
			Matrix queryInput = new Matrix(q);
			Matrix[] res = gp.predict(queryInput);
			return 1 / (1 + Math.exp(-1 * Math.exp(res[1].get(0, 0))));
		} catch (Exception ex) {
			// System.out.println("Predict error: "+ex.getMessage());
			return minBidUtility;
		}
	}

	private double lastTrainingTime = 0;

	private double discountedExpectation(double time) {
		double w = 0.5;
		if (getTime() > 10 * trainingTimeSeperator) {
			w = opponentConcession() / 2.0;
		}
		double predictedUtility = (uTargetForAcceptance
				* Math.pow(0.5, getTime()) * (2 + w)
				+ Math.pow(0.5, 1 - getTime()) * 3 * w
						* (1.0 / PredictUncertainty(time)) * Predict(time)
				+ 2 * w * meanOfOpponentUtilities)
				/ (2 * w + Math.pow(0.5, getTime()) * (2 + w)
						+ Math.pow(0.5, 1 - getTime()) * 3 * w
								* (1.0 / PredictUncertainty(time)));
		return predictedUtility * Math
				.pow(Math.pow(1.0 + (0.1 * theta + 0.05), 0.7), getTime());
	}

	private double lowestExpectation() throws Exception {
		double uFIrst = Math.max(utilitySpace.getUtility(opponentFirstBid1),
				utilitySpace.getUtility(opponentFirstBid2));
		if (uFIrst < estimatedMean) {
			uFIrst = estimatedMean;
		}
		Elow = uFIrst + (maxBidUtility - uFIrst)
				* Math.pow(delta, 0.5 + (0.1) * rand.nextDouble()) / 2.5;
		return Elow * Math.pow(0.9, Math.pow(getTrueTime(), 2.0));
	}

	private double RFunction(double time) {
		try {
			double Elow = lowestExpectation() * Math.pow(0.8 + (0.2 * ppp),
					Math.pow(time, Math.pow(1.0, 0.65)));
			if (time >= 1) {
				return maxBidUtility;
			}
			omega = 1.26;
			if (getTime() > 10 * trainingTimeSeperator && getTime() < 0.3) {
				double comparePoint = opponentConcession()
						* discountedExpectation(time);
				double disToPoint = Math.abs(R - comparePoint)
						/ Math.abs(maxBidUtility - Math.min(
								utilitySpace.getUtility(opponentFirstBid1),
								utilitySpace.getUtility(opponentFirstBid2)));
				ppp = disToPoint;
				omega = 1.16 + 0.5 * (disToPoint);
			}
			omega = Math.min(omega, 1.35);
			double component1 = Math
					.abs(1 - Math.pow(time, 1 / Math.pow(Math.abs(0.5), beta)));
			double component2 = Math
					.cos((1 - Math.pow(1.0, 1.3) * Math.pow(compaction2, 0.5))
							/ (omega));
			double addR = component1 * component2 * (maxBidUtility - Elow);
			addR *= Math.pow(1.06, compaction2);
			R = Math.min(maxBidUtility, Elow + addR);
			return R;
		} catch (Exception e) {
			return maxBidUtility;
		}

	}

	double sigmoid(double x) {
		return 1 / (1 + Math.exp(-1 * x));
	}

	int currentTimeIndex = 0;

	private void recordOffer(Offer offerToRecord) throws Exception {

		Bid bidToRecord = offerToRecord.getBid();
		if (countOpp == lastCountOpp + 1) {
			if (opponentFirstBid1 == null) {
				opponentFirstBid1 = bidToRecord;
				Edelta = utilitySpace.getUtility(opponentFirstBid1);
				minBidUtility = utilitySpace.getUtility(opponentFirstBid1);
				compaction2 = 1 / (maxBidUtility - minBidUtility);
				opponentBestOfferBid1 = opponentFirstBid1;
				controlFreqFunc(opponentFirstBid1, 10);
			} else {
				if (bidToRecord.equals(opponentFirstBid1)) {
					controlFreqFunc(myLastBid, 10);
				}
			}
			opponentBids.add(bidToRecord);
			opponentPreviousBid1 = bidToRecord;
			double utilityToRecord = utilitySpace.getUtility(bidToRecord);
			opponentBestOfferUtility1 = utilitySpace
					.getUtility(opponentBestOfferBid1);
			if (utilityToRecord > opponentBestOfferUtility1) {
				opponentBestOfferBid1 = bidToRecord;
				if (countOpp == lastCountOpp + 1) {
					controlFreqFunc(opponentBestOfferBid1, -1);
				}
			}
			opponentPreviousBidUtility1 = utilityToRecord;
			sumOfOpponentBidUtilities += utilityToRecord;
			double timeToRecord = getTime();
			opponentPreviousBidTime1 = timeToRecord;

			bestReceivedBid1 = opponentBestOfferBid1;
			bestReceivedBidUtility1 = utilitySpace.getUtility(bestReceivedBid1);
		} else if (countOpp == lastCountOpp + 2) {
			if (opponentFirstBid2 == null) {
				opponentFirstBid2 = bidToRecord;
				opponentBestOfferBid2 = opponentFirstBid2;
				controlFreqFunc(opponentFirstBid2, 5);
			} else {
				if (bidToRecord.equals(opponentFirstBid2)) {
					controlFreqFunc(myLastBid, 1);

				}
			}

			opponentPreviousBid2 = bidToRecord;
			double utilityToRecord = utilitySpace.getUtility(bidToRecord);
			opponentBestOfferUtility2 = utilitySpace
					.getUtility(opponentBestOfferBid2);
			if (utilityToRecord > opponentBestOfferUtility2) {
				opponentBestOfferBid2 = bidToRecord;
				controlFreqFunc(opponentBestOfferBid2, -1);

			}
			opponentPreviousBidUtility2 = utilityToRecord;
			bestReceivedBid2 = opponentBestOfferBid2;
			bestReceivedBidUtility2 = utilitySpace.getUtility(bestReceivedBid2);
		}
		if (opponentFirstBid1 != null && opponentBestOfferBid2 != null) {
			compaction2 = 1 / Math.abs(maxBidUtility
					- Math.min(utilitySpace.getUtility(opponentFirstBid1),
							utilitySpace.getUtility(opponentFirstBid2)));
		}
		if (currentTimeIndex != timeSamples.length - 1) {
			if (getTime() > timeSamples[currentTimeIndex + 1]) {
				if (getTime() > 5 * trainingTimeSeperator) {
					EdeltaPast[currentTimeIndex] = discountedExpectation(
							getTime() + trainingTimeSeperator);
				}
				currentTimeIndex++;
			}
		}
		if (countOpp == lastCountOpp + 2) {
			(saveInTermsOfTime[currentTimeIndex]).add(Math.min(
					opponentPreviousBidUtility1, opponentPreviousBidUtility2));
			recordBidInUtilities(bidToRecord);
		}
	}

	private boolean timeToTrain() {
		return (getTime() - lastTrainingTime) > trainingTimeSeperator;
	}

	int lastTrainingIndex = 0;

	private void getTrainingData() {
		try {
			double currentTime = getTime();
			int numberOfRows = 0;

			for (int i = lastTrainingIndex; i < timeSamples.length; i++) {
				double timeIntervalLowerBound = timeSamples[i];
				double timeIntervalUpperBound = 1.0;
				if (i != timeSamples.length - 1) {
					timeIntervalUpperBound = timeSamples[i + 1];
				}

				double sumU = 0;
				double countU = 0;
				double maxU = 0;
				double timePt = (timeIntervalLowerBound
						+ timeIntervalUpperBound) / 2;
				for (double pastUtility : saveInTermsOfTime[i]) {
					if (timePointsAsOutputData.contains(pastUtility))
						continue;
					sumU += pastUtility;
					countU += 1.0;
					maxU = Math.max(maxU, pastUtility);
				}
				if (saveInTermsOfTime[i].size() > 0) {
					double meanU = sumU / countU;
					maxUtilitiesAsInputData.add(maxU);
					timePointsAsOutputData.add(timePt);
					numberOfRows++;
				}
				if (opponentPreviousBidTime1 < timeIntervalUpperBound) {
					lastTrainingIndex = i - 1;
					break;
				}
			}

			while (maxUtilitiesAsInputData.size() > fixedInputSize) {
				maxUtilitiesAsInputData.remove(0);
				timePointsAsOutputData.remove(0);
			}
			numberOfRows = timePointsAsOutputData.size();
			double[][] inputData = new double[numberOfRows][1];
			double[][] outputData = new double[numberOfRows][1];
			for (int i = 0; i < numberOfRows; i++) {
				outputData[i][0] = maxUtilitiesAsInputData.get(i);
				inputData[i][0] = timePointsAsOutputData.get(i);
			}
			inputDataMatrix = new Matrix(inputData);
			outputDataMatrix = new Matrix(outputData);

		} catch (Exception ex) {
			// System.out.println("getTrainingData: "+ex.getMessage());
		}

	}

	private void trainGP() {
		try {
			gp.train(inputDataMatrix, outputDataMatrix, params0);
		} catch (Exception e) {
			// System.out.println("trainGP: "+e.getMessage());
		}

	}

	@Override
	public String getDescription() {
		return "ANAC2015";
	}

}