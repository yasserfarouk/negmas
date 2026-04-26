package agents.anac.y2013.AgentKF;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Random;

import genius.core.Agent;
import genius.core.Bid;
import genius.core.BidHistory;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.ActionWithBid;
import genius.core.actions.EndNegotiation;
import genius.core.actions.Offer;
import genius.core.bidding.BidDetails;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.IssueInteger;
import genius.core.issue.IssueReal;
import genius.core.issue.Value;
import genius.core.issue.ValueInteger;
import genius.core.issue.ValueReal;
import genius.core.utility.AdditiveUtilitySpace;

public class AgentKF extends Agent {

	// I want to print "state" when I print a message about saving data.
	private String state;
	private Action partner;
	private HashMap<Bid, Double> offeredBidMap;
	private double target;
	private double bidTarget;
	private double bidReduction;
	private double sum;
	private double sum2;
	private int rounds;
	private double tremor;
	private int MaxLoopNum;

	private BidHistory currSessOppBidHistory = new BidHistory();
	private BidHistory prevSessOppBidHistory = new BidHistory();
	private double MINIMUM_BID_UTILITY;
	private double MinAutoAcceptUtil;
	private boolean FinalPhase;
	private double PrevMean;

	@Override
	public void init() {

		offeredBidMap = new HashMap<Bid, Double>();
		target = 1.0;
		bidTarget = 1.0;
		bidReduction = 0.01;
		sum = 0.0;
		sum2 = 0.0;
		rounds = 0;
		tremor = 2.0;
		MinAutoAcceptUtil = 0.8;
		MaxLoopNum = 1000;
		PrevMean = 0;
		FinalPhase = false;

		MINIMUM_BID_UTILITY = utilitySpace.getReservationValueUndiscounted();

		myBeginSession();
	}

	public void myBeginSession() {

		// ---- Code for trying save and load functionality
		// First try to load saved data
		// ---- Loading from agent's function "loadSessionData"
		Serializable prev = this.loadSessionData();
		if (prev != null) {
			prevSessOppBidHistory = (BidHistory) prev;
			currSessOppBidHistory = prevSessOppBidHistory;
			PrevMean = prevSessOppBidHistory
					.getAverageDiscountedUtility(utilitySpace);
		} else {
			// If didn't succeed, it means there is no data for this preference
			// profile
			// in this domain.
		}
	}

	@Override
	public String getVersion() {
		return "1.1";
	}

	@Override
	public String getName() {
		return "AgentKF";
	}

	@Override
	public void ReceiveMessage(Action opponentAction) {
		partner = opponentAction;
		if (opponentAction instanceof Offer) {
			Bid bid = ((Offer) opponentAction).getBid();
			// 2. store the opponent's trace
			try {
				BidDetails opponentBid = new BidDetails(bid,
						utilitySpace.getUtility(bid), timeline.getTime());
				currSessOppBidHistory.add(opponentBid);
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
	}

	@Override
	public Action chooseAction() {
		Action action = null;
		try {
			if (partner == null) {
				action = selectBid();
			}
			if (partner instanceof Offer) {
				Bid offeredBid = ((Offer) partner).getBid();

				double p = acceptProbability(offeredBid);

				if (utilitySpace.getUtility(offeredBid) > MinAutoAcceptUtil) {
					p = 1.0;
				}

				if (rounds % 500 == 0) {
					tremor += adjustTremor(timeline.getCurrentTime());
				}

				if (timeline.getCurrentTime() > 0.85) {
					BidHistory FinalBidHistory = currSessOppBidHistory
							.filterBetweenTime(timeline.getCurrentTime(), 1.0);
					double FinalAvg = FinalBidHistory.getAverageUtility();
					if (FinalAvg < sum / rounds) {
						FinalPhase = true;
					}
				}

				if (p > Math.random()) {
					action = new Accept(getAgentID(), offeredBid);
				} else {
					action = selectBid();
				}
				// ---- Code for trying save and load functionality
				// /////////////////////////////////
				state = "Opponet Send the Bid ";
				tryToSaveAndPrintState();
				// /////////////////////////////////
			}
			if (partner instanceof EndNegotiation) {
				// ---- Code for trying save and load
				// functionality///////////////////////////////////
				state = "Got EndNegotiation from opponent. ";
				tryToSaveAndPrintState();
				// /////////////////////////////////
			}
		} catch (Exception e) {
			// ---- Code for trying save and load functionality
			// /////////////////////////////////
			state = "Got Exception. ";
			tryToSaveAndPrintState();
			// /////////////////////////////////
			action = new Accept(getAgentID(),
					((ActionWithBid) partner).getBid());
		}
		return action;
	}

	// ---- Code for trying save and load functionality
	private void tryToSaveAndPrintState() {

		// ---- Saving from agent's function "saveSessionData"
		if (currSessOppBidHistory.size() < Math.pow(10, 5)) {
			this.saveSessionData(currSessOppBidHistory);
		}
	}

	private Action selectBid() {
		Bid nextBid = null;
		double time = timeline.getTime();

		ArrayList<Bid> bidTemp = new ArrayList<Bid>();

		for (Bid bid : offeredBidMap.keySet()) {
			if (offeredBidMap.get(bid) > target) {
				bidTemp.add(bid);
			}
		}

		int size = bidTemp.size();
		if (size > 0) {
			int sindex = (int) Math.floor(Math.random() * size);
			nextBid = bidTemp.get(sindex);
		} else {
			double searchUtil = 0.0;
			try {
				int loop = 0;
				boolean NotFind = true;
				ArrayList<Bid> AltNextBid = new ArrayList<Bid>();
				while (loop < MaxLoopNum) {/* searchUtil < bidTarget */
					if (loop == MaxLoopNum - 1 & NotFind) {
						bidTarget -= bidReduction;
						loop = 0;
					}
					Bid altNextBid = searchBid();
					searchUtil = utilitySpace.getUtilityWithDiscount(altNextBid,
							time);
					if (searchUtil >= bidTarget) {
						NotFind = false;
						AltNextBid.add(altNextBid);
					}
					loop++;
				}

				double minUtil = Double.MAX_VALUE;
				Bid minBid = null;
				for (int i = 0; i < AltNextBid.size(); i++) {
					Bid bufBid = AltNextBid.get(i);
					Double bufUtil = utilitySpace.getUtilityWithDiscount(bufBid,
							time);
					if (minUtil > bufUtil) {
						minUtil = bufUtil;
						minBid = bufBid;
					} else if (minUtil == bufUtil) {
						BidHistory simHistory = currSessOppBidHistory
								.filterBetweenUtility(MINIMUM_BID_UTILITY, 1.0);
						if (this.similarBid(simHistory, bufBid) < this
								.similarBid(simHistory, minBid)) {
							minBid = bufBid;
						}
					}
					nextBid = minBid;
				}
			} catch (Exception e) {
			}
		}

		if (nextBid == null) {
			return (new Accept(getAgentID(),
					((ActionWithBid) partner).getBid()));
		}
		return (new Offer(getAgentID(), nextBid));
	}

	private int similarBid(BidHistory theHistory, Bid theBid) throws Exception {
		int Value = Integer.MAX_VALUE;
		ArrayList<BidDetails> AltList = (ArrayList<BidDetails>) theHistory
				.getNBestBids(theHistory.size() - 1);
		for (int i = 0; i < AltList.size(); i++) {
			Bid targetBid = AltList.get(i).getBid();
			List<Issue> issues = utilitySpace.getDomain().getIssues();
			for (Issue lIssue : issues) {
				switch (lIssue.getType()) {
				case DISCRETE:
					IssueDiscrete lIssueDiscrete = (IssueDiscrete) lIssue;
					double weight_d = ((AdditiveUtilitySpace) utilitySpace)
							.getWeight(lIssueDiscrete.getNumber());
					if (theBid.getValue(lIssueDiscrete.getNumber()) == targetBid
							.getValue(lIssueDiscrete.getNumber()))
						Value += 1.0 * weight_d;
					break;
				case REAL:
					IssueReal lIssueReal = (IssueReal) lIssue;
					double weight_r = ((AdditiveUtilitySpace) utilitySpace)
							.getWeight(lIssueReal.getNumber());
					if (theBid.getValue(lIssueReal.getNumber()) == targetBid
							.getValue(lIssueReal.getNumber()))
						Value += 1.0 * weight_r;
					break;
				case INTEGER:
					IssueInteger lIssueInteger = (IssueInteger) lIssue;
					double weight_i = ((AdditiveUtilitySpace) utilitySpace)
							.getWeight(lIssueInteger.getNumber());
					if (theBid.getValue(lIssueInteger.getNumber()) == targetBid
							.getValue(lIssueInteger.getNumber()))
						Value += 1.0 * weight_i;
					break;
				default:
					throw new Exception("issue type " + lIssue.getType()
							+ " not supported by SimpleAgent2");
				}
			}
		}
		return Value;

	}

	private Bid searchBid() throws Exception {
		HashMap<Integer, Value> values = new HashMap<Integer, Value>();
		List<Issue> issues = utilitySpace.getDomain().getIssues();
		Random randomnr = new Random();

		Bid bid = null;

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
				values.put(lIssueReal.getNumber(),
						new ValueReal(lIssueReal.getLowerBound() + (lIssueReal
								.getUpperBound() - lIssueReal.getLowerBound())
								* (optionInd) / (lIssueReal
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
		return bid;
	}

	private double adjustTremor(double time) {
		if (currSessOppBidHistory.isEmpty()) {
			return 0.0;
		} else {
			double avg = sum / rounds;
			// double histry_avg =
			// currSessOppBidHistory.getAverageDiscountedUtility(utilitySpace);
			double histry_avg = currSessOppBidHistory
					.filterBetweenTime(0.0, timeline.getCurrentTime())
					.getAverageUtility();
			if (avg > histry_avg) {
				return 0.3;
			} else {
				return -0.3;
			}
		}
	}

	double acceptProbability(Bid offeredBid) throws Exception {
		double time = timeline.getTime();
		double offeredUtility = utilitySpace.getUtilityWithDiscount(offeredBid,
				time);
		offeredBidMap.put(offeredBid, offeredUtility);

		sum += offeredUtility;
		sum2 += offeredUtility * offeredUtility;
		rounds++;

		double mean = sum / rounds;
		mean = 0.7 * mean + 0.3 * PrevMean;

		double variance = (sum2 / rounds) - (mean * mean);

		double deviation = Math.sqrt(variance * 12);
		if (Double.isNaN(deviation)) {
			deviation = 0.0;
		}

		double t = time * time * time;

		if (offeredUtility < 0 || offeredUtility > 1.05) {
			throw new Exception("utility " + offeredUtility + " outside [0,1]");
		}

		if (t < 0 || t > 1) {
			throw new Exception("time " + t + " outside [0,1]");
		}

		if (offeredUtility > 1.) {
			offeredUtility = 1;
		}

		double estimateMax = mean + ((1 - mean) * deviation);

		double alpha = 1 + tremor + (10 * mean) - (2 * tremor * mean);
		double beta = alpha + (Math.random() * tremor) - (tremor / 2);

		double preTarget = 1 - (Math.pow(time, alpha) * (1 - estimateMax));
		double preTarget2 = 1 - (Math.pow(time, beta) * (1 - estimateMax));

		double ratio = (deviation + 0.1) / (1 - preTarget);
		if (Double.isNaN(ratio) || ratio > 2.0) {
			ratio = 2.0;
		}

		double ratio2 = (deviation + 0.1) / (1 - preTarget2);
		if (Double.isNaN(ratio2) || ratio2 > 2.0) {
			ratio2 = 2.0;
		}

		target = ratio * preTarget + 1 - ratio;
		bidTarget = ratio2 * preTarget2 + 1 - ratio2;

		double m = t * (-300) + 400;
		if (target > estimateMax) {
			double r = target - estimateMax;
			double f = 1 / (r * r);
			if (f > m || Double.isNaN(f))
				f = m;
			double app = r * f / m;
			target = target - app;
		} else {
			target = estimateMax;
		}

		if (bidTarget > estimateMax) {
			double r = bidTarget - estimateMax;
			double f = 1 / (r * r);
			if (f > m || Double.isNaN(f))
				f = m;
			double app = r * f / m;
			bidTarget = bidTarget - app;
		} else {
			bidTarget = estimateMax;
		}

		// test code for Discount Factor
		if (FinalPhase) {
			double discount_utility = utilitySpace
					.getUtilityWithDiscount(offeredBid, time);
			double discount_ratio = discount_utility / offeredUtility;
			if (!Double.isNaN(discount_utility)) {
				target *= discount_ratio;
				bidTarget *= discount_ratio;
			}
		}
		// test code for Discount Factor

		double utilityEvaluation = offeredUtility - estimateMax;
		double satisfy = offeredUtility - target;

		double p = (Math.pow(time, alpha) / 5) + utilityEvaluation + satisfy;
		if (p < 0.1) {
			p = 0.0;
		}

		return p;
	}

	@Override
	public String getDescription() {
		return "ANAC2013 compatible with non-linear utility spaces";
	}

}