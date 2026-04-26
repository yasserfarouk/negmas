package agents.anac.y2016.agentlight;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;

import genius.core.AgentID;
import genius.core.Bid;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.EndNegotiation;
import genius.core.actions.Inform;
import genius.core.actions.Offer;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.IssueInteger;
import genius.core.issue.Value;
import genius.core.issue.ValueInteger;
import genius.core.parties.AbstractNegotiationParty;
import genius.core.parties.NegotiationInfo;

public class AgentLight extends AbstractNegotiationParty {

	// PARAMETER LIST. THIS IS WHAT YOU'RE SUPPOSED TO PLAY AROUND WITH.

	// newly create variables
	private double discountFactor = 0;
	private double reservationValue = 0;
	private static boolean isPrinting = true;
	private boolean nextOpponentIndicator;
	private int opponentNum;
	private HashMap<Bid, Integer> bidSupport = new HashMap<Bid, Integer>();
	private ArrayList<Bid> bidPotential = new ArrayList<Bid>();
	private HashMap<AgentID, OpponentInfo> opponentInfo = new HashMap<AgentID, OpponentInfo>();
	private ArrayList<Bid> totalBidHistory = new ArrayList<Bid>();
	private OpponentInfo worthyOpponent = null;
	private OpponentInfo nextOpponent = null;
	private Bid lastBidInHistory = null;
	private Bid bestBidInPotential = null;
	private ArrayList<AgentID> agentOrder = new ArrayList<AgentID>();

	// variable in init()
	private double maxUtility, minUtility;
	private List<Issue> issueList;
	private HashMap<Integer, ArrayList<Value>> issue2Value = new HashMap<Integer, ArrayList<Value>>();
	private long possibleBidNum;
	private ArrayList<ComparableBid> bidList = new ArrayList<ComparableBid>();

	// new variable find here
	// private Bid opponentPreBid;
	private int frequencyReward;
	private int unchangeReward;
	private ArrayList<Bid> myBidRecord = new ArrayList<Bid>();
	private ArrayList<Bid> myAcceptBidRecord = new ArrayList<Bid>();

	// variable in receive()
	private Bid opponentCurBid;
	private int recordTimes = 0;
	// variable in chooseAction()
	private double learningTimes = 10, learningExp = 0.1;
	private double discount = 0.7;

	@Override
	public void init(NegotiationInfo info) {
		super.init(info);
		// initialize variables
		this.discountFactor = utilitySpace.getDiscountFactor();
		this.reservationValue = utilitySpace.getReservationValueUndiscounted();

		if (isPrinting) {
			System.out.println("Discount Factor is " + discountFactor);
			System.out.println("Reservation Value is " + reservationValue);
		}

		this.maxUtility = 1;
		this.frequencyReward = 1;
		this.unchangeReward = 1;
		this.nextOpponentIndicator = false;

		issueList = utilitySpace.getDomain().getIssues();

		for (Issue issue : issueList) {
			int issueNum = issue.getNumber();
			switch (issue.getType()) {
			case DISCRETE:
				IssueDiscrete discreteIssue = (IssueDiscrete) issue;
				ArrayList<Value> discreteValue = new ArrayList<Value>();
				for (int i = 0; i < discreteIssue.getNumberOfValues(); i++) {
					discreteValue.add(discreteIssue.getValue(i));
				}
				issue2Value.put(issueNum, discreteValue);
				break;
			case INTEGER:
				IssueInteger integerIssue = (IssueInteger) issue;
				ArrayList<Value> integerValue = new ArrayList<Value>();
				for (int i = integerIssue.getLowerBound(); i <= integerIssue
						.getUpperBound(); i++) {
					integerValue.add(new ValueInteger(i));
				}
				issue2Value.put(issueNum, integerValue);
				break;
			default:
				try {
					throw new Exception(
							"issue type " + issue.getType() + " not supported");
				} catch (Exception e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}
		}

		possibleBidNum = utilitySpace.getDomain().getNumberOfPossibleBids();
		try {
			for (int i = 0; i < possibleBidNum; i++) {
				HashMap<Integer, Value> issueInst;
				issueInst = Int2Bid(i);
				ComparableBid bid = new ComparableBid(
						new Bid(utilitySpace.getDomain(), issueInst));
				bidList.add(bid);
			}

			Collections.sort(bidList);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	@Override
	public Action chooseAction(List<Class<? extends Action>> Actions) {

		if (!agentOrder.contains(getPartyId())) {
			agentOrder.add(getPartyId());
		} else {
			for (AgentID agentId : agentOrder) {
				System.out.println("agentID: " + agentId);
			}
			if (agentOrder.indexOf(getPartyId()) == (opponentNum - 1))
				nextOpponent = opponentInfo.get(agentOrder.get(0));
			else {
				nextOpponent = opponentInfo.get(
						agentOrder.get(agentOrder.indexOf(getPartyId()) + 1));
			}
		}
		recordTimes++;

		double remainingRounds = (timeline.getTotalTime()
				- timeline.getCurrentTime())
				/ (timeline.getCurrentTime() / recordTimes);
		if (recordTimes >= learningTimes) {
			worthyOpponent = findOpponent(opponentInfo);
			if (worthyOpponent == null) {
				worthyOpponent = nextOpponent;
				nextOpponentIndicator = true;
			}
			if (isPrinting) {
				System.out.println("worthyOpponent: "
						+ worthyOpponent.getAgentID().toString());
			}
			int worthyOpponentBidSize = worthyOpponent.getAgentBidHistory()
					.size();
			opponentCurBid = worthyOpponent.getAgentBidHistory()
					.get(worthyOpponentBidSize - 1);
			// opponentPreBid =
			// worthyOpponent.getAgentBidHistory().get(worthyOpponentBidSize-2);
		}
		if (!totalBidHistory.isEmpty()) {
			int totalBidHistorySize = totalBidHistory.size();
			lastBidInHistory = totalBidHistory.get(totalBidHistorySize - 1);
		}
		if (!bidPotential.isEmpty()) {
			bestBidInPotential = findBestBid(bidPotential);
		}

		if (isPrinting) {
			System.out.println("Threshold calculating!");
			System.out.println("remaining Rounds: " + remainingRounds);
			System.out.println("timeline.getType " + timeline.getType());
			System.out.println("getTotalTime: " + timeline.getTotalTime());
			System.out.println("getTime(): " + timeline.getTime());
			System.out.println("currentTime: " + timeline.getCurrentTime());
			// System.out.println("getTimeUtility:
			// "+utilitySpace.getUtilityWithDiscount(lastBidInHistory,
			// timeline.getTime()));
			// System.out.println("currentTimeUtility:
			// "+utilitySpace.getUtilityWithDiscount(lastBidInHistory,
			// timeline.getCurrentTime()));
		}

		double base = ((timeline.getTotalTime() - timeline.getCurrentTime())
				/ timeline.getTotalTime()) > 0.7
						? ((timeline.getTotalTime() - timeline.getCurrentTime())
								/ timeline.getTotalTime())
						: 0.7;
		if (recordTimes >= learningTimes && worthyOpponent != null) {
			learningExp = ((worthyOpponent.getOpponentStandardDeviation())[1]
					+ (worthyOpponent.getOpponentAverage())[1]);
		}
		minUtility = (reservationValue
				+ (maxUtility - reservationValue) * Math.pow(base, learningExp))
				* discountFactor;
		if (remainingRounds <= 3) {
			minUtility = minUtility * (discount);
		}
		if (isPrinting) {
			System.out.println("learningExp: " + learningExp);
			System.out.println("base: " + base);
			System.out.println("discountFactor: " + discountFactor);
			System.out.println("minUtility: " + minUtility);
		}

		if (remainingRounds <= 3 && lastBidInHistory != null
				&& utilitySpace.getUtilityWithDiscount(lastBidInHistory,
						timeline.getTime()) >= minUtility
				&& bidPotential.contains(lastBidInHistory)) {
			if (isPrinting) {
				System.out.println("Accept condition 1!");
			}
			return (new Accept(getPartyId(), opponentCurBid));
		}
		if (lastBidInHistory != null
				&& utilitySpace.getUtilityWithDiscount(lastBidInHistory,
						timeline.getTime()) >= minUtility
				&& !myAcceptBidRecord.contains(lastBidInHistory)
				&& !bidPotential.contains(lastBidInHistory)
				|| (bestBidInPotential != null
						&& !myAcceptBidRecord.contains(lastBidInHistory)
						&& !bidPotential.contains(lastBidInHistory)
						&& utilitySpace.getUtilityWithDiscount(lastBidInHistory,
								timeline.getTime()) >= utilitySpace
										.getUtilityWithDiscount(
												bestBidInPotential,
												timeline.getTime()))) {
			if (isPrinting) {
				System.out.println(bidPotential);
				System.out.println(lastBidInHistory);
				System.out.println(bidPotential.contains(lastBidInHistory));
				System.out.println("Accept condition 2!");
			}
			myAcceptBidRecord.add(lastBidInHistory);
			return new Accept(getPartyId(), opponentCurBid);
		}

		if (bestBidInPotential != null && remainingRounds <= 3
				&& utilitySpace.getUtilityWithDiscount(bestBidInPotential,
						timeline.getTime()) >= minUtility) {
			if (isPrinting) {
				System.out.println("Offer condition 1!");
			}
			return (new Offer(getPartyId(), bestBidInPotential));
		}

		if (remainingRounds > 3 && recordTimes >= learningTimes) {
			if (isPrinting) {
				System.out.println("choose Negotiation Action");
			}
			return NegotiationAction(minUtility);
		}
		if (remainingRounds <= 3) {
			if (isPrinting) {
				System.out.println("choose Concession Action");
			}
			return NegotiationAction(minUtility);
		}
		if (isPrinting) {
			System.out.println("choose Learning Action");
		}
		return LearningAction();

	}

	public OpponentInfo findOpponent(
			HashMap<AgentID, OpponentInfo> opponentInfo) {
		double minValue = 999;
		OpponentInfo minInfo = null;
		for (AgentID agentId : opponentInfo.keySet()) {
			int opponentBidSize = opponentInfo.get(agentId).getAgentBidHistory()
					.size();
			if (opponentBidSize >= 2) {
				Bid opponentCurBid = opponentInfo.get(agentId)
						.getAgentBidHistory().get(opponentBidSize - 1);
				Bid opponentPreBid = opponentInfo.get(agentId)
						.getAgentBidHistory().get(opponentBidSize - 2);
				double value = opponentInfo.get(agentId)
						.getOpponentStandardDeviation()[1]
						+ opponentInfo.get(agentId).getOpponentAverage()[1];
				if (value < minValue
						&& (opponentInfo.get(agentId)
								.getOpponentLastAction() instanceof Offer)
						&& opponentCurBid != opponentPreBid) {
					minValue = value;
					minInfo = opponentInfo.get(agentId);
				}
			}
		}
		return minInfo;
	}

	public Bid findBestBid(ArrayList<Bid> bidList) {
		double maxValue = 0;
		Bid maxBid = null;
		for (Bid bid : bidList) {
			double utility = utilitySpace.getUtility(bid);
			if (utility > maxValue) {
				maxValue = utility;
				maxBid = bid;
			}
		}
		return maxBid;
	}

	public Action LearningAction() {
		Bid bid = null;
		try {
			bid = utilitySpace.getMaxUtilityBid();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		if (bid != null) {
			myBidRecord.add(bid);
			totalBidHistory.add(bid);
			bidSupport.put(bid, 0);
			myAcceptBidRecord.add(bid);
			return (new Offer(getPartyId(), bid));
		} else {
			myBidRecord.add(bidList.get(bidList.size() - 1).bid);
			totalBidHistory.add(bid);
			bidSupport.put(bid, 0);
			myAcceptBidRecord.add(bid);
			return (new Offer(getPartyId(),
					bidList.get(bidList.size() - 1).bid));
		}
	}

	public Action NegotiationAction(double minUtility) {
		// allowedBidList
		// Tit for Tat
		double time = timeline.getTime();
		double tftUtility = minUtility;
		double curUtility = minUtility;
		double offerUtility;
		if (recordTimes > 1) {
			// tftUtility = utilitySpace.getUtility(opponentCurBid)-
			// utilitySpace.getUtility(opponentPreBid);
			// curUtility =
			// utilitySpace.getUtilityWithDiscount(myBidRecord.get(myBidRecord.size()-1),
			// time)-tftUtility*discount;
			tftUtility = (utilitySpace.getUtility(opponentCurBid) - utilitySpace
					.getUtility(worthyOpponent.getAgentBidHistory().get(0)))
					* (worthyOpponent.getOpponentStandardDeviation()[1]
							+ worthyOpponent.getOpponentAverage()[1]);
			curUtility = maxUtility - tftUtility;
		}
		// if(worthyOpponent.getOpponentStandardDeviation()[0]<worthyOpponent.getOpponentStandardDeviation()[1]
		// &&
		// worthyOpponent.getOpponentAverage()[0]<worthyOpponent.getOpponentAverage()[1])
		// offerUtility = (minUtility>curUtility?minUtility:curUtility);
		// else
		offerUtility = curUtility;
		if (isPrinting) {
			System.out.println("OpponentStandardDeviation()[0]: "
					+ worthyOpponent.getOpponentStandardDeviation()[0]);
			System.out.println("OpponentStandardDeviation()[1]: "
					+ worthyOpponent.getOpponentStandardDeviation()[1]);
			System.out.println("OpponentAverage()[0]: "
					+ worthyOpponent.getOpponentAverage()[0]);
			System.out.println("OpponentAverage()[1]: "
					+ worthyOpponent.getOpponentAverage()[1]);
		}
		if (offerUtility < reservationValue)
			offerUtility = reservationValue;

		if (offerUtility < maxUtility
				- (utilitySpace.getUtility(worthyOpponent.getBestBid())
						- utilitySpace.getUtility(
								worthyOpponent.getAgentBidHistory().get(0))))
			offerUtility = maxUtility - (utilitySpace
					.getUtility(worthyOpponent.getBestBid())
					- utilitySpace.getUtility(
							worthyOpponent.getAgentBidHistory().get(0)));
		if (nextOpponentIndicator) {
			offerUtility = minUtility;
			nextOpponentIndicator = false;
		}
		if (isPrinting) {
			// System.out.println("bestBidUtility:
			// "+utilitySpace.getUtility(worthyOpponent.getBestBid()));
			// System.out.println("lastBidInHistor:
			// "+utilitySpace.getUtility(lastBidInHistory));
			// System.out.println("bestBidInPotential:
			// "+utilitySpace.getUtility(bestBidInPotential));
			System.out.println("offerUtility: " + offerUtility);
		}

		Bid bid = chooseBestBid(offerUtility);

		if (utilitySpace.getUtility(lastBidInHistory) > minUtility
				&& utilitySpace.getUtilityWithDiscount(bid, time) < utilitySpace
						.getUtilityWithDiscount(lastBidInHistory, time)
				&& !myAcceptBidRecord.contains(lastBidInHistory)
				|| (bestBidInPotential != null
						&& !myAcceptBidRecord.contains(lastBidInHistory)
						&& utilitySpace.getUtilityWithDiscount(lastBidInHistory,
								time) > utilitySpace.getUtilityWithDiscount(
										bestBidInPotential, time))) {
			if (isPrinting) {
				System.out.println("Accept in Negotiation!");
			}
			myAcceptBidRecord.add(lastBidInHistory);
			return (new Accept(getPartyId(), opponentCurBid));
		}
		if (isPrinting) {
			System.out.println("Offer in Negotiation");
		}
		myBidRecord.add(bid);
		totalBidHistory.add(bid);
		bidSupport.put(bid, 0);
		myAcceptBidRecord.add(bid);
		return (new Offer(getPartyId(), bid));
	}

	public Bid chooseBestBid(double minUtility) {
		double time = timeline.getTime();
		int ceil = (int) Math.ceil(0.5 * bidList.size());
		int exp = 2;
		// int inc;
		// find allowed bid list
		while (((ceil > 0) && (ceil < bidList.size() - 1))
				&& ((utilitySpace.getUtilityWithDiscount(bidList.get(ceil).bid,
						time) > minUtility
						&& utilitySpace.getUtilityWithDiscount(
								bidList.get(ceil + 1).bid, time) > minUtility)
						|| (utilitySpace.getUtilityWithDiscount(
								bidList.get(ceil).bid, time) <= minUtility
								&& utilitySpace.getUtilityWithDiscount(
										bidList.get(ceil + 1).bid,
										time) <= minUtility))) {
			if (utilitySpace.getUtilityWithDiscount(bidList.get(ceil).bid,
					time) > minUtility) {
				if ((int) Math.ceil(bidList.size() * Math.pow(0.5, exp)) == 0) {
					ceil -= 1;
				} else {
					ceil -= (int) Math
							.ceil(bidList.size() * Math.pow(0.5, exp));
				}
			} else {
				if ((int) Math
						.floor(bidList.size() * Math.pow(0.5, exp)) == 0) {
					ceil += 1;
				} else {
					ceil += (int) Math
							.floor(bidList.size() * Math.pow(0.5, exp));
				}
			}
			exp++;
		}
		if (ceil <= 0) {
			ceil = 0;
		} else if (ceil >= bidList.size() - 1) {
			ceil = bidList.size() - 1;
		}
		List<ComparableBid> tempBidList;
		tempBidList = bidList.subList(ceil, bidList.size());

		// find opponent most favorable bids
		ComparableBid bestBid = tempBidList.get(0);
		for (ComparableBid bid : tempBidList) {
			if (getOppUtility(bid.bid) > getOppUtility(bestBid.bid))
				bestBid = bid;
		}
		return bestBid.bid;
	}

	public double getOppUtility(Bid bid) {
		HashMap<Integer, Value> bidTemp = bid.getValues();
		double utility = 0;
		for (Issue issue : bid.getIssues()) {
			switch (issue.getType()) {
			case DISCRETE:
				utility += worthyOpponent.getValueWeights().get(issue)
						.get(bidTemp.get(issue.getNumber()));
				break;
			case INTEGER:
				IssueInteger integerIssue = (IssueInteger) issue;
				double lowerBound = integerIssue.getLowerBound();
				double upperBound = integerIssue.getUpperBound();
				double lowerWeight = utility += worthyOpponent.getIssueWeights()
						.get(issue)[0];
				double upperWeight = utility += worthyOpponent.getIssueWeights()
						.get(issue)[1];
				utility = lowerWeight + (upperWeight - lowerWeight)
						* (((ValueInteger) bidTemp.get(issue.getNumber()))
								.getValue() - lowerBound)
						/ (upperBound - lowerBound);
				break;
			default:
				try {
					throw new Exception(
							"issue type " + issue.getType() + " not supported");
				} catch (Exception e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}
		}
		return utility;
	}

	@Override
	public void receiveMessage(AgentID sender, Action action) {

		super.receiveMessage(sender, action);
		if (!(action instanceof Inform) && !agentOrder.contains(sender)) {
			agentOrder.add(sender);
		}

		Bid bid = null;

		if (isPrinting) {
			System.out.println("Sender:" + sender + ", Action:" + action);
		}

		if (action != null) {
			if (action instanceof Inform
					&& ((Inform) action).getName() == "NumberOfAgents"
					&& ((Inform) action).getValue() instanceof Integer) {
				opponentNum = ((Integer) ((Inform) action).getValue())
						.intValue();
				if (isPrinting) {
					System.out.println("OpponentNum: " + opponentNum);
				}
			} else if (action instanceof Accept) {
				bid = totalBidHistory.get(totalBidHistory.size() - 1);
				int supportNum = bidSupport.get(bid);
				bidSupport.put(bid, supportNum + 1);
				if (bidSupport.get(bid) == (opponentNum - 1)) {
					bidPotential.add(bid);
				}
				if (!opponentInfo.keySet().contains(sender)) {
					opponentInfo.put(sender, new OpponentInfo(sender));
				}
				opponentInfo.get(sender).setOpponentLastAction(action);
			} else if (action instanceof Offer) {
				bid = ((Offer) action).getBid();
				if (!opponentInfo.keySet().contains(sender)) {
					opponentInfo.put(sender, new OpponentInfo(sender));
					opponentInfo.get(sender).setBestBid(bid);
				}
				totalBidHistory.add(bid);
				bidSupport.put(bid, 1);
				opponentInfo.get(sender).updateBid(bid);
				if (utilitySpace.getUtility(
						opponentInfo.get(sender).getBestBid()) < utilitySpace
								.getUtility(bid)) {
					opponentInfo.get(sender).setBestBid(bid);
				}
				opponentInfo.get(sender).setOpponentLastAction(action);
			} else if (action instanceof EndNegotiation) {
				opponentInfo.get(sender).setOpponentLastAction(action);
			}
		}
	}

	public HashMap<Integer, Value> Int2Bid(int number) {
		HashMap<Integer, Value> issueInst = new HashMap<Integer, Value>();
		int numberTemp = number;
		int temp = 0;
		for (int i = issueList.size() - 1; i >= 0; i--) {
			temp = numberTemp
					% (issue2Value.get(issueList.get(i).getNumber()).size());
			issueInst.put(issueList.get(i).getNumber(),
					issue2Value.get(issueList.get(i).getNumber()).get(temp));
			numberTemp = (numberTemp - temp)
					/ (issue2Value.get(issueList.get(i).getNumber()).size());
		}
		return issueInst;
	}

	private class ComparableBid implements Comparable<ComparableBid> {
		public Bid bid;

		public ComparableBid(Bid bid) {
			this.bid = bid;
		}

		@Override
		public int compareTo(ComparableBid cBid) {
			double time = timeline.getTime();
			return (int) (utilitySpace.getUtilityWithDiscount(this.bid, time)
					* 1000)
					- (int) (utilitySpace.getUtilityWithDiscount(cBid.bid, time)
							* 1000);
		}

	}

	private class OpponentInfo {

		private AgentID agentID;
		private ArrayList<Bid> bidHistory, bestBids;
		private double opponentSum, opponentPowerSum, opponentVariance;
		private double[] opponentAverage, opponentStandardDeviation;
		private Bid bestBid;
		private HashMap<Issue, HashMap<Value, Integer>> valueWeights;
		private HashMap<Issue, double[]> issueWeights;
		private Action opponentLastAction;

		public OpponentInfo(AgentID sender) {
			this.agentID = sender;
			this.bidHistory = new ArrayList<Bid>();
			this.bestBids = new ArrayList<Bid>();
			this.opponentSum = 0;
			this.opponentPowerSum = 0;
			this.opponentVariance = 0;
			this.opponentAverage = new double[2];
			this.opponentStandardDeviation = new double[2];
			this.valueWeights = new HashMap<Issue, HashMap<Value, Integer>>();
			initializeOpponentUtilitySpace();

		}

		private void initializeOpponentUtilitySpace() {
			for (Issue issue : utilitySpace.getDomain().getIssues()) {
				switch (issue.getType()) {
				case DISCRETE:
					IssueDiscrete discreteIssue = (IssueDiscrete) issue;
					HashMap<Value, Integer> value = new HashMap<Value, Integer>();
					for (int i = 0; i < discreteIssue
							.getNumberOfValues(); i++) {
						value.put(discreteIssue.getValue(i), 0);
					}
					valueWeights.put(issue, value);
					break;
				case INTEGER:
					double[] boundWeight = new double[2];
					boundWeight[0] = 0.5;
					boundWeight[1] = 0.5;
					issueWeights.put(issue, boundWeight);
					break;
				default:
					try {
						throw new Exception("issue type " + issue.getType()
								+ " not supported");
					} catch (Exception e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					}
				}
			}

		}

		public AgentID getAgentID() {
			return agentID;
		}

		public void setAgentID(AgentID agentID) {
			this.agentID = agentID;
		}

		public ArrayList<Bid> getAgentBidHistory() {
			return bidHistory;
		}

		public void setAgentBidHistory(ArrayList<Bid> bidHistory) {
			this.bidHistory = bidHistory;
		}

		public ArrayList<Bid> getBestBids() {
			return bestBids;
		}

		public void setBestBids(ArrayList<Bid> bestBids) {
			this.bestBids = bestBids;
		}

		public double getOpponentSum() {
			return opponentSum;
		}

		public void setOpponentSum(double opponentSum) {
			this.opponentSum = opponentSum;
		}

		public double getOpponentPowerSum() {
			return opponentPowerSum;
		}

		public void setOpponentPowerSum(double opponentPowerSum) {
			this.opponentPowerSum = opponentPowerSum;
		}

		public double getOpponentVariance() {
			return opponentVariance;
		}

		public void setOpponentVariance(double opponentVariance) {
			this.opponentVariance = opponentVariance;
		}

		public double[] getOpponentAverage() {
			return opponentAverage;
		}

		public void setOpponent(double[] opponentAverage) {
			this.opponentAverage = opponentAverage;
		}

		public double[] getOpponentStandardDeviation() {
			return opponentStandardDeviation;
		}

		public void setOpponentStandardDeviation(
				double[] opponentStandardDeviation) {
			this.opponentStandardDeviation = opponentStandardDeviation;
		}

		public Bid getBestBid() {
			return bestBid;
		}

		public void setBestBid(Bid bestBid) {
			this.bestBid = bestBid;
		}

		public HashMap<Issue, HashMap<Value, Integer>> getValueWeights() {
			return valueWeights;
		}

		public void setValueWeights(
				HashMap<Issue, HashMap<Value, Integer>> valueWeights) {
			this.valueWeights = valueWeights;
		}

		public HashMap<Issue, double[]> getIssueWeights() {
			return issueWeights;
		}

		public void setIssueWeights(HashMap<Issue, double[]> issueWeights) {
			this.issueWeights = issueWeights;
		}

		public Action getOpponentLastAction() {
			return opponentLastAction;
		}

		public void setOpponentLastAction(Action opponentLastAction) {
			this.opponentLastAction = opponentLastAction;
		}

		public void updateBid(Bid bid) {
			this.bidHistory.add(bid);
			// modeling opponent preference
			for (Issue issue : bid.getIssues()) {
				switch (issue.getType()) {
				case DISCRETE:
					HashMap<Value, Integer> value = valueWeights.get(issue);
					int times = value.get(bid.getValue(issue.getNumber()));
					if (recordTimes > 1
							&& bid.getValue(issue.getNumber()) == bidHistory
									.get(bidHistory.size() - 2)
									.getValue(issue.getNumber())) {
						value.put(bid.getValue(issue.getNumber()),
								times + frequencyReward + unchangeReward);
						valueWeights.put(issue, value);
					}
					value.put(bid.getValue(issue.getNumber()),
							times + frequencyReward);
					valueWeights.put(issue, value);
					break;
				case INTEGER:
					int issueNum = ((IssueInteger) issue).getNumber();
					Value opponentValue = bid.getValue(issueNum);
					int opponentValueInteger = ((ValueInteger) opponentValue)
							.getValue();

					int upperBound = ((IssueInteger) issue).getUpperBound();
					int lowerBound = ((IssueInteger) issue).getLowerBound();
					double midPoint = Math.ceil(
							lowerBound + (upperBound - lowerBound) / 2) + 1;

					if (midPoint > opponentValueInteger) {
						double distanceFromMidPoint = midPoint
								- opponentValueInteger;
						double normalizedDistanceFromMidPoint = distanceFromMidPoint
								/ (midPoint - lowerBound);

						double total = 1;
						double lowerBoundWeight = issueWeights.get(issue)[0];
						double upperBoundWeight = issueWeights.get(issue)[1];

						double newLowEndEvaluation = lowerBoundWeight
								+ lowerBoundWeight
										* normalizedDistanceFromMidPoint
										* Math.pow(1 - timeline.getCurrentTime()
												/ timeline.getTotalTime(),
												learningExp);
						double highEndEvaluation = upperBoundWeight;

						if (newLowEndEvaluation > 1) {
							total = newLowEndEvaluation + highEndEvaluation;
						}

						double[] boundWeight = new double[2];
						boundWeight[0] = newLowEndEvaluation / total;
						boundWeight[1] = highEndEvaluation / total;

						issueWeights.put(issue, boundWeight);
					} else {
						double distanceFromMidPoint = opponentValueInteger
								- midPoint + 1;
						double normalizedDistanceFromMidPoint = distanceFromMidPoint
								/ (upperBound - midPoint + 1);
						double total = 1;
						double lowerBoundWeight = issueWeights.get(issue)[0];
						double upperBoundWeight = issueWeights.get(issue)[1];

						double newHighEndEvaluation = upperBoundWeight
								+ upperBoundWeight
										* normalizedDistanceFromMidPoint
										* Math.pow(1 - timeline.getCurrentTime()
												/ timeline.getTotalTime(),
												learningExp);
						double lowEndEvaluation = lowerBoundWeight;

						if (newHighEndEvaluation > 1) {
							total = newHighEndEvaluation + lowEndEvaluation;
						}

						double[] boundWeight = new double[2];
						boundWeight[0] = lowEndEvaluation / total;
						boundWeight[1] = newHighEndEvaluation / total;

						issueWeights.put(issue, boundWeight);
					}
					break;
				default:
					try {
						throw new Exception("issue type " + issue.getType()
								+ " not supported");
					} catch (Exception e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					}
				}

			}
			// learning opponent behavior
			if (opponentAverage != null && opponentStandardDeviation != null) {
				opponentAverage[0] = opponentAverage[1];
				opponentStandardDeviation[0] = opponentStandardDeviation[1];
			} else {
				opponentAverage[0] = 0;
				opponentStandardDeviation[0] = 0;
			}
			opponentSum += utilitySpace.getUtility(bid);
			opponentAverage[1] = opponentSum / recordTimes;
			opponentPowerSum += Math.pow(utilitySpace.getUtility(bid), 2);
			opponentVariance = (opponentPowerSum / recordTimes)
					- Math.pow(opponentAverage[1], 2);
			opponentStandardDeviation[1] = Math
					.sqrt((opponentVariance >= 0 ? opponentVariance : 0));
		}

		public boolean containsBid(Bid bid) {
			return bidHistory.contains(bid);
		}
	}

	@Override
	public String getDescription() {
		return "ANAC2016";
	}
}