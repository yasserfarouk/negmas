package agents.anac.y2014.AgentYK;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;

import agents.SimpleAgent;
import genius.core.Agent;
import genius.core.Bid;
import genius.core.BidHistory;
import genius.core.Domain;
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

public class AgentYK extends Agent {

	private Action actionOfPartner = null;
	/**
	 * Note: {@link SimpleAgent} does not account for the discount factor in its
	 * computations
	 */
	private static double MINIMUM_BID_UTILITY = 0.0;

	private BidElementHistory opponentBidElementHistory;
	private PairBidElementHistory opponentPairBidElementHistory;
	private BidElementHistory myBidElementHistory;
	private PairBidElementHistory myPairBidElementHistory;
	private BidHistory opponentHistory;
	private double oppAverage;
	private double myAverage;
	private BidHistory myHistory;
	private Bid oppMaxBid = null;
	private Bid myMinBid = null;
	private HashSet<String> opponentBidPool;
	private HashSet<String> myBidPool;
	private long totalCombiNum;
	private double logTCN;
	private List<Integer> issueNrs;
	private int stopNr;
	private int seedNr;
	private double now;
	private double baseBE = 0;
	private double basePBE = 0;
	private double basedBE = 0;
	private double basedPBE = 0;
	private double allowedAdditional = 0.0;
	private double df;
	private double noValueUtil = 1.05;
	private boolean timeBonus = true;
	private double termRefresh;
	private int coefRefresh = 1;
	private int preRefresh = 0;
	private LinkedList<Double> timeQue = new LinkedList<Double>();
	private int linkedNum;
	private double preTime = 0.0;
	private int leftTime = -1;
	private int totalTime = -1;
	private int FaceOfBuddha = 3;

	@Override
	public void init() {
		MINIMUM_BID_UTILITY = utilitySpace.getReservationValueUndiscounted();
		df = utilitySpace.getDiscountFactor();
		opponentBidElementHistory = new BidElementHistory();
		opponentPairBidElementHistory = new PairBidElementHistory();
		myBidElementHistory = new BidElementHistory();
		myPairBidElementHistory = new PairBidElementHistory();
		opponentBidPool = new HashSet<String>();
		myBidPool = new HashSet<String>();
		opponentHistory = new BidHistory();
		myHistory = new BidHistory();
		totalCombiNum = getTotalCombinationNumber();
		logTCN = Math.log(totalCombiNum);
		issueNrs = new ArrayList<Integer>();
		for (Issue issue : utilitySpace.getDomain().getIssues())
			issueNrs.add(issue.getNumber());

		stopNr = 1 + (int) Math.pow(Math.log(Math.sqrt(totalCombiNum)), 2);
		if (stopNr > 300)
			stopNr = 300;
		if (stopNr < 100)
			stopNr = 100;
		seedNr = 1 + (int) logTCN;
		if (seedNr > 30)
			seedNr = 30;
		if (seedNr < 10)
			seedNr = 10;
		try {
			myMinBid = getHillClimbBid(stopNr);
			for (int i = 0; i < seedNr; i++) {
				Bid tmpBid = getHillClimbBid(stopNr);
				if (evaluateBid(myMinBid) < evaluateBid(tmpBid)) {
					myMinBid = new Bid(tmpBid);
				}
			}
		} catch (Exception e) {

		}
		now = 0.0;
		termRefresh = 2 / df;
		linkedNum = 30 + 1;
	}

	@Override
	public String getVersion() {
		return "1.0";
	}

	@Override
	public String getName() {
		return "AgentYK";
	}

	@Override
	public void ReceiveMessage(Action opponentAction) {
		actionOfPartner = opponentAction;
	}

	@Override
	public Action chooseAction() {
		Action action = null;
		now = timeline.getTime();
		updateLeftTime();

		try {
			if (actionOfPartner == null)
				action = new Offer(getAgentID(), myMinBid);
			if (actionOfPartner instanceof Offer) {
				Bid offeredBid = ((Offer) actionOfPartner).getBid();
				opponentBidManager(offeredBid);
				if (isAccept(offeredBid)) {
					action = new Accept(getAgentID(), offeredBid);
				} else {
					if (0 <= leftTime
							&& leftTime < (5 + (double) totalTime / 30)
									* FaceOfBuddha
							&& myMinBid != null) {
						FaceOfBuddha--;
						allowedAdditional *= 1.1;
						noValueUtil = utilitySpace.getUtility(myMinBid);
						if (FaceOfBuddha == 0) {
							stopNr /= 2;
							seedNr /= 2;
							noValueUtil = (utilitySpace.getUtility(myMinBid)
									+ utilitySpace.getUtility(oppMaxBid)) / 2;
							MINIMUM_BID_UTILITY = utilitySpace
									.getUtility(oppMaxBid);
						}
					}
					if (0 <= leftTime && leftTime < 2 && myMinBid != null) {
						double diffMyOpp = utilitySpace.getUtility(myMinBid)
								- utilitySpace.getUtility(oppMaxBid);
						if (diffMyOpp <= 0.25
								|| utilitySpace.getUtility(myMinBid)
										/ 2 <= utilitySpace
												.getUtility(oppMaxBid)) {
							myBidManager(oppMaxBid);
							if (utilitySpace
									.getReservationValue() >= utilitySpace
											.getUtility(oppMaxBid))
								return new EndNegotiation(getAgentID());
							return new Offer(getAgentID(), new Bid(oppMaxBid));
						}
					}
					Bid nextBid = getHillClimbBid(stopNr);
					for (int i = 0; i < seedNr; i++) {
						Bid tmpBid = getHillClimbBid(stopNr);
						if (evaluateBid(nextBid) < evaluateBid(tmpBid)) {
							nextBid = new Bid(tmpBid);
						}
					}
					if (utilitySpace.getUtility(nextBid) < MINIMUM_BID_UTILITY)
						nextBid = new Bid(myHistory.getRandom().getBid());
					if (utilitySpace.getUtility(nextBid) < utilitySpace
							.getUtility(oppMaxBid))
						nextBid = new Bid(oppMaxBid);
					if (myBidPool.contains(nextBid.toString()))
						nextBid = new Bid(myHistory.getRandom().getBid());
					myBidManager(nextBid);
					refreshHistory();
					action = new Offer(getAgentID(), nextBid);
					if (utilitySpace.getReservationValue() >= utilitySpace
							.getUtility(myMinBid))
						action = new EndNegotiation(getAgentID());
				}
			}
		} catch (Exception e) {
			action = new Accept(getAgentID(),
					((ActionWithBid) actionOfPartner).getBid());
		}

		return action;
	}

	private void refreshHistory() throws Exception {
		if (now > coefRefresh * termRefresh
				|| (1 <= coefRefresh * termRefresh
						&& coefRefresh * termRefresh < 1.2 && now > 0.9)
				|| opponentHistory.size() + preRefresh > 1
						+ Math.pow(logTCN, 2)) {
			List<BidDetails> oppList = new ArrayList<BidDetails>(opponentHistory
					.filterBetweenUtility(oppAverage, 1.0).getHistory());
			List<BidDetails> myList = new ArrayList<BidDetails>(myHistory
					.filterBetweenUtility(0.0, myAverage).getHistory());
			opponentBidElementHistory = new BidElementHistory();
			opponentPairBidElementHistory = new PairBidElementHistory();
			myBidElementHistory = new BidElementHistory();
			myPairBidElementHistory = new PairBidElementHistory();
			opponentHistory = new BidHistory();
			myHistory = new BidHistory();
			HashSet<String> tmpOppBidPool = new HashSet<String>(
					opponentBidPool);
			HashSet<String> tmpMyBidPool = new HashSet<String>(myBidPool);
			opponentBidPool = new HashSet<String>();
			myBidPool = new HashSet<String>();
			for (BidDetails bd : myList) {
				now = bd.getTime();
				myBidManager(bd.getBid());
			}
			for (BidDetails bd : oppList) {
				now = bd.getTime();
				opponentBidManager(bd.getBid());
			}
			for (String str : tmpOppBidPool) {
				if (!opponentBidPool.contains(str))
					opponentBidPool.add(str);
			}
			for (String str : tmpMyBidPool) {
				if (!myBidPool.contains(str))
					myBidPool.add(str);
			}
			coefRefresh++;
			preRefresh = opponentHistory.size();
		}
	}

	private void opponentBidManager(Bid oppBid) throws Exception {
		double sum;

		BidElementIterator bei = new BidElementIterator(oppBid);
		PairBidElementIterator pbei = new PairBidElementIterator(oppBid);
		if (!opponentBidPool.contains(oppBid.toString())) {
			opponentBidPool.add(oppBid.toString());
			if (opponentHistory.isEmpty()) {
				opponentHistory.add(new BidDetails(oppBid,
						utilitySpace.getUtility(oppBid), now));
				while (bei.hasNext())
					opponentBidElementHistory
							.add(new BidElementDetails(bei.next(), now));
				while (pbei.hasNext())
					opponentPairBidElementHistory
							.add(new PairBidElementDetails(pbei.next(), now));
			} else {
				oppAverage = 0.0;
				for (BidDetails bd : opponentHistory.getHistory()) {
					oppAverage += bd.getMyUndiscountedUtil();
				}
				oppAverage /= opponentHistory.size();

				opponentHistory.add(new BidDetails(oppBid,
						utilitySpace.getUtility(oppBid), now));
				while (bei.hasNext())
					opponentBidElementHistory
							.add(new BidElementDetails(bei.next(), now));
				while (pbei.hasNext())
					opponentPairBidElementHistory
							.add(new PairBidElementDetails(pbei.next(), now));
			}
		}

		if (basedBE > 0 && basedPBE > 0
				&& oppAverage <= utilitySpace.getUtility(oppBid)) {
			double additional = 0.0;
			bei = new BidElementIterator(oppBid);
			sum = 0;
			while (bei.hasNext())
				sum += myBidElementHistory
						.getWeightedAppearanceCount(bei.next());
			additional += (sum / basedBE);
			pbei = new PairBidElementIterator(oppBid);
			sum = 0;
			while (pbei.hasNext())
				sum += myPairBidElementHistory
						.getWeightedAppearanceCount(pbei.next());
			additional += (sum / basedPBE);
			additional *= (Math.pow((now / df), 2) > 1 ? 1
					: Math.pow((now / df), 2));
			if (2 * additional > allowedAdditional) {
				allowedAdditional = 2 * additional;
			}
		}

		if (oppMaxBid == null) {
			oppMaxBid = new Bid(oppBid);
		} else if (utilitySpace.getUtility(oppMaxBid) < utilitySpace
				.getUtility(oppBid)) {
			oppMaxBid = new Bid(oppBid);
		}
		bei = new BidElementIterator(oppMaxBid);
		sum = 0;
		while (bei.hasNext())
			sum += opponentBidElementHistory
					.getWeightedAppearanceCount(bei.next());
		baseBE = sum;
		pbei = new PairBidElementIterator(oppMaxBid);
		sum = 0;
		while (pbei.hasNext())
			sum += opponentPairBidElementHistory
					.getWeightedAppearanceCount(pbei.next());
		basePBE = sum;
	}

	private void myBidManager(Bid myBid) throws Exception {
		BidElementIterator bei = new BidElementIterator(myBid);
		PairBidElementIterator pbei = new PairBidElementIterator(myBid);

		if (!myBidPool.contains(myBid.toString())) {
			myBidPool.add(myBid.toString());
			if (myHistory.isEmpty()) {
				myHistory.add(new BidDetails(myBid,
						utilitySpace.getUtility(myBid), now));
				while (bei.hasNext())
					myBidElementHistory
							.add(new BidElementDetails(bei.next(), now));
				while (pbei.hasNext())
					myPairBidElementHistory
							.add(new PairBidElementDetails(pbei.next(), now));
			} else {
				myAverage = 0.0;
				for (BidDetails bd : myHistory.getHistory()) {
					myAverage += bd.getMyUndiscountedUtil();
				}
				myAverage /= myHistory.size();

				myHistory.add(new BidDetails(myBid,
						utilitySpace.getUtility(myBid), now));
				while (bei.hasNext())
					myBidElementHistory
							.add(new BidElementDetails(bei.next(), now));
				while (pbei.hasNext())
					myPairBidElementHistory
							.add(new PairBidElementDetails(pbei.next(), now));
			}
		}

		if (utilitySpace.getUtility(myMinBid) > utilitySpace
				.getUtility(myBid)) {
			myMinBid = new Bid(myBid);
		}

		basedBE = basedPBE = 0;
		for (BidDetails bd : myHistory.getHistory()) {
			bei = new BidElementIterator(bd.getBid());
			while (bei.hasNext())
				basedBE += myBidElementHistory
						.getWeightedAppearanceCount(bei.next());
			pbei = new PairBidElementIterator(bd.getBid());
			while (pbei.hasNext())
				basedPBE += myPairBidElementHistory
						.getWeightedAppearanceCount(pbei.next());
		}
		basedBE /= myHistory.size();
		basedPBE /= myHistory.size();
	}

	private void updateLeftTime() {
		if (timeQue.size() < linkedNum) {
			timeQue.offer(now - preTime);
			preTime = now;
		} else {
			timeQue.poll();
			timeQue.offer(now - preTime);
			preTime = now;
			double avgTime = 0.0;
			for (int i = 1; i < linkedNum; i++) {
				avgTime += timeQue.get(i);
			}
			avgTime /= (linkedNum - 1);
			leftTime = (int) ((1 - now) / avgTime);
			totalTime = (int) (1 / avgTime);
		}
	}

	private boolean isAccept(Bid offeredBid) throws Exception {
		double offeredUtil = utilitySpace.getUtility(offeredBid);

		if (offeredUtil >= utilitySpace.getUtility(myMinBid) * 0.999)
			return true;

		return false;
	}

	private Bid getHillClimbBid(int stopNr) throws Exception {
		Bid bid = getRandomBid();
		int c = 0;
		double maxEval = evaluateBid(bid);

		while (c++ < stopNr && timeline.getTime() - now < 0.01
				&& timeline.getTime() < 1.0) {
			double r = Math.random();
			int changeNum = 1;
			for (int i = issueNrs.size(); i >= 1; i--) {
				if (r < 1.0 / (i * i)) {
					changeNum = i;
					break;
				}
			}

			Bid nextBid = nextBid(bid, changeNum);

			double eval = evaluateBid(nextBid);
			if (eval > maxEval) {
				maxEval = eval;
				bid = new Bid(nextBid);
			}
		}

		return bid;
	}

	private Bid nextBid(Bid bid, int changeNum) {
		List<Integer> changeIssueNrs = new ArrayList<Integer>();
		Random randomnr = new Random();
		Bid nextBid = new Bid(bid);

		for (int i = 0; i < changeNum; i++) {
			changeIssueNrs.add(issueNrs.get(randomnr.nextInt(issueNrs.size())));
		}

		Domain domain = utilitySpace.getDomain();
		for (Integer issueNumber : changeIssueNrs) {
			int issueN = 0;
			int i = 0;
			for (Issue issue : domain.getIssues()) {
				if (issueNumber == issue.getNumber()) {
					issueN = i;
					break;
				}
				i++;
			}
			switch (domain.getIssues().get(issueN).getType()) {
			case DISCRETE:
				IssueDiscrete lIssueDiscrete = (IssueDiscrete) domain
						.getIssues().get(issueN);
				int optionIndex = randomnr
						.nextInt(lIssueDiscrete.getNumberOfValues());
				nextBid = nextBid.putValue(issueNumber,
						lIssueDiscrete.getValue(optionIndex));
				break;
			case REAL:
				IssueReal lIssueReal = (IssueReal) domain.getIssues()
						.get(issueN);
				int optionInd = randomnr.nextInt(
						lIssueReal.getNumberOfDiscretizationSteps() - 1);
				nextBid = nextBid.putValue(issueNumber,
						new ValueReal(lIssueReal.getLowerBound() + (lIssueReal
								.getUpperBound() - lIssueReal.getLowerBound())
								* (optionInd) / (lIssueReal
										.getNumberOfDiscretizationSteps())));
				break;
			case INTEGER:
				IssueInteger lIssueInteger = (IssueInteger) domain.getIssues()
						.get(issueN);
				int optionIndex2 = lIssueInteger.getLowerBound()
						+ randomnr.nextInt(lIssueInteger.getUpperBound()
								- lIssueInteger.getLowerBound());
				nextBid = nextBid.putValue(issueNumber,
						new ValueInteger(optionIndex2));
				break;
			default:
			}
		}

		return nextBid;
	}

	private double evaluateBid(Bid bid) throws Exception {
		double eval = 0.0;
		eval = utilitySpace.getUtility(bid);

		if (baseBE > 0 && basePBE > 0) {
			double additional = 0.0;
			BidElementIterator bei = new BidElementIterator(bid);
			double sum = 0;
			while (bei.hasNext())
				sum += opponentBidElementHistory
						.getWeightedAppearanceCount(bei.next());
			additional += (sum / baseBE);
			PairBidElementIterator pbei = new PairBidElementIterator(bid);
			sum = 0;
			while (pbei.hasNext())
				sum += opponentPairBidElementHistory
						.getWeightedAppearanceCount(pbei.next());
			additional += (sum / basePBE);
			additional *= ((now / df) > 1 ? 1 : (now / df));
			if (additional > (1 + (double) (3 - FaceOfBuddha) / 8) * eval)
				additional = (1 + (double) (3 - FaceOfBuddha) / 8) * eval;
			if (MINIMUM_BID_UTILITY <= eval && eval < noValueUtil)
				eval += (allowedAdditional < additional ? allowedAdditional
						: additional);
		}

		if (timeBonus && !myBidPool.contains(bid.toString())
				&& myBidPool.size() <= opponentBidPool.size())
			eval += now;

		return eval;
	}

	private Bid getRandomBid() throws Exception {
		HashMap<Integer, Value> values = new HashMap<Integer, Value>();
		List<Issue> issues = utilitySpace.getDomain().getIssues();
		Random randomnr = new Random();

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
				throw new Exception(
						"issue type " + lIssue.getType() + " not supported.");
			}
		}
		return new Bid(utilitySpace.getDomain(), values);
	}

	private long getTotalCombinationNumber() {
		long c = 1;
		try {
			List<Issue> issues = utilitySpace.getDomain().getIssues();
			for (Issue lIssue : issues) {
				switch (lIssue.getType()) {
				case DISCRETE:
					IssueDiscrete lIssueDiscrete = (IssueDiscrete) lIssue;
					c *= lIssueDiscrete.getValues().size();
					break;
				case REAL:
					IssueReal lIssueReal = (IssueReal) lIssue;
					c *= lIssueReal.getNumberOfDiscretizationSteps();
					break;
				case INTEGER:
					IssueInteger lIssueInteger = (IssueInteger) lIssue;
					c *= (lIssueInteger.getUpperBound()
							- lIssueInteger.getLowerBound() + 1);
					break;
				default:
					break;
				}
			}
			return c;
		} catch (Exception e) {
			System.out.println(e.getMessage());
		}
		return -1;
	}

	@Override
	public String getDescription() {
		return "ANAC2014 compatible with non-linear utility spaces";
	}
}
