package agents.anac.y2012.MetaAgent.agents.SimpleAgentNew;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import genius.core.Agent;
import genius.core.Bid;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.ActionWithBid;
import genius.core.actions.Offer;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.Value;
import genius.core.issue.ValueDiscrete;
import genius.core.utility.AdditiveUtilitySpace;

/**
 * Erez Shalom 27325372 Naseem Biadsy 037733029 Fridman Iliya 309411882
 */
public class SimpleAgentNew extends Agent {
	private Action actionOfPartner = null;

	// just here to suggest possibilities, not used in this agent.
	// private int sessionNumber;
	// private int sessionTotalNumber;

	private final int FIRST_THR = 180;
	private final int SECOND_THR = 90;
	private final int THIRD_THR = 30;
	private final int FOURTH_THR = 1;
	private final int FIFTH_THR = 0;
	private final int MAX_BID_TIME = 5;
	private final int TEMP_FIRST_THR = 360;
	private static double MAX_UTILITY = 0;

	Map<Bid, Integer> offeredBidsCounterMap = new HashMap<Bid, Integer>();
	Map<Bid, Integer> rejectedBidsCounterMap = new HashMap<Bid, Integer>();
	Map<Value, Integer> itemsCounterMap = new HashMap<Value, Integer>();

	/**
	 * init is called when a next session starts with the same opponent.
	 */

	public void init() {
		super.init();
		try {
			Bid maxBid = utilitySpace.getMaxUtilityBid();
			MAX_UTILITY = utilitySpace.getUtility(maxBid);
		} catch (Exception e) {
			e.printStackTrace();
		}

	}

	public void init(int sessionNumberP, int sessionTotalNumberP,
			Date startTimeP, Integer totalTimeP, AdditiveUtilitySpace us) {

	}

	public void ReceiveMessage(Action opponentAction) {

		actionOfPartner = opponentAction;
	}

	public Action chooseAction() {
		Action action = null;
		Bid partnerBid = null;
		try {
			if (actionOfPartner == null)
				action = chooseMaxBidAction();
			if (actionOfPartner instanceof Offer) {
				partnerBid = ((Offer) actionOfPartner).getBid();

				if (offeredBidsCounterMap.get(partnerBid) == null)
					offeredBidsCounterMap.put(partnerBid, 0);

				offeredBidsCounterMap.put(partnerBid,
						offeredBidsCounterMap.get(partnerBid) + 1);
				List<Issue> issues = utilitySpace.getDomain().getIssues();
				for (Issue lIssue : issues) {
					Value issueVal = partnerBid.getValue(lIssue.getNumber());
					if (itemsCounterMap.get(issueVal) == null) {
						itemsCounterMap.put(issueVal, 0);
					}
					itemsCounterMap.put(issueVal,
							itemsCounterMap.get(issueVal) + 1);
				}

				double time = ((new Date()).getTime() - startTime.getTime()) / (1000.);
				boolean accept = acceptStrategy(partnerBid, time);
				if (accept)
					action = new Accept(this.getAgentID(), partnerBid);
				else
					action = chooseBidAction(time);
			}

		} catch (Exception e) {
			System.out.println("Exception in ChooseAction:" + e.getMessage());
			action = new Accept(this.getAgentID(), partnerBid); // best guess if
																// things go
			// wrong.
		}
		return action;
	}

	private double getWeight(Bid bid) {
		double weight = 0;
		List<Issue> issues = utilitySpace.getDomain().getIssues();
		for (Issue lIssue : issues) {
			Value issueVal;
			int counter;
			IssueDiscrete lIssueDiscrete = (IssueDiscrete) lIssue;
			double maxCounter = 0;
			for (int i = 0; i < lIssueDiscrete.getNumberOfValues(); i++) {
				int itemCounter = getItemCounter(lIssueDiscrete.getValue(i));
				if (itemCounter > maxCounter)
					maxCounter = itemCounter;
			}

			try {
				issueVal = bid.getValue(lIssue.getNumber());

				if (itemsCounterMap.get(issueVal) == null) {
					return -1;
				} else {
					counter = itemsCounterMap.get(issueVal);
					weight += counter / maxCounter;
				}
			} catch (Exception e) {
				e.printStackTrace();
			}

		}
		return weight;
	}

	private int getItemCounter(ValueDiscrete value) {
		if (itemsCounterMap.get(value) == null)
			return 0;

		return itemsCounterMap.get(value);
	}

	private boolean acceptStrategy(Bid partnerBid, double time) { // time is the
																	// passed
																	// time from
																	// beginning
																	// of the
																	// session
		double currentTime = FIRST_THR - time;
		double minRelativeUtility;
		try {
			double offeredUtility = utilitySpace.getUtility(partnerBid);

			if (currentTime >= SECOND_THR) {
				minRelativeUtility = (1 - time / TEMP_FIRST_THR) * MAX_UTILITY;
				return offeredUtility >= minRelativeUtility;
			} else if (currentTime < SECOND_THR && currentTime > FOURTH_THR) {
				minRelativeUtility = ((180 - time) / SECOND_THR) * MAX_UTILITY;
				if (minRelativeUtility <= 0.66)
					minRelativeUtility *= 1.5;
				System.out.println("second iterval, minutility="
						+ minRelativeUtility);
				return offeredUtility >= minRelativeUtility;
			} else if (currentTime <= FOURTH_THR) {
				return offeredUtility > 0;
			}

		} catch (Exception e) {
			e.printStackTrace(); // To change body of catch statement use File |
									// Settings | File Templates.
		}

		return false;
	}

	private Action chooseMaxBidAction() {
		Bid maxBid = null;
		try {
			maxBid = utilitySpace.getMaxUtilityBid();
		} catch (Exception e) {
			System.out.println("Problem with received bid:" + e.getMessage()
					+ ". cancelling bidding");
		}
		if (maxBid == null)
			return new Accept(this.getAgentID(),
					((ActionWithBid) actionOfPartner).getBid());
		return (new Offer(this.getAgentID(), maxBid));
	}

	private Action chooseBidAction(double time) {

		double currentTime = FIRST_THR - time;
		double utilityToSearch;
		Bid nextBid = null;
		List<Bid> bidsLst = getAllBids();

		if (currentTime > SECOND_THR) { // 90 < t < 180
			utilityToSearch = 0.96;// (1 - time / FIRST_THR) * MAX_UTILITY;
			nextBid = FindClosestBid(utilityToSearch, bidsLst);
		} else if (currentTime <= SECOND_THR
				&& currentTime >= (SECOND_THR - MAX_BID_TIME)) { // 85 < t < 90
			try {
				nextBid = utilitySpace.getMaxUtilityBid();
			} catch (Exception e) {
				e.printStackTrace();
			}
		} else if (currentTime < (SECOND_THR - MAX_BID_TIME)
				&& currentTime >= THIRD_THR) { // 30 =< t < 85
			utilityToSearch = ((180 - time) / SECOND_THR) * MAX_UTILITY;
			if (utilityToSearch <= 0.66)
				utilityToSearch *= 1.5;
			System.out.println("utility to search for proposa"
					+ utilityToSearch);
			nextBid = FindClosestBid(utilityToSearch, bidsLst);
			try {
				System.out.println("found bid utility"
						+ utilitySpace.getUtility(nextBid));
			} catch (Exception e) {
				e.printStackTrace();
			}
		} else if (currentTime >= (THIRD_THR - MAX_BID_TIME)
				&& currentTime < THIRD_THR) { // 25 =< t < 30
			try {
				nextBid = utilitySpace.getMaxUtilityBid();
			} catch (Exception e) {
				e.printStackTrace();
			}
		} else if (currentTime >= FOURTH_THR
				&& currentTime < (THIRD_THR - MAX_BID_TIME)) // 1 < t < 25
		{
			try {
				nextBid = utilitySpace.getMaxUtilityBid();
			} catch (Exception e) {
				e.printStackTrace();
			}
		} else if (currentTime >= FIFTH_THR) // 0 < t < 1
		{
			try {
				nextBid = utilitySpace.getMaxUtilityBid();
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
		if (rejectedBidsCounterMap.get(nextBid) == null)
			rejectedBidsCounterMap.put(nextBid, 0);

		rejectedBidsCounterMap.put(nextBid,
				rejectedBidsCounterMap.get(nextBid) + 1);

		Bid opponentBid = getOpponentBestBid();

		try {

			System.out.println(utilitySpace.getUtility(nextBid));
			System.out.println(utilitySpace.getUtility(opponentBid));
			System.out.println("-------");

			if (utilitySpace.getUtility(nextBid) < utilitySpace
					.getUtility(opponentBid))
				return (new Offer(this.getAgentID(), opponentBid));
		} catch (Exception e) {
			e.printStackTrace();
		}

		return (new Offer(this.getAgentID(), nextBid));

	}

	// find the closest bid to the specified utility

	private Bid getOpponentBestBid() {
		Set<Bid> opponentBidsSet = offeredBidsCounterMap.keySet();
		List<Bid> opponentBidsLst = new ArrayList<Bid>();
		opponentBidsLst.addAll(opponentBidsSet);
		Collections.sort(opponentBidsLst, new BidComparator());

		Bid maxBid = opponentBidsLst.get(0);
		for (int i = 0; i < opponentBidsLst.size() && i < 5; i++) {
			try {
				if (utilitySpace.getUtility(opponentBidsLst.get(i)) < utilitySpace
						.getUtility(maxBid))
					maxBid = opponentBidsLst.get(i);
			} catch (Exception e) {
				e.printStackTrace();
			}
		}

		return maxBid;
	}

	private Bid FindClosestBid(double utilityToSearch, List<Bid> bidsLst) {

		double foundUtility = 0;
		int foundBidIndex = 0;
		for (int i = 0; i < bidsLst.size(); i++) {
			Bid currentBid = bidsLst.get(i);
			double currentBidUtility = this.getUtility(currentBid);

			if ((currentBidUtility <= utilityToSearch)
					&& currentBidUtility > foundUtility) {
				foundUtility = currentBidUtility;
				foundBidIndex = i;
			}
		}

		return bidsLst.get(foundBidIndex);
	}

	/**
	 * @return List<Bid> which includes all the possible bids in the current
	 *         domain
	 */

	private List<Bid> getAllBids() {
		List<List<ValueDiscrete>> allLst = new ArrayList<List<ValueDiscrete>>();
		List<Integer> issuesIdLst = new ArrayList<Integer>();

		List<Issue> issues = utilitySpace.getDomain().getIssues();
		for (Issue lIssue : issues) {
			issuesIdLst.add(lIssue.getNumber());
			switch (lIssue.getType()) {
			case DISCRETE:

				IssueDiscrete lIssueDiscrete = (IssueDiscrete) lIssue;
				if (allLst.isEmpty()) {
					for (int i = 0; i < lIssueDiscrete.getNumberOfValues(); i++) {
						List<ValueDiscrete> l = new ArrayList<ValueDiscrete>();
						l.add(lIssueDiscrete.getValue(i));
						allLst.add(l);
					}
				} else {
					List<List<ValueDiscrete>> tempValuesLst = new ArrayList<List<ValueDiscrete>>();
					tempValuesLst.addAll(allLst);
					allLst = new ArrayList<List<ValueDiscrete>>();

					for (List<ValueDiscrete> oldLst : tempValuesLst)
						for (int i = 0; i < lIssueDiscrete.getNumberOfValues(); i++) {
							List<ValueDiscrete> l = new ArrayList<ValueDiscrete>();
							l.addAll(oldLst);
							l.add(lIssueDiscrete.getValue(i));
							allLst.add(l);
						}
				}
			}

		}
		List<Bid> bidsLst = convertValuesListToBids(allLst, issuesIdLst);

		return bidsLst;
	}

	/**
	 * @param lst
	 *            : list of different issues combination
	 * @param issuesIdLst
	 *            : issues numbers(ID) as they ordered in the combinations
	 * 
	 * @return list of Bids which are converted from the list lst
	 */
	private List<Bid> convertValuesListToBids(List<List<ValueDiscrete>> lst,
			List<Integer> issuesIdLst) {

		List<Bid> bidsLst = new ArrayList<Bid>();
		try {
			for (List<ValueDiscrete> valuesLst : lst) {
				HashMap<Integer, Value> values = new HashMap<Integer, Value>();
				for (int i = 0; i < valuesLst.size(); i++) {
					values.put(issuesIdLst.get(i), valuesLst.get(i));
				}

				bidsLst.add(new Bid(utilitySpace.getDomain(), values));
			}
		} catch (Exception e) {
			e.printStackTrace();
		}

		return bidsLst;
	}

	class BidComparator implements Comparator {

		public int compare(Object bid1, Object bid2) {

			double bid1Weight = getWeight((Bid) bid1);
			double bid2Weight = getWeight((Bid) bid2);

			if (bid1Weight > bid2Weight)
				return 1;
			else if (bid1Weight < bid2Weight)
				return -1;
			else
				return 0;
		}

	}

}
