package agents.anac.y2017.simpleagent;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import agents.BidComparator;
import genius.core.AgentID;
import genius.core.Bid;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.Offer;
import genius.core.issue.Issue;
import genius.core.issue.Value;
import genius.core.list.Tuple;
import genius.core.persistent.StandardInfo;
import genius.core.persistent.StandardInfoList;
import genius.core.timeline.TimeLineInfo;
import genius.core.utility.AbstractUtilitySpace;

/*
* *TODO* We need to choose our constraints specific to domain, must update some values during the negotiation for
* better results
 */
class Predictor {

	private AbstractUtilitySpace utilitySpace;
	private TimeLineInfo timeline;
	private HashMap<String, ArrayList<Offer>> allOffers = new HashMap<>();
	private HashMap<Integer, Value> mostWantedValues = new HashMap<>();
	private List<Bid> compatibleBids = new ArrayList<>();
	private double maxReservationValue = 1;
	private double minReservationValue = 0.83;
	private double currentReservationValue = 0;
	private boolean isBiddedBefore = false;

	public Predictor(AbstractUtilitySpace utilitySpace, TimeLineInfo timeline) {
		this.utilitySpace = utilitySpace;
		this.timeline = timeline;
	}

	public void storeAgentOffer(Offer offer) {
		if (allOffers.containsKey(offer.getAgent().getName()))
			allOffers.get(offer.getAgent().getName()).add(offer);
		else {
			ArrayList<Offer> newOfferList = new ArrayList<>();
			newOfferList.add(offer);
			allOffers.put(offer.getAgent().getName(), newOfferList);
		}
	}

	private boolean isBidAcceptable(Bid bid) {
		return utilitySpace.getUtility(bid) > currentReservationValue;
	}

	private void updateReservationValue() {
		double totalRound = timeline.getTotalTime();
		double currentRound = timeline.getCurrentTime();
		double resDifference = maxReservationValue - minReservationValue;
		double optimizer = resDifference / totalRound;
		currentReservationValue = maxReservationValue - (currentRound * optimizer);
		if (currentRound > totalRound * 0.8)
			System.out.println("Current Rez: " + currentReservationValue);
	}

	private ArrayList<Bid> getAllBids() {
		ArrayList<Bid> allBids = new ArrayList<>();
		for (ArrayList<Offer> offers : allOffers.values()) {
			for (Offer offer : offers) {
				allBids.add(offer.getBid());
			}
		}

		if (allBids.isEmpty()) {
			return null;
		}
		return allBids;
	}

	private List<List<Value>> getAllValues(ArrayList<Bid> allBids) {
		List<List<Value>> allValues = new ArrayList<>();

		List<Issue> allIssues = getAllBids().get(0).getIssues();
		for (int i = 1; i <= allIssues.size(); i++) {
			List<Value> values = new ArrayList<>();

			for (Bid bid : allBids) {
				if (!values.contains(bid.getValue(i)))
					values.add(bid.getValue(i));
			}
			allValues.add(values);
		}
		return allValues;
	}

	private void updateMostWantedValues() {
		mostWantedValues = new HashMap<>();
		ArrayList<Bid> allBids = getAllBids();

		List<Issue> allIssues = allBids.get(0).getIssues();

		for (int i = 1; i <= allIssues.size(); i++) {
			List<Value> values = new ArrayList<>();

			for (Bid bid : allBids) {
				values.add(bid.getValue(i));
			}

			Value mostCommon = getMostCommonValue(values);
			mostWantedValues.put(i, mostCommon);
		}

	}

	private Value getMostCommonValue(List<Value> values) {
		Map<Value, Integer> map = new HashMap<>();

		for (Value value : values) {
			Integer val = map.get(value);
			map.put(value, val == null ? 1 : val + 1);
		}

		Map.Entry<Value, Integer> max = null;

		for (Map.Entry<Value, Integer> e : map.entrySet()) {
			if (max == null || e.getValue() > max.getValue())
				max = e;
		}

		return max.getKey();
	}

	private Bid generateBid() throws Exception {
		updateMostWantedValues();

		if (mostWantedValues.isEmpty())
			return utilitySpace.getMaxUtilityBid();

		calculateBidList();
		double totalRound = timeline.getTotalTime();
		double currentRound = timeline.getCurrentTime();

		double optimizer = currentRound / totalRound;
		double index = compatibleBids.size() * optimizer;
		if (index == compatibleBids.size())
			index = compatibleBids.size() - 1.0;
		return compatibleBids.get((int) index);
	}

	private void calculateBidList() throws Exception {
		compatibleBids = new ArrayList<>();
		ArrayList<Bid> allBids = getAllBids();
		List<List<Value>> allValues = getAllValues(allBids);
		List<List<Value>> cartesianOfAllValues = new CartesianCalculator().calculate(allValues);
		for (int i = 0; i < cartesianOfAllValues.size(); i++) {
			HashMap<Integer, Value> map = new HashMap<>();
			for (int j = 0; j < cartesianOfAllValues.get(i).size(); j++) {
				map.put(j + 1, cartesianOfAllValues.get(i).get(j));
			}
			Bid bid = new Bid(utilitySpace.getDomain(), map);
			if (utilitySpace.getUtility(bid) > currentReservationValue)
				compatibleBids.add(bid);
		}
		if (compatibleBids.isEmpty())
			compatibleBids.add(utilitySpace.getMaxUtilityBid());
		BidComparator comparator = new BidComparator(utilitySpace);
		compatibleBids.sort(comparator);

	}

	public void setHistoryAndUpdateThreshold(StandardInfoList history) {
		double maxUtilsSum = 0;
		double countedOffers = 0;
		for (StandardInfo prevHistory : history) {
			int numberOfAgents = prevHistory.getAgentProfiles().size();
			List<Tuple<String, Double>> agentUtilities = prevHistory.getUtilities();
			int agentUtilitySize = agentUtilities.size();
			List<Tuple<String, Double>> finalUtilities = agentUtilities.subList(agentUtilitySize - numberOfAgents,
					agentUtilitySize);
			double maxUtility = 0;
			for (Tuple<String, Double> agentUtility : finalUtilities) {
				if (agentUtility.get2() > maxUtility)
					maxUtility = agentUtility.get2();
			}
			if (maxUtility != 0) {
				maxUtilsSum += maxUtility;
				countedOffers++;
			}
		}
		if (maxUtilsSum != 0)
			minReservationValue = maxUtilsSum * 0.93 / countedOffers;
	}

	public Action generateAction(List<Class<? extends Action>> validActions, Bid lastReceivedBid, AgentID agentId) {
		try {
			if (!validActions.contains(Accept.class)) {
				return new Offer(agentId, utilitySpace.getMaxUtilityBid());
			} else {
				updateReservationValue();
				if (isBidAcceptable(lastReceivedBid)) {
					return new Accept(agentId, lastReceivedBid);
				} else {
					if (!isBiddedBefore) {
						isBiddedBefore = true;
						return new Offer(agentId, utilitySpace.getMaxUtilityBid());
					}
					Bid bid = generateBid();
					return new Offer(agentId, generateBid());
				}
			}
		} catch (Exception e) {
			e.printStackTrace();
			return new Accept(agentId, lastReceivedBid);
		}
	}
}
