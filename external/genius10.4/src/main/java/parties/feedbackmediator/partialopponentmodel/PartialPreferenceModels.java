package parties.feedbackmediator.partialopponentmodel;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import genius.core.AgentID;
import genius.core.Bid;
import genius.core.actions.GiveFeedback;
import genius.core.issue.Issue;
import genius.core.issue.Value;

public class PartialPreferenceModels {
	/*
	 * The models for each agent
	 */
	private Map<AgentID, ValuePreferenceGraphMap> preferenceOrderingMap;
	private List<Issue> issues;

	/**
	 * 
	 * @param firstBid
	 *            the first bid that was placed.
	 * @param feedbacks
	 *            the feedbacks received on the first bid. This is only used to
	 *            create {@link #preferenceOrderingMap}
	 */
	public PartialPreferenceModels(Bid firstBid, List<GiveFeedback> feedbacks) {
		// maybe we can get the bid directly and use also domain knowledge
		// later...

		preferenceOrderingMap = new HashMap<>();

		issues = firstBid.getIssues();
		// Add all agents in feedbacks to the {@link #preferenceOrderingMap}

		for (GiveFeedback feedback : feedbacks) {
			AgentID partyID = feedback.getAgent();

			if (!preferenceOrderingMap.containsKey(partyID)) {
				preferenceOrderingMap.put(partyID, new ValuePreferenceGraphMap(firstBid));
			}
		}

	}

	public void init() {

	}

	public void updateIssuePreferenceList(int issueIndex, Value previousValue, Value currentValue,
			List<GiveFeedback> feedbacks) {

		/**
		 * Update all preferences with the new feedback
		 */
		for (GiveFeedback feedback : feedbacks) {
			preferenceOrderingMap.get(feedback.getAgent()).updateValuePreferenceGraph(issueIndex, previousValue,
					currentValue, feedback.getFeedback());

		}
	}

	public boolean mayImproveMajority(int issueIndex, Value previousValue, Value newValue) {

		int count = 0;

		for (ValuePreferenceGraphMap partyPreferenceMap : preferenceOrderingMap.values()) {

			if (partyPreferenceMap.isLessPreferredThan(issueIndex, newValue, previousValue))
				count++;
		}

		if (count < ((double) preferenceOrderingMap.size() / 2))
			return true;
		else
			return false;
	}

	public boolean mayImproveAll(int issueIndex, Value previousValue, Value newValue) {

		for (ValuePreferenceGraphMap partyPreferenceMap : preferenceOrderingMap.values()) {

			if (partyPreferenceMap.isLessPreferredThan(issueIndex, newValue, previousValue))
				return false;
		}
		return false;
	}

	public Value getNashValue(int issueIndex) {

		Value nashValue = null;
		double nashProduct = -1.0;
		double currentProduct;
		/*
		 * if there exists only one value, return it.
		 */
		if (getFirstPrefOrdering().getAllValues(issueIndex).size() == 1)
			return getFirstPrefOrdering().getAllValues(issueIndex).get(0);

		for (Value currentValue : getFirstPrefOrdering().getAllValues(issueIndex)) {

			currentProduct = 1.0;
			for (ValuePreferenceGraphMap valueMap : preferenceOrderingMap.values()) {
				currentProduct *= valueMap.getEstimatedUtility(issueIndex, currentValue);
			}
			if (currentProduct > nashProduct) {
				nashProduct = currentProduct;
				nashValue = currentValue;
			}
		}

		return nashValue;

	}

	public ArrayList<Value> getNashValues(int issueIndex) {

		ArrayList<Value> nashValues = new ArrayList<Value>();
		double nashProduct = -1;
		double currentProduct;

		/*
		 * if there exists only one value, return it.
		 */
		if (getFirstPrefOrdering().getAllValues(issueIndex).size() == 1)
			nashValues.add(getFirstPrefOrdering().getAllValues(issueIndex).get(0));
		else {

			for (Value currentValue : getFirstPrefOrdering().getAllValues(issueIndex)) {

				currentProduct = 1.0;
				for (ValuePreferenceGraphMap valueMap : preferenceOrderingMap.values()) {
					currentProduct *= valueMap.getEstimatedUtility(issueIndex, currentValue);
				}
				if (currentProduct > nashProduct) {
					nashValues.clear();
					nashProduct = currentProduct;
					nashValues.add(currentValue);
				} else if (currentProduct == nashProduct)
					nashValues.add(currentValue);
			}
		}

		return nashValues;

	}

	public double estimateSumUtility(Bid currentBid) throws Exception {

		double utility = 0.0;
		for (ValuePreferenceGraphMap valueMap : preferenceOrderingMap.values()) {
			utility += valueMap.estimateUtility(currentBid);
		}
		return utility;

	}

	public double estimateProductUtility(Bid currentBid) throws Exception {

		double utility = 1;
		for (ValuePreferenceGraphMap valueMap : preferenceOrderingMap.values()) {
			utility *= valueMap.estimateUtility(currentBid);
		}
		return utility;

	}

	/**
	 * because of the time constraint, I wrote the simple sorting but not
	 * efficient (since there are no much nash bids, it will not be a problem)
	 */
	public void sortBidsWrtSumUtility(ArrayList<Bid> bidList) throws Exception {

		for (int i = 0; i < bidList.size(); i++) {

			for (int j = i + 1; j < bidList.size(); j++) {

				if (estimateSumUtility(bidList.get(i)) < estimateSumUtility(bidList.get(j))) {
					Bid temp = new Bid(bidList.get(i));
					bidList.set(i, bidList.get(j));
					bidList.set(j, temp);
				}

			}
		}
	}

	/**
	 * because of the time constraint, I wrote the simple sorting but not
	 * efficient (since there are no much nash bids, it will not be a problem)
	 */
	public void sortBidsWrtProductUtility(ArrayList<Bid> bidList) throws Exception {

		for (int i = 0; i < bidList.size(); i++) {

			for (int j = i + 1; j < bidList.size(); j++) {

				if (estimateProductUtility(bidList.get(i)) < estimateProductUtility(bidList.get(j))) {
					Bid temp = new Bid(bidList.get(i));
					bidList.set(i, bidList.get(j));
					bidList.set(j, temp);
				}

			}
		}
	}

	public ArrayList<Bid> estimatePossibleNashBids(Bid sampleBid) throws Exception {

		ArrayList<Bid> nashBids = new ArrayList<Bid>();
		HashMap<Integer, ArrayList<Value>> nashIssueValues = new HashMap<Integer, ArrayList<Value>>();

		Bid firstBid = new Bid(sampleBid);

		for (Issue currentIssue : issues) {
			nashIssueValues.put(currentIssue.getNumber(), getNashValues(currentIssue.getNumber()));
			firstBid = firstBid.putValue(currentIssue.getNumber(),
					nashIssueValues.get(currentIssue.getNumber()).get(0));
		}

		nashBids.add(firstBid);

		int currentIndex;
		for (Issue currentIssue : issues) {
			currentIndex = currentIssue.getNumber();

			for (Value currentValue : nashIssueValues.get(currentIndex)) {

				for (int i = 0; i < nashBids.size(); i++) {
					Bid currentBid = new Bid(nashBids.get(i));
					currentBid = currentBid.putValue(currentIndex, currentValue);
					if (!nashBids.contains(currentBid))
						nashBids.add(currentBid);
				}
			}

		}

		sortBidsWrtSumUtility(nashBids);
		return nashBids;
	}

	public ArrayList<Value> getIncomparableValues(int issueIndex, Value currentValue) {

		ArrayList<Value> incomparableValues = new ArrayList<Value>();
		Value incomparable;
		for (ValuePreferenceGraphMap valueMap : preferenceOrderingMap.values()) {
			incomparable = valueMap.getIncomparableValue(issueIndex, currentValue);
			if (incomparable != null)
				incomparableValues.add(incomparable);
		}

		return incomparableValues;

	}

	public Value getIncomparableValue(int issueIndex, Value currentValue, Random random) {

		ArrayList<Value> incomparableValues = getIncomparableValues(issueIndex, currentValue);

		if (incomparableValues.size() == 0)
			return null;

		return (incomparableValues.get(random.nextInt(incomparableValues.size())));

	}

	public ArrayList<Value> getAllPossibleValues(int issueIndex) {
		return getFirstPrefOrdering().getAllValues(issueIndex);
	}

	/**
	 * Find a value not yet used by first agent.
	 * 
	 * @param issueIndex
	 * @return
	 */
	public Value getMissingValue(int issueIndex) {
		return getFirstPrefOrdering().getMissingValue(issueIndex);
	}

	@Override
	public String toString() {
		StringBuffer buffy = new StringBuffer("Partial Preference Model");

		for (AgentID agentID : preferenceOrderingMap.keySet()) {
			buffy.append("\n For party -" + agentID + "\n");
			buffy.append(preferenceOrderingMap.get(agentID).toString());
		}

		return (buffy.toString());

	}

	/**
	 * HACK Callers of this are using a hack: they use the first agent's info to
	 * access info that is shared between the agents.
	 * 
	 * @return first agent's {@link #preferenceOrderingMap}
	 */
	private ValuePreferenceGraphMap getFirstPrefOrdering() {
		if (preferenceOrderingMap.isEmpty()) {
			throw new IllegalStateException("no agents registered yet");
		}
		return preferenceOrderingMap.values().iterator().next();
	}

}