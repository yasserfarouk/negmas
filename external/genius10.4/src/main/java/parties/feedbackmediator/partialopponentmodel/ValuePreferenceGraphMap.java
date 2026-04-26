package parties.feedbackmediator.partialopponentmodel;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import genius.core.Bid;
import genius.core.Feedback;
import genius.core.issue.Issue;
import genius.core.issue.Value;

import java.util.Set;

public class ValuePreferenceGraphMap {
	/*
	 * for each issue, value-score pairs are kept.
	 */
	private HashMap<Integer, ValuePreferenceGraph> issuePreferenceList;
	private List<Issue> issueList;

	public ValuePreferenceGraphMap() {
		issuePreferenceList = new HashMap<Integer, ValuePreferenceGraph>();
		issueList = new ArrayList<Issue>();
	}

	public ValuePreferenceGraphMap(Bid firstBid) {

		issuePreferenceList = new HashMap<Integer, ValuePreferenceGraph>();
		issueList = firstBid.getIssues();

		Set<Entry<Integer, Value>> valueSet = firstBid.getValues().entrySet();
		Iterator<Entry<Integer, Value>> valueIterator = valueSet.iterator();

		while (valueIterator.hasNext()) {
			Map.Entry<Integer, Value> indexValuePair = (Entry<Integer, Value>) valueIterator.next();
			issuePreferenceList.put((Integer) indexValuePair.getKey(),
					new ValuePreferenceGraph((Value) indexValuePair.getValue()));
		}

		for (Issue issue : issueList)
			issuePreferenceList.get(issue.getNumber()).setIssue(issue);

	}

	public double getEstimatedUtility(int issueIndex, Value issueValue) {
		return issuePreferenceList.get(issueIndex).getEstimatedUtility(issueValue);
	}

	public double estimateUtility(Bid currentBid) throws Exception {

		double utility = 0.0;

		for (Issue issue : issueList) {
			utility += getEstimatedUtility(issue.getNumber(), currentBid.getValue(issue.getNumber()));
		}

		return utility;
	}

	public void updateValuePreferenceGraph(int issueIndex, Value previousValue, Value currentValue, Feedback feedback) {

		issuePreferenceList.get(issueIndex).addPreferenceOrdering(previousValue, currentValue, feedback);
	}

	public ArrayList<Value> getAllValues(int issueIndex) {
		return issuePreferenceList.get(issueIndex).getAllValues();
	}

	/**
	 * @return true iff the new value is less preferred than current value,
	 *         otherwise false (even we cannot compare them)
	 */
	public boolean isLessPreferredThan(int issueIndex, Value newValue, Value currentValue) {

		return issuePreferenceList.get(issueIndex).getAllLessPreferredValues(currentValue).contains(newValue);
	}

	public boolean isEquallyPreferred(int issueIndex, Value newValue, Value currentValue) {
		return issuePreferenceList.get(issueIndex).getEqualPreferredValues(currentValue).contains(newValue);
	}

	public boolean isMorePreferredThan(int issueIndex, Value newValue, Value currentValue) {

		return (issuePreferenceList.get(issueIndex).getAllMorePreferredValues(currentValue).contains(newValue));
	}

	public Value getMissingValue(int issueIndex) {
		return issuePreferenceList.get(issueIndex).getMissingValue();
	}

	public ArrayList<Value> getIncomparableValues(int issueIndex, Value currentValue) {
		return issuePreferenceList.get(issueIndex).getAllIncomparableValues(currentValue);
	}

	public Value getIncomparableValue(int issueIndex, Value currentValue) {
		return issuePreferenceList.get(issueIndex).getIncomparableValue(currentValue);
	}

	@Override
	public String toString() {

		Set<Integer> keySet = this.issuePreferenceList.keySet();

		Iterator<Integer> keyIterator = keySet.iterator();

		StringBuffer buffy = new StringBuffer("Issue Preference Map:\n");
		while (keyIterator.hasNext()) {
			buffy.append("\n\n" + issuePreferenceList.get(keyIterator.next()).toString());
		}

		return (buffy.toString());
	}
}