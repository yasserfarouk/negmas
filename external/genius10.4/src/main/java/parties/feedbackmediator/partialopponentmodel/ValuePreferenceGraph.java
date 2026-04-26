package parties.feedbackmediator.partialopponentmodel;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;

import genius.core.Feedback;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.IssueInteger;
import genius.core.issue.IssueReal;
import genius.core.issue.Value;
import genius.core.issue.ValueDiscrete;
import genius.core.issue.ValueInteger;
import genius.core.issue.ValueReal;

/**
 * 
 * @author Reyhan Aydogan
 * 
 */

public class ValuePreferenceGraph {

	private HashMap<Value, ValuePreferenceNode> partialGraph;
	private Issue issue;
	private int lowestDepth = 1;
	private int highestDepth = 1;

	public ValuePreferenceGraph(Value initialValue) {

		partialGraph = new HashMap<Value, ValuePreferenceNode>();
		partialGraph.put(initialValue, new ValuePreferenceNode(initialValue, 1));
	}

	public boolean containsIssueValue(Value issueValue) {
		return partialGraph.containsKey(issueValue);
	}

	public Integer getDepth(Value issueValue) {

		if (!containsIssueValue(issueValue))
			return -1;
		else
			return partialGraph.get(issueValue).getDepth() - lowestDepth + 1; // to
																				// ensure
																				// the
																				// lowest
																				// depth
																				// is
																				// equal
																				// to
																				// one..

	}

	public double getEstimatedUtility(Value issueValue) { // according to depth
															// heuristic

		return (double) getDepth(issueValue) / (highestDepth - lowestDepth + 1);
	}

	public ArrayList<Value> getAllValues() {

		ArrayList<Value> values = new ArrayList<Value>();

		Iterator<Value> valueIterator = partialGraph.keySet().iterator();

		while (valueIterator.hasNext()) {
			values.add(valueIterator.next());
		}

		return values;
	}

	public void updateLowHighDepths() {

		Iterator<Value> valueIterator = partialGraph.keySet().iterator();
		Value currentValue = null;

		lowestDepth = highestDepth; // lowerDepth can be greater than it was
									// before
		while (valueIterator.hasNext()) {
			currentValue = valueIterator.next();

			if (partialGraph.get(currentValue).getDepth() < lowestDepth)
				lowestDepth = partialGraph.get(currentValue).getDepth();

			if (partialGraph.get(currentValue).getDepth() > highestDepth)
				highestDepth = partialGraph.get(currentValue).getDepth();
		}

	}

	public Value getMissingValue() {

		switch (issue.getType()) {

		case DISCRETE:
			IssueDiscrete issueDiscrete = (IssueDiscrete) issue;
			for (Value value : issueDiscrete.getValues()) {
				if (!containsIssueValue(value))
					return value;
			}
			break;
		case INTEGER:
			IssueInteger issueInteger = (IssueInteger) issue;
			Value value;
			for (int i = 0; i <= (issueInteger.getUpperBound() - issueInteger.getLowerBound()); i++) {
				value = new ValueInteger(issueInteger.getLowerBound() + i);
				if (!containsIssueValue(value))
					return value;
			}
			break;

		case REAL: // needs to be checked!
			IssueReal issueReal = (IssueReal) issue;
			for (int i = 0; i < issueReal.getNumberOfDiscretizationSteps(); i++) {
				value = new ValueReal(
						issueReal.getLowerBound() + (((double) ((issueReal.getUpperBound() - issueReal.getLowerBound()))
								/ (issueReal.getNumberOfDiscretizationSteps())) * i));
				if (!containsIssueValue(value))
					return value;
			}
			break;
		}

		return null;
	}

	private void searchCondition(Value willBeChecked, ArrayList<Value> visitedList, ArrayList<Value> selectedValues,
			ArrayList<Value> forbiddenList) {

		ArrayList<Value> checkList = new ArrayList<Value>();

		// eliminate the forbidden values first
		for (Value value : getAllComparableValues(willBeChecked)) {
			if (!forbiddenList.contains(value))
				checkList.add(value);
		}

		// seach
		for (Value currentValue : checkList) {
			if (!visitedList.contains(currentValue)) {

				if (!selectedValues.contains(currentValue))
					selectedValues.add(currentValue);

				for (Value equalValue : partialGraph.get(currentValue).getEqualPreferredList()) {
					if (!visitedList.contains(equalValue))
						visitedList.add(equalValue);
				}

				searchCondition(currentValue, visitedList, selectedValues, forbiddenList);
			}
		} // for
	}

	// receiveMessage each node's depth except the given value and value list
	private void increasePreferredValueNodeDepths(Value lessPreferredValue, Value morePreferredValue, int difference) {

		ArrayList<Value> forbiddenList = getAllLessPreferredValues(lessPreferredValue);
		for (Value less : getEqualPreferredValues(lessPreferredValue))
			forbiddenList.add(less);

		// the depth of the nodes related to morePreferredValue except forbidden
		// list will be increaded by "difference"

		ArrayList<Value> selectedValues = new ArrayList<Value>();

		searchCondition(morePreferredValue, new ArrayList<Value>(), selectedValues, forbiddenList);

		for (Value currentValue : selectedValues) {
			partialGraph.get(currentValue).increaseDepth(difference);
		}

	}

	// The method assumes that firstValue always exists in the system.
	public void addPreferenceOrdering(Value firstValue, Value secondValue, Feedback feedback) {

		if (firstValue.equals(secondValue)) // if the values are same, do
											// nothing
			return;

		if (!containsIssueValue(secondValue))
			addNewNode(firstValue, secondValue, feedback);
		else {

			ValuePreferenceNode firstValueNode = partialGraph.get(firstValue);
			ValuePreferenceNode secondValueNode = partialGraph.get(secondValue);

			if (feedback == Feedback.BETTER) {

				// receiveMessage the depth of all nodes except first node and
				// all less preferred nodes than it
				if (firstValueNode.getDepth() >= secondValueNode.getDepth()) { // receiveMessage
																				// depth
					increasePreferredValueNodeDepths(firstValue, secondValue,
							firstValueNode.getDepth() - secondValueNode.getDepth() + 1);
				}

				// common part
				secondValueNode.addLessPreferredValue(firstValue);
				firstValueNode.addMorePreferredValue(secondValue);
				partialGraph.put(secondValue, secondValueNode);

			} else if (feedback == Feedback.WORSE) {
				if (firstValueNode.getDepth() <= secondValueNode.getDepth()) {// receiveMessage
																				// depth
					increasePreferredValueNodeDepths(secondValue, firstValue,
							secondValueNode.getDepth() - firstValueNode.getDepth() + 1);
				}
				// common part
				secondValueNode.addMorePreferredValue(firstValue);
				firstValueNode.addLessPreferredValue(secondValue);
				partialGraph.put(secondValue, secondValueNode);

			} else { // SAME

				if (firstValueNode.getDepth() != secondValueNode.getDepth()) {

					if (firstValueNode.getDepth() < secondValueNode.getDepth()) {// second
																					// node
																					// is
																					// higher

						increasePreferredValueNodeDepths(secondValue, firstValue,
								secondValueNode.getDepth() - firstValueNode.getDepth());
						firstValueNode.setDepth(secondValueNode.getDepth());
					} else { // first node is higher
						increasePreferredValueNodeDepths(firstValue, secondValue,
								firstValueNode.getDepth() - secondValueNode.getDepth());
					}

				}
				// common part
				firstValueNode.addEquallyPreferredValues(secondValueNode.getEqualPreferredList());
				firstValueNode.addLessPreferredValues(secondValueNode.getLessPreferredList());
				firstValueNode.addMorePreferredValues(secondValueNode.getMorePreferredList());

				partialGraph.put(secondValue, firstValueNode);

			} // SAME

			updateLowHighDepths(); // common
			partialGraph.put(firstValue, firstValueNode);

		} // else not contain

	}

	// FirstValue always exists in the system but newValudoes not exist for this
	// method
	public void addNewNode(Value firstValue, Value newValue, Feedback feedback) {

		ValuePreferenceNode firstValueNode = partialGraph.get(firstValue);

		if (feedback == Feedback.SAME) {
			firstValueNode.addEquallyPreferredValue(newValue);
			partialGraph.put(newValue, firstValueNode);
			partialGraph.put(firstValue, firstValueNode);
		} else {
			ValuePreferenceNode newValueNode = new ValuePreferenceNode(newValue);

			if (feedback == Feedback.BETTER) {
				newValueNode.setDepth(firstValueNode.getDepth() + 1);
				newValueNode.addLessPreferredValue(firstValue);
				firstValueNode.addMorePreferredValue(newValue);
				if (newValueNode.getDepth() > highestDepth) // receiveMessage
															// the highest depth
					highestDepth = newValueNode.getDepth();

			} else { // WORSE
				newValueNode.setDepth(firstValueNode.getDepth() - 1);
				newValueNode.addMorePreferredValue(firstValue);
				firstValueNode.addLessPreferredValue(newValue);
				if (newValueNode.getDepth() < lowestDepth) // receiveMessage the
															// lowest depth
					lowestDepth = newValueNode.getDepth();
			}

			partialGraph.put(newValue, newValueNode);
			partialGraph.put(firstValue, firstValueNode);
		} // else
	}

	private void searchPreferredValues(Value currentValue, ArrayList<Value> preferredList) {

		for (Value currentPreferred : partialGraph.get(currentValue).getMorePreferredList()) {

			for (Value preferred : partialGraph.get(currentPreferred).getEqualPreferredList()) {

				if (!preferredList.contains(preferred)) {
					preferredList.add(preferred);
					searchPreferredValues(preferred, preferredList);
				}
			}
		} // for
	}

	private void searchLessPreferredValues(Value value, ArrayList<Value> lessPreferredList) {

		for (Value currentLessPreferred : partialGraph.get(value).getLessPreferredList()) {

			for (Value lessPreferred : partialGraph.get(currentLessPreferred).getEqualPreferredList()) {

				if (!lessPreferredList.contains(lessPreferred)) {
					lessPreferredList.add(lessPreferred);
					searchLessPreferredValues(lessPreferred, lessPreferredList);
				}
			}
		}
	}

	public ArrayList<Value> getAllMorePreferredValues(Value currentValue) {

		ArrayList<Value> preferredValueList = new ArrayList<Value>();

		searchPreferredValues(currentValue, preferredValueList);

		return preferredValueList;
	}

	public ArrayList<Value> getAllLessPreferredValues(Value currentValue) {

		ArrayList<Value> lessPreferredValueList = new ArrayList<Value>();

		searchLessPreferredValues(currentValue, lessPreferredValueList);

		return lessPreferredValueList;
	}

	public ArrayList<Value> getAllComparableValues(Value currentValue) {

		ArrayList<Value> comparables = new ArrayList<Value>();
		comparables.addAll(getAllLessPreferredValues(currentValue));
		comparables.addAll(getAllMorePreferredValues(currentValue));
		comparables.addAll(getEqualPreferredValues(currentValue));
		return comparables;
	}

	public ArrayList<Value> getAllIncomparableValues(Value currentValue) {

		ArrayList<Value> incomparables = new ArrayList<Value>();
		ArrayList<Value> comparables = getAllComparableValues(currentValue);

		for (Value value : getAllValues()) {

			if (!comparables.contains(value))
				incomparables.add(value);
		}

		return incomparables;
	}

	public Value getIncomparableValue(Value currentValue) {

		ArrayList<Value> comparables = getAllComparableValues(currentValue);

		for (Value value : getAllValues()) {
			if (!comparables.contains(value))
				return value;
		}
		return null;
	}

	public ArrayList<Value> getEqualPreferredValues(Value currentValue) {
		return partialGraph.get(currentValue).getEqualPreferredList();
	}

	@Override
	public String toString() {

		Iterator<ValuePreferenceNode> nodes = partialGraph.values().iterator();
		StringBuffer buffy = new StringBuffer("Value Preference Graph:\n");

		buffy.append("Lowest Depth: ").append(lowestDepth).append("\n");
		buffy.append("Highest Depth: ").append(highestDepth).append("\n");
		while (nodes.hasNext()) {
			buffy.append("\n");
			buffy.append(nodes.next());
		}
		return buffy.toString();
	}

	// For test purposes....
	public static void main(String[] args) {

		/*
		 * ValuePreferenceGraph mygraph=new ValuePreferenceGraph(new
		 * ValueDiscrete("a")); mygraph.addPreferenceOrdering(new
		 * ValueDiscrete("a"), new ValueDiscrete("b"), Feedback.BETTER);
		 * mygraph.addPreferenceOrdering(new ValueDiscrete("b"), new
		 * ValueDiscrete("c"), Feedback.BETTER);
		 * mygraph.addPreferenceOrdering(new ValueDiscrete("c"), new
		 * ValueDiscrete("d"), Feedback.WORSE);
		 * mygraph.addPreferenceOrdering(new ValueDiscrete("c"), new
		 * ValueDiscrete("e"), Feedback.SAME); mygraph.addPreferenceOrdering(new
		 * ValueDiscrete("e"), new ValueDiscrete("f"), Feedback.BETTER);
		 * mygraph.addPreferenceOrdering(new ValueDiscrete("a"), new
		 * ValueDiscrete("g"), Feedback.WORSE);
		 * mygraph.addPreferenceOrdering(new ValueDiscrete("a"), new
		 * ValueDiscrete("h"), Feedback.SAME); mygraph.addPreferenceOrdering(new
		 * ValueDiscrete("h"), new ValueDiscrete("k"), Feedback.WORSE);
		 * mygraph.addPreferenceOrdering(new ValueDiscrete("f"), new
		 * ValueDiscrete("m"), Feedback.SAME);
		 * 
		 * System.out.println(mygraph.toString());
		 * 
		 * 
		 * for (Value current: mygraph.getAllValues()) {
		 * 
		 * System.out.print("\n \n Value:"+current+" \n More Preferred Values:"
		 * );
		 * 
		 * for (Value preferred: mygraph.getAllMorePreferredValues(current))
		 * System.out.print(" "+preferred.toString());
		 * 
		 * System.out.print("\n Less Preferred Values:");
		 * 
		 * for (Value lessPreferred: mygraph.getAllLessPreferredValues(current))
		 * System.out.print(" "+lessPreferred.toString());
		 * 
		 * System.out.println("\nEstimated Utility:"+mygraph.getEstimatedUtility
		 * (current));
		 * 
		 * }
		 * 
		 * System.out.println("");
		 * 
		 * System.out.println(
		 * "*******************************************************************"
		 * );
		 * System.out.println("Comparables-a :"+mygraph.getAllComparableValues(
		 * new ValueDiscrete("a")));
		 * System.out.println("Comparables-b :"+mygraph.getAllComparableValues(
		 * new ValueDiscrete("b")));
		 * System.out.println("Comparables-c :"+mygraph.getAllComparableValues(
		 * new ValueDiscrete("c")));
		 * System.out.println("Comparables-d :"+mygraph.getAllComparableValues(
		 * new ValueDiscrete("d")));
		 * System.out.println("Comparables-e :"+mygraph.getAllComparableValues(
		 * new ValueDiscrete("e")));
		 * System.out.println("Comparables-f :"+mygraph.getAllComparableValues(
		 * new ValueDiscrete("f")));
		 * System.out.println("Comparables-g :"+mygraph.getAllComparableValues(
		 * new ValueDiscrete("g")));
		 * System.out.println("Comparables-h :"+mygraph.getAllComparableValues(
		 * new ValueDiscrete("h")));
		 * System.out.println("Comparables-k :"+mygraph.getAllComparableValues(
		 * new ValueDiscrete("k")));
		 * System.out.println("Comparables-m :"+mygraph.getAllComparableValues(
		 * new ValueDiscrete("m")));
		 * 
		 * 
		 * System.out.println(
		 * "*******************************************************************"
		 * ); System.out.println("InComparables-a :"+mygraph.
		 * getAllIncomparableValues(new ValueDiscrete("a")));
		 * System.out.println("InComparables-b :"+mygraph.
		 * getAllIncomparableValues(new ValueDiscrete("b")));
		 * System.out.println("InComparables-c :"+mygraph.
		 * getAllIncomparableValues(new ValueDiscrete("c")));
		 * System.out.println("InComparables-d :"+mygraph.
		 * getAllIncomparableValues(new ValueDiscrete("d")));
		 * System.out.println("InComparables-e :"+mygraph.
		 * getAllIncomparableValues(new ValueDiscrete("e")));
		 * System.out.println("InComparables-f :"+mygraph.
		 * getAllIncomparableValues(new ValueDiscrete("f")));
		 * System.out.println("InComparables-g :"+mygraph.
		 * getAllIncomparableValues(new ValueDiscrete("g")));
		 * System.out.println("InComparables-h :"+mygraph.
		 * getAllIncomparableValues(new ValueDiscrete("h")));
		 * System.out.println("InComparables-k :"+mygraph.
		 * getAllIncomparableValues(new ValueDiscrete("k")));
		 * System.out.println("InComparables-m :"+mygraph.
		 * getAllIncomparableValues(new ValueDiscrete("m")));
		 */

		/*
		 * ValuePreferenceGraph mygraph=new ValuePreferenceGraph(new
		 * ValueDiscrete("a")); mygraph.addPreferenceOrdering(new
		 * ValueDiscrete("a"), new ValueDiscrete("b"), Feedback.WORSE);
		 * mygraph.addPreferenceOrdering(new ValueDiscrete("b"), new
		 * ValueDiscrete("c"), Feedback.BETTER);
		 * mygraph.addPreferenceOrdering(new ValueDiscrete("c"), new
		 * ValueDiscrete("d"), Feedback.BETTER);
		 * mygraph.addPreferenceOrdering(new ValueDiscrete("d"), new
		 * ValueDiscrete("a"), Feedback.BETTER);
		 * mygraph.addPreferenceOrdering(new ValueDiscrete("d"), new
		 * ValueDiscrete("e"), Feedback.WORSE);
		 * mygraph.addPreferenceOrdering(new ValueDiscrete("b"), new
		 * ValueDiscrete("e"), Feedback.WORSE);
		 * 
		 * mygraph.addPreferenceOrdering(new ValueDiscrete("d"), new
		 * ValueDiscrete("f"), Feedback.WORSE);
		 * mygraph.addPreferenceOrdering(new ValueDiscrete("d"), new
		 * ValueDiscrete("g"), Feedback.BETTER);
		 * mygraph.addPreferenceOrdering(new ValueDiscrete("b"), new
		 * ValueDiscrete("f"), Feedback.SAME); mygraph.addPreferenceOrdering(new
		 * ValueDiscrete("d"), new ValueDiscrete("k"), Feedback.WORSE);
		 * mygraph.addPreferenceOrdering(new ValueDiscrete("c"), new
		 * ValueDiscrete("m"), Feedback.WORSE);
		 * mygraph.addPreferenceOrdering(new ValueDiscrete("m"), new
		 * ValueDiscrete("n"), Feedback.BETTER);
		 * mygraph.addPreferenceOrdering(new ValueDiscrete("k"), new
		 * ValueDiscrete("m"), Feedback.BETTER); //
		 * mygraph.addPreferenceOrdering(new ValueDiscrete("e"), new
		 * ValueDiscrete("n"), Feedback.SAME);
		 * System.out.println(mygraph.toString());
		 */
		/*
		 * System.out.println(
		 * "*******************************************************************"
		 * ); System.out.println("InComparables-a :"+mygraph.
		 * getAllIncomparableValues(new ValueDiscrete("a")));
		 * System.out.println("InComparables-b :"+mygraph.
		 * getAllIncomparableValues(new ValueDiscrete("b")));
		 * System.out.println("InComparables-c :"+mygraph.
		 * getAllIncomparableValues(new ValueDiscrete("c")));
		 * System.out.println("InComparables-d :"+mygraph.
		 * getAllIncomparableValues(new ValueDiscrete("d")));
		 * System.out.println("InComparables-e :"+mygraph.
		 * getAllIncomparableValues(new ValueDiscrete("e")));
		 * System.out.println("InComparables-f :"+mygraph.
		 * getAllIncomparableValues(new ValueDiscrete("f")));
		 * System.out.println("InComparables-g :"+mygraph.
		 * getAllIncomparableValues(new ValueDiscrete("g")));
		 * System.out.println("InComparables-k :"+mygraph.
		 * getAllIncomparableValues(new ValueDiscrete("k")));
		 * System.out.println("InComparables-m :"+mygraph.
		 * getAllIncomparableValues(new ValueDiscrete("m")));
		 * System.out.println("InComparables-n :"+mygraph.
		 * getAllIncomparableValues(new ValueDiscrete("n")));
		 */

		ValuePreferenceGraph mygraph = new ValuePreferenceGraph(new ValueDiscrete("a"));
		mygraph.addPreferenceOrdering(new ValueDiscrete("a"), new ValueDiscrete("b"), Feedback.BETTER);
		mygraph.addPreferenceOrdering(new ValueDiscrete("b"), new ValueDiscrete("e"), Feedback.BETTER);
		mygraph.addPreferenceOrdering(new ValueDiscrete("e"), new ValueDiscrete("c"), Feedback.BETTER);

		mygraph.addPreferenceOrdering(new ValueDiscrete("c"), new ValueDiscrete("d"), Feedback.WORSE);
		mygraph.addPreferenceOrdering(new ValueDiscrete("a"), new ValueDiscrete("d"), Feedback.BETTER);

		System.out.println(mygraph.toString());
	}

	public Issue getIssue() {
		return issue;
	}

	public void setIssue(Issue issue) {
		this.issue = issue;
	}

}
