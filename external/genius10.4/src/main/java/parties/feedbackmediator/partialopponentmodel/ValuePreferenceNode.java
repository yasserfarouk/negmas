package parties.feedbackmediator.partialopponentmodel;

import java.util.ArrayList;

import genius.core.issue.Value;

/**
 * 
 * @author Reyhan Aydogan
 * 
 */

public class ValuePreferenceNode {

	private ArrayList<Value> morePreferredList; // more preferred values
	private ArrayList<Value> lessPreferredList; // less preferred values
	private ArrayList<Value> equallyPreferredList;
	private int depth; // depth of the node in the preference graph

	public ValuePreferenceNode(Value value, int assignedDepth) {
		morePreferredList = new ArrayList<Value>();
		lessPreferredList = new ArrayList<Value>();
		equallyPreferredList = new ArrayList<Value>();
		addEquallyPreferredValue(value);
		setDepth(assignedDepth);
	}

	public ValuePreferenceNode(Value value) {
		morePreferredList = new ArrayList<Value>();
		lessPreferredList = new ArrayList<Value>();
		equallyPreferredList = new ArrayList<Value>();
		addEquallyPreferredValue(value);
		setDepth(1);
	}

	public ArrayList<Value> getMorePreferredList() {
		return morePreferredList;
	}

	public void setMorePreferredList(ArrayList<Value> morePreferredList) {
		this.morePreferredList = morePreferredList;
	}

	public void addMorePreferredValue(Value preferredValue) {
		if (!morePreferredList.contains(preferredValue))
			this.morePreferredList.add(preferredValue);
	}

	public void addMorePreferredValues(ArrayList<Value> preferredValues) {

		for (Value currentValue : preferredValues) {
			addMorePreferredValue(currentValue);
		}
	}

	// RA: We can also add removeMorePreferredValue or Values method if it is
	// necessary

	public ArrayList<Value> getLessPreferredList() {
		return lessPreferredList;
	}

	public void setLessPreferredList(ArrayList<Value> lessPreferredList) {
		this.lessPreferredList = lessPreferredList;
	}

	public void addLessPreferredValue(Value lessPreferredValue) {
		if (!lessPreferredList.contains(lessPreferredValue))
			this.lessPreferredList.add(lessPreferredValue);
	}

	public void addLessPreferredValues(ArrayList<Value> lessPreferredValues) {

		for (Value currentValue : lessPreferredValues) {
			addLessPreferredValue(currentValue);
		}
	}

	public ArrayList<Value> getEqualPreferredList() {
		return equallyPreferredList;
	}

	public void setEquallyPreferredList(ArrayList<Value> equalPreferredList) {
		this.equallyPreferredList = equalPreferredList;
	}

	public void addEquallyPreferredValue(Value equallyPreferredValue) {
		if (!equallyPreferredList.contains(equallyPreferredValue))
			this.equallyPreferredList.add(equallyPreferredValue);
	}

	public void addEquallyPreferredValues(ArrayList<Value> equallyPreferredValues) {

		for (Value currentValue : equallyPreferredValues) {
			addEquallyPreferredValue(currentValue);
		}
	}

	public int increaseDepth(int amount) {
		this.depth += amount;
		return this.depth;
	}

	public int getDepth() {
		return depth;
	}

	public void setDepth(int depth) {
		this.depth = depth;
	}

	@Override
	public String toString() {
		StringBuffer buffy = new StringBuffer();

		buffy.append("Values:");
		for (Value current : equallyPreferredList)
			buffy.append(" ").append(current);

		buffy.append("\n\tDepth: ").append(depth).append("\n");

		buffy.append("\tLess Preferred Node List:");

		if (lessPreferredList.size() == 0)
			buffy.append(" None");

		for (Value lessPreferredValue : lessPreferredList)
			buffy.append(" ").append(lessPreferredValue.toString());

		buffy.append("\n\tMore Preferred Node List:");

		if (morePreferredList.size() == 0)
			buffy.append(" None");

		for (Value morePreferredValue : morePreferredList)
			buffy.append(" ").append(morePreferredValue.toString());

		return buffy.toString();
	}

}
