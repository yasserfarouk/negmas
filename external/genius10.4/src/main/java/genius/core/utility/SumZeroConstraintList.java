package genius.core.utility;

import java.util.ArrayList;

import genius.core.Bid;

public class SumZeroConstraintList extends RConstraint {

	protected int index;
	protected ArrayList<String> valuesToBeChecked;
	protected int min, max;
	protected ArrayList<Integer> relatedIssues;

	public SumZeroConstraintList(int index) {
		this.index = index;
		relatedIssues = new ArrayList<Integer>();
		valuesToBeChecked = new ArrayList<String>();
	}

	@Override
	public Integer getIssueIndex() {
		return index;
	}

	@Override
	public boolean willZeroUtility(Bid bid) {

		int total = 0;
		for (int i = 0; i < relatedIssues.size(); i++) {
			total += Integer.valueOf(bid.getValue(relatedIssues.get(i))
					.toString());
		}

		if ((min <= total) && (total <= max)) {
			for (int k = 0; k < this.valuesToBeChecked.size(); k++) {
				if (bid.getValue(index).toString()
						.contains(valuesToBeChecked.get(k)))
					return true;
			}
		}

		return false;
	}

	@Override
	public void addContraint(Integer issueIndex, String conditionToBeCheck) {

	}

	public void addRelatedIssues(ArrayList<Integer> relatedIssueIndices) {
		this.relatedIssues = relatedIssueIndices;
	}

	public void addRelatedIssue(Integer relatedIndex) {
		this.relatedIssues.add(relatedIndex);
	}

	public ArrayList<String> getValueToBeChecked() {
		return valuesToBeChecked;
	}

	public void setValueToBeChecked(ArrayList<String> valuesToBeChecked) {
		this.valuesToBeChecked = valuesToBeChecked;
	}

	public void addValueToBeChecked(String valueTobeChecked) {
		this.valuesToBeChecked.add(valueTobeChecked);
	}

	public int getMax() {
		return max;
	}

	public void setMax(int max) {
		this.max = max;
	}

	public int getMin() {
		return min;
	}

	public void setMin(int min) {
		this.min = min;
	}

}
