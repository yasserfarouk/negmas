package genius.core.utility;

import java.util.ArrayList;

import genius.core.Bid;

public class SumZeroNotConstraint extends SumZeroConstraint {

	public SumZeroNotConstraint(int index) {
		super(index);
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
			if (!bid.getValue(index).toString().contains(valueToBeChecked))
				return true;
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

	public String getValueToBeChecked() {
		return valueToBeChecked;
	}

	public void setValueToBeChecked(String valueToBeChecked) {
		this.valueToBeChecked = valueToBeChecked;
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
