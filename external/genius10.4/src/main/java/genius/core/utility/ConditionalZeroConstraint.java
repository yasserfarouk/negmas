package genius.core.utility;

import java.util.HashMap;

import genius.core.Bid;

public class ConditionalZeroConstraint extends RConstraint {

	private Integer issueIndex;
	private HashMap<Integer, String> checkList;
	private String valueToBeChecked;

	public ConditionalZeroConstraint(int index, String valueToBeChecked) {
		this.checkList = new HashMap<Integer, String>();
		this.issueIndex = index;
		this.valueToBeChecked = valueToBeChecked;
	}

	@Override
	public void addContraint(Integer issueIndex, String conditionToBeCheck) {
		this.checkList.put(issueIndex, conditionToBeCheck);
	}

	@Override
	public Integer getIssueIndex() {
		return this.issueIndex;
	}

	@Override
	public boolean willZeroUtility(Bid bid) {

		boolean check = true;
		for (Integer index : this.checkList.keySet()) {
			if (!this.checkList.get(index).equals(
					bid.getValue(index).toString()))
				check = false;
		}

		if (check == true) {
			if (!bid.getValue(issueIndex).toString().contains(valueToBeChecked))
				return true;
		}
		return false;

	}

}
