package genius.core.utility;

import java.util.HashMap;

import genius.core.Bid;

public class ExclusiveZeroConstraint extends RConstraint {

	Integer issueIndex;
	private HashMap<Integer, String> checkList;

	public ExclusiveZeroConstraint(Integer index) {
		this.issueIndex = index;
		this.checkList = new HashMap<Integer, String>();
	}

	@Override
	public void addContraint(Integer issueIndex, String conditionToBeCheck) {
		this.checkList.put(issueIndex, conditionToBeCheck);
	}

	@Override
	public boolean willZeroUtility(Bid bid) {

		for (Integer index : this.checkList.keySet()) {
			if (!bid.getValue(index).toString()
					.equals(this.checkList.get(index)))
				return true;
		}
		return false;
	}

	@Override
	public Integer getIssueIndex() {
		return this.issueIndex;
	}
}
