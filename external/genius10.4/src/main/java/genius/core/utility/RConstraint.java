package genius.core.utility;

import genius.core.Bid;

public abstract class RConstraint {

	public RConstraint() {
	}

	public abstract Integer getIssueIndex();

	public abstract boolean willZeroUtility(Bid bid);

	public abstract void addContraint(Integer issueIndex,
			String conditionToBeCheck);
}
