package genius.core.utility;

import genius.core.Bid;

public interface ZeroOutcomeContraint {

	public abstract void addContraint(Integer issueIndex,
			String conditionToBeCheck);

	public abstract boolean willGetZeroOutcomeUtility(Bid bid);
}
