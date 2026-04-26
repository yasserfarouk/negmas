package genius.core.utility;

import java.util.HashMap;

import genius.core.Bid;

public class ExclusiveZeroOutcomeConstraint implements ZeroOutcomeContraint {

	private HashMap<Integer, String> checkList;

	public ExclusiveZeroOutcomeConstraint() {
		this.checkList = new HashMap<Integer, String>();
	}

	@Override
	public void addContraint(Integer issueIndex, String conditionToBeCheck) {
		this.checkList.put(issueIndex, conditionToBeCheck);
	}

	@Override
	public boolean willGetZeroOutcomeUtility(Bid bid) {

		for (Integer issueIndex : this.checkList.keySet()) {

			if (checkList.get(issueIndex).contains("numeric")) {

				if (((checkList.get(issueIndex).contains("positive"))
						&& (Integer.valueOf(
								bid.getValue(issueIndex).toString()) > 0))) {

					return true;
				} else if ((checkList.get(issueIndex).contains("negative"))
						&& (Integer.valueOf(
								bid.getValue(issueIndex).toString()) > 0)) {
					return true;
				}

			} else // check it involves "condition" keyword
			{
				if (!bid.getValue(issueIndex).toString()
						.contains(this.checkList.get(issueIndex))) {
					return true;
				}

			}

		}
		return false;
	}

}
