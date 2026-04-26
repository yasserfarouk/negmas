package genius.core.utility;

import java.util.ArrayList;

import genius.core.Bid;
import genius.core.issue.ValueInteger;

/**
 * An {@link InclusiveHyperRectangle} has a utility value when all of its bounds
 * are satisfied, and zero otherwise.
 */
public class InclusiveHyperRectangle extends HyperRectangle {
	private ArrayList<Bound> boundlist;
	private double utilityValue;
	private boolean isAllBidsAcceptable;

	public InclusiveHyperRectangle() {
		this.isAllBidsAcceptable = false;
	}

	public InclusiveHyperRectangle(boolean isAllOkay) {
		this.isAllBidsAcceptable = isAllOkay;
	}

	public ArrayList<Bound> getBoundList() {
		return boundlist;
	}

	protected void setBoundList(ArrayList<Bound> boundlist) {
		this.boundlist = boundlist;
	}

	public double getUtilityValue() {
		return utilityValue;
	}

	protected void setUtilityValue(double utilityValue) {
		this.utilityValue = utilityValue;
	}

	@Override
	public double getUtility(Bid bid) {

		if (this.isAllBidsAcceptable) // if there is no constraint at all
			return (utilityValue * weight);

		int issueValue;
		for (int i = 0; i < boundlist.size(); i++) {
			issueValue = (int) ((ValueInteger) bid.getValue(boundlist.get(i)
					.getIssueIndex())).getValue();
			if ((boundlist.get(i).getMin() > issueValue)
					|| (issueValue > boundlist.get(i).getMax()))
				return 0.0;
		}

		return utilityValue * weight;
	}

	public boolean doesCover(Bid bid) throws Exception {
		if (this.isAllBidsAcceptable) // no constraint at all
			return true;

		int issueValue;
		for (int i = 0; i < boundlist.size(); i++) {
			issueValue = (int) ((ValueInteger) bid.getValue(boundlist.get(i)
					.getIssueIndex())).getValue();
			if ((boundlist.get(i).getMin() > issueValue)
					|| (issueValue > boundlist.get(i).getMax()))
				return false;
		}
		return true;
	}
}
