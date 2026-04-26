package genius.core.utility;

import java.util.ArrayList;

import genius.core.Bid;
import genius.core.issue.ValueInteger;

public class ExclusiveHyperRectangle extends HyperRectangle {

	private ArrayList<Bound> boundlist;
	private double utilityValue;

	public ExclusiveHyperRectangle() {

	}

	public ArrayList<Bound> getBoundList() {
		return boundlist;
	}

	public void setBoundList(ArrayList<Bound> boundlist) {
		this.boundlist = boundlist;
	}

	public double getUtilityValue() {
		return utilityValue;
	}

	public void setUtilityValue(double utilityValue) {
		this.utilityValue = utilityValue;
	}

	@Override
	public double getUtility(Bid bid) {

		int issueValue;
		for (int i = 0; i < boundlist.size(); i++) {
			issueValue = (int) ((ValueInteger) bid.getValue(boundlist.get(i)
					.getIssueIndex())).getValue();
			if ((boundlist.get(i).getMin() > issueValue)
					|| (issueValue > boundlist.get(i).getMax()))
				return utilityValue * weight;
		}

		return 0.0;
	}

}
