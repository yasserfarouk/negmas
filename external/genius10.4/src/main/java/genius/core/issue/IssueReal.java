package genius.core.issue;

import genius.core.misc.Range;
import genius.core.xml.SimpleElement;

/**
 * issue with {@link Double} values within some range [min, max].
 */

public class IssueReal extends Issue {

	/**
	 * 
	 */
	private static final long serialVersionUID = 7974044551620253633L;
	// Assumption 1: real-valued issues have a fixed range, with a lower and
	// upper bound.
	// Assumption 2: value ranges for issue are shared between agents.
	private Range range;
	// use this value for discrete operations in the analysis
	private int fNumberOfDiscretizationSteps = 21;

	public IssueReal(String name, int issueNumber, double min, double max) {
		super(name, issueNumber);
		range = new Range(min, max);
	}

	public IssueReal(String name, int issueNumber, double min, double max, Objective objParent) {
		super(name, issueNumber, objParent);
		range = new Range(min, max);
	}

	public boolean checkInRange(Value val) {
		return (((ValueReal) val).getValue() >= range.getLowerbound()
				&& ((ValueReal) val).getValue() <= range.getUpperbound());
	}

	public final double getLowerBound() {
		return range.getLowerbound();
	}

	public final double getUpperBound() {
		return range.getUpperbound();
	}

	public final boolean setUpperBound(double up) {
		if (up > range.getLowerbound()) {
			range.setUpperbound(up);
			return true;
		} else {
			System.out.println("Minimum bound exceeds maximum bound in integer-valued issue!");
			return false;
		}

	}

	public final boolean setLowerBound(double lo) {
		if (lo < range.getUpperbound()) {
			range.setLowerbound(lo);
			return true;
		} else {
			System.out.println("Minimum bound exceeds maximum bound in integer-valued issue!");
			return false;
		}
	}

	public int getNumberOfDiscretizationSteps() {
		return fNumberOfDiscretizationSteps;
	}

	public void setNumberOfDiscretizationSteps(int numberOfDiscretizationSteps) {
		fNumberOfDiscretizationSteps = numberOfDiscretizationSteps;
	}

	/**
	 * Returns a SimpleElement representation of this issue.
	 * 
	 * @return The SimpleElement with this issues attributes
	 */
	public SimpleElement toXML() {
		SimpleElement thisIssue = new SimpleElement("issue");
		thisIssue.setAttribute("name", getName());
		thisIssue.setAttribute("index", "" + getNumber());
		thisIssue.setAttribute("type", "real");
		thisIssue.setAttribute("etype", "real");
		thisIssue.setAttribute("vtype", "real");
		SimpleElement thisRange = new SimpleElement("range");
		thisRange.setAttribute("lowerbound", "" + getLowerBound());
		thisRange.setAttribute("upperbound", "" + getUpperBound());
		thisIssue.addChildElement(thisRange);
		return thisIssue;

	}

	@Override
	public ISSUETYPE getType() {
		return ISSUETYPE.REAL;
	}

	@Override
	public String convertToString() {
		return "real";
	}
}