package negotiator.boaframework.opponentmodel.agentsmith;

import java.util.HashMap;
import java.util.List;

import genius.core.Bid;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.IssueInteger;
import genius.core.issue.IssueReal;
import genius.core.issue.Value;
import genius.core.issue.ValueDiscrete;
import genius.core.issue.ValueInteger;
import genius.core.issue.ValueReal;

/**
 * Class that is used to determine the bounds of an issue.
 */
public class Bounds {
	private double fLower;
	private double fUpper;
	private double fAmtSteps;

	/**
	 * Constructor
	 */
	public Bounds() {
	}

	/**
	 * Constructor. Sets the bounds for a given issue based on its type
	 * (discrete, real or integer)
	 */
	public Bounds(Issue pIssue) {
		switch (pIssue.getType()) {
		case DISCRETE:
			IssueDiscrete lIssueDiscrete = ((IssueDiscrete) pIssue);
			setUpper(lIssueDiscrete.getNumberOfValues());
			setLower(0);
			setAmtSteps(lIssueDiscrete.getNumberOfValues());
			break;
		case REAL:
			IssueReal lIssueReal = ((IssueReal) pIssue);
			setUpper(lIssueReal.getUpperBound());
			setLower(lIssueReal.getLowerBound());
			setAmtSteps(10);
			break;
		case INTEGER:
			IssueInteger lIssueInteger = ((IssueInteger) pIssue);
			setUpper(lIssueInteger.getUpperBound());
			setLower(lIssueInteger.getLowerBound());
			setAmtSteps(10);
			break;
		}
	}

	/**
	 * returns the lower bound
	 */
	public double getLower() {
		return fLower;
	}

	/**
	 * returns the upper bound
	 */
	public double getUpper() {
		return fUpper;
	}

	/**
	 * returns the amount of steps
	 */
	public double getAmtSteps() {
		return fAmtSteps;
	}

	/**
	 * returns the number of steps
	 */
	public double getStepSize() {
		return (fUpper - fLower) / fAmtSteps;
	}

	/**
	 * sets the lower bound
	 */
	public void setLower(double pLower) {
		this.fLower = pLower;
	}

	/**
	 * sets the upper bound
	 */
	public void setUpper(double pUpper) {
		this.fUpper = pUpper;
	}

	/**
	 * set the number of steps to take
	 */
	public void setAmtSteps(double pAmtSteps) {
		this.fAmtSteps = pAmtSteps;
	}

	/**
	 * Creates a hashmap with for each of the issues the bounds
	 */
	public static HashMap<Integer, Bounds> getIssueBounds(List<Issue> pIssues) {
		HashMap<Integer, Bounds> bounds = new HashMap<Integer, Bounds>();

		for (Issue lIssue : pIssues) {
			Bounds b = new Bounds(lIssue);

			bounds.put(lIssue.getNumber(), b);

		}
		return bounds;
	}

	/**
	 * returns a Value object with the value of an issue at the given index
	 * works for real, discrete and integer objects
	 */
	public static Value getIssueValue(Issue pIssue, double pIndex) {
		Value v = null;
		switch (pIssue.getType()) {
		case DISCRETE:
			v = ((IssueDiscrete) pIssue).getValue((int) pIndex);
			break;
		case REAL:
			v = new ValueReal(pIndex);
			break;
		case INTEGER:
			v = new ValueInteger((int) pIndex);
			break;
		}
		return v;
	}

	/**
	 * returns the scaled value of the (discrete, real or integer) issue. it is
	 * scaled by the difference between the value and the lower bound divided by
	 * the length of the bounds
	 */
	public static double getScaledIssueValue(Bounds pBounds, Bid pBid,
			Issue pIssue) throws Exception {

		Value v = pBid.getValue(pIssue.getNumber());

		double value = 0;

		switch (pIssue.getType()) {
		case DISCRETE:
			value = ((IssueDiscrete) pIssue).getValueIndex(((ValueDiscrete) v)
					.getValue());
			break;
		case REAL:
			value = ((ValueReal) v).getValue();
			break;
		case INTEGER:
			value = ((ValueInteger) v).getValue();
			break;
		}

		return (value - pBounds.getLower())
				/ (pBounds.getUpper() - pBounds.getLower());

	}

	/**
	 * returns a normalized version of a value
	 */
	public double normalize(double pValue) {
		return (pValue - getLower()) / (getUpper() - getLower());
	}

	/**
	 * returns a string with the upper and lower bounds
	 */
	public String toString() {
		return "Upper: " + getUpper() + " Lower: " + getLower();
	}

}
