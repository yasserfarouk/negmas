package agents.anac.y2010.AgentSmith;

import java.util.ArrayList;
import java.util.HashMap;

import genius.core.Bid;
import genius.core.issue.Issue;
import genius.core.issue.Value;
import genius.core.issue.ValueDiscrete;
import genius.core.issue.ValueInteger;
import genius.core.issue.ValueReal;

/**
 * Model of one issue, takes all values of the opponent on this issue. Then the
 * utility and it's weight can be calculated on this issue. In the OpponentModel
 * this information is used to determine the utility of the other party.
 */
public class IssueModel {
	private ArrayList<Value> fValues;
	private Issue fIssue;
	private Bounds fBounds;

	/**
	 * Constructor
	 * 
	 * @param lIssue
	 */
	public IssueModel(Issue lIssue) {
		fValues = new ArrayList<Value>();
		fIssue = lIssue;
		fBounds = new Bounds(fIssue);
	}

	/**
	 * The mean of the values in the values list.
	 */
	public double getAverage() {
		double lTotal = 0;
		for (Value lWeight : fValues) {
			lTotal += getNumberValue(lWeight);
		}
		return lTotal / fValues.size();
	}

	/**
	 * The standard deviation utility of the issues.
	 */
	public double getDeviation() {
		double lIssueAverage = getAverage();
		double lTotal = 0;
		for (Value lWeight : fValues) {
			lTotal += Math.pow(getNumberValue(lWeight) - lIssueAverage, 2);
		}

		return Math.sqrt(lTotal / (double) fValues.size());
	}

	/**
	 * Add a value to the list
	 */
	public void addValue(Value pValue) {
		fValues.add(pValue);
	}

	/**
	 * The value of an issue in a double type
	 */
	public double getNumberValue(Value pValue) {
		switch (fIssue.getType()) {
		case DISCRETE:
			throw new RuntimeException("No get value for discret");
		case REAL:
			return ((ValueReal) pValue).getValue();
		case INTEGER:
			return ((ValueInteger) pValue).getValue();
		default:
			return 0;
		}

	}

	/**
	 * The utility of a bid, which can be real, integer or discrete
	 */
	public double getUtility(Bid pBid) {
		double lUtility = 0;
		switch (fIssue.getType()) {
		case INTEGER:
			lUtility = getRealUtility(pBid);
			break;
		case REAL:
			lUtility = getRealUtility(pBid);
			break;
		case DISCRETE:
			lUtility = getDiscreteUtility(pBid);
			break;
		}

		return lUtility;
	}

	/**
	 * The utility of if this issue is discrete. Takes the amount of values that
	 * are the same and divides it by the total amount of values. If there are
	 * lots of same values as the bid, the utility will be high, else it will be
	 * low.
	 */
	public double getDiscreteUtility(Bid pBid) {
		double lSame = 0;
		for (Value lValue : fValues) {
			ValueDiscrete lDiscreteValue = (ValueDiscrete) lValue;
			if (lDiscreteValue.getValue().equals(
					((ValueDiscrete) getBidValue(pBid)).getValue())) {
				lSame++;
			}
		}
		return this.fValues.size() == 0 ? 0 : lSame
				/ (double) this.fValues.size();
	}

	/**
	 * The utility of a bid in the real or integer case
	 */
	public double getRealUtility(Bid pBid) {
		return 1 - Math.abs(fBounds
				.normalize(getNumberValue(getBidValue(pBid)))
				- fBounds.normalize(getAverage()));
	}

	/**
	 * Get's the importance of this issues utility
	 */
	public double getWeight() {
		double lWeight = 0;
		switch (fIssue.getType()) {
		case INTEGER:
			lWeight = getRealWeight();
			break;
		case REAL:
			lWeight = getRealWeight();
			break;
		case DISCRETE:
			lWeight = getDiscreteWeight();
			break;
		}

		return lWeight;
	}

	/**
	 * Gets the weight of this value. If there are lots of changes, it's not
	 * important, else it is important.
	 */
	public double getDiscreteWeight() {
		HashMap<String, Integer> lCounter = new HashMap<String, Integer>();
		for (Value lValue : fValues) {
			ValueDiscrete lDiscreteValue = (ValueDiscrete) lValue;
			int lCount = lCounter.get(lDiscreteValue.getValue()) != null ? lCounter
					.get(lDiscreteValue.getValue()) : 0;
			lCount++;
			lCounter.put(lDiscreteValue.getValue(), lCount);
		}
		double lMax = 0;
		double lTotal = 0;

		for (int lC : lCounter.values()) {
			if (lC > lMax)
				lMax = lC;
			lTotal += lC;
		}

		return lMax / lTotal;
	}

	/**
	 * returns the real (or integer) weight
	 */
	public double getRealWeight() {
		return 1 / (1 + getDeviation());
	}

	/**
	 * returns the value of a bid
	 */
	public Value getBidValue(Bid pBid) {
		return getBidValueByIssue(pBid, fIssue);
	}

	/**
	 * returns the value of an issue in a bid
	 */
	public static Value getBidValueByIssue(Bid pBid, Issue pIssue) {
		Value lValue = null;
		try {
			lValue = pBid.getValue(pIssue.getNumber());
		} catch (Exception e) {
		}

		return lValue;

	}
}
