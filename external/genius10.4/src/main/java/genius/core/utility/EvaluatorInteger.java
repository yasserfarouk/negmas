package genius.core.utility;

import java.text.DecimalFormat;

import genius.core.Bid;
import genius.core.issue.Objective;
import genius.core.issue.ValueInteger;
import genius.core.xml.SimpleElement;

/**
 * This class is used to convert the value of an integer issue to a utility.
 * This object stores the range of the issue and a linear function mapping each
 * value to a utility. Note that this utility is not yet normalized by the issue
 * weight and is therefore in the range [0,1].
 * 
 */
@SuppressWarnings("serial")
public class EvaluatorInteger implements Evaluator {

	// Class fields
	private double fweight; // the weight of the evaluated Objective or Issue.
	private boolean fweightLock;
	private int lowerBound;
	private int upperBound;
	private double slope = 0.0;
	private double offset = 0.0;
	private DecimalFormat f = new DecimalFormat("0.0000");

	/**
	 * Creates a new integer evaluator with weight 0 and no values.
	 */
	public EvaluatorInteger() {
		fweight = 0;
	}

	@Override
	public double getWeight() {
		return fweight;
	}

	@Override
	public void setWeight(double wt) {
		fweight = wt;
	}

	@Override
	public void lockWeight() {
		fweightLock = true;
	}

	@Override
	public void unlockWeight() {
		fweightLock = false;
	}

	@Override
	public boolean weightLocked() {
		return fweightLock;
	}

	@Override
	public Double getEvaluation(AdditiveUtilitySpace uspace, Bid bid,
			int index) {
		Integer lTmp = null;
		try {
			lTmp = ((ValueInteger) bid.getValue(index)).getValue();
		} catch (Exception e) {
			e.printStackTrace();
		}
		return getEvaluation(lTmp);
	}

	@Override
	public EVALUATORTYPE getType() {
		return EVALUATORTYPE.INTEGER;
	}

	@Override
	public void loadFromXML(SimpleElement pRoot) {

		this.lowerBound = Integer.valueOf(pRoot.getAttribute("lowerbound"));
		this.upperBound = Integer.valueOf(pRoot.getAttribute("upperbound"));

		Object[] xml_items = pRoot.getChildByTagName("evaluator");
		if (xml_items.length != 0) {
			this.slope = Double.valueOf(
					((SimpleElement) xml_items[0]).getAttribute("slope"));
			this.offset = Double.valueOf(
					((SimpleElement) xml_items[0]).getAttribute("offset"));
		}
	}

	@Override
	public String toString() {
		return "{Integer: offset=" + f.format(offset) + " slope="
				+ f.format(slope) + " range=[" + lowerBound + ":" + upperBound
				+ "]}";
	}

	@Override
	public String isComplete(Objective whichobj) {
		return null;
	}

	/************** specific for EvaluationInteger ****************/
	/**
	 * @param value
	 *            of an issue.
	 * @return utility of the given value (range: [0,1]).
	 */
	public Double getEvaluation(int value) {
		double utility;

		utility = EVALFUNCTYPE.evalLinear(value - lowerBound, slope, offset);
		if (utility < 0)
			utility = 0;
		else if (utility > 1)
			utility = 1;
		return utility;
	}

	/**
	 * @return evaluation function type.
	 */
	public EVALFUNCTYPE getFuncType() {
		return null;
	}

	/**
	 * @return lowerbound of the integer issue.
	 */
	public int getLowerBound() {
		return lowerBound;
	}

	/**
	 * @return higherbound of the integer issue.
	 */
	public int getUpperBound() {
		return upperBound;
	}

	/**
	 * @return lowest possible utility value.
	 */
	public double getUtilLowestValue() {
		return offset;
	}

	/**
	 * @return highest possible utility value.
	 */
	public double getUtilHighestValue() {
		return (offset + slope * (upperBound - lowerBound));
	}

	/**
	 * Sets the lower bound of this evaluator.
	 * 
	 * @param lb
	 *            The new lower bound
	 */
	public void setLowerBound(int lb) {
		lowerBound = lb;
	}

	/**
	 * Sets the upper bound of this evaluator.
	 * 
	 * @param ub
	 *            The new upper bound
	 */
	public void setUpperBound(int ub) {
		upperBound = ub;
	}

	/**
	 * Specifies the linear utility function of the issue by giving the utility
	 * of the lowest value and the highest value.
	 * 
	 * @param utilLowInt
	 *            utility of the lowest vale.
	 * @param utilHighInt
	 *            utility of the highest value.
	 */
	public void setLinearFunction(double utilLowInt, double utilHighInt) {
		slope = (utilHighInt - utilLowInt) / (-lowerBound + upperBound);
		offset = utilLowInt;
	}

	/**
	 * Sets weights and evaluator properties for the object in SimpleElement
	 * representation that is passed to it.
	 * 
	 * @param evalObj
	 *            The object of which to set the evaluation properties.
	 * @return The modified simpleElement with all evaluator properties set.
	 */
	public SimpleElement setXML(SimpleElement evalObj) {
		return evalObj;
	}

	/**
	 * @return slope of the linear utility function.
	 */
	public double getSlope() {
		return slope;
	}

	/**
	 * Sets the slope of the linear utility function.
	 * 
	 * @param slope
	 *            of the linear utility function.
	 */
	public void setSlope(double slope) {
		this.slope = slope;
	}

	/**
	 * Sets the slope of the linear utility function.
	 * 
	 * @param slope
	 *            of the linear utility function.
	 */
	@Deprecated
	public void setLinearParam(double slope) {
		setSlope(slope);
	}

	/**
	 * @return offset of the linear utility function.
	 */
	public double getOffset() {
		return offset;
	}

	/**
	 * Sets the offset of the linear utility function.
	 * 
	 * @param offset
	 *            of the linear utility function.
	 */
	public void setOffset(double offset) {
		this.offset = offset;
	}

	/**
	 * Sets the offset of the linear utility function.
	 * 
	 * @param offset
	 *            of the linear utility function.
	 */
	@Deprecated
	public void setConstantParam(double offset) {
		setOffset(offset);
	}

	@Override
	public EvaluatorInteger clone() {
		EvaluatorInteger ed = new EvaluatorInteger();
		ed.setWeight(fweight);
		ed.setUpperBound(upperBound);
		ed.setLowerBound(lowerBound);
		ed.slope = slope;
		ed.offset = offset;
		return ed;
	}
}