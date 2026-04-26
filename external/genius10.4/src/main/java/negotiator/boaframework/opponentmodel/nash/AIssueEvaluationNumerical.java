package negotiator.boaframework.opponentmodel.nash;

import java.util.ArrayList;

import genius.core.issue.Issue;
import genius.core.issue.Value;

/**
 * This is the base class for numerical issueEvaluations. All numerical IssueEvaluations assume that the utilityfunction
 * of the issue consist out of 3 points that form a triangle:
 * - A point at the left side of the function, which has zero utility.
 * - A point somewhere in between the left and right point, which has max utility.
 * - A point at the right side of the function, which has zero utility.
 * 
 * We then assume that the function is a straight line from the left point, to the middle point, to the right point.
 * The middle max utility point is calculated by the offered values of the negotiator. The earlier the value is offered,
 * the more important we think it is. To estimate the utility of an offered value, we then interpolate between the left and middle point,
 * or the right and middle point.
 * 
 * @author Roland van der Linden
 *
 */
public abstract class AIssueEvaluationNumerical extends AIssueEvaluation
{
	// **************************************
	// Fields
	// **************************************
	
	//The three points with which we estimate the utility function.
	protected double leftZeroUtilityValue, rightZeroUtilityValue, maxUtilityValue;
	//This is the range of OUR OWN utility function. We assume that the range of the opponent is almost equal to it.
	protected Range ourNonZeroUtilityRange;
	//These are the values that have been offered by the negotiator in the past.
	protected ArrayList<Value> offeredValueHistory;
	
	
	// **************************************
	// Constructor & init
	// **************************************
	
	/**
	 * This will construct a new numerical IssueEvaluation.
	 * @param issue The issue we are evaluating. 
	 * @param ourNonZeroUtilityRange The range of OUR OWN utilityfunction.
	 */
	public AIssueEvaluationNumerical(Issue issue, Range ourNonZeroUtilityRange) 
	{
		super(issue);
		
		this.ourNonZeroUtilityRange = ourNonZeroUtilityRange;
		this.offeredValueHistory = new ArrayList<Value>(250);
	}

	
	// **************************************
	// Update
	// **************************************

	/**
	 * This updates the numerical issueEvaluation with a newly offered value. 
	 * We first save the new value in our history, and then receiveMessage the max utility point and left and right zero utility points.
	 */
	@Override
	public void updateIssueEvaluation(Value chosenValue)
	{
		super.updateIssueEvaluation(chosenValue);
		
		//Save the newly offered value into the list.
		this.offeredValueHistory.add(chosenValue);
		
		//Update the value where the utility is 1.
		this.updateMaxUtilityValue();
		
		//Update the values where the utility becomes 0 (left and right).
		this.updateZeroUtilityValues();
	}
	
	/**
	 * This method updates the max utility value. We do this based on the history of offered values by the negotiator.
	 * We do not just take the average: The earlier a bid has been done, the higher the importance we give it.
	 * To calculate this factor, we use the following mechanism:
	 * 
	 * maxUtilityValue = 0.5 (+rest) * first offered value
	 * 					+ 0.25 * second offered value
	 * 					+ 0.125 * third offeredd value 
	 * 					+ ...
	 * 
	 */
	protected void updateMaxUtilityValue()
	{
		int n = this.offeredValueHistory.size();
		
		double newMaxUtilityValue = 0;
		
		for(int i = 0; i < n; i++)
		{
			double contributionWeight = 0;
			double numericalValue = this.getNumericalValue(this.offeredValueHistory.get(i));
				
			//The first value has an additional contribution because otherwise that is unused.
			if(i == 0)
				contributionWeight = (Math.pow(0.5, i + 1) + Math.pow(0.5, n));
			else
				contributionWeight = (Math.pow(0.5, i + 1));
				
			newMaxUtilityValue += (contributionWeight * numericalValue);
		}
			
		this.maxUtilityValue = newMaxUtilityValue;	
	}
	
	/**
	 * This method updates the zero utility values. We do this based on our max utility value,
	 * and the length of OUR OWN range. We assume this range is equal to ours.
	 */
	protected void updateZeroUtilityValues()
	{
		double ourRangeSize = this.ourNonZeroUtilityRange.getLength();
		double halfOurRangeSize = ourRangeSize / 2.0;
		
		double newLeftZeroUtilityValue = this.maxUtilityValue - halfOurRangeSize;
		double newRightZeroUtilityValue = this.maxUtilityValue + halfOurRangeSize;
		
		if(newLeftZeroUtilityValue < this.getIssueLowerBound())
		{
			double difference = Math.abs(this.getIssueLowerBound() - newLeftZeroUtilityValue);
			double freeSpaceOnRightSide = this.getIssueUpperBound() - newRightZeroUtilityValue;
			double usedDifference = Math.min(difference,  freeSpaceOnRightSide);
			newRightZeroUtilityValue += usedDifference;
		}
		
		if(newRightZeroUtilityValue > this.getIssueUpperBound())
		{
			double difference = Math.abs(newRightZeroUtilityValue - this.getIssueUpperBound());
			double freeSpaceOnLeftSide = newLeftZeroUtilityValue - this.getIssueLowerBound();
			double usedDifference = Math.min(difference,  freeSpaceOnLeftSide);
			newLeftZeroUtilityValue += usedDifference;
		}
		
		this.leftZeroUtilityValue = newLeftZeroUtilityValue;
		this.rightZeroUtilityValue = newRightZeroUtilityValue;
	}
	
	
	// **************************************
	// Getters
	// **************************************
	
	/**
	 * This returns the normalized weight for the given value. We calculate this value on two rules:
	 * - If the value lies in between the left and max point or the right and middle point:
	 * 		We interpolate between the two points, with utility 1 at the middle point and utility 0 at the other point.
	 * - If the value does not lie within this range:
	 * 		We return utility 0 (outside the range of positive utility).
	 */
	@Override
	public double getNormalizedValueWeight(Value value)
	{
		//We do not allow valueWeight requests when no values have been offered yet.
		if(!this.isFirstValueOffered())
			throw new IllegalStateException("ValueWeights can not be calculated when not values have been offered yet.");
		
		double numericalValue = this.getNumericalValue(value);
		
		//Test whether the value lies within the left zero utility and max utility values
		if(numericalValue >= leftZeroUtilityValue && numericalValue <= maxUtilityValue)
			return getNormalizedInterpolatedWeight(leftZeroUtilityValue, maxUtilityValue, numericalValue);
		//Test whether the value lies within the max utility and right zero utility values.
		else if(numericalValue >= maxUtilityValue && numericalValue <= rightZeroUtilityValue)
			return getNormalizedInterpolatedWeight(rightZeroUtilityValue, maxUtilityValue, numericalValue);
		//The value does not lie within the range where the utility is >0, so it's weight is zero.
		else
			return 0;
	}
	
	/**
	 * This method executes the interpolation between the max utility point and the left or right zero utility point.
	 * We return the interpolated utility value for the inBetweenValue
	 * @param zeroPoint The point where the utility is zero.
	 * @param maxPoint The point where the utility is one.
	 * @param inBetweenValue The value for which we wish to know the utility.
	 * @return The interpolated utility, based on the location of the inBetweenValue.
	 */
	protected double getNormalizedInterpolatedWeight(double zeroPoint, double maxPoint, double inBetweenValue)
	{
		return Math.abs((inBetweenValue - zeroPoint) / (maxPoint - zeroPoint));
	}
	
	/**
	 * This return the actual numerical value that resides inside the Value object.
	 * @param value
	 * @return
	 */
	protected abstract double getNumericalValue(Value value);
	
	/**
	 * This function must be implemented by the subclass to return the lower bound of the issue under evaluation.
	 * Since this class only works with abstract classes Value and Issue, we cannot extract the bound
	 * in this class.
	 * @return
	 */
	public abstract double getIssueLowerBound();
	
	/**
	 * This function must be implemented by the subclass to return the upper bound of the issue under evaluation.
	 * Since this class only works with abstract classes Value and Issue, we cannot extract the bound
	 * in this class.
	 * @return
	 */
	public abstract double getIssueUpperBound();
	
	/**
	 * This method returns the length of the range of our issue.
	 * @return
	 */
	public double getIssueRangeLength()
	{
		return this.getIssueUpperBound() - this.getIssueLowerBound();
	}
	
	/**
	 * This returns the standard deviation of the list of offered values by the opponent.
	 * @return
	 */
	public double getOfferedValuesStandardDeviation()
	{
		ArrayList<Double> valueList = convertToNumericalValues(this.offeredValueHistory);
		return StatisticsUtil.getStandardDeviation(valueList);
	}
	
	
	// **************************************
	// Other methods
	// **************************************
	
	/**
	 * This method converts a list of Value objects to a  list containing the actual numerical values that reside inside them.
	 * @param values
	 * @return
	 */
	private ArrayList<Double> convertToNumericalValues(ArrayList<Value> values)
	{
		ArrayList<Double> result = new ArrayList<Double>(values.size());
		for(Value v : values)
			result.add(this.getNumericalValue(v));
		
		return result;
	}
	
	/**
	 * This returns a string representation of the issueEvaluation.
	 * @return The string representation.
	 */
	public String toString()
	{
		String result = super.toString();
		String nl = "\n";
		String pre = "   ";
		
		result += "===== Boundaries =====" + nl;
		result += pre + "lower = " + this.getIssueLowerBound() + nl;
		result += pre + "upper = " + this.getIssueUpperBound() + nl;
		result += "===== Utility Piramid Values =====" + nl;
		result += pre + "left = " + leftZeroUtilityValue + nl;
		result += pre + "max = " + maxUtilityValue + nl;
		result += pre + "right = " + rightZeroUtilityValue + nl;
		result += "===== offeredValueHistory =====" + nl;
		result += pre + offeredValueHistory.toString() + nl;
		result += "##### END #####" + nl;
		
		return result;
	}
	
}