package negotiator.boaframework.opponentmodel.nash;

import genius.core.issue.IssueReal;
import genius.core.issue.Value;
import genius.core.issue.ValueReal;

/**
 * This class evaluates an IssueReal. It uses the principle of an IssueEvaluationNumerical.
 * Since the abstract version already implements a lot of the functionality, this class
 * exists mainly to work with the specific IssueReal and ValueReal objects.
 * 
 * @author Roland van der Linden
 *
 */
public class IssueEvaluationReal extends AIssueEvaluationNumerical
{
	/**
	 * This constructs the issueEvaluationReal.
	 * @param issueR
	 * @param ourNonZeroUtilityRange
	 */
	public IssueEvaluationReal(IssueReal issueR, Range ourNonZeroUtilityRange) 
	{
		super(issueR, ourNonZeroUtilityRange);
	}

	
	// **************************************
	// Getters
	// **************************************
	
	/**
	 * This method gives us the casted IssueReal object we are evaluating.
	 * @return The IssueReal we are evaluating.
	 */
	public IssueReal getIssueReal()
	{
		return (IssueReal)this.issue;
	}
	
	/**
	 * This returns the actual numerical value that resides inside the Value object.
	 * Object must be a ValueReal.
	 */
	@Override
	protected double getNumericalValue(Value value) 
	{
		if(!(value instanceof ValueReal))
			throw new IllegalArgumentException("The IssueEvaluationReal getNumericalValue method requires a ValueReal value. It is now a: " + value.getClass().getSimpleName());
		
		return ((ValueReal)value).getValue(); 
	}

	/**
	 * This method returns the lower bound of the range of the IssueReal.
	 */
	@Override
	public double getIssueLowerBound() 
	{
		return getIssueReal().getLowerBound();
	}

	/**
	 * This method returns the upper bound of the range of the IssueReal.
	 */
	@Override
	public double getIssueUpperBound() 
	{
		return getIssueReal().getUpperBound();
	}
	
}