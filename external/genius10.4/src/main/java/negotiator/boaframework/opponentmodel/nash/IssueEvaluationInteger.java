package negotiator.boaframework.opponentmodel.nash;

import genius.core.issue.IssueInteger;
import genius.core.issue.Value;
import genius.core.issue.ValueInteger;

/**
 * This class evaluates an IssueInteger. It uses the principle of an IssueEvaluationNumerical.
 * Since the abstract version already implements a lot of the functionality, this class
 * exists mainly to work with the specific IssueInteger and ValueInteger objects.
 * 
 * @author Roland van der Linden
 *
 */
public class IssueEvaluationInteger extends AIssueEvaluationNumerical
{
	/**
	 * This constructs the IssueEvaluationInteger.
	 * @param issueI The issue we are evaluating.
	 * @param ourNonZeroUtilityRange Our own range where the utility is not zero.
	 */
	public IssueEvaluationInteger(IssueInteger issueI, Range ourNonZeroUtilityRange)
	{
		super(issueI, ourNonZeroUtilityRange);
	}

	
	// **************************************
	// Getters
	// **************************************
	
	/**
	 * This method gives us the casted IssueInteger object we are evaluating.
	 * @return The IssueInteger we are evaluating.
	 */
	public IssueInteger getIssueInteger()
	{
		return (IssueInteger)this.issue;
	}
	
	/**
	 * This method returns the actual value that resides inside the Value object.
	 * Given objects should be ValueInteger objects.
	 */
	@Override
	protected double getNumericalValue(Value value) 
	{
		if(!(value instanceof ValueInteger))
			throw new IllegalArgumentException("The IssueEvaluationInteger getNumericalValue method requires a ValueInteger value. It is now a: " + value.getClass().getSimpleName());
		
		return ((ValueInteger)value).getValue(); 
	}

	/**
	 * This method returns the lower bound of the range of the IssueInteger.
	 */
	@Override
	public double getIssueLowerBound() 
	{
		return this.getIssueInteger().getLowerBound();
	}

	/**
	 * This method returns the upper bound of the range of the IssueInteger.
	 */
	@Override
	public double getIssueUpperBound() 
	{
		return this.getIssueInteger().getUpperBound();
	}
	
}