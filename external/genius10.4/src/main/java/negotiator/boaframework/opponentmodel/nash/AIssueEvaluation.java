package negotiator.boaframework.opponentmodel.nash;

import genius.core.issue.ISSUETYPE;
import genius.core.issue.Issue;
import genius.core.issue.Value;

/**
 * This is the base version of the evaluation of an issue. It holds the issue we are evaluating for later reference,
 * and keep track of the firstOfferedValue (which we suppose is important). 
 * 
 * This base class offers the guidelines that a specific IssueEvaluation should implement.
 * 
 * @author Roland van der Linden
 *
 */
public abstract class AIssueEvaluation
{
	// ********************************
	// Fields
	// ********************************
	
	//The issue we are evaluating.
	protected Issue issue;
	//This is the value that has been offered the first time (probably the most preferred value).
	protected Value firstOfferedValue;
	
	
	// ********************************
	// Constructor & init
	// ********************************
	
	/**
	 * This constructs the AIssueEvaluation. You need to provide the issue that will 
	 * be evaluated, so that the appropriate fields can be initialized.
	 * @param issue The issue we are evaluating. May not be null.
	 */
	public AIssueEvaluation(Issue issue)
	{
		if(issue == null) 
			throw new IllegalArgumentException("The provided issue was null.");
		
		this.issue = issue;
	}

	
	// ********************************
	// Update
	// ********************************
	
	/**
	 * This method updates the issueEvaluation. Subclasses should override this
	 * method to handle values that have been chosen in a newly offered bid.
	 * 
	 * We also save the first value that has been offered to find out if this always is they're
	 * most important value.
	 * @param chosenValue The value of the issue we are evaluating that has just been chosen. May not be null.
	 */
	public void updateIssueEvaluation(Value chosenValue)
	{
		if(chosenValue == null) 
			throw new IllegalArgumentException("The chosenValue may not be null.");
		
		//We save the first offered value.
		if(!this.isFirstValueOffered())
			this.firstOfferedValue = chosenValue;
	}
	
	
	// ********************************
	// Getters
	// ********************************
	
	/**
	 * This method specifies that each subclass needs to implement a function that gives us
	 * the normalized weight for the given value. Since the value can be either discrete or numerical,
	 * we need to subclass to define how to extract the normalized valueWeight.
	 * 
	 * Note that normalization for the valueWeights is done like this:
	 * 	- The value with the highest estimated utility has normalized weight 1. 
	 *  - The normalized weight of the other values is scaled accordingly.
	 * 
	 * @param value The value (discrete or non-discrete) of which we want to know the normalized weight.
	 * @return The normalized weight of the given value, ranging between 1 and 0.
	 */
	public abstract double getNormalizedValueWeight(Value value);
	
	/**
	 * This method tells us whether a first value has been offered or not.
	 * @return true if first value has been offered.
	 */
	public boolean isFirstValueOffered()
	{
		return this.firstOfferedValue != null;
	}
	
	/**
	 * This returns the ID of the issue we are evaluating.
	 * @return The ID of the issue.
	 */
	public int getIssueID()
	{
		return this.issue.getNumber();
	}
	
	/**
	 * This returns the name of the issue we are evaluating.
	 * @return The name of the issue.
	 */
	public String getIssueName()
	{
		return this.issue.getName();
	}
	
	/**
	 * This returns the type of the issue we are evaluating.
	 * @return The type of the issue.
	 */
	public ISSUETYPE getIssueType()
	{
		return this.issue.getType();
	}
	
	/**
	 * This returns a string representation of the issueEvaluation.
	 * @return The string representation.
	 */
	public String toString()
	{
		String result = "";
		String nl = "\n";
		String pre = "   ";
		
		String firstOffered = "null";
		if(this.firstOfferedValue != null) 
			firstOffered = this.firstOfferedValue.toString();
		
		result += "##### IssueEvaluation #####" + nl;
		result += "===== IssueID, IssueName, IssueType, FirstOfferedValue =====" + nl;
		result += pre + this.getIssueID() + " - " + this.getIssueName() + " - " + this.getIssueType() + " - " + firstOffered + nl;
		
		return result;
	}
}