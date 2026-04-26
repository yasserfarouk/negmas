package negotiator.boaframework.opponentmodel.nash;

import java.util.HashMap;

import genius.core.issue.Value;


/**
 * This class manages all the issue evaluations and provides easy methods to access and change them.
 * Another important function of this class, is that it calculates the normalized weights for each issue(evaluation).
 * This weight is the estimated importance for each issue.
 * 
 * @author Roland van der Linden
 *
 */
public class IssueEvaluationList
{
	// *******************************************
	// Fields
	// *******************************************
	
	//This map holds references to all the issueEvaluations, which can be found by their
	//unique ID.
	private HashMap<Integer, AIssueEvaluation> issueEvaluationMap;
	//This map holds the normalized weight for each issue.
	private HashMap<Integer, Double> normalizedIssueWeightMap;
	
	
	// *******************************************
	// Constructor & init
	// *******************************************
	
	/**
	 * This constructs the IssueEvaluationList.
	 */
	public IssueEvaluationList(int issueCapacity)
	{
		this.issueEvaluationMap = new HashMap<Integer, AIssueEvaluation>(issueCapacity);
		this.normalizedIssueWeightMap = new HashMap<Integer, Double>(issueCapacity);
	}
	
	
	// ********************************************
	// Add & Remove
	// ********************************************
	
	/**
	 * This method will add the issueEvaluation to the hashmap,
	 * or overwrite it's value if the issueNumber already exists.
	 * @param ie The issueEvaluation to be added.
	 */
	public void addIssueEvaluation(AIssueEvaluation ie)
	{
		this.issueEvaluationMap.put(ie.getIssueID(), ie);
	}
	
	
	// ********************************************
	// Update
	// ********************************************
	
	/**
	 * This method will receiveMessage a single issueEvaluation.
	 * @param issueID The issueID of the issueEvaluation that should be updated.
	 * @param offeredValue The offered value for that issue.
	 */
	public void updateIssueEvaluation(int issueID, Value offeredValue)
	{
		AIssueEvaluation issueEvaluation = this.issueEvaluationMap.get(issueID);
		issueEvaluation.updateIssueEvaluation(offeredValue);
	}
	
	/**
	 * This method will receiveMessage the issueWeightMap. It will first calculate the total unnormalized weight of the issues,
	 * and then normalize it by calculating the percentage that each issue has of the total weight.
	 * Note that if the total weight is zero, we have a special case in which all issues receive equal weights.
	 */
	public void updateIssueWeightMap()
	{
		//Calculate the accumulated weight of all issues which we can use to calculate
		//what percentage of that total an individual issue has (this helps us normalize the issue weights).
		double totalIssueWeight = 0;
		for(AIssueEvaluation issueEvaluation : this.issueEvaluationMap.values())
			totalIssueWeight += calculateIndividualIssueWeight(issueEvaluation.getIssueID());
		
		//If the total issue weight is zero, we have a special case in which all issues just get 
		//the same normalized weight (which adds up to 1).
		if(totalIssueWeight <= 0)
		{
			double specialIssueWeight = 1.0 / this.issueEvaluationMap.size();
			for(AIssueEvaluation issueEvaluation : this.issueEvaluationMap.values())
				this.normalizedIssueWeightMap.put(issueEvaluation.getIssueID(), specialIssueWeight);
		
		}
		//Otherwise we iterate over the issueEvaluations again, and calculate what percentage of the weight each
		//issueEvaluation has. This then becomes the normalized weight for each issue, which we save in the map.
		else
		{
			for(AIssueEvaluation issueEvaluation : this.issueEvaluationMap.values())
			{
				double issueWeight = calculateIndividualIssueWeight(issueEvaluation.getIssueID());
				double normalizedWeight = issueWeight / totalIssueWeight;
				this.normalizedIssueWeightMap.put(issueEvaluation.getIssueID(), normalizedWeight);
			}
		}
	}
	
	/**
	 * This method calculates the issueWeight for the issue with the
	 * given issueID. Note that this weight has NOT been normalized,
	 * and is preferably used for the normalization process itself.
	 */
	private double calculateIndividualIssueWeight(int issueID)
	{
		AIssueEvaluation issueEvaluation = this.getIssueEvaluation(issueID);
		if(issueEvaluation instanceof IssueEvaluationDiscrete)
		{
			IssueEvaluationDiscrete issueEvaluationDiscrete = (IssueEvaluationDiscrete)issueEvaluation;
			double percentage = issueEvaluationDiscrete.getPercentageOfHighestFrequency();
			
			return percentage;
		}
		else if(issueEvaluation instanceof AIssueEvaluationNumerical)
		{
			AIssueEvaluationNumerical issueEvaluationNumerical = (AIssueEvaluationNumerical)issueEvaluation;
			double std = issueEvaluationNumerical.getOfferedValuesStandardDeviation();
			double range = issueEvaluationNumerical.getIssueRangeLength();

			double std_per_unit = std / range;
			return 1 - (2 * std_per_unit);
		}
		else 
			throw new IllegalStateException("The individual issue weight cannot be calculated because the issueEvaluation type is unknown.");
	}
	
	
	// ********************************************
	// Getters
	// ********************************************
	
	/**
	 * This method returns the issueEvaluation with the given ID.
	 * @param issueID The unique ID of the issueEvaluation.
	 * @return
	 */
	public AIssueEvaluation getIssueEvaluation(int issueID)
	{
		return this.issueEvaluationMap.get(issueID);
	}
	
	/**
	 * This method returns the estimated weight for the issue with the given ID.
	 * @param issueID The issueID of the issue for which we want to know the estimated weight.
	 * @return
	 */
	public double getNormalizedIssueWeight(int issueID)
	{
		if (isReady()) {
			return this.normalizedIssueWeightMap.get(issueID);
		} else {
			return 0;
		}
	}
	
	/**
	 * This returns the size of the list.
	 * @return
	 */
	public int getSize()
	{
		return this.issueEvaluationMap.size();
	}
	
	/**
	 * This method returns a string representation of the issueEvaluationList.
	 */
	public String toString()
	{
		String result = "";
		String nl = "\n";
		
		result += "================================================" + nl;
	
		result += "IssueWeights: " + nl;

		result += "IssueEvaluations: " + nl;
		for(AIssueEvaluation ie : this.issueEvaluationMap.values())
			result += ie.toString() + nl;
		
		result += "================================================" + nl;
		
		return result;
	}


	public boolean isReady() {
		return (normalizedIssueWeightMap.size() > 0);
	}
}