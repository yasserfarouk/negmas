package negotiator.boaframework.opponentmodel.nash;

import java.util.HashMap;
import java.util.List;
import java.util.Map.Entry;

import genius.core.issue.IssueDiscrete;
import genius.core.issue.Value;
import genius.core.issue.ValueDiscrete;

/**
 * The IssueEvaluationDiscrete class evaluates an issue. It keeps track of the frequency with which the evaluated
 * negotiator picks values, and bases the importance of the values on this.
 * 
 * The class uses two hashmaps, which are constantly up to date:
 * - The list with the chosen frequency per value.
 * - The list of normalized weights based on the frequencies.
 * 
 * @author Roland van der Linden
 *
 */
public class IssueEvaluationDiscrete extends AIssueEvaluation
{
	// ********************************
	// Fields
	// ********************************

	//This hashmap holds the number of times a certain value has been chosen. 
	//Each value of an issue is represented by a String (must be unique).
	protected HashMap<String, Integer> valueFrequencyMap;
	//This hashmap holds the normalized estimated weights of the values in the issue.
	//This is based on the number of times a value has been chosen, relative to the number
	//of times a choice was made. So presumably, the more often a value is chosen,
	//the more preferable it is. 
	//Values range in between 1 and 0, with the most important value having weight 1.
	//Note that the weight of a value is UNRELATED to the weight of the issue itself (in this class).
	//Each value of an issue is represented by a String (must be unique).
	private HashMap<String, Double> normalizedValueWeightMap;
	
	
	// ********************************
	// Constructor & init
	// ********************************
	
	/**
	 * This constructs the IssueEvaluationDiscrete. You need to
	 * provide the IssueDiscrete which needs to be evaluated so we can 
	 * initialize the corresponding fields.
	 * @param issueD The issueDiscrete that needs to be evaluated. May not be null.
	 */
	public IssueEvaluationDiscrete(IssueDiscrete issueD) 
	{
		super(issueD);
		
		this.initValueFrequencyMap();
		this.initValueWeightMap();
	}
	
	/**
	 * This method will initialize the valueFrequencyMap. We insert a representing string
	 * for each value in the issue. We also set the frequency of each value to zero.
	 */
	protected void initValueFrequencyMap() 
	{
		List<ValueDiscrete> discreteValues = this.getIssueDiscrete().getValues();
		this.valueFrequencyMap = new HashMap<String, Integer>(discreteValues.size());
		
		//Get all possible values of the discrete issue and give all of them frequency zero.
		for(ValueDiscrete vd : discreteValues)
			this.valueFrequencyMap.put(vd.getValue(), 0);
	}
	
	/**
	 * This method will initialize the normalizedValueWeightMap. We insert a representing string
	 * for each value in the issue. We set the weight of each value to -1. Note that those
	 * values should never be used.
	 */
	private void initValueWeightMap() 
	{
		List<ValueDiscrete> discreteValues = this.getIssueDiscrete().getValues();
		this.normalizedValueWeightMap = new HashMap<String, Double>(discreteValues.size());
		
		//Get all possible values of the discrete issue and them weight -1.
		for(ValueDiscrete vd : discreteValues)
			this.normalizedValueWeightMap.put(vd.getValue(), -1.0);
	}
	
	
	// ********************************
	// Update
	// ********************************
	
	/**
	 * This method updates the IssueEvaluationDiscrete, based on the value that has been offered to us
	 * in a new bid. We will first receiveMessage the valueFrequencyMap based on the newly chosen value, and then receiveMessage
	 * the normalizedValueWeightMap based on the new frequencies.
	 * @param chosenValue The value of the issue we are evaluating that has just been chosen. May not be null.
	 */
	@Override
	public void updateIssueEvaluation(Value chosenValue)
	{
		super.updateIssueEvaluation(chosenValue);
		
		this.updateValueFrequencyMap(chosenValue);
		this.updateValueWeightMap();
	}
	
	
	/**
	 * This method updates the valueFrequencyMap for the IssueDiscrete we are evaluating. 
	 * Based on the value that has been chosen, we add +1 to the amount of times the 
	 * value has been chosen in the past.
	 */
	protected void updateValueFrequencyMap(Value chosenValue) 
	{
		if(!(chosenValue instanceof ValueDiscrete))
			throw new IllegalArgumentException("The IssueEvaluationDiscrete receiveMessage method requires a ValueDiscrete value. It is now a: " + chosenValue.getClass().getSimpleName());
	
		ValueDiscrete valueD = (ValueDiscrete)chosenValue;
		
		//Calculate the new frequency.
		Integer newFrequency = 1 + this.valueFrequencyMap.get(valueD.getValue());
		
		//Remove the old frequency of the discrete value, and insert the new one.
		this.valueFrequencyMap.put(valueD.getValue(), newFrequency);
	}
	
	/**
	 * This method updates the normalizedValueWeightMap. The weight of the value with the highest frequency
	 * is 1. The weight of the other values is scaled according to their frequency relatively to the highest frequency.
	 * 
	 */
	private void updateValueWeightMap()
	{
		int highestFrequency = this.getHighestFrequency();
		
		//We do not allow calculation if no frequencies have been entered, so we throw
		//an exception if the highestFrequency is zero.
		if(highestFrequency <= 0)
			throw new IllegalStateException("ValueWeight calculations are not supported when no frequencies have been set.");
		
		//And then we normalize the weight of each value based on their frequency divided by the highestFrequency.
		for(Entry<String, Integer> entry : this.valueFrequencyMap.entrySet())
		{
			double newValueWeight = (double)entry.getValue() / (double)highestFrequency;
			this.normalizedValueWeightMap.put(entry.getKey(), newValueWeight);
		}
	}
	
	
	// **************************************
	// Getters
	// **************************************
	
	/**
	 * This method gives us the casted IssueDiscrete object 
	 * we are evaluating.
	 * @return The IssueDiscrete we are evaluating.
	 */
	public IssueDiscrete getIssueDiscrete()
	{
		return (IssueDiscrete)this.issue;
	}

	/**
	 * This returns the number of possible values for the discrete issue.
	 */
	public int getNumberOfDiscreteValues()
	{
		return this.getIssueDiscrete().getNumberOfValues();
	}
	
	/**
	 * This method returns the sum of the frequencies in the valueFrequencyMap.
	 * This should be equal to the number of offered values.
	 */
	protected int getSummedFrequency()
	{
		if(!this.isFirstValueOffered())
			throw new IllegalStateException("Frequency calculations are not supported when no values have been offered yet.");
		
		int result = 0;
		
		for(Integer valueFrequency : this.valueFrequencyMap.values())
			result += valueFrequency;
		
		return result;
	}
	
	/**
	 * This method returns the highest frequency that can be 
	 * found in the valueFrequencyMap.
	 */
	protected int getHighestFrequency()
	{
		if(!this.isFirstValueOffered())
			throw new IllegalStateException("Frequency calculations are not supported when no values have been offered yet.");
		
		int highestFrequency = -1;
		
		for(Integer valueFrequency : this.valueFrequencyMap.values())
			if(valueFrequency > highestFrequency)
				highestFrequency = valueFrequency;
		
		return highestFrequency;
	}

	/**
	 * This method returns the percentage of the highest frequency.
	 * So we return the frequency of the value with the highest frequency and divide it with the total number of values chosen.
	 * @return
	 */
	protected double getPercentageOfHighestFrequency()
	{		
		if(!this.isFirstValueOffered())
			throw new IllegalStateException("Frequency calculations are not supported when no values have been offered yet.");
		
		return (double)getHighestFrequency() / (double)getSummedFrequency();
	}
	
	/**
	 * This method returns the normalized weight of the given value.
	 * 
	 * Note that the weight of the most important value (measured by it's frequency) is 1,
	 * and the others values' weights are scaled relatively to their frequency.
	 */
	@Override
	public double getNormalizedValueWeight(Value value) 
	{
		//We do not offer correct normalized valueWeights when no values have been offered yet,
		//so the correct behavior would be to throw an exception.
		if(!this.isFirstValueOffered())
			throw new IllegalStateException("ValueWeight calculations are not supported when no values have been offered yet.");
		
		ValueDiscrete valueD = (ValueDiscrete)value;
		String discreteValue = valueD.getValue();
		return this.normalizedValueWeightMap.get(discreteValue);
	}
	
	/**
	 * This returns a string representation of the issueEvaluation.
	 * @return The string representation.
	 */
	public String toString()
	{
		String result = super.toString();
		String nl = "\n";
		
		result += "===== ValueFrequencyMap =====" + nl;
		result += "===== NormalizedValueWeightMap =====" + nl;
		result += "##### END #####" + nl;
		
		return result;
	}
}