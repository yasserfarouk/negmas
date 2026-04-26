package agents.anac.y2010.AgentFSEGA;

import genius.core.Domain;

public class WeightHypothesis extends Hypothesis
{
	double dWeight[];

	public WeightHypothesis(Domain pDomain)
	{
		dWeight = new double[pDomain.getIssues().size()];
	}

	public void setWeight(int index, double value)
	{
		dWeight[index] = value;
	}

	public double getWeight(int index)
	{
		return dWeight[index];
	}

	public String toString()
	{
		String lResult = "WeightHypothesis[";
		for(double iWeight : dWeight)
		{
			lResult += String.format("%1.2f; ", iWeight);
		}
		lResult += "]";
		
		return lResult;
	}
}
