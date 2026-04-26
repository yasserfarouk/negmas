package agents.anac.y2019.fsega2019.fsegaoppmodel;

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

    @Override
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
