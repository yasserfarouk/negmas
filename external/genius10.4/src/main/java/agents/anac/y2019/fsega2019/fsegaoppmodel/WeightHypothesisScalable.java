package agents.anac.y2019.fsega2019.fsegaoppmodel;

public class WeightHypothesisScalable extends Hypothesis
{
	double fWeight;

	public WeightHypothesisScalable()
    {  }

	public void setWeight( double value)
    {
		fWeight = value;
	}
	
    public double getWeight()
    {
		return fWeight;
	}

    @Override
	public String toString()
    {
		String lResult = "";
			lResult = String.format("%1.2f", fWeight) +";";
		return lResult;
	}
}
