package agents.anac.y2010.AgentFSEGA;

public abstract class Hypothesis implements Comparable<Hypothesis>
{
	private double dProbability;

	public double getProbability()
	{
		return dProbability;
	}

	public void setProbability(double probability)
	{
		dProbability = probability;
	}
	
	//allows hypothesis compare by utility
	public int compareTo(Hypothesis o)
	{
		if(getProbability() > o.getProbability())
			return 1;
		else if(getProbability() < o.getProbability())
			return -1;
		else
			return 0;
	}
}
