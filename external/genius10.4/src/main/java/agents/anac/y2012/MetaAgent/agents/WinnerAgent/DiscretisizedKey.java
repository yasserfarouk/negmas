package agents.anac.y2012.MetaAgent.agents.WinnerAgent;


public class DiscretisizedKey extends Key{
	// the range of the value
	private double min;
	private double max;

	public DiscretisizedKey(double mn, double mx)
	{
		min=mn;
		max=mx;
	}
	
	public boolean isInRange(double val)
	{
		if	(val>=min && val<=max)
			return true;
		else
			return false;
	}

	public double getMin() 
	{
		return min;
	}

	public void setMin(double min) 
	{
		this.min = min;
	}

	public double getMax() 
	{
		return max;
	}

	public void setMax(double max) 
	{
		this.max = max;
	}
	 
	@Override
	public boolean equals(Object obj) 
	{
		DiscretisizedKey k = (DiscretisizedKey)obj;
		if(this.min == k.min && this.max==k.max)
			return true;
		return false;
	}
	 
	@Override
	public int hashCode()
	{
		String s = Double.toString(min)+Double.toString(max);
		return s.hashCode();
	}
	 
	public String toString()
	{
		String s = min+"-"+max;
		return s;
	}

	@Override
	public boolean contains(Object obj) 
	{
		return obj instanceof Number && isInRange((Double)obj);
	}
}
