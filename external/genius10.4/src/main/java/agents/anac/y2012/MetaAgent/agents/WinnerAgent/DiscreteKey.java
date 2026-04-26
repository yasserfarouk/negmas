package agents.anac.y2012.MetaAgent.agents.WinnerAgent;


public class DiscreteKey extends Key {
	String key;
	
	public DiscreteKey(String k)
	{
		key=k;
	}
	
	@Override
	public boolean equals(Object obj) 
	{
		DiscreteKey k = (DiscreteKey)obj;
		return this.key.equals(k.key);
	}
	 
	@Override
	public int hashCode()
	{
		return this.key.hashCode();
	}

	public String toString()
	{
		return key;
	}

	@Override
	public boolean contains(Object obj) 
	{
		return key.equals(obj);
	}
}
