package agents.anac.y2013.MetaAgent.portfolio.AgentLG;
import java.util.HashMap;

import genius.core.issue.Issue;
import genius.core.issue.Value;

/**
 * Class that is used to learn opponent utility.
 */
public class BidStatistic {

	private HashMap<Value,Integer> valStatis = new HashMap<Value,Integer>();
	private Issue issue= null;
	private double numVotes =0;
	
	public BidStatistic(Issue issue) {
		super();
		this.issue = issue;
		
		
	}
	
	
	public void add(Value v)
	{
		if(valStatis.get(v)== null)
			valStatis.put(v,1);
		else
			valStatis.put(v,valStatis.get(v)+1);
		numVotes++;
	}
	
	public Value getMostBided()
	{
		Value maxval = null;
		Integer maxtimes=0;
		for(Value val:valStatis.keySet())
		{
			if (valStatis.get(val)>maxtimes)
			{
				maxtimes=valStatis.get(val);
				maxval=val;
			}
		}
		return maxval;
	}
	
	public int getMostVotedCount()
	{
		Integer maxtimes=0;
		for(Value val:valStatis.keySet())
		{
			if (valStatis.get(val)>maxtimes)
			{
				maxtimes=valStatis.get(val);
			}
		}
		return maxtimes;
	}
	
	public double getValueUtility(Value value)
	{
		double ret = 0;
		if (valStatis.get(value)!= null)
			ret = ((double)valStatis.get(value))/getMostVotedCount();
		return ret;
	}
	
}
