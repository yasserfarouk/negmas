package negotiator.boaframework.opponentmodel.agentlg;
import java.util.HashMap;

import genius.core.issue.Issue;
import genius.core.issue.Value;

/**
 * Class that is used by the opponent model of the ANAC2012 AgentLG.
 */
public class BidStatistic {

	private HashMap<Value,Integer> valStatis = new HashMap<Value,Integer>();
	
	public BidStatistic(Issue issue) {
		super();
	}

	public void add(Value v)
	{
		if(valStatis.get(v)== null) {
			valStatis.put(v,1);
		} else {
			valStatis.put(v,valStatis.get(v) + 1);
		}
	}
	
	public int getMostVotedCount()
	{
		Integer maxtimes=0;
		for (Value val:valStatis.keySet())
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
			ret = ((double)valStatis.get(value)) / ((double)getMostVotedCount());
		return ret;
	}
	
}
