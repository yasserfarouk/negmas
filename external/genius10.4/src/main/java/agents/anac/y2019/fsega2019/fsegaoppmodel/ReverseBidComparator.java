package agents.anac.y2019.fsega2019.fsegaoppmodel;

import genius.core.Bid;
import genius.core.utility.UtilitySpace;

public class ReverseBidComparator implements java.util.Comparator<Bid>
{
	private UtilitySpace usp;
	
	public ReverseBidComparator(UtilitySpace pUsp)
	{
		usp = pUsp;
	}
	
	public int compare(Bid b1, Bid b2)
	{
		try
		{
			double u1 = usp.getUtility(b1);
			double u2 = usp.getUtility(b2);
			if(u1 > u2)
				return -1; // ! is reversed
			else if(u1 < u2)
				return 1;
			else
				return 0;
		}
		catch(Exception e)
		{ return -1; }
	}
}
