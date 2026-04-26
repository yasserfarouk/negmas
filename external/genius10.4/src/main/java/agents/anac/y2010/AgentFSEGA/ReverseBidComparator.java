package agents.anac.y2010.AgentFSEGA;

import genius.core.Bid;
import genius.core.utility.AdditiveUtilitySpace;

public class ReverseBidComparator implements java.util.Comparator<Bid>
{
	private AdditiveUtilitySpace usp;
	private final boolean TEST_EQUIVALENCE = false;
	
	public ReverseBidComparator(AdditiveUtilitySpace pUsp)
	{
		usp = pUsp;
	}
	
	public int compare(Bid b1, Bid b2)
	{
		try
		{
			double u1 = usp.getUtility(b1);
			double u2 = usp.getUtility(b2);
			if (TEST_EQUIVALENCE) {
				if (u1 == u2) {
					return String.CASE_INSENSITIVE_ORDER.compare(b1.toString(), b2.toString());
				}
			}
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