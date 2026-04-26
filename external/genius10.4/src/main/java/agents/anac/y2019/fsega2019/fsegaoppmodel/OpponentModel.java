package agents.anac.y2019.fsega2019.fsegaoppmodel;

import java.util.ArrayList;

import genius.core.Bid;
import genius.core.Domain;


public abstract class OpponentModel
{
	protected Domain dDomain;
    protected Domain fDomain;
    public ArrayList<Bid> fBiddingHistory;

	public Domain getDomain()
	{
		return dDomain;
	}

	public abstract double getExpectedUtility(Bid pBid) throws Exception;
	
	public abstract void updateBeliefs(Bid pBid) throws Exception;
	
	public abstract double getExpectedWeight(int pIssueNumber);
}
