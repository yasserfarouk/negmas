package agents.anac.y2010.AgentSmith;

import genius.core.Bid;


/*
 * interface for the bidhistory
 */
public interface IBidHistoryListener {

	public void myBidAdded(BidHistory vHistory, Bid pBid);
	
	public void opponentBidAdded(BidHistory vHistory, Bid pBid);
	
}
