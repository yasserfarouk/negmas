package agents.anac.y2010.AgentSmith;

import genius.core.Bid;

/*
 * Interface for the OpponentModel
 */
public interface IOpponentModel {

	public double getUtility(Bid pBid);
	public void addBid(Bid pBid);
	
}
