package agents.anac.y2011.Nice_Tit_for_Tat;

import genius.core.Bid;

public interface BidHistoryKeeper
{
	public BidHistory getOpponentHistory();

	public Bid getMyLastBid();

	public Bid getMySecondLastBid();

	public Bid getOpponentLastBid();
}
