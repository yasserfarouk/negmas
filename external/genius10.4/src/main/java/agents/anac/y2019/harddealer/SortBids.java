package agents.anac.y2019.harddealer;

import java.util.Comparator;

import genius.core.bidding.BidDetails;

public class SortBids extends HardDealer_OMS implements Comparator<BidDetails> {
	public int compare(BidDetails a, BidDetails b)
	{
		if (a.getMyUndiscountedUtil() - b.getMyUndiscountedUtil() < 0)
		{
			return -1;
		} else if (a.getMyUndiscountedUtil() == b.getMyUndiscountedUtil())
		{
			return 0;
		}
		else
		{
			return 1;
		}

	}
}
