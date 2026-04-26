package agents.anac.y2019.harddealer;
import genius.core.bidding.BidDetails;
import genius.core.boaframework.OpponentModel;

import java.util.*;

public class SortBidsOpponent extends HardDealer_OMS implements Comparator<BidDetails> {
	OpponentModel om;
	public SortBidsOpponent(OpponentModel Om)
	{
		om = Om;
	}

	public int compare(BidDetails a, BidDetails b)
	{
		if (om.getBidEvaluation(a.getBid()) - om.getBidEvaluation(b.getBid()) < 0)
		{
			return -1;
		} else if (om.getBidEvaluation(a.getBid()) == om.getBidEvaluation(b.getBid()))
		{
			return 0;
		}
		else
		{
			return 1;
		}
	}
}
