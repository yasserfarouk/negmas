package agents.anac.y2013.MetaAgent.portfolio.AgentLG;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import genius.core.Bid;
import genius.core.Domain;
import genius.core.issue.Issue;
import genius.core.issue.Value;
import genius.core.utility.UtilitySpace;

/**
 * Class that is used to save opponents bid and learn opponent utility
 *
 */
public class OpponentBids {

	private ArrayList<Bid> oppBids = new ArrayList<Bid>();
	private HashMap<Issue, BidStatistic> statistic = new HashMap<Issue, BidStatistic>();
	private Bid maxUtilityBidForMe = null;
	private UtilitySpace utilitySpace;

	/**
	 * add opponent bid and updates statistics
	 *
	 */
	public void addBid(Bid bid) {
		oppBids.add(bid);
		try {
			// updates statistics
			for (Issue issue : statistic.keySet()) {
				Value v = bid.getValue(issue.getNumber());
				statistic.get(issue).add(v);
			}

			// receiveMessage the max bid for the agent from the opponent bids
			if (oppBids.size() == 1)
				maxUtilityBidForMe = bid;
			else if (utilitySpace.getUtility(maxUtilityBidForMe) < utilitySpace
					.getUtility(bid))
				maxUtilityBidForMe = bid;
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	/**
	 * return opponents Bids
	 *
	 */
	public ArrayList<Bid> getOpponentsBids() {
		return oppBids;
	}

	public OpponentBids(UtilitySpace utilitySpace) {
		this.utilitySpace = utilitySpace;
		List<Issue> issues = utilitySpace.getDomain().getIssues();
		for (Issue issue : issues) {
			statistic.put(issue, new BidStatistic(issue));
		}
	}

	/**
	 * returns opponents Bids
	 *
	 */
	public Bid getMaxUtilityBidForMe() {
		return maxUtilityBidForMe;
	}

	/**
	 * returns the most voted value for an isuue
	 *
	 */
	public Value getMostVotedValueForIsuue(Issue issue) {
		return statistic.get(issue).getMostBided();
	}

	/**
	 * returns opponent bid utility that calculated from the vote statistics.
	 *
	 */
	public double getOpponentBidUtility(Domain domain, Bid bid) {
		double ret = 0;
		List<Issue> issues = domain.getIssues();
		for (Issue issue : issues) {
			try {
				ret += statistic.get(issue).getValueUtility(
						bid.getValue(issue.getNumber()));
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
		System.out.println(ret / domain.getIssues().size());
		return ret;
	}
}
