package agents.anac.y2010.AgentSmith;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;

import genius.core.AgentID;
import genius.core.Bid;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.Offer;
import genius.core.issue.Issue;
import genius.core.issue.Value;
import genius.core.utility.AdditiveUtilitySpace;

/**
 * The strategy that is used in our agent. It samples the bid space, takes the
 * bids that are near Pareto and returns them as next actions.
 * 
 * NOTE There are two bugs in this strategy: 1. The agent incorrectly assumes
 * that there are only 2 minutes 2. The agent checks if it gave "enough" utility
 * to the opponent instead of itself
 */
public class SmithBidStrategy extends ABidStrategy {
	private int fIndex = 0;

	static private double sUtilyMargin = 0.7;
	static private double sTimeMargin = 110; // in seconds

	static private double UTILITY_THRESHOLD = 0.7;
	private final boolean TEST_EQUIVALENCE = false;

	/**
	 * Constructor
	 * 
	 * @param pHist
	 *            The bidhistory
	 * @param utilitySpace
	 *            Utilityspace
	 * @param pPreferenceProfile
	 *            Preference profile
	 * @param pId
	 *            Our id
	 */
	public SmithBidStrategy(BidHistory pHist,
			AdditiveUtilitySpace utilitySpace,
			PreferenceProfileManager pPreferenceProfile, AgentID pId) {
		super(pHist, utilitySpace, pPreferenceProfile, pId);
		if (TEST_EQUIVALENCE) {
			sTimeMargin = 170;
		}
	}

	/**
	 * Generate in new action
	 */
	public Action getNextAction(double time) {
		double normalTime = time * 180;
		Action lAction = null;
		// Time in seconds.
		try {
			// Check if the session (2 min) is almost finished
			if (normalTime >= sTimeMargin) {
				// If the session is almost finished check if the utility is
				// "high enough"
				Bid lastBid = fBidHistory.getOpponentLastBid();

				boolean result = false;
				if (TEST_EQUIVALENCE) {
					result = fPreferenceProfile.getMyUtility(lastBid) >= sUtilyMargin;
				} else {
					result = fPreferenceProfile.getOpponentUtility(lastBid) >= sUtilyMargin;
				}
				if (result) {
					lAction = new Accept(fAgentID, lastBid);
				} else {
					lAction = new Offer(fAgentID, getBestOpponentOffer());
				}
			} else {
				lAction = new Offer(fAgentID, getMostOptimalBid());
			}
		} catch (Exception e) {
			lAction = null;
		}
		return lAction;
	}

	/**
	 * Calculate the most optimal bid
	 * 
	 * @return the most optimal bid
	 * @throws Exception
	 */
	public Bid getMostOptimalBid() {
		ArrayList<Bid> lBids = getSampledBidList();

		// Log.logger.info("Size of bid space: " + lBids.size());
		BidComparator lComparator = new BidComparator(this.fPreferenceProfile);

		// sort the bids in order of highest utility
		Collections.sort(lBids, lComparator);

		Bid lBid = lBids.get(fIndex);
		if (fIndex < lBids.size() - 1)
			fIndex++;

		return lBid;
	}

	/**
	 * Return the best offer made by your opponent. When time is running out
	 * this method is called to find the best bid available.
	 * 
	 * @return
	 * @throws Exception
	 */
	public Bid getBestOpponentOffer() {
		double util = 0;
		Bid bestBid = null;

		try {
			// check for the highest bid offered by the opponent
			for (int i = 0; i < fBidHistory.getOpponentBidCount(); i++) {
				if (fUtilitySpace.getUtility(fBidHistory.getOpponentBid(i)) > util) {
					util = fUtilitySpace.getUtility(fBidHistory
							.getOpponentBid(i));
					bestBid = fBidHistory.getOpponentBid(i);
				}
			}
		} catch (Exception e) {
			bestBid = null;
		}

		return bestBid;
	}

	/**
	 * returns an ArrayList with possible bids This function constructs all
	 * samples of the entire bidspace.
	 */
	private ArrayList<Bid> getSampledBidList() {
		ArrayList<Bid> lBids = new ArrayList<Bid>();
		List<Issue> lIssues = this.fPreferenceProfile.getIssues();
		HashMap<Integer, Bounds> lBounds = Bounds.getIssueBounds(lIssues);

		// first createFrom a new list
		HashMap<Integer, Value> lBidValues = new HashMap<Integer, Value>();
		for (Issue lIssue : lIssues) {
			Bounds b = lBounds.get(lIssue.getNumber());
			Value v = Bounds.getIssueValue(lIssue, b.getLower());
			lBidValues.put(lIssue.getNumber(), v);
		}
		try {
			lBids.add(new Bid(this.fPreferenceProfile.getDomain(), lBidValues));
		} catch (Exception e) {
		}

		// for each item permutate with issue values, like binary
		// 0 0 0
		// 0 0 1
		// 0 1 0
		// 0 1 1
		// etc.
		for (Issue lIssue : lIssues) {
			ArrayList<Bid> lTempBids = new ArrayList<Bid>();
			Bounds b = lBounds.get(lIssue.getNumber());
			for (Bid lTBid : lBids) {
				for (double i = b.getLower(); i < b.getUpper(); i += b
						.getStepSize()) {
					HashMap<Integer, Value> lNewBidValues = getBidValues(lTBid);
					lNewBidValues.put(lIssue.getNumber(),
							Bounds.getIssueValue(lIssue, i));

					try {
						Bid iBid = new Bid(this.fPreferenceProfile.getDomain(),
								lNewBidValues);
						lTempBids.add(iBid);

					} catch (Exception e) {

					}
				}
			}
			lBids = lTempBids;
		}

		ArrayList<Bid> lToDestroy = new ArrayList<Bid>();
		for (Bid lBid : lBids) {

			if (this.fPreferenceProfile.getMyUtility(lBid) < UTILITY_THRESHOLD) {
				lToDestroy.add(lBid);
			}
		}
		for (Bid lBid : lToDestroy) {
			lBids.remove(lBid);
		}

		return lBids;
	}

	/**
	 * Get the values of a bid
	 * 
	 * @param pBid
	 * @return
	 */
	private HashMap<Integer, Value> getBidValues(Bid pBid) {
		HashMap<Integer, Value> lNewBidValues = new HashMap<Integer, Value>();
		for (Issue lIssue : this.fPreferenceProfile.getIssues()) {
			try {
				lNewBidValues.put(lIssue.getNumber(),
						pBid.getValue(lIssue.getNumber()));
			} catch (Exception e) {

			}
		}
		return lNewBidValues;
	}

}
