package agents.anac.y2013.MetaAgent.portfolio.thenegotiatorreloaded;

import java.util.List;

import genius.core.Bid;
import genius.core.Domain;
import genius.core.issue.Issue;
import genius.core.timeline.TimeLineInfo;
import genius.core.utility.AbstractUtilitySpace;
import genius.core.utility.AdditiveUtilitySpace;

/**
 * This is an abstract class which manages all the negotiation Session pertinent
 * information to a single agent
 * 
 * @author Alex Dirkzwager
 */
public class NegotiationSession {

	/** Optional outcomespace which should be set manually */
	protected OutcomeSpace outcomeSpace;
	/** History of bids made by the opponent */
	protected BidHistory opponentBidHistory;
	/** History of bids made by the agent */
	protected BidHistory ownBidHistory;
	/** Reference to the negotiation domain */
	protected Domain domain;
	/** Reference to the agent's preference profile for the domain */
	protected AbstractUtilitySpace utilitySpace;
	/** Reference to the timeline */
	protected TimeLineInfo timeline;

	public NegotiationSession(AbstractUtilitySpace utilitySpace, TimeLineInfo timeline) {
		this.utilitySpace = utilitySpace;
		this.timeline = timeline;
		this.domain = utilitySpace.getDomain();
		this.opponentBidHistory = new BidHistory();
		this.ownBidHistory = new BidHistory();
	}

	/**
	 * Returns a list of bids offered by the opponent.
	 * 
	 * @return a list of of opponent bids
	 */
	public BidHistory getOpponentBidHistory() {
		return opponentBidHistory;
	}

	public BidHistory getOwnBidHistory() {
		return ownBidHistory;
	}

	public double getDiscountFactor() {
		return utilitySpace.getDiscountFactor();
	}

	public List<Issue> getIssues() {
		return domain.getIssues();
	}

	public TimeLineInfo getTimeline() {
		return timeline;
	}

	/**
	 * gets the normalized time (t = [0,1])
	 * 
	 * @return time normalized
	 */
	public double getTime() {
		return timeline.getTime();
	}

	public AdditiveUtilitySpace getUtilitySpace() {
		return (AdditiveUtilitySpace) utilitySpace;
	}

	public OutcomeSpace getOutcomeSpace() {
		return outcomeSpace;
	}

	public void setOutcomeSpace(OutcomeSpace space) {
		this.outcomeSpace = space;
	}

	/**
	 * Returns the best bid in the domain.
	 */
	public BidDetails getMaxBidinDomain() {
		BidDetails maxBid = null;
		if (outcomeSpace == null) {
			try {
				Bid maximumBid = utilitySpace.getMaxUtilityBid();
				maxBid = new BidDetails(maximumBid, utilitySpace.getUtility(maximumBid));
			} catch (Exception e) {
				e.printStackTrace();
			}
		} else {
			maxBid = outcomeSpace.getMaxBidPossible();
		}
		return maxBid;
	}
}