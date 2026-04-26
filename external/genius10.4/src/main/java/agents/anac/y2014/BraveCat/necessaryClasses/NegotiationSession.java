package agents.anac.y2014.BraveCat.necessaryClasses;

import java.io.Serializable;
import java.util.List;

import genius.core.Bid;
import genius.core.Domain;
import genius.core.bidding.BidDetails;
import genius.core.boaframework.BoaType;
import genius.core.boaframework.SessionData;
import genius.core.issue.Issue;
import genius.core.timeline.TimeLineInfo;
import genius.core.utility.AbstractUtilitySpace;

public class NegotiationSession {
	protected OutcomeSpace outcomeSpace;
	protected BidHistory opponentBidHistory;
	protected BidHistory ownBidHistory;
	protected Domain domain;
	protected AbstractUtilitySpace utilitySpace;
	protected TimeLineInfo timeline;
	private SessionData sessionData;

	protected NegotiationSession() {
	}

	public NegotiationSession(SessionData sessionData, AbstractUtilitySpace utilitySpace, TimeLineInfo timeline) {
		this(sessionData, utilitySpace, timeline, null);
	}

	public NegotiationSession(SessionData sessionData, AbstractUtilitySpace utilitySpace, TimeLineInfo timeline,
			OutcomeSpace outcomeSpace) {
		this.sessionData = sessionData;
		this.utilitySpace = utilitySpace;
		this.timeline = timeline;
		this.domain = utilitySpace.getDomain();
		this.opponentBidHistory = new BidHistory();
		this.ownBidHistory = new BidHistory();
		this.outcomeSpace = outcomeSpace;
		sessionData = new SessionData();
	}

	public BidHistory getOpponentBidHistory() {
		return this.opponentBidHistory;
	}

	public BidHistory getOwnBidHistory() {
		return this.ownBidHistory;
	}

	public double getDiscountFactor() {
		return this.utilitySpace.getDiscountFactor();
	}

	public List<Issue> getIssues() {
		return this.domain.getIssues();
	}

	public TimeLineInfo getTimeline() {
		return this.timeline;
	}

	public double getTime() {
		return this.timeline.getTime();
	}

	public Domain getDomain() {
		if (this.utilitySpace != null) {
			return this.utilitySpace.getDomain();
		}
		return null;
	}

	public AbstractUtilitySpace getUtilitySpace() {
		return this.utilitySpace;
	}

	public OutcomeSpace getOutcomeSpace() {
		return this.outcomeSpace;
	}

	public void setOutcomeSpace(OutcomeSpace outcomeSpace) {
		this.outcomeSpace = outcomeSpace;
	}

	public BidDetails getMaxBidinDomain() {
		BidDetails maxBid = null;
		if (this.outcomeSpace == null)
			try {
				Bid maximumBid = this.utilitySpace.getMaxUtilityBid();
				maxBid = new BidDetails(maximumBid, this.utilitySpace.getUtility(maximumBid), -1.0D);
			} catch (Exception e) {
				e.printStackTrace();
			}
		else {
			maxBid = this.outcomeSpace.getMaxBidPossible();
		}
		return maxBid;
	}

	public BidDetails getMinBidinDomain() {
		BidDetails minBid = null;
		if (this.outcomeSpace == null)
			try {
				Bid minimumBidBid = this.utilitySpace.getMinUtilityBid();
				minBid = new BidDetails(minimumBidBid, this.utilitySpace.getUtility(minimumBidBid), -1.0D);
			} catch (Exception e) {
				e.printStackTrace();
			}
		else {
			minBid = this.outcomeSpace.getMinBidPossible();
		}
		return minBid;
	}

	public void setData(BoaType component, Serializable data) {
		this.sessionData.setData(component, data);
	}

	public Serializable getData(BoaType component) {
		return this.sessionData.getData(component);
	}

	public double getDiscountedUtility(Bid bid, double time) {
		return this.utilitySpace.getUtilityWithDiscount(bid, time);
	}

	public SessionData getSessionData() {
		return this.sessionData;
	}
}