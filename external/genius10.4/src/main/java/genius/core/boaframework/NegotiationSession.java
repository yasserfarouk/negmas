package genius.core.boaframework;

import java.util.List;

import java.io.Serializable;

import genius.core.Bid;
import genius.core.BidHistory;
import genius.core.Domain;
import genius.core.bidding.BidDetails;
import genius.core.issue.Issue;
import genius.core.timeline.TimeLineInfo;
import genius.core.uncertainty.User;
import genius.core.uncertainty.UserModel;
import genius.core.utility.AbstractUtilitySpace;

/**
 * This is a class which manages all the negotiation session pertinent
 * information to a single agent. This class is designed for bilateral
 * negotiations.
 * 
 */
public class NegotiationSession 
{
	/** Optional outcomespace which should be set manually. */
	protected OutcomeSpace outcomeSpace;
	/** History of bids made by the opponent. */
	protected final BidHistory opponentBidHistory;
	/** History of bids made by the agent. */
	protected final BidHistory ownBidHistory;
	/** If not null, this overrides utilitySpace. */
	private final UserModel userModel;
	/** Reference to the agent's preference profile for the domain. */
	protected final AbstractUtilitySpace utilitySpace;
	/**Reference to the agent's user; */
	protected final User user;
	/** Reference to the timeline. */
	protected final TimeLineInfo timeline;
	private final SessionData sessionData;

	/**
	 * Create a negotiation session which is used to keep track of the
	 * negotiation state.
	 * 
	 * @param utilitySpace
	 *            of the agent. May be overridden by outcomeSpace
	 * @param timeline
	 *            of the current negotiation.
	 * @param outcomeSpace
	 *            representation of the possible outcomes. Can be null.
	 * @param userModel
	 *            the usermodel. Can be null. If not null, this overrides
	 *            utilitySpace.
	 */
	public NegotiationSession(SessionData sessionData,
			AbstractUtilitySpace utilitySpace, TimeLineInfo timeline,
			OutcomeSpace outcomeSpace, UserModel userModel, User user) {
		// this.sessionData = sessionData;
		this.utilitySpace = utilitySpace;
		this.timeline = timeline;
		this.opponentBidHistory = new BidHistory();
		this.ownBidHistory = new BidHistory();
		this.outcomeSpace = outcomeSpace;
		this.sessionData = new SessionData();
		this.userModel = userModel;
		this.user = user;
	
	}

	/**
	 * Returns the bidding history of the opponent.
	 * 
	 * @return bidding history of the opponent.
	 */
	public BidHistory getOpponentBidHistory() {
		return opponentBidHistory;
	}

	/**
	 * Returns the bidding history of the agent.
	 * 
	 * @return bidding history of the agent.
	 */
	public BidHistory getOwnBidHistory() {
		return ownBidHistory;
	}

	/**
	 * Returns the discount factor of the utilityspace. Each utilityspace has a
	 * unique discount factor.
	 * 
	 * @return discount factor of the utilityspace.
	 */
	public double getDiscountFactor() {
		return utilitySpace.getDiscountFactor();
	}

	/**
	 * @return issues of the domain.
	 */
	public List<Issue> getIssues() {
		return utilitySpace.getDomain().getIssues();
	}

	/**
	 * @return timeline of the negotiation.
	 */
	public TimeLineInfo getTimeline() {
		return timeline;
	}

	/**
	 * Returns the normalized time (t = [0,1])
	 * 
	 * @return normalized time.
	 */
	public double getTime() {
		return timeline.getTime();
	}

	/**
	 * Returns the negotiation domain.
	 * 
	 * @return domain of the negotiation.
	 */
	public Domain getDomain() {
		if (utilitySpace != null) {
			return utilitySpace.getDomain();
		}
		return null;
	}

	/**
	 * Returns the utilityspace of the agent.
	 * 
	 * @return utilityspace of the agent.
	 */
	public AbstractUtilitySpace getUtilitySpace() {
		return utilitySpace;
	}

	
	/**
	 * Returns the space of possible outcomes in the domain. The returned value
	 * may be null.
	 * 
	 * @return outcomespace if available.
	 */
	public OutcomeSpace getOutcomeSpace() {
		return outcomeSpace;
	}

	/**
	 * Method used to set the outcomespace. Setting an outcomespace makes method
	 * such as getMaxBidinDomain much more efficient.
	 * 
	 * @param outcomeSpace
	 *            to be set.
	 */
	public void setOutcomeSpace(OutcomeSpace outcomeSpace) {
		this.outcomeSpace = outcomeSpace;
	}

	/**
	 * Returns the best bid in the domain. If the outcomespace is set, it is
	 * used in this step. Else a highly inefficient method is used.
	 * 
	 * @return bid with highest possible utility.
	 */
	public BidDetails getMaxBidinDomain() {
		BidDetails maxBid = null;
		if (outcomeSpace == null) {
			try {
				Bid maximumBid = utilitySpace.getMaxUtilityBid();
				maxBid = new BidDetails(maximumBid,
						utilitySpace.getUtility(maximumBid), -1);
			} catch (Exception e) {
				e.printStackTrace();
			}
		} else {
			maxBid = outcomeSpace.getMaxBidPossible();
		}
		return maxBid;
	}

	/**
	 * Returns the worst bid in the domain. If the outcomespace is set, it is
	 * used in this step. Else a highly inefficient method is used.
	 * 
	 * @return bid with lowest possible utility.
	 */
	public BidDetails getMinBidinDomain() {
		BidDetails minBid = null;
		if (outcomeSpace == null) {
			try {
				Bid minimumBidBid = utilitySpace.getMinUtilityBid();
				minBid = new BidDetails(minimumBidBid,
						utilitySpace.getUtility(minimumBidBid), -1);
			} catch (Exception e) {
				e.printStackTrace();
			}
		} else {
			minBid = outcomeSpace.getMinBidPossible();
		}
		return minBid;
	}

	/**
	 * Method used o store the data of a component. For agent programming please
	 * use the storeData() method of the BOA component.
	 * 
	 * @param component
	 *            from which the data is stored.
	 * @param data
	 *            to be stored.
	 */
	public void setData(BoaType component, Serializable data) {
		sessionData.setData(component, data);
	}

	/**
	 * Method used to load the data saved by a component. For agent programming
	 * please use the loadData() method of the BOA component.
	 * 
	 * @param component
	 *            from which the data is requested.
	 * @return data saved by the component.
	 */
	public Serializable getData(BoaType component) {
		return sessionData.getData(component);
	}

	/**
	 * Returns the discounted utility of a bid given the bid and the time at
	 * which it was offered.
	 * 
	 * @param bid
	 *            which discount utility is requested.
	 * @param time
	 *            at which the bid was offered.
	 * @return discounted utility of the given bid at the given time.
	 */
	public double getDiscountedUtility(Bid bid, double time) {
		return utilitySpace.getUtilityWithDiscount(bid, time);
	}

	public SessionData getSessionData() {
		return sessionData;
	}

	/**
	 * @return A partial profile. If not null, this supersedes the value
	 *         returned by {@link #getUtilitySpace()}.
	 */
	public UserModel getUserModel() {
		return userModel;
	}
	
	/**
	 * @return the agent's user
	 */
	public User getUser(){
		return user;
	}
}