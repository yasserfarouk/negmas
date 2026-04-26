package genius.core.actions;

import genius.core.AgentID;
import genius.core.Bid;

/**
 * This class is used to createFrom an action which symbolizes that an agent
 * rejects an offer. immutable.
 * 
 * @author Dmytro Tykhonov
 */
public class Reject extends DefaultActionWithBid {

	/**
	 * Action to accept an opponent's bid.
	 * 
	 * @param agentID
	 *            ID of agent rejecting the bid.
	 * @param bid
	 *            the Bid that is rejected
	 */
	public Reject(AgentID agentID, Bid bid) {
		super(agentID, bid);
	}

	/**
	 * @return string representation of action: "(Reject)".
	 */
	public String toString() {
		return "(Reject " + getContent() + ")";
	}
}