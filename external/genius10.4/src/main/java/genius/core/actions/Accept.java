package genius.core.actions;

import genius.core.AgentID;
import genius.core.Bid;

/**
 * This class is used to createFrom an action which symbolizes that an agent
 * accepts an offer. Immutable.
 * 
 * @author Dmytro Tykhonov
 */
public class Accept extends DefaultActionWithBid {

	/**
	 * @param agentID
	 *            id of the agent which creates this accept.
	 * @param bid
	 *            the accepted bid.
	 */
	public Accept(AgentID agentID, Bid bid) {
		super(agentID, bid);
	}

	/**
	 * @return string representation of action: "(Accept)".
	 */
	public String toString() {
		return "(Accept " + getContent() + ")";
	}
}