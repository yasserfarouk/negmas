package genius.core.actions;

import genius.core.AgentID;
import genius.core.Bid;

/**
 * Symbolizes an offer of an agent for the opponent. Immutable.
 * 
 * @author Tim Baarslag and Dmytro Tykhonov
 */
public class Offer extends DefaultActionWithBid {

	public Offer(AgentID agentID, Bid bid) {
		super(agentID, bid);
	}

	/**
	 * @return string representation of action
	 */
	public String toString() {
		return "(Offer " + getContent() + ")";
	}
}