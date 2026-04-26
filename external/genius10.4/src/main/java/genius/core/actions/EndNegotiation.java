package genius.core.actions;

import genius.core.AgentID;

/**
 * Class which symbolizes the action to leave a negotiation. Immutable.
 * 
 * @author Dmytro Tykhonov
 */
public class EndNegotiation extends DefaultAction {

	/**
	 * Action to end the negotiation.
	 * 
	 * @param agentID
	 *            of the opponent
	 */
	public EndNegotiation(AgentID agentID) {
		super(agentID);
	}

	/**
	 * @return string representation of action: "(EndNegotiation)".
	 */
	public String toString() {
		return "(EndNegotiation)";
	}
}