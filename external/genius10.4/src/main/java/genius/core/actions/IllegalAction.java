package genius.core.actions;

import genius.core.AgentID;

/**
 * This action represents that the agent did an illegal action (not fitting the
 * protocol), eg kill his agent. Immutable.
 * 
 * @author W.Pasman 17sept08
 */
public class IllegalAction extends DefaultAction {

	private String details;

	/**
	 * Specifies that an agent returned an action not fitting the protocol.
	 * 
	 * @param agentID
	 *            id of agent to blame.
	 * @param details
	 *            of the error.
	 */
	public IllegalAction(AgentID agentID, String details) {
		super(agentID);
		this.details = details;
	}

	/**
	 * @return string representation of action: "(IllegalAction-DETAILS)".
	 */
	public String toString() {
		return "(IllegalAction- " + details + ")";
	}
}