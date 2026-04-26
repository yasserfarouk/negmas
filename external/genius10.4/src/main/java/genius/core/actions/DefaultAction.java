package genius.core.actions;

import genius.core.AgentID;
import genius.core.Bid;

/**
 * @author Tim Baarslag and Dmytro Tykhonov
 *
 */
public abstract class DefaultAction implements Action {
	protected AgentID agentID;

	/**
	 * Constructor which sets the agentID of an agent.
	 * 
	 * @param agentID
	 *            of the agent which created the action.
	 */
	public DefaultAction(AgentID agentID) {
		this.agentID = agentID;
		if (agentID == null) {
			throw new IllegalArgumentException("AgentID =null");
		}
	}

	@Override
	public AgentID getAgent() {
		return agentID;
	}

	/**
	 * Enforces that actions implements a string-representation.
	 */
	public abstract String toString();

	/**
	 * Method which returns the bid of the current action if it is of the type
	 * Offer or else Null.
	 * 
	 * @param currentAction
	 *            of which we want the offer.
	 * @return bid specifies by this action or null if there is none.
	 */
	public static Bid getBidFromAction(Action currentAction) {

		Bid currentBid = null;
		if (currentAction instanceof EndNegotiationWithAnOffer) // RA
			currentBid = ((EndNegotiationWithAnOffer) currentAction).getBid();
		else if (currentAction instanceof Offer)
			currentBid = ((Offer) currentAction).getBid();

		return currentBid;
	}

}
