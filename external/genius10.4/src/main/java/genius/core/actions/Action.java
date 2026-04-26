package genius.core.actions;

import java.io.Serializable;

import genius.core.AgentID;

/**
 * Interface for actions that are taken by an Agent and part of a negotiation.
 * All actions must be immutable.
 * 
 */
public interface Action extends Serializable {
	/**
	 * Returns the ID of the agent which created the action.
	 * 
	 * @return ID of the agent. Only actions returned from the protocol like
	 *         {@link Inform} can return the null ID.
	 */
	AgentID getAgent();

}