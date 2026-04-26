package genius.core.session;

import java.util.List;

import genius.core.actions.Action;
import genius.core.parties.NegotiationParty;

/**
 * Error that will be thrown when an action that is not valid for the given
 * round. At this moment (September 2014) The
 * {@link genius.core.session.SessionManager} detects these invalid actions and
 * throws this error.
 */
@SuppressWarnings("serial")
public class InvalidActionError extends ActionException {
	/**
	 * Holds the party that did an invalid action
	 */
	private final NegotiationParty instigator;
	private List<Class<? extends Action>> allowed;
	private Action found;

	/**
	 * Initializes a new instance of the {@link InvalidActionError} class.
	 *
	 * @param instigator
	 *            The party that did an invalid action.
	 * @param allowed
	 *            the list of allowed actions
	 * @param found
	 *            the actual taken action
	 */
	public InvalidActionError(NegotiationParty instigator,
			List<Class<? extends Action>> allowed, Action found) {
		this.instigator = instigator;
		this.allowed = allowed;
		this.found = found;
	}

	/**
	 * Gets the party that did an invalid action
	 *
	 * @return The party that did an invalid action.
	 */
	@SuppressWarnings("UnusedDeclaration")
	// might be used in future
	public NegotiationParty getInstigator() {
		return instigator;
	}

	public String toString() {
		return "Invalid action by " + instigator + ". Expected one from "
				+ allowed + " but actual action was " + found;
	}
}
