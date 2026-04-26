package genius.core.actions;

import genius.core.Bid;

/**
 * Interface for actions that involve a {@link Bid}
 * 
 * @author W.Pasman
 *
 */
public interface ActionWithBid extends Action {
	/**
	 * Returns the bid that is involved with this action.
	 * 
	 * @return the involved Bid.
	 */

	public Bid getBid();
}
