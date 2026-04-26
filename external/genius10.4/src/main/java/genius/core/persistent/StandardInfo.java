package genius.core.persistent;

import java.io.Serializable;
import java.util.List;
import java.util.Map;

import genius.core.Bid;
import genius.core.Deadline;
import genius.core.actions.ActionWithBid;
import genius.core.list.Tuple;

/**
 * Contains info about a previous session. Contains agents and profile played
 * against, list of received actors that proposed bids and their utilities.
 * Immutable. All components are also immutable. StandardInfo is updated for you
 * by the Genius system.
 */
public interface StandardInfo extends Serializable {
	/**
	 * 
	 * @return immutable map containing a set of pairs [agent, profile]. The
	 *         order is as provided to the protocol.
	 */
	Map<String, String> getAgentProfiles();

	/**
	 * 
	 * @return the starting agent: the agent that did the first action. With
	 *         most protocols, this will be the same as the agent in the first
	 *         tuple in {@link #getUtilities()}
	 */
	String getStartingAgent();

	/**
	 * @return tuples [agent, utility] for all the {@link ActionWithBid}s that
	 *         were done, in that order (0 being the first action with a bid,
	 *         etc). Agent is the party that did the action. Utility is the
	 *         non-discounted utility of his offer. All utilities are in the
	 *         utility space of the agent receiving this info.
	 */
	List<Tuple<String, Double>> getUtilities();

	/**
	 * 
	 * @return the deadline used for the session.
	 */
	Deadline getDeadline();

	/**
	 * get the agreement. Returns null if no agreement was reached.
	 */
	Tuple<Bid, Double> getAgreement();

}
