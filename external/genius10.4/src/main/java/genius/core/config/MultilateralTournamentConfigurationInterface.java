package genius.core.config;

import java.util.List;

import genius.core.Deadline;
import genius.core.persistent.PersistentDataType;
import genius.core.repository.MultiPartyProtocolRepItem;
import genius.core.repository.ParticipantRepItem;
import genius.core.repository.PartyRepItem;
import genius.core.repository.ProfileRepItem;
import genius.core.repository.RepItem;

/**
 * Stores the configuration variables
 *
 * <p>
 * This object must be serializable so that the configuration can be read from
 * file. Therefore should not contain complex runtime objects but rather
 * {@link RepItem}s. Implementations should be immutable.
 * 
 */
public interface MultilateralTournamentConfigurationInterface {

	/**
	 * Gets the protocol to run
	 *
	 * @return the protocol to run
	 */
	MultiPartyProtocolRepItem getProtocolItem();

	/**
	 * @return Deadline for all sessions.
	 */
	Deadline getDeadline();

	/**
	 * Gets the number of times to run the tournament.
	 *
	 * @return the number of tournaments
	 */
	int getRepeats();

	/**
	 * @return The mediator in use. ignored if protocol does not use mediator.
	 *         Must be non-null if protocol needs mediator.
	 */
	PartyRepItem getMediator();

	/**
	 * read-only list of party repository items. This may include mediator(s)
	 *
	 * @return a list of all chosen parties. Only a sub-set of the non-mediators
	 *         will be used in each session, see also
	 *         {@link #getNumPartiesPerSession()}
	 */
	List<ParticipantRepItem> getPartyItems();

	/**
	 * Gets read-only list of profiles used by the parties. These protocols are
	 * used to generate the sessions. The number of available items determines
	 * the maximum number of parties in one session.
	 *
	 * @return list of profiles used by the parties
	 */
	List<ProfileRepItem> getProfileItems();

	/**
	 * @return a pool of parties for side B . This is only used if not empty and
	 *         if {@link #getNumPartiesPerSession()} =2.
	 */
	List<ParticipantRepItem> getPartyBItems();

	/**
	 * 
	 * @return a pool of profiles for side B. Must be non empty if
	 *         {@link #getPartyBItems()} is not empty.
	 */
	List<ProfileRepItem> getProfileBItems();

	/**
	 *
	 * @return the number of parties (excluding mediators) per session. This can
	 *         be smaller than size of {@link #getPartyItems()}, then subsets of
	 *         the party items will be used in each session.
	 */
	int getNumPartiesPerSession();

	/**
	 * Gets whether repetition is allowed when generating combinations of
	 * agents.
	 * 
	 * @return true if allowed
	 */
	boolean isRepetitionAllowed();

	/**
	 * 
	 * @return true if the sessions inside a tournament must be randomized.
	 */
	boolean isRandomSessionOrder();

	/**
	 * 
	 * @return the persistent data setting for this tournament.
	 */
	PersistentDataType getPersistentDataType();

	/**
	 * @return true iff print (using System.out.println) is enabled. If not, it
	 *         is suppressed.
	 */
	boolean isPrintEnabled();
}
