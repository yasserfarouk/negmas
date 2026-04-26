package genius.core.parties;

import genius.core.AgentID;
import genius.core.Bid;
import genius.core.utility.UtilitySpace;

/**
 * Lightweight interface for some party that gives {@link Bid}s a utility.
 *
 */
public interface PartyWithUtility {

	/**
	 * @return the unique ID of this agent.
	 */
	public AgentID getID();

	/**
	 * @return the utility space for this agent.
	 */
	public UtilitySpace getUtilitySpace();

	/**
	 * Note, this does not enforce proper implementation as equals already
	 * exists in Object. This is just to remind implementors.
	 * 
	 * @param obj
	 * @return can depend only on AgentID as these should be unique.
	 */
	public boolean equals(Object obj);

}
