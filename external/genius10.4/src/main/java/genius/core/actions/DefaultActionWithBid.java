package genius.core.actions;

import genius.core.AgentID;
import genius.core.Bid;

/**
 * Default implementation for ActionWithBid
 * 
 * @author W.Pasman
 *
 */
public abstract class DefaultActionWithBid extends DefaultAction implements
		ActionWithBid {
	/** The involved bid. */
	protected Bid bid;

	/**
	 * Creates an action symbolizing an offer for the opponent.
	 * 
	 * @param agentID
	 *            id of the agent which created the offer.
	 * @param bid
	 *            for the opponent.
	 */
	public DefaultActionWithBid(AgentID agentID, Bid bid) {
		super(agentID);
		if (bid == null) {
			throw new NullPointerException("bid can not be null");
		}
		// FIXME we need to check actual Bid class; but we want junit tests to
		// work.
		// if (bid.getClass() != Bid.class) {
		// throw new IllegalArgumentException("Extending Bid is not allowed");
		// }
		this.bid = bid;
	}

	/**
	 * Returns the bid offered by the agent which created this offer.
	 * 
	 * @return bid to offer.
	 */
	public Bid getBid() {
		return bid;
	}

	/**
	 * @return hashcode of this object.
	 */
	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + ((bid == null) ? 0 : bid.hashCode());
		return result;
	}

	/**
	 * @param obj
	 *            object to which this object is compared.
	 * @return true if this object is equal to the given object.
	 */
	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		Offer other = (Offer) obj;
		if (bid == null) {
			if (other.bid != null)
				return false;
		} else if (!bid.equals(other.bid))
			return false;
		return true;
	}

	public String getContent() {
		return " bid:" + bid;
	}

}
