package genius.core.session;

import genius.core.AgentID;
import genius.core.parties.Mediator;
import genius.core.repository.ParticipantRepItem;
import genius.core.repository.ProfileRepItem;

/**
 * Contains full participant info: the party name, strategy and profile.
 * Equality only involves the party NAME, as names should be unique in a
 * negotiation. mediators are also participants. immutable.
 */
public class Participant {
	private AgentID id;
	private ParticipantRepItem strategy;
	private ProfileRepItem profile;

	/**
	 * 
	 * @param id
	 *            the agent ID
	 * @param party
	 *            the {@link ParticipantRepItem} to use for this participant.
	 * @param profile
	 *            The profile that this participant uses. can be null if this
	 *            participant does not need a profile (eg, {@link Mediator}.
	 */
	public Participant(AgentID id, ParticipantRepItem party,
			ProfileRepItem profile) {
		if (id == null || party == null || profile == null) {
			throw new IllegalArgumentException("parameter is null");
		}
		this.id = id;
		this.strategy = party;
		this.profile = profile;
	}

	public AgentID getId() {
		return id;
	}

	public ParticipantRepItem getStrategy() {
		return strategy;
	}

	public ProfileRepItem getProfile() {
		return profile;
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + ((id == null) ? 0 : id.hashCode());
		return result;
	}

	/**
	 * Equals is based only on the agent id.
	 */
	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		Participant other = (Participant) obj;
		if (id == null) {
			if (other.id != null)
				return false;
		} else if (!id.equals(other.id))
			return false;
		return true;
	}

	@Override
	public String toString() {
		return "Participant[" + id + "," + strategy + "," + profile + "]";
	}

}
