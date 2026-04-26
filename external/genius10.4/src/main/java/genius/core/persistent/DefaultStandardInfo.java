package genius.core.persistent;

import java.util.Collections;
import java.util.List;
import java.util.Map;

import genius.core.Bid;
import genius.core.Deadline;
import genius.core.list.Tuple;

/**
 * immutable implementation of StandardInfo
 */
public final class DefaultStandardInfo implements StandardInfo {

	private static final long serialVersionUID = 1L;
	private Map<String, String> profiles;
	private String startingAgent;
	private List<Tuple<String, Double>> utilities;
	private Deadline deadline;
	private Tuple<Bid, Double> agreement;

	public DefaultStandardInfo(Map<String, String> profiles, String startingAgent,
			List<Tuple<String, Double>> utilities, Deadline deadline, Tuple<Bid, Double> agreement) {
		this.profiles = profiles;
		this.startingAgent = startingAgent;
		this.utilities = utilities;
		this.deadline = deadline;
		this.agreement = agreement;
	}

	@Override
	public Map<String, String> getAgentProfiles() {
		return Collections.unmodifiableMap(profiles);
	}

	@Override
	public String getStartingAgent() {
		return startingAgent;
	}

	@Override
	public List<Tuple<String, Double>> getUtilities() {
		return Collections.unmodifiableList(utilities);
	}

	@Override
	public Deadline getDeadline() {
		return deadline;
	}

	@Override
	public Tuple<Bid, Double> getAgreement() {
		return agreement;
	}

	public String toString() {
		return "Info[" + profiles + "," + startingAgent + "," + utilities + "," + deadline + "]";
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + ((deadline == null) ? 0 : deadline.hashCode());
		result = prime * result + ((profiles == null) ? 0 : profiles.hashCode());
		result = prime * result + ((startingAgent == null) ? 0 : startingAgent.hashCode());
		result = prime * result + ((utilities == null) ? 0 : utilities.hashCode());
		return result;
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		DefaultStandardInfo other = (DefaultStandardInfo) obj;
		if (deadline == null) {
			if (other.deadline != null)
				return false;
		} else if (!deadline.equals(other.deadline))
			return false;
		if (profiles == null) {
			if (other.profiles != null)
				return false;
		} else if (!profiles.equals(other.profiles))
			return false;
		if (startingAgent == null) {
			if (other.startingAgent != null)
				return false;
		} else if (!startingAgent.equals(other.startingAgent))
			return false;
		if (utilities == null) {
			if (other.utilities != null)
				return false;
		} else if (!utilities.equals(other.utilities))
			return false;
		return true;
	}

}
