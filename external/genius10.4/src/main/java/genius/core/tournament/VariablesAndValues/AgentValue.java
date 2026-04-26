package genius.core.tournament.VariablesAndValues;

import genius.core.repository.AgentRepItem;

public class AgentValue extends TournamentValue
{
	private static final long serialVersionUID = -1479458519909188852L;
	AgentRepItem value;	
	
	public AgentValue(AgentRepItem val) { value=val; }
	public String toString() { return value.getName(); }
	public AgentRepItem getValue() { return value; }
	/* (non-Javadoc)
	 * @see java.lang.Object#hashCode()
	 */
	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + ((value == null) ? 0 : value.hashCode());
		return result;
	}
	/* (non-Javadoc)
	 * @see java.lang.Object#equals(java.lang.Object)
	 */
	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		AgentValue other = (AgentValue) obj;
		if (value == null) {
			if (other.value != null)
				return false;
		} else if (!value.equals(other.value))
			return false;
		return true;
	}
}