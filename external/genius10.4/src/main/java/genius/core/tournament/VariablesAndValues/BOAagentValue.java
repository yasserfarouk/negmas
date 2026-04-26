package genius.core.tournament.VariablesAndValues;

import genius.core.boaframework.BOAagentInfo;

public class BOAagentValue extends TournamentValue
{
	private static final long serialVersionUID = 4154311572147986731L;
	BOAagentInfo value;	
	
	public BOAagentValue(BOAagentInfo val) { value = val; }
	public String toString() { return value.toString(); }
	public BOAagentInfo getValue() { return value; }
	
	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + ((value == null) ? 0 : value.hashCode());
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
		BOAagentValue other = (BOAagentValue) obj;
		if (value == null) {
			if (other.value != null)
				return false;
		} else if (!value.equals(other.value))
			return false;
		return true;
	}
}