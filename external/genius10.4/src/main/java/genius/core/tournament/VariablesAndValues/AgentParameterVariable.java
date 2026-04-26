package genius.core.tournament.VariablesAndValues;

import genius.core.AgentParam;

/**
 * ProfileVariable is a variable for a tournament,
 * indicating that the profile is to be manipulated.
 * It just is an indicator for the TournamentVariable that its
 * value array contains a ProfileValue.
 * 
 * @author wouter
 *
 */
public class AgentParameterVariable extends TournamentVariable
{
	private static final long serialVersionUID = -8223126402840072070L;

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
	//	result = prime * result
		//		+ ((agentparam == null) ? 0 : agentparam.hashCode());
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
		AgentParameterVariable other = (AgentParameterVariable) obj;
		if (agentparam == null) {
			if (other.agentparam != null)
				return false;
		} else if (!agentparam.equals(other.agentparam))
			return false;
		return true;
	}

	AgentParam agentparam; // the name and other info about the parameter
	
	/** 
	 * @param para the parameter info
	 */
	public AgentParameterVariable(AgentParam para) {
		agentparam=para;
	}
	
	public void addValue(TournamentValue v) throws Exception
	{
		if (!(v instanceof AgentParamValue))
			throw new IllegalArgumentException("Expected AgentParamValue but received "+v);
		values.add(v);
	}
	
	public AgentParam getAgentParam() { return agentparam; }

	public String varToString() {
		return "AgentParamVar:"+agentparam.name;
	}
	
}