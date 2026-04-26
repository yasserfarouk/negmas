package genius.core.tournament.VariablesAndValues;

/** simple datastructure to couple a parameter to an specific agent.
 * We need to do this because the AgentParam in the tournament are bound to a CLASS, not a particular agent,
 * while in the nego session we need to bind params to particular agents. */

public class AssignedParameterVariable {
	public AgentParameterVariable parameter;
	public String agentname;
	public AssignedParameterVariable(AgentParameterVariable param,String name) {
		parameter = param;
		agentname = name;
	}
}
