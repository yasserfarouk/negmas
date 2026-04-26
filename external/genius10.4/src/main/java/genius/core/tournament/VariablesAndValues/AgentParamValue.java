package genius.core.tournament.VariablesAndValues;

/**
 * This class contains a possible parameter value for a nego session 
 * A parameter value is a value that will appear as a start-up argument for the agent,
 * for instance the random-seed value, a tau value or debug options
 * @author wouter
 *
 */
public class AgentParamValue extends TournamentValue
{
	private static final long serialVersionUID = 1391633175859262227L;
	Double value;

	public AgentParamValue(Double v) {
		value=v;
	}
	
	public Double getValue() { return value; }
	
	public String toString() { return value.toString(); }
	
}