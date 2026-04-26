package genius.core.tournament.VariablesAndValues;

/**
 * {@link AgentVariable} indicates the agents used in a tournament.
 */
public class BOAagentVariable extends TournamentVariable
{
	private static final long serialVersionUID = 6409851887801713416L;
	private String side = null;
	public void addValue(TournamentValue a) throws Exception {
		if (!(a instanceof BOAagentValue))
			throw new IllegalArgumentException("Expected DecoupledAgentValue but received "+a);
		values.add(a);
	}
	
	public String varToString() {
		String res = "BOA Agent";
		if(side != null) res = res + " side " +side;
		return res;
	}
	public void setSide(String val) 
	{
		side = val;
	}
	
	public String getSide() {
		return side;
	}
}