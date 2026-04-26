package genius.core.tournament.VariablesAndValues;

public class MultipleAgentsVariable extends TournamentVariable {

	public void addValue(TournamentValue a) throws Exception 
	{
		if (!(a instanceof AgentValue))
		{
			throw new IllegalArgumentException("Expected AgentValue but received " + a);
		}
		
		values.add(a);
	}

	public String varToString() 
	{
		return "Agents";
	}
}
