package genius.core.tournament.VariablesAndValues;

/**
 * @author Mark Hendrikx
 * Stores the sessionname of the database.
 */
public class DBPasswordVariable extends TournamentVariable
{
	private static final long serialVersionUID = 563404623019296025L;

	@Override
	public void addValue(TournamentValue a) throws Exception {
	}
	
	public String varToString() {
		return "Database password";
	}
}