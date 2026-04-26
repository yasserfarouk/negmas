package genius.core.tournament.VariablesAndValues;

/**
 * @author Mark
 * Stores the location of the database.
 */
public class DBUserVariable extends TournamentVariable
{
	private static final long serialVersionUID = -1851516553702855611L;

	@Override
	public void addValue(TournamentValue a) throws Exception {
	}
	
	public String varToString() {
		return "Database user";
	}
}