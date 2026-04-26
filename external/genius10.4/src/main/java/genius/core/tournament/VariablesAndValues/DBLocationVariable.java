package genius.core.tournament.VariablesAndValues;

/**
 * @author Mark
 * Stores the location of the database.
 */
public class DBLocationVariable extends TournamentVariable
{
	private static final long serialVersionUID = 3916000985404639616L;

	@Override
	public void addValue(TournamentValue a) throws Exception {
	}
	
	public String varToString() {
		return "Database address";
	}
}