package genius.core.tournament.VariablesAndValues;

/**
 * @author Mark
 * Stores the sessionname of the database.
 */
public class DBSessionVariable extends TournamentVariable
{
	private static final long serialVersionUID = -4335662542882961866L;

	@Override
	public void addValue(TournamentValue a) throws Exception {
	}
	
	public String varToString() {
		return "Database sessionname";
	}
}