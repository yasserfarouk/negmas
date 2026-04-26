package genius.core.tournament.VariablesAndValues;

/**
 * @author tim
 * Used for having values in experiments.
 */
public class ExperimentalVariable extends TournamentVariable
{
	private static final long serialVersionUID = -8560060884848539056L;
	private String side = null;
	
	@Override
	public void addValue(TournamentValue a) throws Exception {
		if (!(a instanceof ExperimentalValue))
			throw new IllegalArgumentException("Expected ExperimentalValue but received "+a);
		values.add(a);
	}
	
	public String varToString() {
		String res = "Experimental variable";
		if(side != null) res = res + " " +side;
		return res;
	}
	
	public void setSide(String val) {
		side = val;
	}
}