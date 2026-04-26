package genius.core.tournament.VariablesAndValues;

import java.util.HashMap;

public class TournamentOptionsValue extends TournamentValue {
	
	private static final long serialVersionUID = 1L;
	HashMap<String, Integer> options = new HashMap<String, Integer>();	
	
	public TournamentOptionsValue(HashMap<String, Integer> options) { this.options = options; }
	
	public String toString() { return options.toString(); }
	
	public HashMap<String, Integer> getValue(){ return options;	}
}