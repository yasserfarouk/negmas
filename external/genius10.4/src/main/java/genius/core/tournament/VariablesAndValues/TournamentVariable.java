package genius.core.tournament.VariablesAndValues;

import java.io.Serializable;
import java.util.ArrayList;

/**
 * TournamentVariable is an abstract class, it is either a ProfileVariable,
 * AgentVariable or AgentParameterVariable, and it has a set of values that it
 * can take. During the tournament, all of the values will be used one after
 * another, in the order given in the ArrayList.
 * 
 * @author wouter
 * 
 */
public abstract class TournamentVariable implements Serializable {
	private static final long serialVersionUID = 5326059723219308268L;
	ArrayList<TournamentValue> values = new ArrayList<TournamentValue>();

	/** ordered list of values this var can take */

	/** add given value to the array of values */
	public abstract void addValue(TournamentValue value) throws Exception;

	public ArrayList<TournamentValue> getValues() {
		return values;
	}

	public void setValues(ArrayList<TournamentValue> newvals) {
		values = newvals;
	}

	/**
	 * varToString converts the variable name into a string. It shound NOT
	 * convert the values, only the variable name and its parameters (eg
	 * AgentParam[tau])
	 */
	public abstract String varToString();

}