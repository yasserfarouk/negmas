package genius.core;

import java.io.Serializable;
import java.util.HashMap;

/**
 * Simple class which stores the parameters given to a negotiation strategy, for
 * example an concession factor.
 * 
 * @author Mark Hendrikx
 */
public class StrategyParameters implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 265616574821536192L;
	/** Parameters as specified in the agent repository. */
	protected HashMap<String, String> strategyParam;

	/**
	 * Create an empty hashmap of parameters.
	 */
	public StrategyParameters() {
		strategyParam = new HashMap<String, String>();
	}

	/**
	 * Add a parameter to the list of parameters.
	 * 
	 * @param name
	 *            of the parameter.
	 * @param value
	 *            of the parameter.
	 */
	public void addVariable(String name, String value) {
		strategyParam.put(name, value);
	}

	/**
	 * Returns the value of the parameter with the given name as string.
	 * 
	 * @param name
	 *            of the given parameter.
	 * @return value of the parameter as string.
	 */
	public String getValueAsString(String name) {
		return strategyParam.get(name);
	}

	/**
	 * Returns the value of the parameter with the given name as double.
	 * 
	 * @param name
	 *            of the given parameter.
	 * @return value of the parameter as double.
	 */
	public double getValueAsDouble(String name) {
		return Double.parseDouble(strategyParam.get(name));
	}
	
	@Override
	public String toString() {
		String result = "";
		for (String key: this.strategyParam.keySet()) {
			result += ";" + key + "=" + this.strategyParam.get(key);
		}
		return result;
	}
}