package genius.core.boaframework;

import java.io.Serializable;
import java.util.HashSet;

import genius.core.misc.Pair;

/**
 * This stores the parameter specification of a BOA component: the name
 * description and range of valid values. Used in in {@link Tournament}.
 * 
 * If lower and higher bound is used, it also requires a step size and all
 * in-between values in the range are being generated immediately. Basically,
 * what is stored is [Lowerbound:Stepsize:Upperbound]. [1:5:20] = {1, 6, 11,
 * 16}.
 * 
 */
public class BOAparameter implements Serializable {

	private static final long serialVersionUID = 2555736049221913613L;
	/** Name of the parameter. */
	private String name;
	/** Lowerbound of the specified range. */
	private Double low;
	/** Upperbound of the specified range. */
	private Double high;
	/** Step size of the specified range. */
	private Double step;
	/** set of separate values which the specified variable should attain */
	private HashSet<Pair<String, Double>> valuePairs;
	/** description of the parameter */
	private String description;

	/**
	 * Describes a parameter for a BOA component. A parameter consists of a
	 * name, and the possible values for the parameter.
	 * 
	 * @param name
	 *            of the parameter.
	 * @param low
	 *            value of the range.
	 * @param high
	 *            value of the range.
	 * @param step
	 *            of the range.
	 */
	public BOAparameter(String name, Double low, Double high, Double step) {
		this.name = name;
		this.low = low;
		this.high = high;
		this.step = step;
		description = "";
		generatePairs();
	}

	/**
	 * Describes a parameter for a BOA component with a fixed single value.
	 * 
	 * @param name
	 * @param defaultValue
	 * @param description
	 */
	public BOAparameter(String name, Double defaultValue, String description) {
		this.name = name;
		this.description = description;
		this.low = defaultValue;
		this.high = defaultValue;
		this.step = 1D;
		generatePairs();
	}

	/**
	 * Describes a parameter for a decoupled component. A parameter consists of
	 * a name, a description, and the possible values for the parameter.
	 * 
	 * @param name
	 *            of the parameter.
	 * @param low
	 *            value of the range.
	 * @param high
	 *            value of the range.
	 * @param step
	 *            of the range.
	 * @param description
	 *            of the parameter.
	 */
	public BOAparameter(String name, Double low, Double high, Double step, String description) {
		this.name = name;
		this.low = low;
		this.high = high;
		this.step = step;
		this.description = description;
		generatePairs();
	}

	/**
	 * Generates the set of all possible configurations for the parameter given
	 * the range and step size of the component.
	 */
	private void generatePairs() {
		valuePairs = new HashSet<Pair<String, Double>>();
		for (Double value = low; value <= high; value += step) {
			valuePairs.add(new Pair<String, Double>(name, value));
		}
	}

	/**
	 * Returns all values of the parameters which satisfy
	 * [Lowerbound:Stepsize:Upperbound].
	 * 
	 * @return possible values for the parameter specified.
	 */
	public HashSet<Pair<String, Double>> getValuePairs() {
		return valuePairs;
	}

	/**
	 * @return name of the parameter.
	 */
	public String getName() {
		return name;
	}

	/**
	 * @return value for the lowerbound.
	 */
	public Double getLow() {
		return low;
	}

	/**
	 * @return upperbound of the range.
	 */
	public Double getHigh() {
		return high;
	}

	/**
	 * @return stepsize of the range.
	 */
	public Double getStep() {
		return step;
	}

	public String toString() {
		if (!name.equals("null")) {
			if (low.compareTo(high) == 0) {
				/*
				 * without doubleValue we get a crazy number of digits
				 */
				return name + ": " + low.doubleValue();
			}
			return name + ": [" + low + " : " + step + " : " + high + "]";
		} else {
			return "";
		}
	}

	/**
	 * @return description of the parameter.
	 */
	public String getDescription() {
		return description;
	}

	public String toXML() {
		return "<parameter name=\"" + name + "\" default=\"" + high + "\" description=\"" + description + "\"/>";
	}

	/**
	 * 
	 * @param newDescr
	 *            the new description for this parameter
	 * @return new {@link BOAparameter} with same settings as this but new
	 *         description as given
	 */
	public BOAparameter withDescription(String newDescr) {
		return new BOAparameter(name, low, high, step, newDescr);
	}

	/**
	 * @param newLow
	 *            the new low value for this parameter.
	 * @return new BOAparameter with same settings as this but with new low
	 *         value. If the high value is < newLow, it is also set to newLow.
	 */
	public BOAparameter withLow(Double newLow) {
		return new BOAparameter(name, newLow, Math.max(newLow, high), step, description);
	}

	/**
	 * 
	 * @param newHigh
	 *            the new high value for this parameter
	 * @return new {@link BOAparameter} with same settings as this but with the
	 *         new high value. If the high value is < low, then low is also set
	 *         to newHigh.
	 */
	public BOAparameter withHigh(Double newHigh) {
		return new BOAparameter(name, Math.min(low, newHigh), newHigh, step, description);
	}

	/**
	 * 
	 * @param step
	 *            the new step value
	 * @return new {@link BOAparameter} with same settings as this but with step
	 *         value as given
	 */
	public BOAparameter withStep(Double step) {
		return new BOAparameter(name, low, high, step, description);
	}

}