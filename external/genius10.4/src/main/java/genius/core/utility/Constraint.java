package genius.core.utility;

import genius.core.Bid;

/**
 * Specifies an area that has a utility.
 */
public class Constraint {

	protected double weight = 1.0;

	// The following method will be overridden by super classes (classes
	// extending "Constraint" e.g. InclusiveHyperRectangle)

	public double getUtility(Bid bid) {
		return 0.0;
	}

	protected double getWeight() {
		return weight;
	}

	protected void setWeight(double weight) {
		this.weight = weight;
	}
}