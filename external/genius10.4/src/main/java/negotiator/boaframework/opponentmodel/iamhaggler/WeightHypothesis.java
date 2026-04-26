package negotiator.boaframework.opponentmodel.iamhaggler;

import agents.bayesianopponentmodel.Hypothesis;

/**
 * @author Colin Williams
 * 
 */
public class WeightHypothesis extends Hypothesis {

	double weight;

	public void setWeight(double weight) {
		this.weight = weight;
	}

	public double getWeight() {
		return this.weight;
	}

	public String toString() {
		return String.format("%1.2f", this.weight) + ";";
	}
}
