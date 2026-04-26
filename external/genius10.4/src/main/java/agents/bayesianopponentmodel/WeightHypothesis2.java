package agents.bayesianopponentmodel;

import genius.core.Domain;

public class WeightHypothesis2 extends Hypothesis {
	double fWeight;
//	double fAprioriProbability;
	public WeightHypothesis2 (Domain pDomain) {

	}
	public void setWeight( double value) {
		fWeight = value;
	}
	public double getWeight() {
		return fWeight;
	}
	public String toString() {
		String lResult = "";
			lResult = String.format("%1.2f", fWeight) +";";
		return lResult;
	}
}
