package agents.anac.y2010.Southampton.analysis;

public class RealEvaluationSection extends ContinuousEvaluationSection {

	public RealEvaluationSection(double lowerBound, double evalLowerBound, double upperBound, double evalUpperBound) {
		this.lowerBound = lowerBound;
		this.evalLowerBound = evalLowerBound;
		this.upperBound = upperBound;
		this.evalUpperBound = evalUpperBound;
	}

	/**
	 * @param weight
	 * @param lowerBound
	 * @param evalLowerBound
	 * @param upperBound
	 * @param evalUpperBound
	 * @return
	 */
	private double getNormalPart(double weight, double lowerBound, double evalLowerBound, double upperBound, double evalUpperBound) {
		if (lowerBound >= upperBound)
			throw new AssertionError("lowerBound cannot be greater than or equal to upperBound");
		if (evalLowerBound != 0 && evalUpperBound != 0)
			throw new AssertionError("evalLowerBound or evalUpperBound must be zero");
		if (evalLowerBound != 1 && evalUpperBound != 1)
			throw new AssertionError("evalLowerBound or evalUpperBound must be one");
		if (evalUpperBound > evalLowerBound)
			return weight / (upperBound - lowerBound);
		else
			return -weight / (upperBound - lowerBound);
	}

	@Override
	public double getNormal(double weight) {
		return getNormalPart(weight, lowerBound, evalLowerBound, upperBound, evalUpperBound);
	}
}
