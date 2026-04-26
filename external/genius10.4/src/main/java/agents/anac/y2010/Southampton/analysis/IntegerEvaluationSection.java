package agents.anac.y2010.Southampton.analysis;

public class IntegerEvaluationSection extends ContinuousEvaluationSection {

	public IntegerEvaluationSection(int lowerBound, double evalLowerBound, int upperBound, double evalUpperBound) {
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
	private double getNormalPart(double weight, int lowerBound, double evalLowerBound, int upperBound, double evalUpperBound) {
		if (lowerBound >= upperBound)
			throw new AssertionError("lowerBound cannot be greater than or equal to upperBound");
		if (evalLowerBound != 0 && evalUpperBound != 0)
			throw new AssertionError("evalLowerBound or evalUpperBound must be zero");
		if (evalLowerBound != 1 && evalUpperBound != 1)
			throw new AssertionError("evalLowerBound or evalUpperBound must be one");
		if (evalUpperBound > evalLowerBound)
			return weight / (double) (upperBound - lowerBound);
		else
			return -weight / (double) (upperBound - lowerBound);
	}

	@Override
	public double getNormal(double weight) {
		return getNormalPart(weight, (int) lowerBound, evalLowerBound, (int) upperBound, evalUpperBound);
	}
}
