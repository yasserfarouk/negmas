package negotiator.boaframework.offeringstrategy.anac2010.IAMhaggler2010;

public abstract class ContinuousEvaluationSection {

	protected double evalLowerBound;
	protected double evalUpperBound;
	protected double lowerBound;
	protected double upperBound;

	public ContinuousEvaluationSection() {
		super();
	}

	/**
	 * @param weight
	 * @return
	 */
	public abstract double getNormal(double weight);

	/**
	 * @return
	 */
	public double getEvalLowerBound() {
		return this.evalLowerBound;
	}

	/**
	 * @return
	 */
	public double getEvalUpperBound() {
		return this.evalUpperBound;
	}

	/**
	 * @return
	 */
	public double getLowerBound() {
		return this.lowerBound;
	}

	/**
	 * @return
	 */
	public double getUpperBound() {
		return this.upperBound;
	}

	/**
	 * @return
	 */
	public double getTopPoint() {
		if (this.evalLowerBound < this.evalUpperBound)
			return upperBound;
		else
			return lowerBound;
	}
}