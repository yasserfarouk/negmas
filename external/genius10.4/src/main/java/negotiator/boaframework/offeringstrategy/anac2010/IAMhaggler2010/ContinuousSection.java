package negotiator.boaframework.offeringstrategy.anac2010.IAMhaggler2010;

public class ContinuousSection {

	private double[] minBounds;
	private double[] maxBounds;
	private double[] normal;
	private double[] knownPoint1;
	private double evalKnownPoint1;
	private double[] knownPoint2;
	private double evalKnownPoint2;

	public double[] getMinBounds() {
		return this.minBounds;
	}

	public double[] getMaxBounds() {
		return this.maxBounds;
	}

	public double[] getNormal() {
		return this.normal;
	}

	public double[] getKnownPoint1() {
		return knownPoint1;
	}

	public double[] getKnownPoint2() {
		return knownPoint2;
	}

	public double getEvalKnownPoint1() {
		return evalKnownPoint1;
	}

	public double getEvalKnownPoint2() {
		return evalKnownPoint2;
	}

	public void setMinBounds(double[] minBounds) {
		this.minBounds = minBounds;
	}

	public void setMaxBounds(double[] maxBounds) {
		this.maxBounds = maxBounds;
	}

	public void setNormal(double[] normal) {
		this.normal = normal;
	}

	public void setKnownPoint1(double[] knownPoint1, double evalKnownPoint1) {
		this.knownPoint1 = knownPoint1;
		this.evalKnownPoint1 = evalKnownPoint1;
	}

	public void setKnownPoint2(double[] knownPoint2, double evalKnownPoint2) {
		this.knownPoint2 = knownPoint2;
		this.evalKnownPoint2 = evalKnownPoint2;
	}
}
