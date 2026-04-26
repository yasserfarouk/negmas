package agents.uk.ac.soton.ecs.gp4j.gp.covariancefunctions;

public class ConstantFunction implements MultivariateRealFunction{

	private double constant;

	public ConstantFunction(double constant) {
		this.constant = constant;
	}

	public double evaluate(double[] x) {
		return constant;
	}

}
