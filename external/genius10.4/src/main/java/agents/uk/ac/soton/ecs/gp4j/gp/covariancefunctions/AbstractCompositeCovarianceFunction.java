package agents.uk.ac.soton.ecs.gp4j.gp.covariancefunctions;

import agents.Jama.Matrix;
import agents.org.apache.commons.lang.ArrayUtils;
import agents.org.apache.commons.lang.Validate;

public abstract class AbstractCompositeCovarianceFunction implements
		CovarianceFunction {
	protected final CovarianceFunction[] functions;

	public AbstractCompositeCovarianceFunction(CovarianceFunction... functions) {
		Validate.notEmpty(functions);
		this.functions = functions;
	}

	public int getHyperParameterCount(Matrix trainX) {
		int result = 0;

		for (int i = 0; i < functions.length; i++)
			result += functions[i].getHyperParameterCount(trainX);

		return result;
	}

	public CovarianceFunction[] getCovarianceFunctions() {
		return functions;
	}
	
	protected double[][] partitionHyperParameters(double[] hyperParameters,
			Matrix x) {
		double[][] partition = new double[functions.length][];

		int start = 0;

		for (int i = 0; i < functions.length; i++) {
			partition[i] = ArrayUtils.subarray(hyperParameters, start, start
					+ functions[i].getHyperParameterCount(x));
			start += functions[i].getHyperParameterCount(x);
		}

		return partition;
	}
}
