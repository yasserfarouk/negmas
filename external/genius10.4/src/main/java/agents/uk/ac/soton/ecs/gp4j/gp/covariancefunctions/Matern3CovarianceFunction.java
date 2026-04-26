package agents.uk.ac.soton.ecs.gp4j.gp.covariancefunctions;

import java.util.Arrays;

import agents.Jama.Matrix;

public class Matern3CovarianceFunction implements CovarianceFunction {

	private Matern3ARDCovarianceFunction function = new Matern3ARDCovarianceFunction();

	private static Matern3CovarianceFunction instance;

	protected Matern3CovarianceFunction() {

	}

	public static Matern3CovarianceFunction getInstance() {
		if (instance == null)
			instance = new Matern3CovarianceFunction();
		
		return instance;
	}

	public Matrix calculateCovarianceMatrix(double[] loghyper, Matrix trainX) {
		return function.calculateCovarianceMatrix(translateHyperParameters(
				loghyper, trainX.getColumnDimension()), trainX);
	}

	public Matrix calculateTrainTestCovarianceMatrix(double[] loghyper,
			Matrix trainX, Matrix testX) {
		return function
				.calculateTrainTestCovarianceMatrix(translateHyperParameters(
						loghyper, trainX.getColumnDimension()), trainX, testX);
	}

	public Matrix calculateTestCovarianceMatrix(double[] loghyper, Matrix testX) {
		return function.calculateTestCovarianceMatrix(translateHyperParameters(
				loghyper, testX.getColumnDimension()), testX);
	}

	private double[] translateHyperParameters(double[] loghyper,
			int inputDimension) {
		double[] result = new double[inputDimension + 1];
		Arrays.fill(result, loghyper[0]);
		result[result.length - 1] = loghyper[1];

		return result;
	}

	public int getHyperParameterCount(Matrix testX) {
		return 2;
	}
}