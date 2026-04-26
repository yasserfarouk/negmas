package agents.uk.ac.soton.ecs.gp4j.gp.covariancefunctions;

import agents.Jama.Matrix;

public class NoiseCovarianceFunction implements CovarianceFunction {

	private static NoiseCovarianceFunction noise;

	private NoiseCovarianceFunction() {
	}

	public Matrix calculateCovarianceMatrix(double[] loghyper, Matrix trainX) {
		int rowCount = trainX.getRowDimension();
		double noiseVariance = Math.exp(2 * loghyper[0]);

		Matrix result = new Matrix(rowCount, rowCount);

		for (int i = 0; i < rowCount; i++)
			result.set(i, i, noiseVariance);

		return result;
	}

	public static NoiseCovarianceFunction getInstance() {
		if (noise == null)
			noise = new NoiseCovarianceFunction();

		return noise;
	}

	public Matrix calculateTestCovarianceMatrix(double[] loghyper, Matrix testX) {
		return new Matrix(testX.getRowDimension(), 1, Math.exp(2 * loghyper[0]));
	}

	public Matrix calculateTrainTestCovarianceMatrix(double[] loghyper,
			Matrix trainX, Matrix testX) {
		return new Matrix(trainX.getRowDimension(), testX.getRowDimension());
	}

	public int getHyperParameterCount(Matrix testX) {
		return 1;
	}

}
