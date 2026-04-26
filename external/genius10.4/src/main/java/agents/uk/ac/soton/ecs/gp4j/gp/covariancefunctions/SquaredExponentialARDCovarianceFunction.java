package agents.uk.ac.soton.ecs.gp4j.gp.covariancefunctions;

import agents.Jama.Matrix;

public class SquaredExponentialARDCovarianceFunction implements
		CovarianceFunction {
	private static SquaredExponentialARDCovarianceFunction instance;

	public Matrix calculateCovarianceMatrix(double[] loghyper, Matrix trainX) {
		return calculateTrainTestCovarianceMatrix(loghyper, trainX, trainX);
	}

	/**
	 * @param loghyper
	 *            loghyper[0] to loghyper[n-2] are lengthscales for every input
	 *            dimension, loghyper[n-1] is the signal variance
	 */
	public Matrix calculateTrainTestCovarianceMatrix(double[] loghyper,
			Matrix trainX, Matrix testX) {
		int samplesTrain = trainX.getRowDimension();
		int samplesTest = testX.getRowDimension();

		if (samplesTrain == 0 || samplesTest == 0)
			return new Matrix(samplesTrain, samplesTest);

		double signalVariance = Math.exp(2 * loghyper[loghyper.length - 1]);

		double[][] trainXVals = scaleValues(trainX, loghyper);
		double[][] testXVals = scaleValues(testX, loghyper);

		double[][] result = new double[samplesTrain][samplesTest];

		for (int i = 0; i < samplesTrain; i++) {
			for (int j = 0; j < samplesTest; j++) {
				double sq_dist = calculateSquareDistance(trainXVals[i],
						testXVals[j]);
				result[i][j] = signalVariance * Math.exp(-sq_dist / 2);
			}
		}

		return new Matrix(result);
	}

	private double[][] scaleValues(Matrix matrix, double[] loghyper) {
		double[][] array = matrix.getArrayCopy();

		for (int i = 0; i < loghyper.length - 1; i++) {
			double lengthScale = Math.exp(loghyper[i]);

			for (int j = 0; j < array.length; j++) {
				array[j][i] /= lengthScale;
			}
		}

		return array;
	}

	public Matrix calculateTestCovarianceMatrix(double[] loghyper, Matrix testX) {
		return new Matrix(testX.getRowDimension(), 1, Math
				.exp(2 * loghyper[loghyper.length - 1]));
	}

	public int getHyperParameterCount(Matrix testX) {
		return testX.getColumnDimension() + 1;
	}

	private double calculateSquareDistance(double[] ds, double[] ds2) {
		double sq_dist = 0;

		for (int i = 0; i < ds.length; i++) {
			double diff = ds[i] - ds2[i];
			sq_dist += diff * diff;
		}

		return sq_dist;
	}

	public static CovarianceFunction getInstance() {
		if (instance == null)
			instance = new SquaredExponentialARDCovarianceFunction();

		return instance;
	}
}
