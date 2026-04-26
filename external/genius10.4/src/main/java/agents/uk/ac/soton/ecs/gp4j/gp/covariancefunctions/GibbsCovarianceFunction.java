package agents.uk.ac.soton.ecs.gp4j.gp.covariancefunctions;

import agents.Jama.Matrix;
import agents.org.apache.commons.lang.Validate;

public class GibbsCovarianceFunction implements CovarianceFunction {

	MultivariateRealFunction[] lds;

	public GibbsCovarianceFunction() {

	}

	public GibbsCovarianceFunction(MultivariateRealFunction[] lds) {
		this.lds = lds;
	}

	public void setLds(MultivariateRealFunction... lds) {
		this.lds = lds;
	}

	public Matrix calculateCovarianceMatrix(double[] loghyper, Matrix trainX) {
		return calculateTrainTestCovarianceMatrix(loghyper, trainX, trainX);
	}

	public Matrix calculateTestCovarianceMatrix(double[] loghyper, Matrix testX) {
		return new Matrix(testX.getRowDimension(), 1, 1.0);
	}

	public Matrix calculateTrainTestCovarianceMatrix(double[] loghyper,
			Matrix trainX, Matrix testX) {
		int samplesTrain = trainX.getRowDimension();
		int samplesTest = testX.getRowDimension();

		if (samplesTrain == 0 || samplesTest == 0)
			return new Matrix(samplesTrain, samplesTest);

		double[][] result = new double[samplesTrain][samplesTest];
		double[][] trainXVals = trainX.getArray();
		double[][] testXVals = testX.getArray();

		int dimensions = trainXVals[0].length;
		Validate.isTrue(testXVals[0].length == dimensions);
		Validate.isTrue(lds.length == dimensions);

		double[][] precomputedLsTrain = precomputeLs(trainXVals);
		double[][] precomputedLsTest = precomputeLs(testXVals);

		for (int i = 0; i < trainX.getRowDimension(); i++) {
			for (int j = 0; j < testX.getRowDimension(); j++) {
				result[i][j] = calculateCovariance(trainXVals[i], testXVals[j],
						precomputedLsTrain[i], precomputedLsTest[j]);
			}
		}

		return new Matrix(result);
	}

	private double[][] precomputeLs(double[][] testXVals) {
		double[][] result = new double[testXVals.length][lds.length];

		for (int i = 0; i < testXVals.length; i++) {
			for (int j = 0; j < lds.length; j++) {
				result[i][j] = lds[j].evaluate(testXVals[i]);
			}
		}

		return result;
	}

	private double calculateCovariance(double[] train, double[] test,
			double[] trainL, double[] testL) {
		double term1 = 1.0;
		double term2 = 0.0;

		for (int d = 0; d < train.length; d++) {
			double denominator = trainL[d] * trainL[d] + testL[d] * testL[d];

			term1 *= Math.sqrt(2 * trainL[d] * testL[d] / denominator);

			term2 += (train[d] - test[d]) * (train[d] - test[d]) / denominator;
		}

		term2 = Math.exp(-term2);

		return term1 * term2;
	}

	public int getHyperParameterCount(Matrix trainX) {
		return 0;
	}
}
