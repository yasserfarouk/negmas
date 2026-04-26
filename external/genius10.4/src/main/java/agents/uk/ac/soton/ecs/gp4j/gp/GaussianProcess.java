package agents.uk.ac.soton.ecs.gp4j.gp;

import agents.Jama.Matrix;
import agents.org.apache.commons.lang.ArrayUtils;
import agents.org.apache.commons.lang.Validate;
import agents.uk.ac.soton.ecs.gp4j.gp.covariancefunctions.CovarianceFunction;
import agents.uk.ac.soton.ecs.gp4j.util.MatrixUtils;

public class GaussianProcess implements
		GaussianPredictor<GaussianProcessPrediction> {

	private final Matrix trainX;
	private final Matrix alpha;
	private final Matrix cholTrainingCovarianceMatrix;
	private final CovarianceFunction function;
	private final double logLikelihood;
	private final double[] loghyper;
	private final Matrix trainY;

	protected GaussianProcess(Matrix trainX, Matrix trainY, Matrix alpha,
			Matrix cholTrainingCovarianceMatrix, double[] loghyper,
			CovarianceFunction function, double logLikelihood) {
		Validate.notNull(trainX);
		Validate.notNull(trainY);
		Validate.notNull(alpha);
		Validate.notNull(cholTrainingCovarianceMatrix);
		Validate.notNull(function);
		Validate.notNull(loghyper);
		Validate.notNull(logLikelihood);

		this.trainX = trainX;
		this.trainY = trainY;
		this.alpha = alpha;
		this.cholTrainingCovarianceMatrix = cholTrainingCovarianceMatrix;
		this.function = function;
		this.loghyper = loghyper;
		this.logLikelihood = logLikelihood;

		Validate
				.isTrue(function.getHyperParameterCount(trainX) == loghyper.length);
		Validate.isTrue(trainY.getColumnDimension() == 1);
	}

	public GaussianProcessPrediction calculatePrediction(Matrix testX) {
		return calculatePrediction(testX, false);
	}

	public GaussianProcessPrediction calculatePrediction(Matrix testX,
			boolean calculateCovarianceMatrix) {
		Matrix trainTestCovarianceMatrix = function
				.calculateTrainTestCovarianceMatrix(loghyper, trainX, testX);

		Matrix testCovarianceMatrix = function.calculateTestCovarianceMatrix(
				loghyper, testX);

		Matrix mean = trainTestCovarianceMatrix.transpose().times(alpha);

		Matrix L = cholTrainingCovarianceMatrix;

		Matrix v = null;
		try {
			v = L.solve(trainTestCovarianceMatrix);
		} catch (RuntimeException e) {
			System.out.println(ArrayUtils.toString(loghyper));
			System.out.println(function);
			trainX.print(10, 10);

			throw e;
		}

		Matrix covariance = testCovarianceMatrix.minus(MatrixUtils.sum(
				v.arrayTimes(v)).transpose());

		if (calculateCovarianceMatrix) {
			Matrix testTestCovarianceMatrix = function
					.calculateCovarianceMatrix(loghyper, testX);

			Matrix conditionedTestCovarianceMatrix = testTestCovarianceMatrix
					.minus(v.transpose().times(v));
			return new GaussianProcessPrediction(testX, mean, covariance,
					conditionedTestCovarianceMatrix);
		} else {
			return new GaussianProcessPrediction(testX, mean, covariance);
		}

	}

	public Matrix getTrainX() {
		return trainX;
	}

	public Matrix getTrainY() {
		return trainY;
	}

	protected Matrix getAlpha() {
		return alpha;
	}

	protected Matrix getCholTrainingCovarianceMatrix() {
		return cholTrainingCovarianceMatrix;
	}

	public double[] getLogHyperParameters() {
		return loghyper;
	}

	public CovarianceFunction getCovarianceFunction() {
		return function;
	}

	public double getLogLikelihood() {
		return logLikelihood;
	}
}
