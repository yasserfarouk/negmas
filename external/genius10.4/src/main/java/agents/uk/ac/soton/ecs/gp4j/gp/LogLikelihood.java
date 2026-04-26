package agents.uk.ac.soton.ecs.gp4j.gp;

import agents.Jama.Matrix;
import agents.uk.ac.soton.ecs.gp4j.util.MatrixUtils;

/**
 * In Rasmussen's Gaussian Processes for Machine Learning (2006), the
 * loglikelihood is broken up into three components. This class uses the same
 * decomposition and terminology as used in the book (see p. 113).
 * 
 * @author Ruben Stranders
 * 
 */
class LogLikelihood {

	private double value;
	private Matrix trainY;
	private double datafit;

	protected LogLikelihood() {
		value = Double.NaN;
		datafit = Double.NaN;
		trainY = new Matrix(0, 0);
	}

	public LogLikelihood(Matrix trainY, Matrix alpha,
			Matrix cholTrainingCovarianceMatrix) {
		this.trainY = trainY;

		datafit = calculateDatafit(trainY, alpha);
		double complexityPanelty = calculateComplexityPanelty(cholTrainingCovarianceMatrix);
		double normalizationConstant = calculateNormalizationConstant(trainY
				.getRowDimension());

		value = datafit + complexityPanelty + normalizationConstant;
	}

	private LogLikelihood(double value, Matrix trainY, double datafit) {
		this.value = value;
		this.trainY = trainY.copy();
		this.datafit = datafit;
	}

	public void update(Matrix addedTrainY, Matrix alpha, Matrix U) {
		value -= datafit;

		trainY = MatrixUtils.append(trainY, addedTrainY);

		datafit = calculateDatafit(trainY, alpha);

		value += datafit + calculateComplexityPanelty(U)
				+ calculateNormalizationConstant(addedTrainY.getRowDimension());
	}

	public double getValue() {
		return value;
	}

	private double calculateNormalizationConstant(int n) {
		return -0.5 * n * Math.log(2 * Math.PI);
	}

	private double calculateDatafit(Matrix trainY, Matrix alpha) {
		try {
			return -0.5 * trainY.transpose().times(alpha).get(0, 0);
		} catch (IllegalArgumentException e) {
			System.out.println("train: " + trainY.getRowDimension());
			trainY.print(10, 5);
			System.out.println("alpha: " + alpha.getRowDimension());
			alpha.print(10, 4);
			throw e;
		}
	}

	private double calculateComplexityPanelty(
			Matrix cholTrainingCovarianceMatrix) {
		return -0.5 * MatrixUtils.logDetChol(cholTrainingCovarianceMatrix);
	}

	public void downdate(int epochs) {
		// don't recalculate anything, just remove old datapoint
		trainY = trainY.getMatrix(epochs, trainY.getRowDimension() - 1, 0, 0);
	}

	public LogLikelihood copy() {
		return new LogLikelihood(value, trainY, datafit);
	}
}
