package agents.uk.ac.soton.ecs.gp4j.gp.covariancefunctions;

import agents.Jama.Matrix;

/**
 * Selectively feeds coordinates of inputs to a covariance function. This is
 * particularly handy when time and space have differently correlation
 * structures, and need to be fed into different covariance functions
 * 
 * @author rs06r
 * 
 */

public class SelectiveCoordinateCovarianceFunction implements
		CovarianceFunction {

	private CovarianceFunction function;

	private int[] indices;

	public SelectiveCoordinateCovarianceFunction(CovarianceFunction function,
			int... indices) {
		this.function = function;
		this.indices = indices;
	}

	/**
	 * Returns a submatrix with appropriate column indices. This corresponds to
	 * selecting only the necessary coordinates
	 * 
	 * @param matrix
	 * @return
	 */
	private Matrix filterCoordinates(Matrix matrix) {
		return matrix.getMatrix(0, matrix.getRowDimension() - 1, indices);
	}

	public Matrix calculateCovarianceMatrix(double[] loghyper, Matrix trainX) {
		return function.calculateCovarianceMatrix(loghyper,
				filterCoordinates(trainX));
	}

	public Matrix calculateTestCovarianceMatrix(double[] loghyper, Matrix testX) {
		return function.calculateTestCovarianceMatrix(loghyper,
				filterCoordinates(testX));
	}

	public Matrix calculateTrainTestCovarianceMatrix(double[] loghyper,
			Matrix trainX, Matrix testX) {
		return function.calculateTrainTestCovarianceMatrix(loghyper,
				filterCoordinates(trainX), filterCoordinates(testX));
	}

	public int getHyperParameterCount(Matrix trainX) {
		return function.getHyperParameterCount(filterCoordinates(trainX));
	}
}
