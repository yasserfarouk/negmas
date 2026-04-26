package agents.uk.ac.soton.ecs.gp4j.gp;

import java.util.List;

import agents.Jama.CholeskyDecomposition;
import agents.Jama.Matrix;
import agents.org.apache.commons.lang.Validate;
import agents.uk.ac.soton.ecs.gp4j.gp.covariancefunctions.CovarianceFunction;
import agents.uk.ac.soton.ecs.gp4j.util.MatrixUtils;

public class GaussianProcessRegression implements
		GaussianRegression<GaussianProcess> {

	private double[] loghyper;

	private CovarianceFunction covarianceFunction;

	private Matrix cholTrainingCovarianceMatrix = new Matrix(0, 0);

	private Matrix alpha = new Matrix(0, 0);

	private LogLikelihood logLikelihood = new LogLikelihood();

	private Matrix trainX = new Matrix(0, 0);

	private Matrix trainY = new Matrix(0, 0);

	private boolean initialized;

	public GaussianProcessRegression(double[] loghyper,
			CovarianceFunction function) {
		super();
		setLogHyperParameters(loghyper);
		setCovarianceFunction(function);
		reset();
	}

	public void reset() {
		cholTrainingCovarianceMatrix = new Matrix(0, 0);
		alpha = new Matrix(0, 0);
		logLikelihood = new LogLikelihood();
		trainX = new Matrix(0, 0);
		trainY = new Matrix(0, 0);

		for (int i = 0; i < 100; i++) {
			if (covarianceFunction.getHyperParameterCount(new Matrix(0, i)) == loghyper.length) {
				calculateRegression(new Matrix(0, i), new Matrix(0, 1));
				break;
			}
		}

	}

	private GaussianProcessRegression(double[] loghyper,
			CovarianceFunction covarianceFunction,
			Matrix cholTrainingCovarianceMatrix, Matrix alpha,
			LogLikelihood logLikelihood, Matrix trainX, Matrix trainY,
			boolean initialized) {
		this(loghyper, covarianceFunction);
		Validate.notNull(cholTrainingCovarianceMatrix);
		Validate.notNull(alpha);
		Validate.notNull(logLikelihood);
		Validate.notNull(trainX);
		Validate.notNull(trainY);
		this.cholTrainingCovarianceMatrix = cholTrainingCovarianceMatrix;
		this.alpha = alpha;
		this.logLikelihood = logLikelihood;
		this.trainX = trainX;
		this.trainY = trainY;
		this.initialized = initialized;
	}

	public GaussianProcessRegression() {

	}

	public void setHyperParameters(List<Double> hyper) {
		this.loghyper = new double[hyper.size()];

		for (int i = 0; i < hyper.size(); i++) {
			loghyper[i] = Math.log(hyper.get(i));
		}
	}

	public void setLogHyperParameters(double[] hyper) {
		Validate.notNull(hyper);
		this.loghyper = hyper;
	}

	public Double[] getHyperParameters() {
		Double[] hyper = new Double[loghyper.length];

		for (int i = 0; i < loghyper.length; i++) {
			hyper[i] = Math.exp(loghyper[i]);
		}
		return hyper;
	}

	public void setCovarianceFunction(CovarianceFunction covarianceFunction) {
		Validate.notNull(covarianceFunction);
		this.covarianceFunction = covarianceFunction;
	}

	public GaussianProcess updateRegression(Matrix addedTrainX,
			Matrix addedTrainY, boolean downDate) {
		return updateRegression(addedTrainX, addedTrainY);
	}

	// core functionality
	public GaussianProcess calculateRegression(Matrix trainX, Matrix trainY) {
		checkDimensions(trainX, trainY);

		this.trainX = trainX;
		this.trainY = trainY;

		Matrix trainingCovarianceMatrix = covarianceFunction
				.calculateCovarianceMatrix(loghyper, trainX);

		CholeskyDecomposition chol = trainingCovarianceMatrix.chol();
		cholTrainingCovarianceMatrix = chol.getL();
		alpha = chol.solve(trainY);
		logLikelihood = new LogLikelihood(trainY, alpha,
				cholTrainingCovarianceMatrix);

		initialized = true;

		return createGaussianProcess();
	}

	private void checkDimensions(Matrix x, Matrix y) {
		Validate.notNull(covarianceFunction);
		Validate.notNull(loghyper);

		if (covarianceFunction.getHyperParameterCount(x) != loghyper.length)
			throw new IllegalArgumentException(
					"Dimensionality of training points is incorrect. Expected "
							+ covarianceFunction.getHyperParameterCount(x)
							+ " hyperparameters, but got " + loghyper.length);

		if (y.getColumnDimension() != 1)
			throw new IllegalArgumentException(
					"Dimensionality of training output should be 1");

		// if (x.getRowDimension() != y.getRowDimension())
		// throw new IllegalArgumentException(
		// "Number of training points in X should be equals to the number of points in Y. Got "
		// + x.getRowDimension()
		// + " points in X, and "
		// + y.getRowDimension() + " in Y");
	}

	public GaussianProcess updateRegression(Matrix addedTrainX,
			Matrix addedTrainY) {
		if (!initialized)
			return calculateRegression(addedTrainX, addedTrainY);

		checkDimensions(addedTrainX, addedTrainY);

		// number of extra samples
		int deltaN = addedTrainX.getRowDimension();
		// number of existing samples
		int oldN = getTrainingSampleCount();
		int n = oldN + deltaN;

		Matrix trainAddedTrainCovarianceMatrix = covarianceFunction
				.calculateTrainTestCovarianceMatrix(loghyper, trainX,
						addedTrainX);
		Matrix addedTrainCovarianceMatrix = covarianceFunction
				.calculateCovarianceMatrix(loghyper, addedTrainX);

		// update the Cholesky decomposition using equations 4.1.4 and 4.1.5
		Matrix S = cholTrainingCovarianceMatrix.solve(
				trainAddedTrainCovarianceMatrix).transpose();
		Matrix U = addedTrainCovarianceMatrix.minus(S.times(S.transpose()))
				.chol().getL();

		// construct the new Cholesky matrix
		Matrix updatedChol = new Matrix(n, n);
		updatedChol.setMatrix(0, oldN - 1, 0, oldN - 1,
				cholTrainingCovarianceMatrix);
		updatedChol.setMatrix(oldN, n - 1, 0, oldN - 1, S);
		updatedChol.setMatrix(oldN, n - 1, oldN, n - 1, U);

		cholTrainingCovarianceMatrix = updatedChol;

		trainX = MatrixUtils.append(trainX, addedTrainX);
		trainY = MatrixUtils.append(trainY, addedTrainY);

		alpha = MatrixUtils.solveChol(updatedChol, trainY);

		logLikelihood.update(addedTrainY, alpha, U);

		return createGaussianProcess();
	}

	public GaussianProcess downdateRegression(int epochs) {

		for (int i = 0; i < epochs; i++) {
			cholTrainingCovarianceMatrix = MatrixUtils
					.choleskyDowndate(cholTrainingCovarianceMatrix);

			int m = trainX.getColumnDimension();

			trainX = trainX
					.getMatrix(1, trainX.getRowDimension() - 1, 0, m - 1);
			trainY = trainY.getMatrix(1, trainY.getRowDimension() - 1, 0, 0);
		}

		alpha = MatrixUtils.solveChol(cholTrainingCovarianceMatrix, trainY);
		logLikelihood.downdate(epochs);

		return createGaussianProcess();
	}

	public GaussianProcess downdateRegression() {
		return downdateRegression(1);
	}

	// Simple getters
	public Matrix getTrainX() {
		return trainX;
	}

	public Matrix getTrainY() {
		return trainY;
	}

	protected Matrix getAlpha() {
		return alpha;
	}

	public double[] getLogHyperParameters() {
		return loghyper;
	}

	public CovarianceFunction getCovarianceFunction() {
		return covarianceFunction;
	}

	public int getTrainingSampleCount() {
		return trainX.getRowDimension();
	}

	public CovarianceFunction getFunction() {
		return covarianceFunction;
	}

	// convenience methods
	public GaussianProcess calculateRegression(double[] trainX, double[] trainY) {
		return calculateRegression(MatrixUtils.createColumnVector(trainX),
				MatrixUtils.createColumnVector(trainY));
	}

	public GaussianProcess updateRegression(double[] addedTrainX,
			double[] addedTrainY) {
		return updateRegression(MatrixUtils.createColumnVector(addedTrainX),
				MatrixUtils.createColumnVector(addedTrainY));
	}

	// internal utility methods
	private GaussianProcess createGaussianProcess() {
		return new GaussianProcess(trainX, trainY, alpha,
				cholTrainingCovarianceMatrix, loghyper, covarianceFunction,
				getLogLikelihood());
	}

	public double getLogLikelihood() {
		return logLikelihood.getValue();
	}

	public GaussianProcessRegression copy() {
		return new GaussianProcessRegression(loghyper, covarianceFunction,
				cholTrainingCovarianceMatrix.copy(), alpha.copy(),
				logLikelihood.copy(), trainX.copy(), trainY.copy(), initialized);
	}

	public GaussianProcessRegression shallowCopy() {
		return new GaussianProcessRegression(loghyper, covarianceFunction);
	}

	public GaussianPredictor<?> getCurrentPredictor() {
		return createGaussianProcess();
	}
}
