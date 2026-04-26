package agents.uk.ac.soton.ecs.gp4j.gp;

import agents.Jama.Matrix;

public interface GaussianRegression<T extends GaussianPredictor<?>> {
	T calculateRegression(Matrix trainX, Matrix trainY);

	T updateRegression(Matrix addedTrainX, Matrix addedTrainY, boolean downDate);

	T updateRegression(Matrix addedTrainX, Matrix addedTrainY);

	T downdateRegression();

	int getTrainingSampleCount();

	Matrix getTrainX();

	Matrix getTrainY();

	void reset();

	/**
	 * Copy the regression, including any learned datapoints and state matrices
	 * 
	 * @return
	 */
	GaussianRegression<T> copy();

	/**
	 * Create a shallow copy of the regression, containing parameters and
	 * covariance function, but not actual learned datapoints
	 * 
	 * @return
	 */
	GaussianRegression<T> shallowCopy();

	GaussianPredictor<?> getCurrentPredictor();

	T downdateRegression(int i);
}
