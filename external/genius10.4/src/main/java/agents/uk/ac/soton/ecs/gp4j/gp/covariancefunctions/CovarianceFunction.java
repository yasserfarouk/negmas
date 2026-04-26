package agents.uk.ac.soton.ecs.gp4j.gp.covariancefunctions;

import agents.Jama.Matrix;

public interface CovarianceFunction {

	int getHyperParameterCount(Matrix trainX);

	Matrix calculateCovarianceMatrix(double[] loghyper, Matrix trainX);

	Matrix calculateTrainTestCovarianceMatrix(double[] loghyper, Matrix trainX,
			Matrix testX);

	Matrix calculateTestCovarianceMatrix(double[] loghyper, Matrix testX);
}
