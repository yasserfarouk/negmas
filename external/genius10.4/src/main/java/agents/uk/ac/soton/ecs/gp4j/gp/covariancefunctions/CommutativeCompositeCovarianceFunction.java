package agents.uk.ac.soton.ecs.gp4j.gp.covariancefunctions;

import agents.Jama.Matrix;
import agents.org.apache.commons.lang.Validate;

public abstract class CommutativeCompositeCovarianceFunction extends
		AbstractCompositeCovarianceFunction {

	public CommutativeCompositeCovarianceFunction(
			CovarianceFunction... functions) {
		super(functions);
	}

	public final Matrix calculateCovarianceMatrix(double[] loghyper,
			Matrix trainX) {
		Validate.isTrue(getHyperParameterCount(trainX) == loghyper.length,
				"Incorrect number of hyperparameters. Expected "
						+ getHyperParameterCount(trainX) + ", got "
						+ loghyper.length);

		double[][] hyperParameters = partitionHyperParameters(loghyper, trainX);
		Matrix result = functions[0].calculateCovarianceMatrix(
				hyperParameters[0], trainX);

		for (int i = 1; i < functions.length; i++) {
			result = operation(result, functions[i].calculateCovarianceMatrix(
					hyperParameters[i], trainX));
		}

		return result;
	}

	public final Matrix calculateTestCovarianceMatrix(double[] loghyper,
			Matrix testX) {
		Validate.isTrue(getHyperParameterCount(testX) == loghyper.length);

		double[][] hyperParameters = partitionHyperParameters(loghyper, testX);

		Matrix result = functions[0].calculateTestCovarianceMatrix(
				hyperParameters[0], testX);

		for (int i = 1; i < functions.length; i++) {
			result = operation(result, functions[i]
					.calculateTestCovarianceMatrix(hyperParameters[i], testX));
		}

		return result;
	}

	public final Matrix calculateTrainTestCovarianceMatrix(double[] loghyper,
			Matrix trainX, Matrix testX) {

		Validate.isTrue(getHyperParameterCount(trainX) == loghyper.length);

		double[][] hyperParameters = partitionHyperParameters(loghyper, trainX);

		Matrix result = functions[0].calculateTrainTestCovarianceMatrix(
				hyperParameters[0], trainX, testX);

		for (int i = 1; i < functions.length; i++) {
			result = operation(result, functions[i]
					.calculateTrainTestCovarianceMatrix(hyperParameters[i],
							trainX, testX));
		}

		return result;
	}

	protected abstract Matrix operation(Matrix result, Matrix matrix);
}
