package agents.uk.ac.soton.ecs.gp4j.gp.covariancefunctions;

import agents.Jama.Matrix;

public class CovarianceFunctionFactory {
	public static CovarianceFunction getNoisySquaredExponentialCovarianceFunction() {
		return new SumCovarianceFunction(SquaredExponentialCovarianceFunction
				.getInstance(), NoiseCovarianceFunction.getInstance());
	}

	public static CovarianceFunction getNoisySquaredExponentialARDCovarianceFunction() {
		return new SumCovarianceFunction(
				SquaredExponentialARDCovarianceFunction.getInstance(),
				NoiseCovarianceFunction.getInstance());
	}

	public static CovarianceFunction getNoisy2DTimeSquaredExponentialCovarianceFunction() {
		return new SumCovarianceFunction(
				get2DTimeSquaredExponentialCovarianceFunction(),
				NoiseCovarianceFunction.getInstance());
	}

	public static CovarianceFunction getNoisy2DTimeMatern3CovarianceFunction() {
		return new SumCovarianceFunction(get2DTimeMatern3CovarianceFunction(),
				NoiseCovarianceFunction.getInstance());
	}

	public static CovarianceFunction get2DTimeSquaredExponentialCovarianceFunction() {
		return new TwoDimensionalTimeCovarianceFunction(
				SquaredExponentialARDCovarianceFunction.getInstance());
	}

	public static CovarianceFunction get2DTimeMatern3CovarianceFunction() {
		return new TwoDimensionalTimeCovarianceFunction(
				new Matern3ARDCovarianceFunction());
	}

	private static class TwoDimensionalTimeCovarianceFunction implements
			CovarianceFunction {
		private final CovarianceFunction instance;

		public TwoDimensionalTimeCovarianceFunction(CovarianceFunction instance) {
			this.instance = instance;
		}

		public int getHyperParameterCount(Matrix trainX) {
			// 2 spatial dimensions, 1 time dimension
			// Validate.isTrue(trainX.getColumnDimension() == 3, trainX
			// .getColumnDimension()
			// + "");

			// lengthscale, timescale, signalvariance
			return 3;
		}

		public Matrix calculateTrainTestCovarianceMatrix(double[] loghyper,
				Matrix trainX, Matrix testX) {
			return instance.calculateTrainTestCovarianceMatrix(
					translate(loghyper), trainX, testX);
		}

		private double[] translate(double[] loghyper) {
			return new double[] { loghyper[0], loghyper[0], loghyper[1],
					loghyper[2] };
		}

		public Matrix calculateTestCovarianceMatrix(double[] loghyper,
				Matrix testX) {
			return instance.calculateTestCovarianceMatrix(translate(loghyper),
					testX);
		}

		public Matrix calculateCovarianceMatrix(double[] loghyper, Matrix trainX) {
			return instance.calculateCovarianceMatrix(translate(loghyper),
					trainX);
		}
	}
}
