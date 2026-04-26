package agents.uk.ac.soton.ecs.gp4j.gp;

import java.io.PrintWriter;
import java.io.StringWriter;

import agents.Jama.Matrix;

public class GaussianProcessPrediction implements GaussianPrediction {

	private final Matrix variance;
	private final Matrix mean;
	private final Matrix testX;
	private final Matrix standardDeviation;
	private Matrix covarianceMatrix;

	public GaussianProcessPrediction(Matrix testX, Matrix mean, Matrix variance) {
		this(testX, mean, variance, null);
	}

	public GaussianProcessPrediction(Matrix testX, Matrix mean,
			Matrix variance, Matrix minus) {
		this.testX = testX;
		this.mean = mean;
		this.variance = variance;
		standardDeviation = new Matrix(variance.getArrayCopy());
		this.covarianceMatrix = minus;

		for (int i = 0; i < standardDeviation.getRowDimension(); i++) {
			standardDeviation.set(i, 0, Math.sqrt(standardDeviation.get(i, 0)));
		}
	}

	public Matrix getCovarianceMatrix() {
		return covarianceMatrix;
	}

	public Matrix getVariance() {
		return variance;
	}

	public Matrix getMean() {
		return mean;
	}

	public Matrix getTestX() {
		return testX;
	}

	public Matrix getStandardDeviation() {
		return standardDeviation;
	}

	@Override
	public String toString() {
		StringWriter writer = new StringWriter();
		PrintWriter printWriter = new PrintWriter(writer);

		int m = testX.getColumnDimension();
		int n = testX.getRowDimension();

		Matrix resultMatrix = new Matrix(n, m + 2);
		resultMatrix.setMatrix(0, n - 1, 0, m - 1, testX);
		resultMatrix.setMatrix(0, n - 1, m, m, mean);
		resultMatrix.setMatrix(0, n - 1, m + 1, m + 1, variance);

		resultMatrix.print(printWriter, 10, 4);

		return writer.getBuffer().toString();
	}

	public int size() {
		return variance.getRowDimension();
	}
}
