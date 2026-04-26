package agents.uk.ac.soton.ecs.gp4j.util;

import agents.Jama.Matrix;
import agents.org.apache.commons.lang.Validate;

public class MathUtils {
	public static double normPDF(double x, double mean, double standardDeviation) {
		return Math.exp(-(Math.pow((x - mean) / standardDeviation, 2)) / 2)
				/ (Math.sqrt(2 * Math.PI) * standardDeviation);
	}

	public static double mvnPDF(double[] x, double[] mu, double[][] sigma) {
		double N = x.length;

		Validate.isTrue(N == mu.length);
		Validate.isTrue(N == sigma.length);

		Matrix sigmaMat = new Matrix(sigma);
		Matrix xMat = new Matrix(x, 1);
		Matrix muMat = new Matrix(mu, 1);

		double constant = 1 / Math.sqrt(Math.pow(2 * Math.PI, N)
				* sigmaMat.det());

		Matrix xMinMu = xMat.minus(muMat);

		double d = xMinMu.times(sigmaMat.inverse()).times(xMinMu.transpose())
				.get(0, 0);

		return constant * Math.exp(-0.5 * d);
	}
}
