package agents.uk.ac.soton.ecs.gp4j.util;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.Locale;
import java.util.Map;

import agents.Jama.Matrix;
import agents.uk.ac.soton.ecs.gp4j.bmc.BasicPrior;
import agents.uk.ac.soton.ecs.gp4j.bmc.GaussianProcessRegressionBMC;
import agents.uk.ac.soton.ecs.gp4j.gp.covariancefunctions.CovarianceFunctionFactory;

public class DataLearner {
	public static void main(String[] args) throws IOException {
		BufferedReader reader = new BufferedReader(new FileReader(new File(
				"/mnt/data/berkeley-dataset/sensor1_times_0.1.txt")));
		Matrix trainX = Matrix.read(reader);

		reader = new BufferedReader(new FileReader(new File(
				"/mnt/data/berkeley-dataset/sensor1_temps_0.1.txt")));
		Matrix trainY = Matrix.read(reader);

		GaussianProcessRegressionBMC regression = new GaussianProcessRegressionBMC();
		regression.setCovarianceFunction(CovarianceFunctionFactory
				.getNoisySquaredExponentialARDCovarianceFunction());
		// .getNoisy2DTimeSquaredExponentialCovarianceFunction());

		// BasicPrior lengthScalePrior = new BasicPrior(5, 5.0, 0.5);
		BasicPrior timeScalePrior = new BasicPrior(10, 5000, 0.15);
		BasicPrior signalVariance = new BasicPrior(10, 10, 0.15);
		BasicPrior noise = new BasicPrior(1, 0.4, 0.3);

		regression.setPriors(timeScalePrior, signalVariance, noise);

		// int batchSize = 1;

		regression.updateRegression(trainX, trainY);

		printHyperParamWeights(regression.getHyperParameterWeights(), 1);

		// for (int i = 0; i < trainX.getRowDimension(); i++) {
		//
		// System.out.println(i);
		//
		// regression.updateRegression(trainX.getMatrix(i, i, 0, 0), trainY
		// .getMatrix(i, i, 0, 0));
		//
		// // if (regression.getTrainingSampleCount() > 120)
		// // regression.downdateRegression();
		//
		// printHyperParamWeights(regression.getHyperParameterWeights(), i);
		// }
	}

	private static void printHyperParamWeights(
			Map<Double[], Double> hyperParameterWeights, int round)
			throws IOException {

		StringBuffer buffer = new StringBuffer();

		for (Double[] hyper : hyperParameterWeights.keySet()) {
			Double weight = hyperParameterWeights.get(hyper);

			buffer.append(String.format(Locale.US, " %6d", round));

			for (int j = 0; j < hyper.length; j++) {
				buffer.append(String.format(Locale.US, " %15.5f", hyper[j]));
			}

			buffer.append(String.format(Locale.US, " %15.5f\n", weight));
		}

		// FileUtils.writeStringToFile(new File("params", "params-" + round
		// + ".txt"), buffer.toString());
	}
}
