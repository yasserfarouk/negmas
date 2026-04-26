package agents.uk.ac.soton.ecs.gp4j.bmc;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import agents.Jama.Matrix;
import agents.org.apache.commons.lang.NotImplementedException;
import agents.org.apache.commons.lang.Validate;
import agents.org.apache.commons.math.stat.StatUtils;
import agents.uk.ac.soton.ecs.gp4j.gp.GaussianPredictor;
import agents.uk.ac.soton.ecs.gp4j.gp.GaussianProcess;
import agents.uk.ac.soton.ecs.gp4j.gp.GaussianProcessRegression;
import agents.uk.ac.soton.ecs.gp4j.gp.GaussianRegression;
import agents.uk.ac.soton.ecs.gp4j.gp.covariancefunctions.CovarianceFunction;
import agents.uk.ac.soton.ecs.gp4j.util.ArrayUtils;
import agents.uk.ac.soton.ecs.gp4j.util.MathUtils;
import agents.uk.ac.soton.ecs.gp4j.util.MatrixUtils;

public class GaussianProcessRegressionBMC implements
		GaussianRegression<GaussianProcessMixture> {

	// private static Log log =
	// LogFactory.getLog(GaussianProcessRegressionBMC.class);

	private CovarianceFunction function;

	private List<BasicPrior> priors;

	private List<GaussianProcessRegression> gpRegressions;

	private List<Double> weights;

	private Matrix KSinv_NS_KSinv;

	private boolean initialized;

	private int dataPointsProcessed = 0;

	private GaussianProcessMixture currentPredictor;

	public GaussianProcessRegressionBMC() {

	}

	public void reset() {
		throw new NotImplementedException();
	}

	public GaussianProcessRegressionBMC(CovarianceFunction function,
			List<BasicPrior> priors) {
		this.function = function;
		this.priors = priors;
		initialize();
	}

	// copy constructor
	private GaussianProcessRegressionBMC(GaussianProcessRegressionBMC toCopy) {
		this.function = toCopy.function;
		this.priors = new ArrayList<BasicPrior>(toCopy.priors);

		this.weights = new ArrayList<Double>(toCopy.weights);

		this.KSinv_NS_KSinv = toCopy.KSinv_NS_KSinv.copy();

		this.currentPredictor = toCopy.currentPredictor;
		initialize();
		this.gpRegressions = new ArrayList<GaussianProcessRegression>();
	}

	public void initialize() {
		if (!initialized) {
			initializeSamples();
			calculateKSinv_NS_KSinv();
		}

		initialized = true;
	}

	private void initializeSamples() {
		int i = 0;
		double[][] independentSamples = new double[priors.size()][0];
		for (BasicPrior prior : priors)
			independentSamples[i++] = prior.getLogSamples();

		double[][] samples = ArrayUtils.allCombinations(independentSamples);

		gpRegressions = new ArrayList<GaussianProcessRegression>(samples.length);

		for (i = 0; i < samples.length; i++) {
			gpRegressions.add(new GaussianProcessRegression(samples[i],
					function));
		}

		// for (int j = 0; j < samples.length; j++) {
		// log.debug("Sample : " + ArrayUtils.toString(samples[j]));
		// }
	}

	/**
	 * Calculate the first three terms of equation 3.8.13, which is the product
	 * of the inverse of KS, NS, and the inverse of KS
	 */
	private void calculateKSinv_NS_KSinv() {
		KSinv_NS_KSinv = new Matrix(1, 1, 1.0);

		for (BasicPrior prior : priors) {
			Matrix KS = calculateKS(prior.getWidth(), prior.getLogSamples());
			Matrix NS = calculateNS(prior.getWidth(), prior.getLogSamples(),
					prior.getLogMean(), prior.getStandardDeviation());

			Matrix cholKS = KS.chol().getL();
			Matrix result = MatrixUtils.solveChol(cholKS, NS).transpose();
			result = cholKS.transpose().solve(result);
			result = cholKS.solve(result);

			KSinv_NS_KSinv = MatrixUtils.kronecker(KSinv_NS_KSinv, result);
		}
	}

	/**
	 * Calculate the NS Matrix using equation 3.8.10
	 */
	private Matrix calculateNS(double width, double[] samples, double mean,
			double standardDeviation) {
		Matrix NS = new Matrix(samples.length, samples.length);

		double variance = standardDeviation * standardDeviation;
		double lambda = variance + width * width;
		double precX = lambda - variance * variance / lambda;
		double precY = 1 / (variance - lambda * lambda / variance);
		double multConst = 1 / Math.sqrt(Math.pow(2 * Math.PI, 2) * lambda
				* precX);

		for (int i = 0; i < samples.length; i++) {
			for (int j = 0; j < samples.length; j++) {
				double xDev = samples[i] - mean;
				double yDev = samples[j] - mean;

				NS.set(i, j, multConst
						* Math.exp(-0.5 / precX * (xDev * xDev + yDev * yDev)
								- precY * xDev * yDev));
			}
		}

		return NS;
	}

	/**
	 * Calculate the KS matrix
	 */
	private Matrix calculateKS(double width, double[] samples) {
		Matrix KS = new Matrix(samples.length, samples.length);

		for (int i = 0; i < samples.length; i++) {
			for (int j = 0; j < samples.length; j++) {
				KS.set(i, j, MathUtils.normPDF(samples[i], samples[j], width));
			}
		}

		return KS;
	}

	public List<GaussianProcessRegression> getGpRegressions() {
		return gpRegressions;
	}

	public GaussianProcessMixture calculateRegression(Matrix trainX,
			Matrix trainY) {
		initialize();
		List<GaussianProcess> gaussianProcesses = new ArrayList<GaussianProcess>();

		for (GaussianProcessRegression gpRegression : gpRegressions) {
			GaussianProcess gp = gpRegression.calculateRegression(trainX,
					trainY);
			gaussianProcesses.add(gp);
		}

		calculateWeights();
		Validate.isTrue(gpRegressions.size() == weights.size());

		currentPredictor = new GaussianProcessMixture(gaussianProcesses,
				weights);
		return currentPredictor;
	}

	public GaussianProcessMixture downdateRegression(int i) {
		List<GaussianProcess> gaussianProcesses = new ArrayList<GaussianProcess>();

		for (GaussianProcessRegression gpRegression : gpRegressions) {
			GaussianProcess gp = gpRegression.downdateRegression(i);
			gaussianProcesses.add(gp);
		}

		// log-likelihoods will not change during a downdate. Therefore, weights
		// need not be recalculated
		// calculateWeights();

		currentPredictor = new GaussianProcessMixture(gaussianProcesses,
				weights);
		return currentPredictor;

	}

	public GaussianProcessMixture downdateRegression() {
		return downdateRegression(1);
	}

	public GaussianProcessMixture updateRegression(Matrix addedTrainX,
			Matrix addedTrainY, boolean downDate) {
		return updateRegression(addedTrainX, addedTrainY);
	}

	public GaussianProcessMixture updateRegression(Matrix addedTrainX,
			Matrix addedTrainY) {
		initialize();

		dataPointsProcessed += addedTrainX.getRowDimension();

		List<GaussianProcess> gaussianProcesses = new ArrayList<GaussianProcess>();

		for (GaussianProcessRegression gpRegression : gpRegressions) {
			GaussianProcess gp = gpRegression.updateRegression(addedTrainX,
					addedTrainY);
			gaussianProcesses.add(gp);
		}

		calculateWeights();

		Validate.isTrue(gpRegressions.size() == weights.size());

		currentPredictor = new GaussianProcessMixture(gaussianProcesses,
				weights);

		recalculateSamples();

		return currentPredictor;
	}

	private void calculateWeights() {
		int size = gpRegressions.size();

		double[] logLikelihoods = new double[size];

		// calculate the weights using equation 3.8.16
		for (int i = 0; i < size; i++)
			logLikelihoods[i] = gpRegressions.get(i).getLogLikelihood();

		// scale log-likelihoods for numerical stability
		double maxLogLikelihood = StatUtils.max(logLikelihoods);
		Matrix rs = new Matrix(size, 1);
		for (int i = 0; i < size; i++)
			rs.set(i, 0, Math.exp(logLikelihoods[i] - maxLogLikelihood));

		Matrix numerator = KSinv_NS_KSinv.times(rs);
		double denominator = MatrixUtils.sum(numerator).get(0, 0);
		Matrix weightsMatrix = numerator.times(1 / denominator);

		weights = Arrays.asList(ArrayUtils.toObject(weightsMatrix
				.getColumnPackedCopy()));
	}

	private void recalculateSamples() {
		if (dataPointsProcessed % 50 == 0) {
			double threshold = 1e-3;

			for (int j = 0; j < priors.size(); j++) {
				if (priors.get(j).getSampleCount() <= 5)
					continue;

				double[][] marginalizedWeights = getMarginalizedHyperParameterWeights(j);
				double maxWeight = Double.NEGATIVE_INFINITY;
				double maxWeightedParam = Double.NEGATIVE_INFINITY;
				int underThreshold = 0;

				for (int i = 0; i < marginalizedWeights.length; i++) {
					if (marginalizedWeights[i][1] < threshold)
						underThreshold++;

					if (marginalizedWeights[i][1] > maxWeight) {
						maxWeight = marginalizedWeights[i][1];
						maxWeightedParam = marginalizedWeights[i][0];
					}
				}

				BasicPrior oldPrior = priors.get(j);
				int newSampleCount = Math.max(5, oldPrior.getSampleCount()
						- underThreshold / 2);
				double newStandardDeviation = oldPrior.getStandardDeviation() * 0.8;
				BasicPrior newPrior = new BasicPrior(newSampleCount, Math
						.exp(maxWeightedParam), newStandardDeviation);

				priors.set(j, newPrior);
			}

			initialized = false;
			calculateRegression(getTrainX(), getTrainY());
		}
	}

	protected Matrix getKSinv_NS_KSinv() {
		return KSinv_NS_KSinv;
	}

	protected List<Double> getWeights() {
		return weights;
	}

	public Map<Double[], Double> getHyperParameterWeights() {
		HashMap<Double[], Double> weighing = new HashMap<Double[], Double>();

		for (int i = 0; i < weights.size(); i++) {
			weighing.put(gpRegressions.get(i).getHyperParameters(), weights
					.get(i));
		}

		return weighing;
	}

	/**
	 * Returns a 3D matrix of. First dimension specifies hyperparameter index.
	 * Second and third dimensions form a 2D matrix with (param value, weight)
	 * tuples
	 * 
	 * @return
	 */
	public double[][] getMarginalizedHyperParameterWeights(int paramIndex) {
		int n = priors.get(paramIndex).getSampleCount();
		double[] samples = priors.get(paramIndex).getLogSamples();
		double[][] result = new double[n][];

		for (int i = 0; i < samples.length; i++) {
			result[i] = new double[2];
			result[i][0] = samples[i];

			for (int j = 0; j < gpRegressions.size(); j++) {
				GaussianProcessRegression gpr = gpRegressions.get(j);
				if (gpr.getLogHyperParameters()[paramIndex] == samples[i])
					result[i][1] += weights.get(j);
			}
		}

		return result;
	}

	public GaussianProcessMixture calculateRegression(double[] trainX,
			double[] trainY) {
		return calculateRegression(new Matrix(trainX, 1).transpose(),
				new Matrix(trainY, 1).transpose());
	}

	public int getTrainingSampleCount() {
		return gpRegressions.get(0).getTrainingSampleCount();
	}

	public Matrix getTrainX() {
		return gpRegressions.get(0).getTrainX();
	}

	public Matrix getTrainY() {
		return gpRegressions.get(0).getTrainY();
	}

	public GaussianProcessRegressionBMC copy() {
		Validate.isTrue(initialized, "Cannot copy before initialized");

		GaussianProcessRegressionBMC regressionBMC = new GaussianProcessRegressionBMC(
				this);

		for (GaussianProcessRegression regression : gpRegressions) {
			regressionBMC.gpRegressions.add(regression.copy());
		}

		Validate.isTrue(regressionBMC.gpRegressions.size() == KSinv_NS_KSinv
				.getColumnDimension());

		return regressionBMC;
	}

	public GaussianProcessRegressionBMC shallowCopy() {
		GaussianProcessRegressionBMC regressionBMC = new GaussianProcessRegressionBMC(
				this);

		for (GaussianProcessRegression regression : gpRegressions) {
			regressionBMC.gpRegressions.add(regression.shallowCopy());
		}

		Validate.isTrue(gpRegressions.size() == KSinv_NS_KSinv
				.getColumnDimension());

		return regressionBMC;
	}

	public void setCovarianceFunction(CovarianceFunction function) {
		this.function = function;
	}

	public void setPriors(List<BasicPrior> priors) {
		this.priors = priors;
	}

	public GaussianPredictor<?> getCurrentPredictor() {
		return currentPredictor;
	}

	public void setPriors(BasicPrior... priors) {
		this.priors = Arrays.asList(priors);
	}
}
