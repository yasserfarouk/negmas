package agents.anac.y2013.MetaAgent.portfolio.IAMhaggler2012.agents2011;

import java.util.ArrayList;

import agents.Jama.Matrix;
import agents.anac.y2013.MetaAgent.portfolio.IAMhaggler2012.agents2011.southampton.utils.BidCreator;
import agents.anac.y2013.MetaAgent.portfolio.IAMhaggler2012.agents2011.southampton.utils.Pair;
import agents.anac.y2013.MetaAgent.portfolio.IAMhaggler2012.agents2011.southampton.utils.RandomBidCreator;
import agents.org.apache.commons.math.MathException;
import agents.org.apache.commons.math.MaxIterationsExceededException;
import agents.org.apache.commons.math.special.Erf;
import agents.uk.ac.soton.ecs.gp4j.bmc.BasicPrior;
import agents.uk.ac.soton.ecs.gp4j.bmc.GaussianProcessMixture;
import agents.uk.ac.soton.ecs.gp4j.bmc.GaussianProcessMixturePrediction;
import agents.uk.ac.soton.ecs.gp4j.bmc.GaussianProcessRegressionBMC;
import agents.uk.ac.soton.ecs.gp4j.gp.covariancefunctions.CovarianceFunction;
import agents.uk.ac.soton.ecs.gp4j.gp.covariancefunctions.Matern3CovarianceFunction;
import agents.uk.ac.soton.ecs.gp4j.gp.covariancefunctions.NoiseCovarianceFunction;
import agents.uk.ac.soton.ecs.gp4j.gp.covariancefunctions.SumCovarianceFunction;
import genius.core.Bid;
import genius.core.utility.AdditiveUtilitySpace;

/**
 * @author Colin Williams
 * 
 *         The IAMhaggler Agent, created for ANAC 2011. Designed by C. R.
 *         Williams, V. Robu, E. H. Gerding and N. R. Jennings.
 * 
 */
public class IAMhaggler2011 extends SouthamptonAgent {

	protected double RISK_PARAMETER = 1;

	private Matrix utilitySamples;
	private Matrix timeSamples;
	private Matrix utility;
	private GaussianProcessRegressionBMC regression;
	private double lastRegressionTime = 0;
	private double lastRegressionUtility = 1;
	private ArrayList<Double> opponentTimes = new ArrayList<Double>();
	private ArrayList<Double> opponentUtilities = new ArrayList<Double>();

	private double maxUtilityInTimeSlot;
	private int lastTimeSlot = -1;
	private Matrix means;
	private Matrix variances;

	private double maxUtility;

	private Bid bestReceivedBid;

	private double previousTargetUtility;

	protected BidCreator bidCreator;

	private double intercept;

	private Matrix matrixTimeSamplesAdjust;

	private double maxOfferedUtility = Double.MIN_VALUE;
	private double minOfferedUtility = Double.MAX_VALUE;

	public IAMhaggler2011() {
		debug = true;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see agents.southampton.SouthamptonAgent#init()
	 */
	@Override
	public void init() {
		log("Run init...");
		try {
			super.init();
		} catch (Exception ex) {
			ex.printStackTrace();
		}
		double discountingFactor = 0.5;
		try {
			discountingFactor = adjustDiscountFactor(utilitySpace
					.getDiscountFactor());
		} catch (Exception ex) {
			logError("Unable to get discounting factor, assuming 0.5");
			ex.printStackTrace();
		}
		if (discountingFactor == 0)
			discountingFactor = 1;
		log("Discounting factor is " + discountingFactor);
		makeUtilitySamples(100);
		makeTimeSamples(100);
		Matrix discounting = generateDiscountingFunction(discountingFactor);
		Matrix risk = generateRiskFunction(RISK_PARAMETER);
		utility = risk.arrayTimes(discounting);

		log(utility);

		log("Setting up GP");
		flushLog();

		BasicPrior[] bps = { new BasicPrior(11, 0.252, 0.5),
				new BasicPrior(11, 0.166, 0.5), new BasicPrior(1, .01, 1.0) };
		CovarianceFunction cf = new SumCovarianceFunction(
				Matern3CovarianceFunction.getInstance(),
				NoiseCovarianceFunction.getInstance());

		regression = new GaussianProcessRegressionBMC();
		regression.setCovarianceFunction(cf);
		regression.setPriors(bps);

		// regression.calculateRegression(new Matrix(new double[] {}, 0), new
		// Matrix(new double[] {}, 0));

		maxUtility = 0;
		previousTargetUtility = 1;

		bidCreator = new RandomBidCreator();

		log("init complete.");
		flushLog();
	}

	@Override
	public String getName() {
		return "IAMhaggler2011";
	}

	/**
	 * Create an m-by-1 matrix of utility samples.
	 * 
	 * @param m
	 *            The sample size.
	 */
	private void makeUtilitySamples(int m) {
		double[] utilitySamplesArray = new double[m];
		{
			for (int i = 0; i < utilitySamplesArray.length; i++) {
				utilitySamplesArray[i] = 1.0 - ((double) i + 0.5)
						/ ((double) m + 1.0);
			}
		}
		utilitySamples = new Matrix(utilitySamplesArray,
				utilitySamplesArray.length);
	}

	/**
	 * Create a 1-by-n matrix of time samples.
	 * 
	 * @param n
	 *            The sample size.
	 */
	private void makeTimeSamples(int n) {
		double[] timeSamplesArray = new double[n + 1];
		{
			for (int i = 0; i < timeSamplesArray.length; i++) {
				timeSamplesArray[i] = ((double) i) / ((double) n);
			}
		}
		timeSamples = new Matrix(timeSamplesArray, 1);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see agents.southampton.SouthamptonAgent#proposeInitialBid()
	 */
	@Override
	protected Bid proposeInitialBid() throws Exception {
		return ((AdditiveUtilitySpace) utilitySpace).getMaxUtilityBid();
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see agents.southampton.SouthamptonAgent#proposeNextBid(negotiator.Bid)
	 */
	@Override
	protected Bid proposeNextBid(Bid opponentBid) throws Exception {
		double opponentUtility = utilitySpace.getUtility(opponentBid);

		if (opponentUtility > maxUtility) {
			bestReceivedBid = opponentBid;
			maxUtility = opponentUtility;
		}

		log("Opponent utility is " + opponentUtility);

		double targetUtility = getTarget(opponentUtility, getTime());

		log("Target utility is " + targetUtility);

		if (targetUtility <= maxUtility && previousTargetUtility > maxUtility)
			return bestReceivedBid;
		previousTargetUtility = targetUtility;

		flushLog();

		// Now get a random bid in the range targetUtility � 0.025
		return bidCreator.getBid(((AdditiveUtilitySpace) utilitySpace),
				targetUtility - 0.025, targetUtility + 0.025);
	}

	/**
	 * Get the target at a given time, recording the opponent's utility.
	 * 
	 * @param opponentUtility
	 *            The utility of the most recent offer made by the opponent.
	 * @param time
	 *            The current time.
	 * @return the target.
	 */
	protected double getTarget(double opponentUtility, double time) {
		log("++>>> IAMhaggler 2011 <<<++");

		log("getTarget: " + opponentUtility);

		maxOfferedUtility = Math.max(maxOfferedUtility, opponentUtility);
		minOfferedUtility = Math.min(minOfferedUtility, opponentUtility);

		// Calculate the current time slot
		int timeSlot = (int) Math.floor(time * 36);

		boolean regressionUpdateRequired = false;
		if (lastTimeSlot == -1) {
			regressionUpdateRequired = true;
		}

		// If the time slot has changed
		if (timeSlot != lastTimeSlot) {
			if (lastTimeSlot != -1) {
				// Store the data from the time slot
				opponentTimes.add((lastTimeSlot + 0.5) / 36.0);
				if (opponentUtilities.size() == 0) {
					intercept = Math.max(0.5, maxUtilityInTimeSlot);
					double[] timeSamplesAdjust = new double[timeSamples
							.getColumnDimension()];
					int i = 0;
					double gradient = 0.9 - intercept;
					for (double d : timeSamples.getRowPackedCopy()) {
						timeSamplesAdjust[i++] = intercept + (gradient * d);
					}
					matrixTimeSamplesAdjust = new Matrix(timeSamplesAdjust,
							timeSamplesAdjust.length);
				}
				opponentUtilities.add(maxUtilityInTimeSlot);
				// Flag regression receiveMessage required
				regressionUpdateRequired = true;
			}
			// Update the time slot
			lastTimeSlot = timeSlot;
			// Reset the max utility
			maxUtilityInTimeSlot = 0;
		}

		log("intercept: " + intercept);

		// Calculate the maximum utility observed in the current time slot
		maxUtilityInTimeSlot = Math.max(maxUtilityInTimeSlot, opponentUtility);

		if (timeSlot == 0) {
			return 1.0 - time / 2.0;
		}

		if (regressionUpdateRequired) {
			double gradient = 0.9 - intercept;
			/*
			 * double[] x = new double[opponentTimes.size()]; double[] yAdjust =
			 * new double[opponentTimes.size()]; double[] y = new
			 * double[opponentUtilities.size()];
			 * 
			 * int i; i = 0; for (double d : opponentTimes) { x[i++] = d; } i =
			 * 0; for (double d : opponentTimes) { yAdjust[i++] = intercept +
			 * (gradient * d); } i = 0; for (double d : opponentUtilities) {
			 * y[i++] = d; }
			 * 
			 * Matrix matrixX = new Matrix(x, x.length); Matrix matrixYAdjust =
			 * new Matrix(yAdjust, yAdjust.length); Matrix matrixY = new
			 * Matrix(y, y.length);
			 * 
			 * matrixY.minusEquals(matrixYAdjust);
			 * 
			 * //GaussianProcessMixture predictor =
			 * regression.calculateRegression(matrixX, matrixY);
			 */

			GaussianProcessMixture predictor;

			if (lastTimeSlot == -1) {
				predictor = regression.calculateRegression(new double[] {},
						new double[] {});
			} else {
				double x;
				double y;
				try {
					x = opponentTimes.get(opponentTimes.size() - 1);
					y = opponentUtilities.get(opponentUtilities.size() - 1);
				} catch (Exception ex) {
					System.out.println("Error getting x or y");
					throw new Error(ex);
				}

				predictor = regression.updateRegression(new Matrix(
						new double[] { x }, 1), new Matrix(new double[] { y
						- intercept - (gradient * x) }, 1));
			}

			GaussianProcessMixturePrediction prediction = predictor
					.calculatePrediction(timeSamples.transpose());

			// Store the means and variances
			means = prediction.getMean().plus(matrixTimeSamplesAdjust);
			variances = prediction.getVariance();

			log(means.transpose());
			log(variances.transpose());
		}

		Pair<Matrix, Matrix> acceptMatrices = generateProbabilityAccept(means,
				variances, time);
		Matrix probabilityAccept = acceptMatrices.fst;
		Matrix cumulativeAccept = acceptMatrices.snd;

		Matrix probabilityExpectedUtility = probabilityAccept
				.arrayTimes(utility);
		Matrix cumulativeExpectedUtility = cumulativeAccept.arrayTimes(utility);

		if (regressionUpdateRequired) {
			log(probabilityAccept);
			log(cumulativeAccept);
			log(probabilityExpectedUtility);
			log(cumulativeExpectedUtility);
		}

		Pair<Double, Double> bestAgreement = getExpectedBestAgreement(
				probabilityExpectedUtility, cumulativeExpectedUtility, time);
		double bestTime = bestAgreement.fst;
		double bestUtility = bestAgreement.snd;

		double targetUtility = lastRegressionUtility
				+ ((time - lastRegressionTime)
						* (bestUtility - lastRegressionUtility) / (bestTime - lastRegressionTime));

		log(time + "," + bestTime + "," + bestUtility + ","
				+ lastRegressionTime + "," + lastRegressionUtility + ","
				+ targetUtility);

		// Store the target utility and time
		lastRegressionUtility = targetUtility;
		lastRegressionTime = time;

		log("-->>> IAMhaggler 2011 <<<--");

		return limitConcession(targetUtility);
	}

	private double limitConcession(double targetUtility) {
		double limit = 1.0 - ((maxOfferedUtility - minOfferedUtility) + 0.1);
		if (limit > targetUtility) {
			log("Limiting concession to " + limit);
			return limit;
		}
		return targetUtility;
	}

	/**
	 * Generate an n-by-m matrix representing the effect of the discounting
	 * factor for a given utility-time combination. The combinations are given
	 * by the time and utility samples stored in timeSamples and utilitySamples
	 * respectively.
	 * 
	 * @param discountingFactor
	 *            The discounting factor, in the range (0, 1].
	 * @return An n-by-m matrix representing the discounted utilities.
	 */
	private Matrix generateDiscountingFunction(double discountingFactor) {
		double[] discountingSamples = timeSamples.getRowPackedCopy();
		double[][] m = new double[utilitySamples.getRowDimension()][timeSamples
				.getColumnDimension()];
		for (int i = 0; i < m.length; i++) {
			for (int j = 0; j < m[i].length; j++) {
				m[i][j] = Math.pow(discountingFactor, discountingSamples[j]);
			}
		}
		return new Matrix(m);
	}

	/**
	 * Generate an (n-1)-by-m matrix representing the probability of acceptance
	 * for a given utility-time combination. The combinations are given by the
	 * time and utility samples stored in timeSamples and utilitySamples
	 * respectively.
	 * 
	 * @param mean
	 *            The means, at each of the sample time points.
	 * @param variance
	 *            The variances, at each of the sample time points.
	 * @param time
	 *            The current time, in the range [0, 1].
	 * @return An (n-1)-by-m matrix representing the probability of acceptance.
	 */
	private Pair<Matrix, Matrix> generateProbabilityAccept(Matrix mean,
			Matrix variance, double time) {
		int i = 0;
		for (; i < timeSamples.getColumnDimension(); i++) {
			if (timeSamples.get(0, i) > time)
				break;
		}
		Matrix cumulativeAccept = new Matrix(utilitySamples.getRowDimension(),
				timeSamples.getColumnDimension(), 0);
		Matrix probabilityAccept = new Matrix(utilitySamples.getRowDimension(),
				timeSamples.getColumnDimension(), 0);

		double interval = 1.0 / utilitySamples.getRowDimension();

		for (; i < timeSamples.getColumnDimension(); i++) {
			double s = Math.sqrt(2 * variance.get(i, 0));
			double m = mean.get(i, 0);

			double minp = (1.0 - (0.5 * (1 + erf((utilitySamples.get(0, 0)
					+ (interval / 2.0) - m)
					/ s))));
			double maxp = (1.0 - (0.5 * (1 + erf((utilitySamples.get(
					utilitySamples.getRowDimension() - 1, 0) - (interval / 2.0) - m)
					/ s))));

			for (int j = 0; j < utilitySamples.getRowDimension(); j++) {
				double utility = utilitySamples.get(j, 0);
				double p = (1.0 - (0.5 * (1 + erf((utility - m) / s))));
				double p1 = (1.0 - (0.5 * (1 + erf((utility - (interval / 2.0) - m)
						/ s))));
				double p2 = (1.0 - (0.5 * (1 + erf((utility + (interval / 2.0) - m)
						/ s))));

				cumulativeAccept.set(j, i, (p - minp) / (maxp - minp));
				probabilityAccept.set(j, i, (p1 - p2) / (maxp - minp));
			}
		}
		return new Pair<Matrix, Matrix>(probabilityAccept, cumulativeAccept);
	}

	/**
	 * Wrapper for the erf function.
	 * 
	 * @param x
	 * @return
	 */
	private double erf(double x) {
		if (x > 6)
			return 1;
		if (x < -6)
			return -1;
		try {
			double d = Erf.erf(x);
			if (d > 1)
				return 1;
			if (d < -1)
				return -1;
			return d;
		} catch (MaxIterationsExceededException e) {
			if (x > 0)
				return 1;
			else
				return -1;
		} catch (MathException e) {
			e.printStackTrace();
			return 0;
		}
	}

	/**
	 * Generate an n-by-m matrix representing the risk based utility for a given
	 * utility-time combination. The combinations are given by the time and
	 * utility samples stored in timeSamples and utilitySamples
	 * 
	 * @param riskParameter
	 *            The risk parameter.
	 * @return an n-by-m matrix representing the risk based utility.
	 */
	protected Matrix generateRiskFunction(double riskParameter) {
		double mmin = generateRiskFunction(riskParameter, 0.0);
		double mmax = generateRiskFunction(riskParameter, 1.0);
		double range = mmax - mmin;

		double[] riskSamples = utilitySamples.getColumnPackedCopy();
		double[][] m = new double[utilitySamples.getRowDimension()][timeSamples
				.getColumnDimension()];
		for (int i = 0; i < m.length; i++) {
			double val;
			if (range == 0) {
				val = riskSamples[i];
			} else {
				val = (generateRiskFunction(riskParameter, riskSamples[i]) - mmin)
						/ range;
			}
			for (int j = 0; j < m[i].length; j++) {
				m[i][j] = val;
			}
		}
		return new Matrix(m);
	}

	/**
	 * Generate the risk based utility for a given actual utility.
	 * 
	 * @param riskParameter
	 *            The risk parameter.
	 * @param utility
	 *            The actual utility to calculate the risk based utility from.
	 * @return the risk based utility.
	 */
	protected double generateRiskFunction(double riskParameter, double utility) {
		return Math.pow(utility, riskParameter);
	}

	/**
	 * Get a pair representing the time and utility value of the expected best
	 * agreement.
	 * 
	 * @param expectedValues
	 *            A matrix of expected utility values at the sampled time and
	 *            utilities given by timeSamples and utilitySamples
	 *            respectively.
	 * @param time
	 *            The current time.
	 * @return a pair representing the time and utility value of the expected
	 *         best agreement.
	 */
	private Pair<Double, Double> getExpectedBestAgreement(
			Matrix probabilityExpectedValues, Matrix cumulativeExpectedValues,
			double time) {
		log("probabilityExpectedValues is "
				+ probabilityExpectedValues.getRowDimension() + "x"
				+ probabilityExpectedValues.getColumnDimension());
		log("time is " + time);
		Matrix probabilityFutureExpectedValues = getFutureExpectedValues(
				probabilityExpectedValues, time);
		Matrix cumulativeFutureExpectedValues = getFutureExpectedValues(
				cumulativeExpectedValues, time);

		log("probabilityFutureExpectedValues is "
				+ probabilityFutureExpectedValues.getRowDimension() + "x"
				+ probabilityFutureExpectedValues.getColumnDimension());

		double[][] probabilityFutureExpectedValuesArray = probabilityFutureExpectedValues
				.getArray();
		double[][] cumulativeFutureExpectedValuesArray = cumulativeFutureExpectedValues
				.getArray();

		Double bestX = null;
		Double bestY = null;

		double[] colSums = new double[probabilityFutureExpectedValuesArray[0].length];
		double bestColSum = 0;
		int bestCol = 0;

		for (int x = 0; x < probabilityFutureExpectedValuesArray[0].length; x++) {
			colSums[x] = 0;
			for (int y = 0; y < probabilityFutureExpectedValuesArray.length; y++) {
				colSums[x] += probabilityFutureExpectedValuesArray[y][x];
			}

			if (colSums[x] >= bestColSum) {
				bestColSum = colSums[x];
				bestCol = x;
			}
		}

		log(new Matrix(colSums, 1));

		int bestRow = 0;
		double bestRowValue = 0;

		for (int y = 0; y < cumulativeFutureExpectedValuesArray.length; y++) {
			double expectedValue = cumulativeFutureExpectedValuesArray[y][bestCol];
			if (expectedValue > bestRowValue) {
				bestRowValue = expectedValue;
				bestRow = y;
			}
		}

		bestX = timeSamples.get(0,
				bestCol + probabilityExpectedValues.getColumnDimension()
						- probabilityFutureExpectedValues.getColumnDimension());
		bestY = utilitySamples.get(bestRow, 0);

		log("About to return the best agreement at " + bestX + ", " + bestY);
		return new Pair<Double, Double>(bestX, bestY);
	}

	/**
	 * Get a matrix of expected utility values at the sampled time and utilities
	 * given by timeSamples and utilitySamples, for times in the future.
	 * 
	 * @param expectedValues
	 *            A matrix of expected utility values at the sampled time and
	 *            utilities given by timeSamples and utilitySamples
	 *            respectively.
	 * @param time
	 *            The current time.
	 * @return a matrix of expected utility values for future time.
	 */
	private Matrix getFutureExpectedValues(Matrix expectedValues, double time) {
		int i = 0;
		for (; i < timeSamples.getColumnDimension(); i++) {
			if (timeSamples.get(0, i) > time)
				break;
		}
		return expectedValues.getMatrix(0,
				expectedValues.getRowDimension() - 1, i,
				expectedValues.getColumnDimension() - 1);
	}
}
