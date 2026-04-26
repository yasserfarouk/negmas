package negotiator.boaframework.opponentmodel;

import java.util.ArrayList;

import agents.Jama.Matrix;
import agents.uk.ac.soton.ecs.gp4j.bmc.BasicPrior;
import agents.uk.ac.soton.ecs.gp4j.bmc.GaussianProcessMixture;
import agents.uk.ac.soton.ecs.gp4j.bmc.GaussianProcessMixturePrediction;
import agents.uk.ac.soton.ecs.gp4j.bmc.GaussianProcessRegressionBMC;
import agents.uk.ac.soton.ecs.gp4j.gp.covariancefunctions.CovarianceFunction;
import agents.uk.ac.soton.ecs.gp4j.gp.covariancefunctions.Matern3CovarianceFunction;
import agents.uk.ac.soton.ecs.gp4j.gp.covariancefunctions.NoiseCovarianceFunction;
import agents.uk.ac.soton.ecs.gp4j.gp.covariancefunctions.SumCovarianceFunction;
import genius.core.utility.AdditiveUtilitySpace;

/**
 * This class is a component of the ANAC 2011 agent IAMHaggler where it estimates
 * the concession rate of the opponent.  For more information on how it works see
 * "Using Gaussian Processes to Optimise Concession in Complex Negotiations against Unknown Opponents"
 * by Colins et al.
 * @author Alex Dirkzwager
 *
 */

public class IAMHagglerOpponentConcessionModel {
	
	private Matrix timeSamples;
	private Matrix means;
	private Matrix variances;
	private GaussianProcessRegressionBMC regression;
	private int lastTimeSlot = -1;
	private ArrayList<Double> opponentTimes = new ArrayList<Double>();
	private ArrayList<Double> opponentUtilities = new ArrayList<Double>();
	private double intercept;
	private double maxUtilityInTimeSlot;
	private Matrix matrixTimeSamplesAdjust;
	private int slots;



	/**
	 * 
	 * @param numberOfSlots (within the total negotiation
	 * @param utilitySpace
	 */
	public IAMHagglerOpponentConcessionModel(int numberOfSlots, AdditiveUtilitySpace utilitySpace, int amountOfSamples){
		double discountingFactor = 0.5;
		try
		{
			discountingFactor = utilitySpace.getDiscountFactor();
		}
		catch(Exception ex)
		{
			ex.printStackTrace();
		}
		if(discountingFactor == 0)
			discountingFactor = 1;
		makeTimeSamples(amountOfSamples);
		this.slots = numberOfSlots;
		BasicPrior[] bps = { new BasicPrior(11, 0.252, 0.5),
				new BasicPrior(11, 0.166, 0.5), new BasicPrior(1, .01, 1.0) };
		CovarianceFunction cf = new SumCovarianceFunction(
				Matern3CovarianceFunction.getInstance(),
				NoiseCovarianceFunction.getInstance());
		
		regression = new GaussianProcessRegressionBMC();
		regression.setCovarianceFunction(cf);
		regression.setPriors(bps);
	}
	
	/**
	 * updates the model with the most recent opponent bid
	 * @param opponentUtility
	 * @param time
	 */
	public void updateModel(double opponentUtility, double time) {
		// Calculate the current time slot
		int timeSlot = (int) Math.floor(time * slots);
		
		boolean regressionUpdateRequired = false;
		if (lastTimeSlot == -1) {
			regressionUpdateRequired = true;
		}

		// If the time slot has changed
		if (timeSlot != lastTimeSlot) {
			if (lastTimeSlot != -1) {
				// Store the data from the time slot
				opponentTimes.add((lastTimeSlot + 0.5) / slots);
				if(opponentUtilities.size() == 0)
				{
					intercept = Math.max(0.5, maxUtilityInTimeSlot);
					double[] timeSamplesAdjust = new double[timeSamples.getColumnDimension()];
					System.out.println("timeSampleAdjusted[15]: " + timeSamplesAdjust.length);
					
					
					int i = 0;
					double gradient = 0.9 - intercept;
					for (double d : timeSamples.getRowPackedCopy()) {
						timeSamplesAdjust[i++] = intercept + (gradient * d);
					}
					matrixTimeSamplesAdjust = new Matrix(timeSamplesAdjust, timeSamplesAdjust.length);
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
		
		// Calculate the maximum utility observed in the current time slot
		maxUtilityInTimeSlot = Math.max(maxUtilityInTimeSlot, opponentUtility);

		if (timeSlot == 0) {
			return;
		}

		if (regressionUpdateRequired && opponentTimes.size() > 0) {
			double gradient = 0.9 - intercept;

			GaussianProcessMixture predictor;

			if(lastTimeSlot == -1)
			{
				predictor = regression.calculateRegression(new double[] {}, new double[] {});
			}
			else
			{
				double x;
				double y;
				try {
				x = opponentTimes.get(opponentTimes.size() - 1);
				y = opponentUtilities.get(opponentUtilities.size() - 1);

				} catch(Exception ex) {
					System.out.println("Error getting x or y");
					throw new Error(ex);
				}
			
				predictor = regression.updateRegression(
						new Matrix(new double[] {x}, 1),
						new Matrix(new double[] {y - intercept - (gradient * x)}, 1));
			}

			GaussianProcessMixturePrediction prediction = predictor
					.calculatePrediction(timeSamples.transpose());

			// Store the means and variances
			means = prediction.getMean().plus(matrixTimeSamplesAdjust);
			
			variances = prediction.getVariance();
		}
	}
	
	/**
	 * Gets all means in a n-by-1 Matrix
	 * @return
	 */
	public Matrix getMeans(){
		return means;
	}
	
	/**
	 * Gets a specific mean point corresponding to the timeSlot
	 * @param timeSlot
	 * @return
	 */
	public double getMeanAt(int timeSlot){
		return means.get(timeSlot, 0);
	}
	
	/**
	 * Gets all Variances in a n-by-1 Matrix
	 * @return
	 */
	public Matrix getVariance(){
		return variances;
	}
	
	/**
	 * Gets a specific variance point corresponding to the timeSlot
	 * @param timeSlot
	 * @return
	 */
	public double getVarianceAt(int timeSlot){
		return variances.get(timeSlot, 0);
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
		System.out.println("timeSampleArray[15]: " + timeSamplesArray.length);
		timeSamples = new Matrix(timeSamplesArray, 1);
	}
}
