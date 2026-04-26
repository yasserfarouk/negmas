package negotiator.boaframework.opponentmodel.nash;

import java.util.ArrayList;


/**
 * This class contains methods that can be used for statistical measurements on lists.
 * 
 * @author Roland van der Linden
 *
 */
public class StatisticsUtil
{
	/**
	 * This method adds the values in the list together and returns the sum.
	 * @param list The list containing the values.
	 * @return The sum of the values.
	 */
	public static double getSum(ArrayList<Double> list)
	{
		if(list == null)
			throw new IllegalArgumentException("The sum of a list that is null cannot be calculated.");
		
		double result = 0;
		
		for(Double d : list)
			result += d;
		
		return result;
	}
	
	/**
	 * This methods calculates the average value of the values in the list.
	 * @param list The list containing the values.
	 * @return The mean of the values.
	 */
	public static double getMean(ArrayList<Double> list)
	{
		if(list == null)
			throw new IllegalArgumentException("The mean of a list that is null cannot be calculated.");
		
		if(list.size() == 0)
			return 0;
		else
			return (getSum(list) / (double)list.size());
	}
	
	/**
	 * This method calculates the variance of the values in the list.
	 * @param list The list containing the values.
	 * @return The variance of the values.
	 */
	public static double getVariance(ArrayList<Double> list)
	{
		if(list == null)
			throw new IllegalArgumentException("The variance of a list that is null cannot be calculated.");
		
		if(list.size() == 0)
			return 0;
		else
		{
			double mean = getMean(list);
			double totalDifference = 0;
			for(Double d : list)
			{
				double absdiff = Math.abs(d - mean);
				double absdiff_squared = absdiff * absdiff;
				totalDifference += absdiff_squared;
			}
		
			return (totalDifference / (double)list.size());
		}
	}
	
	/**
	 * This method calculates the standard deviation of the values in the list.
	 * @param list The list containing the values.
	 * @return The standard deviation of the values.
	 */
	public static double getStandardDeviation(ArrayList<Double> list)
	{
		if(list == null)
			throw new IllegalArgumentException("The standardDeviation of a list that is null cannot be calculated.");
		
		if(list.size() == 0)
			return 0;
		else
		{
			double variance = getVariance(list);
			return Math.sqrt(variance);
		}
	}
}