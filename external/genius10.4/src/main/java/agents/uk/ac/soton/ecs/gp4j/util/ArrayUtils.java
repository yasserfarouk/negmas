package agents.uk.ac.soton.ecs.gp4j.util;

import agents.org.apache.commons.math.stat.StatUtils;

public class ArrayUtils extends agents.org.apache.commons.lang.ArrayUtils {
	public static double[][] combine(double[][] combinations, double[] addition) {
		double[][] result;

		if (combinations == null || combinations.length == 0) {
			result = new double[addition.length][1];

			for (int i = 0; i < addition.length; i++)
				result[i][0] = addition[i];
		} else {
			result = new double[combinations.length * addition.length][combinations[0].length + 1];

			for (int i = 0; i < combinations.length; i++) {
				for (int j = 0; j < addition.length; j++) {
					System.arraycopy(combinations[i], 0, result[i
							* addition.length + j], 0, combinations[0].length);
					result[i * addition.length + j][result[0].length - 1] = addition[j];
				}
			}

		}
		return result;
	}

	public static int[][] combine(int[][] combinations, int[] addition) {
		int[][] result;

		if (combinations == null || combinations.length == 0) {
			result = new int[addition.length][1];

			for (int i = 0; i < addition.length; i++)
				result[i][0] = addition[i];
		} else {
			result = new int[combinations.length * addition.length][combinations[0].length + 1];

			for (int i = 0; i < combinations.length; i++) {
				for (int j = 0; j < addition.length; j++) {
					System.arraycopy(combinations[i], 0, result[i
							* addition.length + j], 0, combinations[0].length);
					result[i * addition.length + j][result[0].length - 1] = addition[j];
				}
			}

		}
		return result;
	}

	public static int[][] allCombinations(int[][] values) {
		int result[][] = combine(null, values[0]);

		for (int i = 1; i < values.length; i++)
			result = combine(result, values[i]);

		return result;
	}

	public static int[][] allCombinations(int[] values, int repetitions) {
		int[][] input = new int[repetitions][values.length];

		for (int i = 0; i < input.length; i++) {
			input[i] = values;
		}

		return allCombinations(input);
	}

	public static double[][] allCombinations(double[][] values) {
		double result[][] = combine(null, values[0]);

		for (int i = 1; i < values.length; i++)
			result = combine(result, values[i]);

		return result;
	}

	public static double[] linspace(double min, double max, int sampleCount) {
		if (sampleCount == 1)
			return new double[] { (max - min) / 2 };

		double[] samples = new double[sampleCount];
		double increment = (max - min) / (sampleCount - 1);
		samples[0] = min;
		samples[samples.length - 1] = max;

		for (int i = 1; i < sampleCount - 1; i++)
			samples[i] = samples[i - 1] + increment;

		return samples;
	}

	public static double[] log(double[] array) {
		double[] result = new double[array.length];

		for (int i = 0; i < array.length; i++)
			result[i] = Math.log(array[i]);

		return result;
	}

	public static double[] exp(double[] array) {
		double[] result = new double[array.length];

		for (int i = 0; i < array.length; i++)
			result[i] = Math.exp(array[i]);

		return result;
	}

	public static double[] normalize(double[] array) {
		double sum = StatUtils.sum(array);
		return divide(array, sum);
	}

	public static double[] divide(double[] array, double scalar) {
		double[] result = new double[array.length];

		for (int i = 0; i < array.length; i++) {
			result[i] = array[i] / scalar;
		}

		return result;
	}
}
