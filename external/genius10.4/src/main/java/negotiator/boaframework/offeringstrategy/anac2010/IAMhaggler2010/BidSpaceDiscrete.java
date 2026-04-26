package negotiator.boaframework.offeringstrategy.anac2010.IAMhaggler2010;

import java.util.ArrayList;
import java.util.HashMap;

import genius.core.issue.ValueDiscrete;

public class BidSpaceDiscrete {

	/**
	 * @param discreteEvaluationFunctions
	 * @param discreteWeights
	 * @return
	 */
	public static ValueDiscrete[][] getDiscreteCombinations(ArrayList<HashMap<ValueDiscrete, Double>> discreteEvaluationFunctions,
			ArrayList<Double> discreteWeights) {
		return getDiscrete(getDiscreteCombinationValues(discreteEvaluationFunctions), discreteEvaluationFunctions, discreteWeights);
	}

	/**
	 * @param discreteCombinationValues
	 * @param discreteEvaluationFunctions
	 * @param discreteWeights
	 * @return
	 */
	private static ValueDiscrete[][] getDiscrete(ArrayList<int[]> discreteCombinationValues,
			ArrayList<HashMap<ValueDiscrete, Double>> discreteEvaluationFunctions, ArrayList<Double> discreteWeights) {
		ValueDiscrete[][] result = new ValueDiscrete[discreteCombinationValues.size()][discreteEvaluationFunctions.size()];

		int i = 0;
		for (int[] discreteCombinationValue : discreteCombinationValues) {
			for (int j = 0; j < discreteEvaluationFunctions.size(); j++) {
				result[i][j] = (ValueDiscrete) discreteEvaluationFunctions.get(j).keySet().toArray()[discreteCombinationValue[j]];
			}
			i++;
		}

		return result;
	}

	/**
	 * @param discreteEvaluationFunctions
	 * @return
	 */
	private static ArrayList<int[]> getDiscreteCombinationValues(ArrayList<HashMap<ValueDiscrete, Double>> discreteEvaluationFunctions) {
		int[] space = new int[discreteEvaluationFunctions.size() + 1];
		space[0] = 1;
		int i = 0;

		// Work out how many 'sections' there are
		for (HashMap<ValueDiscrete, Double> discreteEvaluationFunction : discreteEvaluationFunctions) {
			i++;
			space[i] = space[i - 1] * discreteEvaluationFunction.size();
		}

		// Work out what the combinations of spaces are
		return BidSpace.getCombinationValues(space);
	}
}
