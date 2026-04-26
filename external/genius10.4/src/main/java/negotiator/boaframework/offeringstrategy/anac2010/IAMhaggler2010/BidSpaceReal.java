package negotiator.boaframework.offeringstrategy.anac2010.IAMhaggler2010;

import java.util.ArrayList;

public class BidSpaceReal {

	/**
	 * @param continuousEvaluationFunctions
	 * @param continuousWeights
	 * @return
	 */
	public static ArrayList<ContinuousSection> getContinuousCombinations(ArrayList<ContinuousEvaluationFunction> continuousEvaluationFunctions,
			ArrayList<Double> continuousWeights) {
		ArrayList<ContinuousSection> continuousSections = new ArrayList<ContinuousSection>();

		for (int[] combination : getContinuousCombinationValues(continuousEvaluationFunctions)) {
			continuousSections.add(getContinuousSection(continuousEvaluationFunctions, continuousWeights, combination));
		}

		return continuousSections;
	}

	/**
	 * @param continuousEvaluationFunctions
	 * @param continuousWeights
	 * @param sectionNos
	 * @return
	 */
	private static ContinuousSection getContinuousSection(ArrayList<ContinuousEvaluationFunction> continuousEvaluationFunctions,
			ArrayList<Double> continuousWeights, int[] sectionNos) {
		double[] minBounds = new double[continuousEvaluationFunctions.size()];
		double[] maxBounds = new double[continuousEvaluationFunctions.size()];
		double[] normal = new double[continuousEvaluationFunctions.size()];
		double[] knownPoint1 = new double[continuousEvaluationFunctions.size()];
		double[] knownPoint2 = new double[continuousEvaluationFunctions.size()];
		double evalKnownPoint1 = 0;
		double evalKnownPoint2 = 0;
		int nonZeroWeightDimension = -1;
		for (int i = 0; i < continuousEvaluationFunctions.size(); i++) {
			if (continuousWeights.get(i) != 0) {
				nonZeroWeightDimension = i;
				break;
			}
		}
		for (int i = 0; i < continuousEvaluationFunctions.size(); i++) {
			double[] tmp = getSectionDimension(continuousEvaluationFunctions.get(i), continuousWeights.get(i), sectionNos[i]);
			minBounds[i] += tmp[0];
			maxBounds[i] += tmp[1];
			normal[i] = tmp[2];
			if (i == nonZeroWeightDimension) {
				knownPoint1[i] = tmp[0];
				evalKnownPoint1 += tmp[3];
				knownPoint2[i] = tmp[1];
				evalKnownPoint2 += tmp[4];
			} else {
				knownPoint1[i] = tmp[0];
				evalKnownPoint1 += tmp[3];
				knownPoint2[i] = tmp[0];
				evalKnownPoint2 += tmp[3];
			}
		}

		ContinuousSection continuousSection = new ContinuousSection();
		continuousSection.setMinBounds(minBounds);
		continuousSection.setMaxBounds(maxBounds);
		continuousSection.setNormal(normal);
		continuousSection.setKnownPoint1(knownPoint1, evalKnownPoint1);
		continuousSection.setKnownPoint2(knownPoint2, evalKnownPoint2);
		return continuousSection;
	}

	/**
	 * @param continuousEvaluationFunction
	 * @param continuousWeight
	 * @param sectionNo
	 * @return
	 */
	private static double[] getSectionDimension(ContinuousEvaluationFunction continuousEvaluationFunction, double continuousWeight, int sectionNo) {

		double normalPart;
		double lowerPart;
		double upperPart;

		ContinuousEvaluationSection section = continuousEvaluationFunction.getSection(sectionNo);
		lowerPart = section.getLowerBound();
		upperPart = section.getUpperBound();
		normalPart = section.getNormal(continuousWeight);

		double[] result = new double[5];
		result[0] = lowerPart;
		result[1] = upperPart;
		result[2] = normalPart;
		result[3] = section.getEvalLowerBound() * continuousWeight;
		result[4] = section.getEvalUpperBound() * continuousWeight;
		return result;
	}

	/**
	 * @param continuousEvaluationFunctions
	 * @return
	 */
	private static ArrayList<int[]> getContinuousCombinationValues(ArrayList<ContinuousEvaluationFunction> continuousEvaluationFunctions) {
		int[] space = new int[continuousEvaluationFunctions.size() + 1];
		space[0] = 1;
		int i = 0;

		// Work out how many 'sections' there are
		for (ContinuousEvaluationFunction continuousEvaluationFunction : continuousEvaluationFunctions) {
			i++;
			space[i] = space[i - 1] * continuousEvaluationFunction.getSectionCount();
		}

		// Work out what the combinations of spaces are
		return BidSpace.getCombinationValues(space);
	}

}
