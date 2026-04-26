package agents.anac.y2010.Southampton.analysis;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;

import agents.anac.y2010.Southampton.utils.OpponentModel;
import agents.anac.y2010.Southampton.utils.Pair;
import agents.anac.y2010.Southampton.utils.concession.ConcessionUtils;
import genius.core.Bid;
import genius.core.Domain;
import genius.core.issue.ISSUETYPE;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.Value;
import genius.core.issue.ValueDiscrete;
import genius.core.issue.ValueInteger;
import genius.core.issue.ValueReal;
import genius.core.utility.AdditiveUtilitySpace;
import genius.core.utility.EvaluatorDiscrete;
import genius.core.utility.EvaluatorInteger;
import genius.core.utility.EvaluatorReal;

public class BidSpace {

	private final boolean TEST_EQUIVALENCE = false;

	/**
	 * @return the continuousWeights
	 */
	public ArrayList<Double> getContinuousWeights() {
		return continuousWeights;
	}

	/**
	 * @return the count of discreteCombinations
	 */
	public int getDiscreteCombinationsCount() {
		return discreteCombinations.length;
	}

	/**
	 * @return the continuousWeightsZero
	 */
	public boolean isContinuousWeightsZero() {
		return continuousWeightsZero;
	}

	/**
	 * @return the discreteWeights
	 */
	public ArrayList<Double> getDiscreteWeights() {
		return discreteWeights;
	}

	public class EvaluatedDiscreteCombination implements
			Comparable<EvaluatedDiscreteCombination> {

		/**
		 * @return the discreteCombination
		 */
		public ValueDiscrete[] getDiscreteCombination() {
			return discreteCombination;
		}

		private ValueDiscrete[] discreteCombination;
		private double jointUtility;

		public EvaluatedDiscreteCombination(
				ValueDiscrete[] discreteCombination, double jointUtility) {
			this.discreteCombination = discreteCombination;
			this.jointUtility = jointUtility;
		}

		public int compareTo(EvaluatedDiscreteCombination o) {
			return this.jointUtility < o.jointUtility ? -1
					: (this.jointUtility == o.jointUtility ? 0 : 1);
		}
	}

	private Domain domain;

	ArrayList<Double> continuousWeights;
	double[] continuousPreference;
	double[] range;
	double[] offset;
	ArrayList<ContinuousSection> continuousSections;
	ValueDiscrete[][] discreteCombinations;
	ISSUETYPE[] issueTypes;
	List<Issue> issues;

	ArrayList<EvaluatorDiscrete> discreteEvaluators;
	ArrayList<EvaluatorReal> realEvaluators;
	ArrayList<EvaluatorInteger> integerEvaluators;

	private boolean continuousWeightsZero;

	private ArrayList<HashMap<ValueDiscrete, Double>> discreteEvaluationFunctions;
	private ArrayList<Double> discreteWeights;

	private double discountFactor;

	/**
	 * Build the bid space based on a utility space.
	 * 
	 * @param space
	 *            the utility space.
	 * @throws Exception
	 */
	public BidSpace(AdditiveUtilitySpace space) throws Exception {

		domain = space.getDomain();

		discountFactor = space.getDiscountFactor();
		if (!space.isDiscounted()) {
			discountFactor = 0.0; // in 2011 no discount had value 0.0 instead
									// of 1.0
		}

		issues = domain.getIssues();
		double[] weights = new double[issues.size()];
		issueTypes = new ISSUETYPE[issues.size()];

		discreteEvaluationFunctions = new ArrayList<HashMap<ValueDiscrete, Double>>();
		discreteWeights = new ArrayList<Double>();

		ArrayList<ContinuousEvaluationFunction> continuousEvaluationFunctions = new ArrayList<ContinuousEvaluationFunction>();
		continuousWeights = new ArrayList<Double>();
		continuousWeightsZero = true;
		ArrayList<Double> tmpContinuousPreference = new ArrayList<Double>();

		ArrayList<Double> tmpRange = new ArrayList<Double>();
		ArrayList<Double> tmpOffset = new ArrayList<Double>();

		realEvaluators = new ArrayList<EvaluatorReal>();
		integerEvaluators = new ArrayList<EvaluatorInteger>();
		discreteEvaluators = new ArrayList<EvaluatorDiscrete>();

		int i = 0;
		for (Issue issue : issues) {
			weights[i] = space.getWeight(issue.getNumber());
			issueTypes[i] = issue.getType();
			switch (issueTypes[i]) {
			case DISCRETE:
				IssueDiscrete issueDiscrete = (IssueDiscrete) issue;
				List<ValueDiscrete> values = issueDiscrete.getValues();
				HashMap<ValueDiscrete, Double> evaluationFunction = new HashMap<ValueDiscrete, Double>();
				EvaluatorDiscrete evaluatorDiscrete = (EvaluatorDiscrete) space
						.getEvaluator(issue.getNumber());
				discreteEvaluators.add(evaluatorDiscrete);
				for (ValueDiscrete value : values) {
					evaluationFunction.put(value,
							evaluatorDiscrete.getEvaluation(value));
				}
				discreteEvaluationFunctions.add(evaluationFunction);
				discreteWeights.add(space.getWeight(issue.getNumber()));
				break;
			case REAL:
				EvaluatorReal evaluatorReal = (EvaluatorReal) space
						.getEvaluator(issue.getNumber());
				realEvaluators.add(evaluatorReal);

				tmpRange.add(evaluatorReal.getUpperBound()
						- evaluatorReal.getLowerBound());
				tmpOffset.add(evaluatorReal.getLowerBound());

				ArrayList<RealEvaluationSection> realSections = new ArrayList<RealEvaluationSection>();
				switch (evaluatorReal.getFuncType()) {
				case LINEAR: {
					double lb = normalise(evaluatorReal.getLowerBound(),
							evaluatorReal.getLowerBound(),
							evaluatorReal.getUpperBound());
					double ub = normalise(evaluatorReal.getUpperBound(),
							evaluatorReal.getLowerBound(),
							evaluatorReal.getUpperBound());
					RealEvaluationSection res = new RealEvaluationSection(lb,
							evaluatorReal.getEvaluation(evaluatorReal
									.getLowerBound()), ub,
							evaluatorReal.getEvaluation(evaluatorReal
									.getUpperBound()));
					realSections.add(res);
					tmpContinuousPreference.add(res.getTopPoint());
					break;
				}
				case TRIANGULAR: {
					double lb = normalise(evaluatorReal.getLowerBound(),
							evaluatorReal.getLowerBound(),
							evaluatorReal.getUpperBound());
					double tp = normalise(evaluatorReal.getTopParam(),
							evaluatorReal.getLowerBound(),
							evaluatorReal.getUpperBound());
					double ub = normalise(evaluatorReal.getUpperBound(),
							evaluatorReal.getLowerBound(),
							evaluatorReal.getUpperBound());
					realSections
							.add(new RealEvaluationSection(lb, evaluatorReal
									.getEvaluation(evaluatorReal
											.getLowerBound()), tp,
									evaluatorReal.getEvaluation(evaluatorReal
											.getTopParam())));
					realSections.add(new RealEvaluationSection(tp,
							evaluatorReal.getEvaluation(evaluatorReal
									.getTopParam()), ub, evaluatorReal
									.getEvaluation(evaluatorReal
											.getUpperBound())));
					tmpContinuousPreference.add(tp);
					break;
				}
				}
				continuousEvaluationFunctions
						.add(new ContinuousEvaluationFunction(realSections,
								space.getWeight(issue.getNumber())));
				if (space.getWeight(issue.getNumber()) > 0)
					continuousWeightsZero = false;
				continuousWeights.add(space.getWeight(issue.getNumber()));
				break;
			case INTEGER:
				EvaluatorInteger evaluatorInteger = (EvaluatorInteger) space
						.getEvaluator(issue.getNumber());
				integerEvaluators.add(evaluatorInteger);

				tmpRange.add((double) evaluatorInteger.getUpperBound()
						- evaluatorInteger.getLowerBound());
				tmpOffset.add((double) evaluatorInteger.getLowerBound());

				ArrayList<IntegerEvaluationSection> integerSections = new ArrayList<IntegerEvaluationSection>();
				switch (evaluatorInteger.getFuncType()) {
				case LINEAR: {
					int lb = normalise(evaluatorInteger.getLowerBound(),
							evaluatorInteger.getLowerBound(),
							evaluatorInteger.getUpperBound());
					int ub = normalise(evaluatorInteger.getUpperBound(),
							evaluatorInteger.getLowerBound(),
							evaluatorInteger.getUpperBound());
					IntegerEvaluationSection ies = new IntegerEvaluationSection(
							lb, evaluatorInteger.getEvaluation(evaluatorInteger
									.getLowerBound()), ub,
							evaluatorInteger.getEvaluation(evaluatorInteger
									.getUpperBound()));
					integerSections.add(ies);
					tmpContinuousPreference.add(ies.getTopPoint());
					break;
				}
				}
				continuousEvaluationFunctions
						.add(new ContinuousEvaluationFunction(integerSections,
								space.getWeight(issue.getNumber())));
				if (space.getWeight(issue.getNumber()) > 0)
					continuousWeightsZero = false;
				continuousWeights.add(space.getWeight(issue.getNumber()));
				break;
			}
			i++;
		}

		range = new double[tmpRange.size()];
		for (i = 0; i < range.length; i++) {
			range[i] = tmpRange.get(i);
		}

		offset = new double[tmpOffset.size()];
		for (i = 0; i < offset.length; i++) {
			offset[i] = tmpOffset.get(i);
		}

		continuousPreference = new double[tmpContinuousPreference.size()];
		for (i = 0; i < continuousPreference.length; i++) {
			continuousPreference[i] = tmpContinuousPreference.get(i);
		}

		// Print out the discrete issues.
		// BidSpacePrinter.printDiscreteIssues(discreteEvaluationFunctions);

		// Print out the continuous issues.
		// BidSpacePrinter.printContinuousIssues(continuousEvaluationFunctions);

		// Work out what the combinations of the discrete issues are...
		discreteCombinations = BidSpaceDiscrete.getDiscreteCombinations(
				discreteEvaluationFunctions, discreteWeights);

		// Work out what the continuous sections are...
		continuousSections = BidSpaceReal.getContinuousCombinations(
				continuousEvaluationFunctions, continuousWeights);
	}

	public double getBeta(
			ArrayList<Pair<Double, Double>> bestOpponentBidUtilityHistory,
			double time, double utility0, double utility1) {
		return ConcessionUtils.getBeta(bestOpponentBidUtilityHistory, time,
				discountFactor, utility0, utility1);
	}

	public double getBeta(
			ArrayList<Pair<Double, Double>> bestOpponentBidUtilityHistory,
			double time, double utility0, double utility1,
			double minDiscounting, double minBeta, double maxBeta,
			double defaultBeta, double ourTime, double opponentTime) {
		return ConcessionUtils.getBeta(bestOpponentBidUtilityHistory, time,
				discountFactor, utility0, utility1, minDiscounting, minBeta,
				maxBeta, defaultBeta, ourTime, opponentTime);
	}

	private double normalise(double value, double lowerBound, double upperBound) {
		return (value - lowerBound) / (upperBound - lowerBound);
	}

	private int normalise(int value, double lowerBound, double upperBound) {
		return (int) normalise((double) value, lowerBound, upperBound);
	}

	/**
	 * Get a point in an iso-utility space.
	 * 
	 * @param utility
	 *            the utility of the iso-utility space.
	 * @param normal
	 *            the normal to the space.
	 * @return a point in an iso-utility space.
	 */
	private double[] getPointOnLine(double utility, double[] normal,
			double utilityA, double[] pointA, double utilityB, double[] pointB) {
		if (utilityA == utilityB)
			throw new AssertionError("utilityA must not equal utilityB");

		double m = (utility - utilityA) / (utilityB - utilityA);

		double[] pointX = new double[normal.length];

		for (int i = 0; i < normal.length; i++) {
			pointX[i] = pointA[i] + m * (pointB[i] - pointA[i]);
		}

		return pointX;
	}

	/**
	 * Project a point onto an iso-utility space.
	 * 
	 * @param pointToProject
	 *            the point to project.
	 * @param utility
	 *            the utility of the iso-utility space.
	 * @param opponentModel
	 * @param utilitySpace
	 * @return an array list of bids that lie closest to the point (for all
	 *         combinations of discrete values) and have the given utility.
	 */
	public ArrayList<Bid> Project(double[] pointToProject, double utility,
			int limit, AdditiveUtilitySpace utilitySpace,
			OpponentModel opponentModel) {
		ArrayList<Bid> bids = new ArrayList<Bid>();
		if (discreteCombinations.length == 0) {
			ArrayList<double[]> tmpPoints = new ArrayList<double[]>();
			for (ContinuousSection continuousSection : continuousSections) {
				Project(tmpPoints, pointToProject, utility, continuousSection,
						null, range);
			}
			for (double[] point : getClosestPoints(tmpPoints, pointToProject)) {
				addUniqueBid(bids, createBid(point, null));
			}
		} else {
			ValueDiscrete[][] bestCombinations = getBestCombinations(
					discreteCombinations, limit, continuousPreference,
					utilitySpace, opponentModel);

			for (ValueDiscrete[] discreteCombination : bestCombinations) {
				ArrayList<double[]> tmpPoints = new ArrayList<double[]>();
				if (continuousWeightsZero) {
					if (evaluate(discreteCombination,
							discreteEvaluationFunctions, discreteWeights) >= utility) {
						Project(tmpPoints, pointToProject, 0, null, null, null);
					}
				} else {
					for (ContinuousSection continuousSection : continuousSections) {
						Project(tmpPoints,
								pointToProject,
								utility
										- evaluate(discreteCombination,
												discreteEvaluationFunctions,
												discreteWeights),
								continuousSection, discreteCombination, range);
					}
				}
				for (double[] point : getClosestPoints(tmpPoints,
						pointToProject)) {
					addUniqueBid(bids, createBid(point, discreteCombination));
				}
			}
		}

		return bids;
	}

	private double evaluate(
			ValueDiscrete[] discreteCombination,
			ArrayList<HashMap<ValueDiscrete, Double>> discreteEvaluationFunctions,
			ArrayList<Double> discreteWeights) {
		double value = 0;
		for (int j = 0; j < discreteCombination.length; j++) {
			value += discreteWeights.get(j)
					* discreteEvaluationFunctions.get(j).get(
							discreteCombination[j]);
		}
		return value;
	}

	private ValueDiscrete[][] getBestCombinations(
			ValueDiscrete[][] discreteCombinations, int limit,
			double[] continuousPreference, AdditiveUtilitySpace utilitySpace,
			OpponentModel opponentModel) {
		if (limit == 0 || limit >= discreteCombinations.length)
			return discreteCombinations;
		List<EvaluatedDiscreteCombination> options = new ArrayList<EvaluatedDiscreteCombination>();
		for (ValueDiscrete[] discreteCombination : discreteCombinations) {
			Bid b = createBid(continuousPreference, discreteCombination);
			double jointUtility = 0;
			try {
				jointUtility = utilitySpace.getUtility(b)
						+ opponentModel.getNormalizedUtility(b);
			} catch (Exception e) {
				e.printStackTrace();
			}
			options.add(new EvaluatedDiscreteCombination(discreteCombination,
					jointUtility));
		}
		Collections.sort(options);
		options = options.subList(options.size() - limit, options.size());
		ValueDiscrete[][] bestCombinations = new ValueDiscrete[limit][discreteCombinations[0].length];
		int i = 0;
		for (EvaluatedDiscreteCombination edc : options) {
			bestCombinations[i] = edc.getDiscreteCombination();
			i++;
		}
		return bestCombinations;
	}

	/**
	 * @param points
	 * @param pointToProject
	 * @param utility
	 * @param continuousSection
	 * @param discreteCombination
	 */
	private void Project(ArrayList<double[]> points, double[] pointToProject,
			double utility, ContinuousSection continuousSection,
			ValueDiscrete[] discreteCombination, double[] range) {

		if (continuousSection == null) {
			addUniquePoint(points, pointToProject);
			return;
		}

		double[] pointA = continuousSection.getKnownPoint1();
		double[] pointB = continuousSection.getKnownPoint2();

		double utilityA = continuousSection.getEvalKnownPoint1();
		double utilityB = continuousSection.getEvalKnownPoint2();

		double[] pointOnLine = getPointOnLine(utility,
				continuousSection.getNormal(), utilityA, pointA, utilityB,
				pointB);
		double[] projectedPoint = Project(pointToProject,
				continuousSection.getNormal(), pointOnLine);
		if (WithinConstraints(projectedPoint, continuousSection.getMinBounds(),
				continuousSection.getMaxBounds())) {

			addUniquePoint(points, projectedPoint);
		} else {
			projectedPoint = getEndPoint(continuousSection.getMinBounds(),
					continuousSection.getMaxBounds(),
					continuousSection.getNormal(), utility,
					discreteCombination, pointToProject, utilityA, pointA,
					utilityB, pointB, range);
			if (projectedPoint != null) {
				addUniquePoint(points, projectedPoint);
			}
		}
	}

	/**
	 * Project a point onto a hyperplane.
	 * 
	 * @param pointToProject
	 *            the point to project onto the hyperplane.
	 * @param normal
	 *            the normal to the hyperplane.
	 * @param pointOnLine
	 *            a point on the hyperplane.
	 * @return the projected point.
	 */
	private double[] Project(double[] pointToProject, double[] normal,
			double[] pointOnLine) {
		if (pointToProject.length != normal.length)
			throw new AssertionError(
					"Lengths of pointToProject and normal do not match");
		if (pointOnLine.length != normal.length)
			throw new AssertionError(
					"Lengths of pointOnLine and normal do not match");

		int dimensions = pointToProject.length;

		double projectedPoint[] = new double[dimensions];
		double suma = 0;
		double sumb = 0;
		double sumc = 0;
		for (int i = 0; i < dimensions; i++) {
			suma += (normal[i] * pointOnLine[i]);
			sumb += (normal[i] * pointToProject[i]);
			sumc += (normal[i] * normal[i]);
		}
		double sum = (suma - sumb) / sumc;
		for (int i = 0; i < dimensions; i++) {
			projectedPoint[i] = pointToProject[i] + (sum * normal[i]);
		}
		return projectedPoint;
	}

	/**
	 * Add a bid to an array list of bids, but only if the array list does not
	 * already contain an identical bid.
	 * 
	 * @param bids
	 *            the array list of bids.
	 * @param bid
	 *            the bid to try to add.
	 */
	private void addUniquePoint(ArrayList<double[]> points, double[] point) {
		for (double[] p : points) {
			if (p.equals(point)) {
				return;
			}
		}
		points.add(point);
	}

	/**
	 * Add a bid to an array list of bids, but only if the array list does not
	 * already contain an identical bid.
	 * 
	 * @param bids
	 *            the array list of bids.
	 * @param bid
	 *            the bid to try to add.
	 */
	private void addUniqueBid(ArrayList<Bid> bids, Bid bid) {
		for (Bid b : bids) {
			if (b.equals(bid)) {
				return;
			}
		}
		bids.add(bid);
	}

	/**
	 * Get the endpoints of a bounded hyperplane that are closest to a target
	 * point.
	 * 
	 * @param min
	 *            the minimum bound of the space.
	 * @param max
	 *            the maximum bound of the space.
	 * @param normal
	 *            the normal to the hyperplane.
	 * @param utility
	 *            the utility value of the hyperplane.
	 * @param discreteCombination
	 *            the combination of discrete values to use in the bids.
	 * @param target
	 *            the target point.
	 * @return the endpoints of a bounded hyperplane that are closest to a
	 *         target point.
	 */
	private double[] getEndPoint(double[] min, double[] max, double[] normal,
			double utility, ValueDiscrete[] discreteCombination,
			double[] target, double utilityA, double[] pointA, double utilityB,
			double[] pointB, double[] range) {
		if (min.length != normal.length)
			throw new AssertionError("Lengths of min and normal do not match");
		if (max.length != normal.length)
			throw new AssertionError("Lengths of max and normal do not match");
		if (pointA.length != normal.length)
			throw new AssertionError(
					"Lengths of pointA and normal do not match");
		if (pointB.length != normal.length)
			throw new AssertionError(
					"Lengths of pointB and normal do not match");
		if (target.length != normal.length)
			throw new AssertionError(
					"Lengths of target and normal do not match");

		int dimension = normal.length;

		double[] pointOnLine = getHillClimbStartPoint(min, max, normal,
				utility, discreteCombination, target, utilityA, pointA,
				utilityB, pointB);

		if (pointOnLine == null)
			return null;

		if (!WithinConstraints(pointOnLine, min, max)) {
			throw new AssertionError("Get Intersection fail");
		}

		for (int precision = 1; precision < 5; precision++)
			while (true) {
				double step = Math.pow(0.1, precision);
				ArrayList<double[]> nearbyPointsOnLine = new ArrayList<double[]>();
				nearbyPointsOnLine.add(pointOnLine);
				Double[] nearbyPoint = new Double[dimension];
				double[] proposedPoint;
				for (int shiftDimension = 0; shiftDimension < dimension; shiftDimension++) {
					for (int unknownDimension = 0; unknownDimension < dimension; unknownDimension++) {
						if (shiftDimension == unknownDimension)
							continue;
						for (int i = 0; i < dimension; i++) {
							nearbyPoint[i] = pointOnLine[i];
						}
						nearbyPoint[unknownDimension] = null;

						nearbyPoint[shiftDimension] += step
								* range[shiftDimension];
						proposedPoint = getIntersection(nearbyPoint, normal,
								pointOnLine);
						if (WithinConstraints(proposedPoint, min, max))
							nearbyPointsOnLine.add(proposedPoint);

						nearbyPoint[shiftDimension] -= 2 * step
								* range[shiftDimension];
						proposedPoint = getIntersection(nearbyPoint, normal,
								pointOnLine);
						if (WithinConstraints(proposedPoint, min, max))
							nearbyPointsOnLine.add(proposedPoint);
					}
				}

				ArrayList<double[]> closestPoints = getClosestPoints(
						nearbyPointsOnLine, target);
				if (closestPoints.size() == 0)
					break;

				double[] closestPoint;
				if (TEST_EQUIVALENCE) {
					closestPoint = closestPoints.get(0);
				} else {
					closestPoint = closestPoints.get((int) Math.random()
							* closestPoints.size());
				}

				if (getDistance(closestPoint, target) == getDistance(
						pointOnLine, target))
					break;
				else
					pointOnLine = closestPoint;
			}
		return pointOnLine;
	}

	private double[] getHillClimbStartPoint(double[] min, double[] max,
			double[] normal, double utility,
			ValueDiscrete[] discreteCombination, double[] target,
			double utilityA, double[] pointA, double utilityB, double[] pointB) {
		if (min.length != normal.length)
			throw new AssertionError("Lengths of min and normal do not match");
		if (max.length != normal.length)
			throw new AssertionError("Lengths of max and normal do not match");
		if (pointA.length != normal.length)
			throw new AssertionError(
					"Lengths of pointA and normal do not match");
		if (pointB.length != normal.length)
			throw new AssertionError(
					"Lengths of pointB and normal do not match");
		if (target.length != normal.length)
			throw new AssertionError(
					"Lengths of target and normal do not match");

		ArrayList<Double[]> bounds = getBounds(min, max);

		ArrayList<double[]> endPoints = new ArrayList<double[]>();

		double[] pointOnLine = getPointOnLine(utility, normal, utilityA,
				pointA, utilityB, pointB);

		for (Double[] bound : bounds) {
			double[] endPoint = getIntersection(bound, normal, pointOnLine);
			if (WithinConstraints(endPoint, min, max)) {
				endPoints.add(endPoint);
			}
		}

		ArrayList<double[]> closestPoints = getClosestPoints(endPoints, target);
		if (closestPoints.size() == 0)
			return null;
		return closestPoints.get((int) Math.random() * closestPoints.size());
	}

	private ArrayList<double[]> getClosestPoints(ArrayList<double[]> endPoints,
			double[] target) {
		double closestDistance = Double.MAX_VALUE;

		for (double[] endPoint : endPoints) {
			double distance = getDistance(endPoint, target);
			closestDistance = Math.min(closestDistance, distance);
		}

		ArrayList<double[]> closestEndPoints = new ArrayList<double[]>();

		for (double[] endPoint : endPoints) {
			if (getDistance(endPoint, target) == closestDistance) {
				closestEndPoints.add(endPoint);
			}
		}

		return closestEndPoints;
	}

	/**
	 * Get the distance between two points in multi-dimensional space.
	 * 
	 * @param pointA
	 *            the first point.
	 * @param pointB
	 *            the second point.
	 * @return the distance between two points in multi-dimensional space.
	 */
	private double getDistance(double[] pointA, double[] pointB) {
		if (pointA.length != pointB.length)
			throw new AssertionError(
					"Lengths of pointA and pointB do not match");
		if (pointA.length != range.length)
			throw new AssertionError("Lengths of pointA and range do not match");

		double distance = 0;
		for (int i = 0; i < pointB.length; i++) {
			distance += Math.pow((pointA[i] - pointB[i]), 2);
		}
		return Math.sqrt(distance);
	}

	/**
	 * Create a bid.
	 * 
	 * @param point
	 *            the point in the multi-dimensional space that represents the
	 *            continuous issues of the domain.
	 * @param discreteCombination
	 *            the combination of discrete values.
	 * @return a bid.
	 */
	private Bid createBid(double[] point, ValueDiscrete[] discreteCombination) {
		HashMap<Integer, Value> bidInternals = new HashMap<Integer, Value>();
		int continuousPos = 0;
		int discretePos = 0;
		for (int i = 0; i < issueTypes.length; i++) {
			if (issueTypes[i] == ISSUETYPE.REAL) {
				bidInternals.put(issues.get(i).getNumber(), new ValueReal(
						(point[continuousPos] * range[continuousPos])
								+ offset[continuousPos]));
				continuousPos++;
			} else if (issueTypes[i] == ISSUETYPE.INTEGER) {
				bidInternals
						.put(issues.get(i).getNumber(),
								new ValueInteger(
										(int) Math
												.round((point[continuousPos] * range[continuousPos])
														+ offset[continuousPos])));
				continuousPos++;
			} else if (issueTypes[i] == ISSUETYPE.DISCRETE) {
				bidInternals.put(issues.get(i).getNumber(),
						discreteCombination[discretePos]);
				discretePos++;
			}
		}
		try {
			return new Bid(domain, bidInternals);
		} catch (Exception e) {
			return null;
		}
	}

	/**
	 * Get the intersection between a bound and a hyperplane.
	 * 
	 * @param bound
	 *            the bound.
	 * @param normal
	 *            the normal to the hyperplane.
	 * @param utility
	 *            the utility of the hyperplane.
	 * @return the intersection between a bound and a hyperplane.
	 */
	private double[] getIntersection(Double[] bound, double[] normal,
			double[] pointOnLine) {
		if (bound.length != normal.length)
			throw new AssertionError("Lengths of bound and normal do not match");
		int dimensions = normal.length;

		double c = 0;
		for (int i = 0; i < dimensions; i++) {
			c += normal[i] * pointOnLine[i];
		}

		int unknown = -1;
		double sum = 0;
		double[] intersection = new double[dimensions];
		for (int i = 0; i < dimensions; i++) {
			if (bound[i] == null) {
				unknown = i;
			} else {
				sum += bound[i] * normal[i];
				intersection[i] = bound[i];
			}
		}

		if (unknown < 0)
			throw new AssertionError("bound has no unknown");

		intersection[unknown] = (c - sum) / normal[unknown];

		return intersection;
	}

	/**
	 * Get all bounds of a space.
	 * 
	 * @param min
	 *            the minimum bound of the space.
	 * @param max
	 *            the maximum bound of the space.
	 * @return all bounds of a space.
	 */
	private ArrayList<Double[]> getBounds(double[] min, double[] max) {
		if (min.length != max.length)
			throw new AssertionError("Lengths of min and max do not match");

		int dimensions = min.length;

		ArrayList<Double[]> bounds = new ArrayList<Double[]>();

		int boundCount = dimensions * (int) Math.pow(2, dimensions - 1);
		for (int i = 0; i < boundCount; i++) {
			int dimension = (int) Math.floor(i / (boundCount / dimensions));
			Double[] bound = new Double[dimensions];
			for (int j = 0; j < dimensions; j++) {
				if (j == dimension)
					continue;
				if (j < dimension)
					bound[j] = (i & (1 << j)) == 0 ? min[j] : max[j];
				else
					bound[j] = (i & (1 << (j - 1))) == 0 ? min[j] : max[j];
			}
			bounds.add(bound);
		}

		return bounds;
	}

	/**
	 * Check whether a point lies within a set of bounds.
	 * 
	 * @param point
	 *            the point to check.
	 * @param min
	 *            the minimum bounds.
	 * @param max
	 *            the maximum bounds.
	 * @return true if the point lies within the bounds, false otherwise.
	 */
	private boolean WithinConstraints(double[] point, double[] min, double[] max) {
		if (min.length != point.length)
			throw new AssertionError("Lengths of min and point do not match");
		if (max.length != point.length)
			throw new AssertionError("Lengths of max and point do not match");

		for (int i = 0; i < point.length; i++) {
			if (point[i] < min[i] || point[i] > max[i]) {
				return false;
			}
		}
		return true;
	}

	/**
	 * Get all combinations of integers in a space.
	 * 
	 * @param space
	 *            the size of the space.
	 * @return all combinations of integers in a space.
	 */
	public static ArrayList<int[]> getCombinationValues(int[] space) {
		ArrayList<int[]> combinationValues = new ArrayList<int[]>();
		if (space.length == 1) {
			return combinationValues;
		}
		for (int i = 0; i < space[space.length - 1]; i++) {
			int[] combination = new int[space.length - 1];
			for (int j = 0; j < combination.length; j++) {
				combination[j] = (int) Math.floor((i / space[j])
						% (space[j + 1] / space[j]));
			}

			combinationValues.add(combination);
		}
		return combinationValues;
	}

	/**
	 * Get the point in multi-dimensional space that represents a bid.
	 * 
	 * @param bid
	 *            the bid.
	 * @return the point in multi-dimensional space that represents a bid.
	 */
	public double[] getPoint(Bid bid) {
		double[] point = new double[continuousWeights.size()];
		int i = 0;
		int j = 0;
		for (ISSUETYPE issueType : issueTypes) {
			if (issueType == ISSUETYPE.REAL) {
				ValueReal valueReal;
				try {
					valueReal = (ValueReal) bid.getValue(issues.get(j)
							.getNumber());
					point[i] = (valueReal.getValue() - offset[i]) / range[i];
				} catch (Exception e) {
					e.printStackTrace();
				}
				i++;
			}
			if (issueType == ISSUETYPE.INTEGER) {
				ValueInteger valueInteger;
				try {
					valueInteger = (ValueInteger) bid.getValue(issues.get(j)
							.getNumber());
					point[i] = (valueInteger.getValue() - offset[i]) / range[i];
				} catch (Exception e) {
					e.printStackTrace();
				}
				i++;
			}
			j++;
		}
		return point;
	}

	public Bid getMaxUtilityBid() {

		int discreteIssues = 0;
		int continuousIssues = 0;

		for (int i = 0; i < issueTypes.length; i++) {
			switch (issueTypes[i]) {
			case DISCRETE:
				discreteIssues++;
				break;
			case REAL:
			case INTEGER:
				continuousIssues++;
				break;
			}
		}

		int discreteCount = 0;
		int realCount = 0;
		int integerCount = 0;

		ValueDiscrete[] discreteCombination = new ValueDiscrete[discreteIssues];
		double[] continuousValues = new double[continuousIssues];

		for (Issue issue : issues) {
			switch (issue.getType()) {
			case DISCRETE:
				IssueDiscrete issueDiscrete = (IssueDiscrete) issue;
				List<ValueDiscrete> values = issueDiscrete.getValues();
				double maxEval = 0;
				ValueDiscrete maxValue = null;
				for (ValueDiscrete value : values) {
					double tmpEval;
					try {
						tmpEval = discreteEvaluators.get(discreteCount)
								.getEvaluation(value);
						if (tmpEval > maxEval) {
							maxEval = tmpEval;
							maxValue = value;
						}
					} catch (Exception e) {
						e.printStackTrace();
					}
				}
				discreteCombination[discreteCount] = maxValue;
				discreteCount++;
				break;
			case REAL:
				EvaluatorReal realEvaluator = realEvaluators.get(realCount);
				switch (realEvaluator.getFuncType()) {
				case LINEAR:
					if (realEvaluator.getEvaluation(realEvaluator
							.getLowerBound()) < realEvaluator
							.getEvaluation(realEvaluator.getUpperBound()))
						continuousValues[realCount + integerCount] = 1;
					else
						continuousValues[realCount + integerCount] = 0;
					break;
				case TRIANGULAR:
					continuousValues[realCount + integerCount] = normalise(
							realEvaluator.getTopParam(),
							realEvaluator.getLowerBound(),
							realEvaluator.getUpperBound());
					break;
				}
				realCount++;
				break;
			case INTEGER:
				EvaluatorInteger integerEvaluator = integerEvaluators
						.get(integerCount);
				switch (integerEvaluator.getFuncType()) {
				case LINEAR:
					if (integerEvaluator.getEvaluation(integerEvaluator
							.getLowerBound()) < integerEvaluator
							.getEvaluation(integerEvaluator.getUpperBound()))
						continuousValues[realCount + integerCount] = 1;
					else
						continuousValues[realCount + integerCount] = 0;
					break;
				/*
				 * case TRIANGULAR: realValues[integerCount] =
				 * integerEvaluator.getTopParam(); break;
				 */
				}
				integerCount++;
				break;
			}
		}

		return createBid(continuousValues, discreteCombination);
	}
}
