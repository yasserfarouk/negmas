package agents.anac.y2010.Southampton.utils;

import java.util.ArrayList;
import agents.anac.y2010.Southampton.utils.concession.TimeConcessionFunction;
import genius.core.Bid;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.IssueInteger;
import genius.core.issue.IssueReal;
import genius.core.issue.ValueDiscrete;
import genius.core.utility.AdditiveUtilitySpace;
import genius.core.utility.EVALFUNCTYPE;
import genius.core.utility.Evaluator;
import genius.core.utility.EvaluatorDiscrete;
import genius.core.utility.EvaluatorInteger;
import genius.core.utility.EvaluatorReal;

/**
 * @author Colin Williams
 * 
 */
public class OpponentModel extends agents.bayesianopponentmodel.OpponentModel{

	private AdditiveUtilitySpace utilitySpace;

	private ArrayList<Bid> biddingHistory;
	private ArrayList<ArrayList<EvaluatorHypothesis>> evaluatorHypotheses;
	private ArrayList<ArrayList<WeightHypothesis>> weightHypotheses;
	private double previousBidUtility;
	private Double maxUtility;
	private Double minUtility;
	private double[] expectedWeights;
	private double SIGMA = 0.25;
	private final int totalTriangularFunctions = 4;
	private TimeConcessionFunction opponentConcessionFunction;

	/**
	 * Default constructor.
	 * @param utilitySpace The utility space of the agent.
	 * @param agent The agent (used only for logging purposes).
	 */
	public OpponentModel(AdditiveUtilitySpace utilitySpace) {
		opponentConcessionFunction = new TimeConcessionFunction(TimeConcessionFunction.Beta.LINEAR, TimeConcessionFunction.BREAKOFF);
		this.utilitySpace = utilitySpace;
		
		previousBidUtility = 1;
		
		biddingHistory = new ArrayList<Bid>();
		weightHypotheses = new ArrayList<ArrayList<WeightHypothesis>>();
		evaluatorHypotheses = new ArrayList<ArrayList<EvaluatorHypothesis>>();
		expectedWeights = new double[utilitySpace.getDomain().getIssues().size()];

		initWeightHypotheses();

		initEvaluatorHypotheses();
	}

	/**
	 * Initialise the weight hypotheses.
	 */
	private void initWeightHypotheses() {
		int weightHypothesesNumber = 11;
		for (int i = 0; i < utilitySpace.getDomain().getIssues().size(); ++i) {
			ArrayList<WeightHypothesis> weightHypothesis = new ArrayList<WeightHypothesis>();
			for (int j = 0; j < weightHypothesesNumber; ++j) {
				WeightHypothesis weight = new WeightHypothesis();
				weight.setProbability((1.0 - (((double) j + 1.0) / weightHypothesesNumber)) * (1.0 - (((double) j + 1.0) / weightHypothesesNumber))
						* (1.0 - (((double) j + 1.0D) / weightHypothesesNumber)));
				weight.setWeight((double) j / (weightHypothesesNumber - 1));
				weightHypothesis.add(weight);
			}

			// Normalization
			double n = 0.0D;
			for (int j = 0; j < weightHypothesesNumber; ++j) {
				n += weightHypothesis.get(j).getProbability();
			}
			for (int j = 0; j < weightHypothesesNumber; ++j) {
				weightHypothesis.get(j).setProbability(weightHypothesis.get(j).getProbability() / n);
			}

			weightHypotheses.add(weightHypothesis);
		}
	}

	/**
	 * Initialise the evaluator hypotheses.
	 */
	private void initEvaluatorHypotheses() {
		evaluatorHypotheses = new ArrayList<ArrayList<EvaluatorHypothesis>>();
		for (int i = 0; i < utilitySpace.getNrOfEvaluators(); ++i) {
			ArrayList<EvaluatorHypothesis> lEvalHyps;
			EvaluatorReal lHypEvalReal;
			EvaluatorInteger lHypEvalInteger;
			EvaluatorHypothesis lEvaluatorHypothesis;
			switch (utilitySpace.getEvaluator(utilitySpace.getIssue(i).getNumber()).getType()) {
			
			case REAL:
			{
				lEvalHyps = new ArrayList<EvaluatorHypothesis>();
				evaluatorHypotheses.add(lEvalHyps);

				IssueReal lIssue = (IssueReal) utilitySpace.getIssue(i);

				/* Uphill */
				lHypEvalReal = new EvaluatorReal();
				lHypEvalReal.setUpperBound(lIssue.getUpperBound());
				lHypEvalReal.setLowerBound(lIssue.getLowerBound());
				lHypEvalReal.setType(EVALFUNCTYPE.LINEAR);
				lHypEvalReal.addParam(1, 1.0 / (lHypEvalReal.getUpperBound() - lHypEvalReal.getLowerBound()));
				lHypEvalReal.addParam(0, -lHypEvalReal.getLowerBound() / (lHypEvalReal.getUpperBound() - lHypEvalReal.getLowerBound()));
				lEvaluatorHypothesis = new EvaluatorHypothesis(lHypEvalReal);
				lEvaluatorHypothesis.setDesc("uphill");
				lEvalHyps.add(lEvaluatorHypothesis);

				/* Triangular */
				for (int k = 1; k <= totalTriangularFunctions; ++k) {
					lHypEvalReal = new EvaluatorReal();
					lHypEvalReal.setUpperBound(lIssue.getUpperBound());
					lHypEvalReal.setLowerBound(lIssue.getLowerBound());
					lHypEvalReal.setType(EVALFUNCTYPE.TRIANGULAR);
					lHypEvalReal.addParam(0, lHypEvalReal.getLowerBound());
					lHypEvalReal.addParam(1, lHypEvalReal.getUpperBound());
					double lMaxPoint = lHypEvalReal.getLowerBound() + (double) k * (lHypEvalReal.getUpperBound() - lHypEvalReal.getLowerBound())
							/ (totalTriangularFunctions + 1);
					lHypEvalReal.addParam(2, lMaxPoint);
					lEvaluatorHypothesis = new EvaluatorHypothesis(lHypEvalReal);
					lEvalHyps.add(lEvaluatorHypothesis);
					lEvaluatorHypothesis.setDesc("triangular " + String.format("%1.2f", lMaxPoint));
				}

				/* Downhill */
				lHypEvalReal = new EvaluatorReal();
				lHypEvalReal.setUpperBound(lIssue.getUpperBound());
				lHypEvalReal.setLowerBound(lIssue.getLowerBound());
				lHypEvalReal.setType(EVALFUNCTYPE.LINEAR);
				lHypEvalReal.addParam(1, -1.0 / (lHypEvalReal.getUpperBound() - lHypEvalReal.getLowerBound()));
				lHypEvalReal.addParam(0, 1.0 + lHypEvalReal.getLowerBound() / (lHypEvalReal.getUpperBound() - lHypEvalReal.getLowerBound()));
				lEvaluatorHypothesis = new EvaluatorHypothesis(lHypEvalReal);
				lEvaluatorHypothesis.setDesc("downhill");
				lEvalHyps.add(lEvaluatorHypothesis);
				
				for (int k = 0; k < lEvalHyps.size(); ++k) {
					lEvalHyps.get(k).setProbability(1.0 / lEvalHyps.size());
				}

				break;
			}
			case INTEGER:
			{
				lEvalHyps = new ArrayList<EvaluatorHypothesis>();
				evaluatorHypotheses.add(lEvalHyps);

				IssueInteger lIssue = (IssueInteger) utilitySpace.getIssue(i);

				/* Uphill */
				lHypEvalInteger = new EvaluatorInteger();
				lHypEvalInteger.setUpperBound(lIssue.getUpperBound());
				lHypEvalInteger.setLowerBound(lIssue.getLowerBound());
				lHypEvalInteger.setOffset(-lHypEvalInteger.getLowerBound() / (lHypEvalInteger.getUpperBound() - lHypEvalInteger.getLowerBound()));
				lHypEvalInteger.setSlope(1.0 / (lHypEvalInteger.getUpperBound() - lHypEvalInteger.getLowerBound()));
				lEvaluatorHypothesis = new EvaluatorHypothesis(lHypEvalInteger);
				lEvaluatorHypothesis.setDesc("uphill");
				lEvalHyps.add(lEvaluatorHypothesis);

				/* Downhill */
				lHypEvalInteger = new EvaluatorInteger();
				lHypEvalInteger.setUpperBound(lIssue.getUpperBound());
				lHypEvalInteger.setLowerBound(lIssue.getLowerBound());
				lHypEvalInteger.setOffset(1.0 + lHypEvalInteger.getLowerBound() / (lHypEvalInteger.getUpperBound() - lHypEvalInteger.getLowerBound()));
				lHypEvalInteger.setSlope(-1.0 / (lHypEvalInteger.getUpperBound() - lHypEvalInteger.getLowerBound()));
				lEvaluatorHypothesis = new EvaluatorHypothesis(lHypEvalInteger);
				lEvaluatorHypothesis.setDesc("downhill");
				lEvalHyps.add(lEvaluatorHypothesis);
				
				for (int k = 0; k < lEvalHyps.size(); ++k) {
					lEvalHyps.get(k).setProbability(1.0 / lEvalHyps.size());
				}

				break;
			}
			case DISCRETE:
			{
				lEvalHyps = new ArrayList<EvaluatorHypothesis>();
				evaluatorHypotheses.add(lEvalHyps);

				IssueDiscrete lDiscIssue = (IssueDiscrete) utilitySpace.getIssue(i);

				/* Uphill */
				EvaluatorDiscrete lDiscreteEval = new EvaluatorDiscrete();
				for (int j = 0; j < lDiscIssue.getNumberOfValues(); ++j)
					lDiscreteEval.addEvaluation(lDiscIssue.getValue(j), Integer.valueOf(1000 * j + 1));
				lEvaluatorHypothesis = new EvaluatorHypothesis(lDiscreteEval);
				lEvaluatorHypothesis.setDesc("uphill");
				lEvalHyps.add(lEvaluatorHypothesis);

				/* Triangular */
				if (lDiscIssue.getNumberOfValues() > 2) {
					for (int k = 1; k < lDiscIssue.getNumberOfValues() - 1; ++k) {
						lDiscreteEval = new EvaluatorDiscrete();
						for (int j = 0; j < lDiscIssue.getNumberOfValues(); ++j) {
							if (j < k) {
								lDiscreteEval.addEvaluation(lDiscIssue.getValue(j), 1000 * j / k);
							} else
								lDiscreteEval.addEvaluation(lDiscIssue.getValue(j), 1000 * (lDiscIssue.getNumberOfValues() - j - 1
										/ (lDiscIssue.getNumberOfValues() - k - 1) + 1));
						}
						lEvaluatorHypothesis = new EvaluatorHypothesis(lDiscreteEval);
						lEvalHyps.add(lEvaluatorHypothesis);
						lEvaluatorHypothesis.setDesc("triangular " + String.valueOf(k));
					}
				}

				/* Downhill */
				lDiscreteEval = new EvaluatorDiscrete();
				for (int j = 0; j < lDiscIssue.getNumberOfValues(); ++j)
					lDiscreteEval.addEvaluation(lDiscIssue.getValue(j), Integer.valueOf(1000 * (lDiscIssue.getNumberOfValues() - j - 1) + 1));
				lEvaluatorHypothesis = new EvaluatorHypothesis(lDiscreteEval);
				lEvaluatorHypothesis.setDesc("downhill");
				lEvalHyps.add(lEvaluatorHypothesis);
				
				for (int k = 0; k < lEvalHyps.size(); ++k) {
					lEvalHyps.get(k).setProbability(1.0 / lEvalHyps.size());
				}

				break;
			}
			}
		}

		for (int i = 0; i < expectedWeights.length; ++i)
			expectedWeights[i] = getExpectedWeight(i);

		normalize(expectedWeights);
	}

	/**
	 * Get the normalised utility of a bid.
	 * @param bid The bid to get the normalised utility of.
	 * @return the normalised utility of a bid.
	 * @throws Exception
	 */
	public double getNormalizedUtility(Bid bid) throws Exception {
		return getNormalizedUtility(bid, false);
	}
	
	/**
	 * Get the normalised utility of a bid.
	 * @param bid The bid to get the normalised utility of.
	 * @param debug Whether or not to output debugging information
	 * @return the normalised utility of a bid.
	 * @throws Exception
	 */
	public double getNormalizedUtility(Bid bid, boolean debug) throws Exception {
		double u = getExpectedUtility(bid);
		if (minUtility == null || maxUtility == null)
			findMinMaxUtility();
		return (u - minUtility) / (maxUtility - minUtility);
	}

	/**
	 * Get the expected utility of a bid.
	 * @param bid The bid to get the expected utility of.
	 * @return the expected utility of the bid.
	 * @throws Exception
	 */
	public double getExpectedUtility(Bid bid) throws Exception {
		double u = 0;
		for (int i = 0; i < utilitySpace.getDomain().getIssues().size(); i++) {
			u += expectedWeights[i] * getExpectedEvaluationValue(bid, i);
		}
		return u;
	}

	/**
	 * Update the beliefs about the opponent, based on an observation.
	 * @param opponentBid The opponent's bid that was observed.
	 * @throws Exception
	 */
	public void updateBeliefs(Bid opponentBid, long currentTime, double totalTime) throws Exception {
		if (biddingHistory.contains(opponentBid))
			return;
		biddingHistory.add(opponentBid);

		if (biddingHistory.size() > 1) {
			updateWeights();
		}
		updateEvaluationFunctions();
		
		previousBidUtility = opponentConcessionFunction.getConcession(1, currentTime, totalTime);
		for (int i = 0; i < expectedWeights.length; ++i)
			expectedWeights[i] = getExpectedWeight(i);

		normalize(expectedWeights);

		findMinMaxUtility();
	}

	/**
	 * Normalise the values in an array so that they sum to 1.
	 * @param array The array to normalise;
	 */
	private void normalize(double[] array) {
		double n = 0;
		for (int i = 0; i < array.length; ++i) {
			n += array[i];
		}
		if(n == 0)
		{
			for (int i = 0; i < array.length; ++i) {
				array[i] = 1.0/array.length;
			}
			return;
		}

		for (int i = 0; i < array.length; ++i) {
			array[i] = array[i] / n;
		}
	}

	/**
	 * Find the minimum and maximum utilities of the bids in the utility space.
	 * @throws Exception
	 */
	protected void findMinMaxUtility() throws Exception {
		maxUtility = getExtremeUtility(Extreme.MAX);
		minUtility = getExtremeUtility(Extreme.MIN);
	}

	public enum Extreme { MIN, MAX }
	
	private double getExtremeUtility(Extreme extreme) {
		double u = 0;
		for (int i = 0; i < utilitySpace.getDomain().getIssues().size(); i++) {
			u += expectedWeights[i] * getExtremeEvaluationValue(i, extreme);
		}
		return u;
	}

	private double getExtremeEvaluationValue(int number, Extreme extreme) {
		double expectedEval = 0;
		for (EvaluatorHypothesis evaluatorHypothesis : evaluatorHypotheses.get(number)) {
			expectedEval += evaluatorHypothesis.getProbability()
				* getExtremeEvaluation(evaluatorHypothesis.getEvaluator(), extreme);
		}
		return expectedEval;
	}

	public double getExtremeEvaluation(Evaluator evaluator, Extreme extreme) {
		double extremeEval = initExtreme(extreme);
		switch(evaluator.getType())
		{
		case DISCRETE:
			EvaluatorDiscrete discreteEvaluator = (EvaluatorDiscrete)evaluator;
			for(ValueDiscrete value : discreteEvaluator.getValues())
			{
				try {
					switch(extreme)
					{
					case MAX:
						extremeEval = Math.max(extremeEval, discreteEvaluator.getEvaluation(value));
						break;
					case MIN:
						extremeEval = Math.min(extremeEval, discreteEvaluator.getEvaluation(value));
						break;
					}
				} catch (Exception e) {
					e.printStackTrace();
				}
			}
			break;
		case INTEGER:
			EvaluatorInteger integerEvaluator = (EvaluatorInteger)evaluator;
			switch(extreme)
			{
			case MAX:
				extremeEval = Math.max(integerEvaluator.getEvaluation(integerEvaluator.getUpperBound()), integerEvaluator.getEvaluation(integerEvaluator.getLowerBound()));
				//if(integerEvaluator.getFuncType() == EVALFUNCTYPE.TRIANGULAR)
				//{
				//	extremeEval = Math.max(extremeEval, integerEvaluator.getEvaluation(integerEvaluator.getTopParam()));
				//}
				break;
			case MIN:
				extremeEval = Math.min(integerEvaluator.getEvaluation(integerEvaluator.getUpperBound()), integerEvaluator.getEvaluation(integerEvaluator.getLowerBound()));
				//if(integerEvaluator.getFuncType() == EVALFUNCTYPE.TRIANGULAR)
				//{
				//	extremeEval = Math.min(extremeEval, integerEvaluator.getEvaluation(integerEvaluator.getTopParam()));
				//}
				break;
			}
			break;
		case REAL:
			EvaluatorReal realEvaluator = (EvaluatorReal)evaluator;
			switch(extreme)
			{
			case MAX:
				extremeEval = Math.max(realEvaluator.getEvaluation(realEvaluator.getUpperBound()), realEvaluator.getEvaluation(realEvaluator.getLowerBound()));
				if(realEvaluator.getFuncType() == EVALFUNCTYPE.TRIANGULAR)
				{
					extremeEval = Math.max(extremeEval, realEvaluator.getEvaluation(realEvaluator.getTopParam()));
				}
				break;
			case MIN:
				extremeEval = Math.min(realEvaluator.getEvaluation(realEvaluator.getUpperBound()), realEvaluator.getEvaluation(realEvaluator.getLowerBound()));
				if(realEvaluator.getFuncType() == EVALFUNCTYPE.TRIANGULAR)
				{
					extremeEval = Math.min(extremeEval, realEvaluator.getEvaluation(realEvaluator.getTopParam()));
				}
				break;
			}
			break;
		}
		return extremeEval;
	}

	private double initExtreme(Extreme extreme) {
		switch(extreme)
		{
		case MAX:
			return Double.MIN_VALUE;
		case MIN:
			return Double.MAX_VALUE;
		}
		return 0;
	}
	
	/**
	 * Update the evaluation functions.
	 * @throws Exception
	 */
	private void updateEvaluationFunctions() throws Exception {
		maxUtility = null;
		minUtility = null;
		
		Bid bid = biddingHistory.get(biddingHistory.size() - 1);
		ArrayList<ArrayList<EvaluatorHypothesis>> evaluatorHypotheses = new ArrayList<ArrayList<EvaluatorHypothesis>>();

		for (int i = 0; i < this.evaluatorHypotheses.size(); ++i) {
			ArrayList<EvaluatorHypothesis> tmp = new ArrayList<EvaluatorHypothesis>();
			for (int j = 0; j < this.evaluatorHypotheses.get(i).size(); ++j) {
				EvaluatorHypothesis evaluatorHypothesis = new EvaluatorHypothesis(this.evaluatorHypotheses.get(i).get(j).getEvaluator());
				evaluatorHypothesis.setDesc(this.evaluatorHypotheses.get(i).get(j).getDesc());
				evaluatorHypothesis.setProbability(this.evaluatorHypotheses.get(i).get(j).getProbability());
				tmp.add(evaluatorHypothesis);
			}
			evaluatorHypotheses.add(tmp);
		}

		for (int i = 0; i < this.utilitySpace.getDomain().getIssues().size(); i++) {
			double n = 0.0D;
			double utility = 0.0D;
			for (EvaluatorHypothesis evaluatorHypothesis : evaluatorHypotheses.get(i)) {
				utility = getPartialUtility(bid, i) +
						getExpectedWeight(i) * evaluatorHypothesis.getEvaluator().getEvaluation(utilitySpace, bid, utilitySpace.getIssue(i).getNumber());
				n += evaluatorHypothesis.getProbability() * conditionalDistribution(utility, previousBidUtility);
			}

			for (EvaluatorHypothesis evaluatorHypothesis : evaluatorHypotheses.get(i)) {
				utility = getPartialUtility(bid, i) +
						getExpectedWeight(i) * evaluatorHypothesis.getEvaluator().getEvaluation(utilitySpace, bid, utilitySpace.getIssue(i).getNumber());
				evaluatorHypothesis.setProbability(evaluatorHypothesis.getProbability()	* conditionalDistribution(utility, previousBidUtility) / n);
			}
		}

		this.evaluatorHypotheses = evaluatorHypotheses;
	}

	/**
	 * Update the weights.
	 * @throws Exception
	 */
	private void updateWeights() throws Exception {
		maxUtility = null;
		minUtility = null;
		
		Bid bid = biddingHistory.get(biddingHistory.size() - 1);
		ArrayList<ArrayList<WeightHypothesis>> weightHypotheses = new ArrayList<ArrayList<WeightHypothesis>>();

		for (int i = 0; i < this.weightHypotheses.size(); ++i) {
			ArrayList<WeightHypothesis> tmp = new ArrayList<WeightHypothesis>();
			for (int j = 0; j < this.weightHypotheses.get(i).size(); ++j) {
				WeightHypothesis weightHypothesis = new WeightHypothesis();
				weightHypothesis.setWeight(this.weightHypotheses.get(i).get(j).getWeight());
				weightHypothesis.setProbability(this.weightHypotheses.get(i).get(j).getProbability());
				tmp.add(weightHypothesis);
			}
			weightHypotheses.add(tmp);
		}

		for (int i = 0; i < this.utilitySpace.getDomain().getIssues().size(); i++) {
			double n = 0.0D;
			double utility = 0.0D;
			for (WeightHypothesis weightHypothesis : weightHypotheses.get(i)) {
				utility = getPartialUtility(bid, i) +
						weightHypothesis.getWeight() * getExpectedEvaluationValue(bid, i);
				n += weightHypothesis.getProbability() * conditionalDistribution(utility, previousBidUtility);
			}

			for (WeightHypothesis weightHypothesis : weightHypotheses.get(i)) {
				utility = getPartialUtility(bid, i) +
						weightHypothesis.getWeight() * getExpectedEvaluationValue(bid, i);
				weightHypothesis.setProbability(weightHypothesis.getProbability() * conditionalDistribution(utility, previousBidUtility) / n);
			}
		}

		this.weightHypotheses = weightHypotheses;

	}

	/**
	 * The conditional distribution function.
	 * @param utility The utility.
	 * @param previousBidUtility The utility of the previous bid.
	 * @return
	 */
	private double conditionalDistribution(double utility, double previousBidUtility) {
		double x = (previousBidUtility - utility) / previousBidUtility;
		return (1.0 / (SIGMA * Math.sqrt(2 * Math.PI))) * Math.exp(-(x * x) / (2 * SIGMA * SIGMA));
	}

	/**
	 * Get the expected evaluation value of a bid for a particular issue.
	 * @param bid The bid to get the expected evaluation value of.
	 * @param number The number of the issue to get the expected evaluation value of.
	 * @return the expected evaluation value of a bid for a particular issue.
	 * @throws Exception
	 */
	private double getExpectedEvaluationValue(Bid bid, int number) throws Exception {
		double expectedEval = 0;
		for (EvaluatorHypothesis evaluatorHypothesis : evaluatorHypotheses.get(number)) {
			expectedEval += evaluatorHypothesis.getProbability()
				* evaluatorHypothesis.getEvaluator().getEvaluation(utilitySpace, bid, utilitySpace.getIssue(number).getNumber());
		}
		return expectedEval;
	}

	/**
	 * Get the partial utility of a bid, excluding a specific issue.
	 * @param bid The bid to get the partial utility of.
	 * @param number The number of the issue to exclude.
	 * @return the partial utility of a bid, excluding a specific issue.
	 * @throws Exception
	 */
	private double getPartialUtility(Bid bid, int number) throws Exception {
		double u = 0;
		for (int i = 0; i < utilitySpace.getDomain().getIssues().size(); i++) {
			if (number == i) {
				continue;
			}
			double w = 0;
			for (WeightHypothesis weightHypothesis : weightHypotheses.get(i))
				w += weightHypothesis.getProbability() * weightHypothesis.getWeight();
			u += w * getExpectedEvaluationValue(bid, i);
		}
		return u;
	}

	/**
	 * Get the expected weight of a particular issue.
	 * @param number The issue number.
	 * @return the expected weight of a particular issue.
	 */
	public double getExpectedWeight(int number) {
		double expectedWeight = 0;
		for (WeightHypothesis weightHypothesis : weightHypotheses.get(number)) {
			expectedWeight += weightHypothesis.getProbability() * weightHypothesis.getWeight();
		}
		return expectedWeight;
	}

	/**
	 * Print the best hypothesis.
	 */
	public void printBestHypothesis() {
		double[] bestWeights = new double[utilitySpace.getDomain().getIssues().size()];
		EvaluatorHypothesis[] bestEvaluatorHypotheses = new EvaluatorHypothesis[utilitySpace.getDomain().getIssues().size()];
		for (int i = 0; i < this.utilitySpace.getDomain().getIssues().size(); i++) {
			for (WeightHypothesis weightHypothesis : weightHypotheses.get(i)) {
				bestWeights[i] += weightHypothesis.getWeight() * weightHypothesis.getProbability();
			}

			bestEvaluatorHypotheses[i] = getBestHypothesis(i);
		}

		normalize(bestWeights);
	}

	
	public EvaluatorHypothesis getBestHypothesis(int issue) {
		double maxEvaluatorProbability = -1;
		EvaluatorHypothesis bestEvaluatorHypothesis = null;
		for (EvaluatorHypothesis evaluatorHypothesis : evaluatorHypotheses.get(issue)) {
			if (evaluatorHypothesis.getProbability() > maxEvaluatorProbability) {
				maxEvaluatorProbability = evaluatorHypothesis.getProbability();
				bestEvaluatorHypothesis = evaluatorHypothesis;
			}
		}
		return bestEvaluatorHypothesis;
	}
	
	/**
	 * Get the first bid.
	 * @return the first bid.
	 */
	public Bid getFirstBid() {
		return biddingHistory.get(0);
	}
	
	public Hypothesis getHypothesis(int index) {
		return this.evaluatorHypotheses.get(index).get(index);
	}
}
