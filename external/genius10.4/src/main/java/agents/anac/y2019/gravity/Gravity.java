package agents.anac.y2019.gravity;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import javax.swing.event.DocumentListener;

import genius.core.AgentID;
import genius.core.Bid;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.Offer;
import genius.core.boaframework.SortedOutcomeSpace;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.ValueDiscrete;
import genius.core.parties.AbstractNegotiationParty;
import genius.core.parties.NegotiationInfo;
import genius.core.timeline.DiscreteTimeline;
import genius.core.uncertainty.AdditiveUtilitySpaceFactory;
import genius.core.utility.AbstractUtilitySpace;

public class Gravity extends AbstractNegotiationParty {
	private static final double HALF_OF_TIME = 0.5;

	private AbstractUtilitySpace utilitySpace = null;
	private SortedOutcomeSpace outcomeSpace = null;

	Bid sampleBid;
	private List<Bid> orderedBidListInAscendingOrder;
	private List<Issue> listOfIssues;
	private int numberOfIssues;

	Map<Integer, double[][]> eachIssueAnItsIndexingValuesDoubleArrayMap;
	Map<Integer, double[]> eachIssueAndItsValuesUtilityMap;

	double[] sumOfSquaredErrorsOfIssueValues;
	Map<Integer, Double> eachIssueAndItsUtilityMap;

	// opponent modeling
	Bid lastReceivedBid;
	Map<Integer, double[]> eachIssueAndItsValuesFrequencyArrayMap;
	Map<Integer, double[]> opponentEachIssueAndItsValuesUtilityMap;
	double[] opponentSumOfSquaredErrosOfEachIssueMatrix;
	Map<Integer, Double> opponentEachIssueAndItsUtilityMap;

	double lastOfferingTime;
	Bid lastOfferedBid;
	Bid nextBid;

	@Override
	public void init(NegotiationInfo info) {
		super.init(info);
		initAgentVariables();
		initOpponentVars();
	}

	// uncertain preferences
	private void initAgentVariables() {
		initUncertainPrefVariables();
		createCopelandMatrices();
		fillAgentVars();
	}

	private void fillAgentVars() {
		fillCopelandMatrices();
		setValueUtilities();
		fillMatriceOfNoChangeOfEachIssue();
		setIssueUtilitiesWithNormalization();
	}

	// opponent modeling
	private void initOpponentVars() {
		initOpponentModelingVars();
		createFrequencyArrays();
	}

	private void initUncertainPrefVariables() {
		this.utilitySpace = getUtilitySpace();
		this.outcomeSpace = new SortedOutcomeSpace(utilitySpace);

		this.orderedBidListInAscendingOrder = getUserModel().getBidRanking().getBidOrder();
		this.sampleBid = orderedBidListInAscendingOrder.get(0);
		this.listOfIssues = sampleBid.getIssues();
		this.numberOfIssues = listOfIssues.size();

		this.eachIssueAnItsIndexingValuesDoubleArrayMap = new HashMap<Integer, double[][]>();
		this.eachIssueAndItsValuesUtilityMap = new HashMap<Integer, double[]>();

		this.sumOfSquaredErrorsOfIssueValues = new double[numberOfIssues];
		this.eachIssueAndItsUtilityMap = new HashMap<Integer, Double>();
	}

	private void initOpponentModelingVars() {
		this.eachIssueAndItsValuesFrequencyArrayMap = new HashMap<Integer, double[]>();
		this.opponentEachIssueAndItsValuesUtilityMap = new HashMap<Integer, double[]>();
		this.opponentSumOfSquaredErrosOfEachIssueMatrix = new double[numberOfIssues];
		this.opponentEachIssueAndItsUtilityMap = new HashMap<Integer, Double>();
	}

	private void createCopelandMatrices() {
		for (int i = 0; i < numberOfIssues; i++) {
			IssueDiscrete issueDiscrete = (IssueDiscrete) sampleBid.getIssues().get(i);
			int valueSize = issueDiscrete.getValues().size();
			eachIssueAnItsIndexingValuesDoubleArrayMap.put(i, new double[valueSize][valueSize]);
		}
	}

	private void createFrequencyArrays() {
		for (int i = 0; i < numberOfIssues; i++) {
			IssueDiscrete issueDiscrete = (IssueDiscrete) sampleBid.getIssues().get(i);
			int valueSize = issueDiscrete.getValues().size();
			eachIssueAndItsValuesFrequencyArrayMap.put(i, new double[valueSize]);
		}
	}

	/*
	 * Gets the bigger bid from the sorted list . Gets the smaller bids from the
	 * sorted list. Does pairwise Copeland comparison with one bigger bid and
	 * smaller bids. depending on the bigger bid > all smaller bids
	 */
	private void fillCopelandMatrices() {
		for (int i = orderedBidListInAscendingOrder.size() - 1; i > 0; i--) {
			Bid biggerBid = orderedBidListInAscendingOrder.get(i);
			for (int j = i - 1; j >= 0; j--) {
				Bid smallerBid = orderedBidListInAscendingOrder.get(j);
				fillCopelandMatriceWithBiggerAndSmaller(biggerBid, smallerBid, i, j);
			}
			System.out.println(biggerBid);
		}
		printIntDoubleDoubleArrMap(eachIssueAnItsIndexingValuesDoubleArrayMap);
	}

	/*
	 * pairwise Copeland comparison in each issue value, update the frequency
	 * matrices for different issue values depending on the similarity.
	 */
	private void fillCopelandMatriceWithBiggerAndSmaller(Bid biggerBid, Bid smallerBid, int biggerIndex,
			int smallerIndex) {
		for (int i = 0; i < numberOfIssues; i++) {
			ValueDiscrete biggerBidValue = (ValueDiscrete) (biggerBid.getValue(i + 1));
			int biggerBidValueIndex = getIndexOfValueInIssue(biggerBid, i, biggerBidValue.getValue());
			ValueDiscrete smallerValue = (ValueDiscrete) (smallerBid.getValue(i + 1));
			int smallerBidValueIndex = getIndexOfValueInIssue(smallerBid, i, smallerValue.getValue());
			int numberOfSimilarities = biggerBid.countEqualValues(smallerBid);
			if (numberOfSimilarities > 0) {
				eachIssueAnItsIndexingValuesDoubleArrayMap.get(i)[biggerBidValueIndex][smallerBidValueIndex] += (1d
						/ (biggerIndex - smallerIndex)) * numberOfSimilarities;
			}
		}
	}

	private int getIndexOfValueInIssue(Bid bid, int issueIndex, String value) {
		IssueDiscrete is = (IssueDiscrete) bid.getIssues().get(issueIndex);
		return is.getValueIndex(value);
	}

	private void setValueUtilities() {
		for (int i = 0; i < numberOfIssues; i++) {
			IssueDiscrete issueDiscrete = (IssueDiscrete) sampleBid.getIssues().get(i);
			int valueSize = issueDiscrete.getValues().size();
			double[] valuesBeingBigInfoArray = new double[valueSize];
			double[][] matrix = eachIssueAnItsIndexingValuesDoubleArrayMap.get(i);
			for (int j = 0; j < valueSize; j++) {
				double sumOfRowInMatrix = getSumOfRowInMatrix(matrix, j);
				double sumOfColInMatrix = getSumOfColInMatrix(matrix, j);
				double total = sumOfColInMatrix + sumOfRowInMatrix;
				if (total == 0) {
					valuesBeingBigInfoArray[j] = 0;
				} else {
					double beingBigPercentage = (sumOfRowInMatrix) / total;
					valuesBeingBigInfoArray[j] = beingBigPercentage;
				}
			}
			normalize(i, valueSize, valuesBeingBigInfoArray);
		}
		System.out.println("------------AGENT------------");
		printIntDoubleArrMap(eachIssueAndItsValuesUtilityMap);
	}

	private void normalize(int i, int valueSize, double[] valuesBeingBigInfoArray) {
		double[] utilityArr = new double[valueSize];
		double totalSum = getSumOfRowInOneDimensionalMatrix(valuesBeingBigInfoArray);
		for (int j = 0; j < valueSize; j++) {
			if (totalSum == 0) {
				utilityArr[j] = 0;
			} else {
				utilityArr[j] = valuesBeingBigInfoArray[j] / totalSum;
			}
		}
		eachIssueAndItsValuesUtilityMap.put(i, utilityArr);
	}

	private void fillMatriceOfNoChangeOfEachIssue() {
		for (int i = 0; i < numberOfIssues; i++) {
			double[][] matrix = eachIssueAnItsIndexingValuesDoubleArrayMap.get(i);
			double sumOfMatrix = 0;
			for (int j = 0; j < matrix.length; j++) {
				sumOfMatrix += getSumOfRowInMatrix(matrix, j);
			}
			System.out.println("Sum of matrix: " + sumOfMatrix);
			double average = sumOfMatrix / (matrix.length * matrix.length);
			System.out.println("average of matrix: " + average);
			double sumOfSquaredErrors = 0;
			for (int j = 0; j < matrix.length; j++) {
				for (int k = 0; k < matrix.length; k++) {
					sumOfSquaredErrors += Math.pow(matrix[j][k] - average, 2);
				}
			}
			sumOfSquaredErrorsOfIssueValues[i] = Math.sqrt(sumOfSquaredErrors / (matrix.length * matrix.length));
		}
	}

	private int getSumOfRowInMatrix(double[][] matrix, int row) {
		int rowSum = 0;
		for (int col = 0; col < matrix[row].length; col++) {
			rowSum += matrix[row][col];
		}
		return rowSum;
	}

	private int getSumOfColInMatrix(double[][] matrix, int col) {
		int colSum = 0;
		for (int row = 0; row < matrix.length; row++) {
			colSum += matrix[row][col];
		}
		return colSum;
	}

	private void setIssueUtilitiesWithNormalization() {
		double totalOfsumOfSquares = 0;
		for (int i = 0; i < numberOfIssues; i++) {
			totalOfsumOfSquares += sumOfSquaredErrorsOfIssueValues[i];
		}
		for (int i = 0; i < numberOfIssues; i++) {
			eachIssueAndItsUtilityMap.put(i, sumOfSquaredErrorsOfIssueValues[i] / totalOfsumOfSquares);
		}
		
		System.out.println("----------------------AGENT------------------");
		printIntDoubleMap(eachIssueAndItsUtilityMap);
	}

	private void printIntDoubleDoubleArrMap(Map<Integer, double[][]> eachIssueAnItsValuesDoubleArrayMap) {
		System.out.println("EACH ISSUE AND ITS VALUES");
		for (Entry<Integer, double[][]> entry : eachIssueAnItsValuesDoubleArrayMap.entrySet()) {
			System.out.println(entry.getKey() + " ");
			double[][] values = entry.getValue();
			for (int j = 0; j < values.length; j++) {
				for (int k = 0; k < values.length; k++) {
					System.out.print(values[j][k] + " ");
				}
				System.out.println();
			}
			System.out.println();
		}
	}

	private void printIntDoubleArrMap(Map<Integer, double[]> map) {
		System.out.println("EACH ISSUE AND ITS VALUES UTILITIES");
		for (Entry<Integer, double[]> entry : map.entrySet()) {
			System.out.print(entry.getKey() + " ");
			double[] values = entry.getValue();
			for (int j = 0; j < values.length; j++) {
				System.out.print(values[j] + " ");
			}
			System.out.println();
		}
		System.out.println();
	}

	private void printIntDoubleMap(Map<Integer, Double> map) {
		System.out.println("----------------------EACH ISSUE AND ITS UTILITIES------------------");
		for (Entry<Integer, Double> entry : map.entrySet()) {
			System.out.println(entry.getKey() + " " + entry.getValue());
		}
		System.out.println();
	}

	@Override
	public void receiveMessage(AgentID sender, Action action) {
		try {
			super.receiveMessage(sender, action);
			if (action instanceof Offer) {
				lastReceivedBid = ((Offer) action).getBid();
				updateFrequencyArrays();
				updateIssueValueUtilities();
				fillOpponentMatriceOfSumOfSquaresOfEachIssue();
				updateIsseUtilities();
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	private void updateFrequencyArrays() {
		for (int i = 0; i < numberOfIssues; i++) {
			ValueDiscrete valueDiscrete = (ValueDiscrete) (lastReceivedBid.getValue(i + 1));
			int valueIndex = getIndexOfValueInIssue(lastReceivedBid, i, valueDiscrete.getValue());
			eachIssueAndItsValuesFrequencyArrayMap.get(i)[valueIndex] += 1;
		}
	}

	private void updateIssueValueUtilities() {
		for (int i = 0; i < numberOfIssues; i++) {
			double[] frequencyArr = eachIssueAndItsValuesFrequencyArrayMap.get(i);
			double[] utilityArr = new double[frequencyArr.length];
			for (int j = 0; j < frequencyArr.length; j++) {
				double freqSumOfEachIssue = getSumOfRowInOneDimensionalMatrix(frequencyArr);
				utilityArr[j] = ((double) frequencyArr[j] / freqSumOfEachIssue);
			}
			opponentEachIssueAndItsValuesUtilityMap.put(i, utilityArr);
		}
		System.out.println("------------OPPONENT ISSUE VALUE UTULITIES------------");
		printIntDoubleArrMap(opponentEachIssueAndItsValuesUtilityMap);
	}

	private void fillOpponentMatriceOfSumOfSquaresOfEachIssue() {
		for (int i = 0; i < numberOfIssues; i++) {
			double[] frequencyArr = eachIssueAndItsValuesFrequencyArrayMap.get(i);
			double averageOfMatrix = getAverageOfOneDimensionalMatrix(frequencyArr);
			double sumOfSquaresForMatrix = getSumOfSquaredErrorsForMatrix(frequencyArr, averageOfMatrix);
			opponentSumOfSquaredErrosOfEachIssueMatrix[i] = sumOfSquaresForMatrix / frequencyArr.length;
		}
	}

	private double getAverageOfOneDimensionalMatrix(double[] matrix) {
		return getSumOfRowInOneDimensionalMatrix(matrix) / (matrix.length);
	}

	private double getSumOfRowInOneDimensionalMatrix(double[] matrix) {
		double rowSum = 0;
		for (int i = 0; i < matrix.length; i++) {
			rowSum += matrix[i];
		}
		return rowSum;
	}

	private double getSumOfSquaredErrorsForMatrix(double[] frequencyArr, double averageOfMatrix) {
		double totalOfsumOfSquares = 0;
		for (int j = 0; j < frequencyArr.length; j++) {
			totalOfsumOfSquares += Math.pow((frequencyArr[j] - averageOfMatrix), 2);
		}
		return Math.sqrt(totalOfsumOfSquares);
	}

	private void updateIsseUtilities() {
		double totalOfsumOfSquares = 0;
		for (int i = 0; i < numberOfIssues; i++) {
			totalOfsumOfSquares += opponentSumOfSquaredErrosOfEachIssueMatrix[i];
		}
		for (int i = 0; i < numberOfIssues; i++) {
			opponentEachIssueAndItsUtilityMap.put(i,
					opponentSumOfSquaredErrosOfEachIssueMatrix[i] / totalOfsumOfSquares);
		}
		System.out.println("---------OPPONENT ISSUE UTILITIES-----------");
		printIntDoubleMap(opponentEachIssueAndItsUtilityMap);
	}

	@Override
	public Action chooseAction(List<Class<? extends Action>> validActions) {
		try {
			double agentUtility = getUtilityForBid(eachIssueAndItsValuesUtilityMap, eachIssueAndItsUtilityMap,
					lastReceivedBid);
			double agentUtilityThatAgentOkayLastTime = getUtilityForBid(eachIssueAndItsValuesUtilityMap,
					eachIssueAndItsUtilityMap, lastOfferedBid);
			double oppoentUtility = getUtilityForBid(opponentEachIssueAndItsValuesUtilityMap,
					opponentEachIssueAndItsUtilityMap, lastReceivedBid);
			double oppoentUtilityThatItsNotOkay = getUtilityForBid(opponentEachIssueAndItsValuesUtilityMap,
					opponentEachIssueAndItsUtilityMap, lastOfferedBid);

			List<Bid> nextOnes = getOfferingBidsWithMostImportantIssueIsFixed();
			nextBid = getBestBidInsidePossibleBids(nextOnes);
			double agentNextUtility = getUtilityForBid(eachIssueAndItsValuesUtilityMap, eachIssueAndItsUtilityMap,
					nextBid);

			double offset = timeline instanceof DiscreteTimeline ? 1d / ((DiscreteTimeline) timeline).getTotalRounds()
					: 0d;
			double time = timeline.getTime() - offset;
			if (time > HALF_OF_TIME) {
				System.out.println(time);
			}

			// yeni gönderdiği bi sonraki göndereceğimden daha iyiyse // agentNextUtility <=
			// agentUtility
			// en son gonderdiğinden benim için daha iyi bir teklifle gelmişse //
			// agentUtilityThatAgentOkayLastTime <= agentUtility
			// en son gönderdiğinden kendisi için daha kötü bir teklifle gelmişse //
			// oppoentUtility <= oppoentUtilityThatItsNotOkay
			if (agentNextUtility <= agentUtility && agentUtilityThatAgentOkayLastTime <= agentUtility
					&& oppoentUtility <= oppoentUtilityThatItsNotOkay
					&& utilitySpace.getReservationValue() < agentUtility) {
				return new Accept(getPartyId(), lastReceivedBid);
			}
			return createOffer(validActions, time);
		} catch (Exception e) {
			lastOfferedBid = generateBestBid();
			return new Offer(getPartyId(), lastOfferedBid);
		}
	}

	private Action createOffer(List<Class<? extends Action>> validActions, double time) {
		if (lastReceivedBid == null || !validActions.contains(Accept.class)) {
			lastOfferedBid = generateBestBid();
			return new Offer(getPartyId(), lastOfferedBid);
		} else {
			if (time < HALF_OF_TIME) {
				lastOfferedBid = generateBestBid();
				return new Offer(getPartyId(), lastOfferedBid);
			} else {
				String currentTimeStr = String.valueOf(time);
				String lastOfferingTimeStr = String.valueOf(lastOfferingTime);
				String afterDotCurrentStr = currentTimeStr.substring(2, currentTimeStr.length());
				String afterDotLastOfferingStr = lastOfferingTimeStr.substring(2, lastOfferingTimeStr.length());
				// in 0.6, 0.7, 0.8, 0.9, send best bit to confuse opponent
				if(afterDotCurrentStr.charAt(0) != afterDotLastOfferingStr.charAt(0)) {
					lastOfferedBid = generateBestBid();
					lastOfferingTime = time;
					return new Offer(getPartyId(), lastOfferedBid);
				} else {
					List<Bid> offeringBidsWithMostImportantIssueIsFixed = getOfferingBidsWithMostImportantIssueIsFixed();
					lastOfferedBid = getBestBidInsidePossibleBids(offeringBidsWithMostImportantIssueIsFixed);
					lastOfferingTime = time;
					return new Offer(getPartyId(), lastOfferedBid);
				}
			}
		}
	}

	private List<Bid> getOfferingBidsWithMostImportantIssueIsFixed() {
		List<Bid> possibleOfferingBids = new ArrayList<Bid>();
		List<Bid> allBidsWithoutUtilities = outcomeSpace.getAllBidsWithoutUtilities();
		for (Bid bid : allBidsWithoutUtilities) {
			int mostImportantIssueForOpponent = getMostImportantIssueIndexForOpponent();
			for (int i = 0; i < numberOfIssues; i++) {
				// if most important issue is the one we are looking for
				if (mostImportantIssueForOpponent == i) {
					// if issue values are the same in any bid and last received
					// then add this any bid to possible bid.
					ValueDiscrete anyBidValue = (ValueDiscrete) (bid.getValue(i + 1));

					int anyBidValueIndex = getIndexOfValueInIssue(bid, i, anyBidValue.getValue());

					ValueDiscrete lastReceivedValue = (ValueDiscrete) (lastReceivedBid.getValue(i + 1));
					int lastReceivedValueIndex = getIndexOfValueInIssue(lastReceivedBid, i,
							lastReceivedValue.getValue());

					if (anyBidValueIndex == lastReceivedValueIndex) {
						if (bid.countEqualValues(lastReceivedBid) == 1) {
							possibleOfferingBids.add(bid);
						}
					}
				}
			}
		}
		return possibleOfferingBids;
	}

	// returns 0, 1, 2....
	private int getMostImportantIssueIndexForOpponent() {
		int biggestIssueNumberForOpponent = 0;
		double biggestIssueUtility = 0;
		for (Entry<Integer, Double> entry : opponentEachIssueAndItsUtilityMap.entrySet()) {
			if (biggestIssueUtility < entry.getValue()) {
				biggestIssueUtility = entry.getValue();
				biggestIssueNumberForOpponent = entry.getKey();
			}
		}
		List<Integer> mostImportantIssueIndexes = new ArrayList<Integer>();
		for (Entry<Integer, Double> entry : opponentEachIssueAndItsUtilityMap.entrySet()) {
			if (entry.getValue() == biggestIssueUtility) {
				mostImportantIssueIndexes.add(entry.getKey());
			}
		}

		if (mostImportantIssueIndexes.size() == 1) {
			return biggestIssueNumberForOpponent;
		} else {
			return getMostImportantIssueIndexForAgent(mostImportantIssueIndexes);
		}
	}

	private int getMostImportantIssueIndexForAgent(List<Integer> mostImportantIssueIndexes) {
		int biggestIssueNumberForAgent = 0;
		double biggestIssueUtilityForAgent = 0;
		for (Entry<Integer, Double> entry : eachIssueAndItsUtilityMap.entrySet()) {
			for (Integer mostImportantIssueIndexForOpponent : mostImportantIssueIndexes) {
				if (mostImportantIssueIndexForOpponent == entry.getKey()) {
					if (biggestIssueUtilityForAgent < entry.getValue()) {
						biggestIssueUtilityForAgent = entry.getValue();
						biggestIssueNumberForAgent = entry.getKey();
					}
				}
			}
		}
		return biggestIssueNumberForAgent;
	}

	private Bid getBestBidInsidePossibleBids(List<Bid> possibleBids) {
		Bid bestBid = null;
		double bestUtility = 0;
		for (Bid bid : possibleBids) {
			double agentUtility = getUtilityForBid(eachIssueAndItsValuesUtilityMap, eachIssueAndItsUtilityMap, bid);
			if (bestUtility <= agentUtility) {
				bestBid = bid;
				bestUtility = agentUtility;
			}
		}
		return bestBid;
	}

	private Bid generateBestBid() {
		List<Bid> possibleBids = outcomeSpace.getAllBidsWithoutUtilities();
		Bid bestBid = null;
		double bestUtility = 0;
		for (Bid bid : possibleBids) {
			double agentUtility = getUtilityForBid(eachIssueAndItsValuesUtilityMap, eachIssueAndItsUtilityMap, bid);
			if (bestUtility <= agentUtility) {
				bestBid = bid;
				bestUtility = agentUtility;
			}
		}
		return bestBid;
	}

	private double getUtilityForBid(Map<Integer, double[]> valueUtilityMap, Map<Integer, Double> issueUtilityMap,
			Bid bid) {
		double totalUtility = 0;
		if (bid != null) {
			for (int i = 0; i < numberOfIssues; i++) {
				Double utilityOfIssue = issueUtilityMap.get(i);
				ValueDiscrete biggerBidValue = (ValueDiscrete) (bid.getValue(i + 1));
				int indexOfValue = getIndexOfValueInIssue(bid, i, biggerBidValue.getValue());
				double[] valueUtilities = valueUtilityMap.get(i);
				double utilityOfValue = valueUtilities[indexOfValue];

				totalUtility += utilityOfIssue * utilityOfValue;
			}
			return totalUtility;
		}
		return totalUtility;
	}

	@Override
	public AbstractUtilitySpace estimateUtilitySpace() {
		return new AdditiveUtilitySpaceFactory(getDomain()).getUtilitySpace();
	}

	@Override
	public String getDescription() {
		return "ANAC 2019";
	}

}
