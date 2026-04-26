package negotiator.boaframework.sharedagentstate.anac2011;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;

import genius.core.Bid;
import genius.core.bidding.BidDetails;
import genius.core.bidding.BidDetailsStrictSorterUtility;
import genius.core.boaframework.NegotiationSession;
import genius.core.boaframework.SharedAgentState;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.Value;
import genius.core.issue.ValueDiscrete;
import genius.core.misc.Pair;
import genius.core.misc.Queue;

/**
 * This is the shared code of the acceptance condition and bidding strategy of
 * ANAC 2011 TheNegotiator. The code was taken from the ANAC2011 TheNegotiator
 * and adapted to work within the BOA framework.
 * 
 * @author Mark Hendrikx
 */
public class TheNegotiatorSAS extends SharedAgentState {

	private final boolean TEST_EQUIVALENCE = false;

	private NegotiationSession negotiationSession;

	private final double totalTime = 180;
	private final double[] endPhases1 = { 140.0 / totalTime, 35.0 / totalTime,
			5.0 / totalTime };
	private double[] endPhases2;
	private double[] maxThresArray1;
	private double[] maxThresArray2;
	private int propArray[];
	private ArrayList<BidDetails> possibleBids;

	// queue holding last 30 lapsed time
	private Queue queue = new Queue();
	private double lastTime;
	// queue size
	private int queueSize;
	private double discount;

	private int phase;
	private double threshold;
	private int movesLeft;

	public TheNegotiatorSAS(NegotiationSession negoSession) {
		negotiationSession = negoSession;
		NAME = "TheNegotiator";
		possibleBids = new ArrayList<BidDetails>();
		createAllBids();
		if (TEST_EQUIVALENCE) {
			Collections.sort(possibleBids, new BidDetailsStrictSorterUtility());
		} else {
			Collections.sort(possibleBids);
		}

		endPhases2 = new double[3];
		Arrays.fill(endPhases2, 0);

		maxThresArray1 = new double[4];
		Arrays.fill(maxThresArray1, 0);

		maxThresArray2 = new double[4];
		Arrays.fill(maxThresArray2, 0);

		propArray = new int[3];
		Arrays.fill(propArray, 0);

		queueSize = 15;
		lastTime = 0;
		discount = negoSession.getDiscountFactor();

		// no discounts
		calculateEndPhaseThresholds();

		// discounts
		if (negotiationSession.getDiscountFactor() != 0) {
			calculatePropArray();
			calculateEndPhases();
		}
	}

	public ArrayList<BidDetails> getPossibleBids() {
		return possibleBids;
	}

	/**
	 * Calculates the time which should be spend on each phase based on the
	 * distribution of the utilities of the bids.
	 */
	public void calculateEndPhases() {
		int sum = 0;
		for (int i = 0; i < propArray.length; i++) {
			sum += propArray[i];
		}

		endPhases2[0] = discount
				+ (((double) propArray[0] / (double) sum) * (1 - discount));
		endPhases2[1] = (((double) propArray[1] / (double) sum) * (1 - discount));
		endPhases2[2] = (((double) propArray[2] / (double) sum) * (1 - discount));
	}

	/**
	 * Returns the current phase of the negotiation.
	 * 
	 * @return phase of the negotiation
	 */
	public int calculateCurrentPhase(double time) {

		int phase = 1;
		double[] endPhases = endPhases1;

		if (discount != 0 && time > discount) {
			endPhases = endPhases2;
		}

		if (time > (endPhases[1] + endPhases[0])) {
			double lapsedTime = time - lastTime;
			queue.enqueue(lapsedTime);
			if (queue.size() > queueSize) {
				queue.dequeue();
			}
			phase = 3;
		} else if (time > endPhases[0]) {
			phase = 2;
		}
		lastTime = time;

		this.phase = phase;
		return phase;
	}

	/**
	 * Returns the time dependent threshold which specifies how good a bid of an
	 * opponent should be to be accepted. This threshold is also used as a
	 * minimum for the utility of the bid of our agent.
	 * 
	 * @return threshold
	 */
	public double calculateThreshold(double time) {
		int phase = calculateCurrentPhase(time);
		double threshold = 0.98; // safe value
		double[] maxThresArray = maxThresArray1;
		double[] endPhases = endPhases1;

		if (discount != 0 && time > discount) {
			maxThresArray = maxThresArray2;
			endPhases = endPhases2;
		}
		double discountActive = discount;
		if (time <= discount) {
			discountActive = 0;
		}

		switch (phase) {
		case 1:
			threshold = maxThresArray[0]
					- ((time - discountActive) / (endPhases[0] - discountActive))
					* (maxThresArray[0] - maxThresArray[1]);
			break;
		case 2:
			threshold = maxThresArray[1]
					- (((time - endPhases[0]) / (endPhases[1])) * (maxThresArray[1] - maxThresArray[2]));
			break;
		case 3:
			threshold = maxThresArray[2]
					- (((time - endPhases[0] - endPhases[1]) / (endPhases[2])) * (maxThresArray[2] - maxThresArray[3]));
			break;
		}
		this.threshold = threshold;
		return threshold;
	}

	public int calculateMovesLeft() {
		int movesLeft = -1;

		if (queue.isEmpty()) {
			movesLeft = 500; // to avoid an error
		} else {
			Double[] lapsedTimes = queue.toArray();
			double total = 0;
			for (int i = 0; i < queueSize; i++) {
				if (lapsedTimes[i] != null) {
					total += lapsedTimes[i];
				}
			}
			movesLeft = (int) Math.floor((1.0 - negotiationSession.getTime())
					/ (total / (double) queueSize));
		}
		this.movesLeft = movesLeft;
		return movesLeft;
	}

	public void calculateEndPhaseThresholds() {
		maxThresArray1[0] = possibleBids.get(0).getMyUndiscountedUtil();
		int size = possibleBids.size();
		maxThresArray1[3] = possibleBids.get(size - 1).getMyUndiscountedUtil();
		double range = maxThresArray1[0] - maxThresArray1[3];
		maxThresArray1[1] = maxThresArray1[0] - ((1.0 / 8) * range);
		maxThresArray1[2] = maxThresArray1[0] - ((3.0 / 8) * range);
	}

	/**
	 * Calculate how many possible bids are within a certain threshold interval.
	 * This is done for all the bins (phases).
	 */
	public void calculatePropArray() {
		double max = calculateThreshold(discount); // 0.0001 is just to be sure
													// :)
		double min = possibleBids.get(possibleBids.size() - 1)
				.getMyUndiscountedUtil();
		double range = max - min;
		double rangeStep = range / 3;

		for (int i = 0; i < possibleBids.size(); i++) {
			double util = possibleBids.get(i).getMyUndiscountedUtil();

			// calculate if a utility of a bid is within a certain interval.
			// Intervals should be calculated!!!!
			if (util >= max - rangeStep && util <= max) {
				propArray[0]++;
			} else if (util >= max - 2 * rangeStep && util < max - rangeStep) {
				propArray[1]++;
			} else if (util >= max - 3 * rangeStep
					&& util < max - 2 * rangeStep) {
				propArray[2]++;
			}
		}
		// find the maximum possible utility within a bin (plus the lowest bid
		// of the last bin)
		maxThresArray2[0] = max;

		if (propArray[0] == 0) {
			maxThresArray2[1] = possibleBids.get(0).getMyUndiscountedUtil();
		} else {
			maxThresArray2[1] = possibleBids.get(propArray[0] - 1)
					.getMyUndiscountedUtil(); // -1 to correct for array offset
												// of zero
		}

		if (propArray[0] == 0 && propArray[1] == 0) {
			maxThresArray2[2] = possibleBids.get(0).getMyUndiscountedUtil();

		} else {
			maxThresArray2[2] = possibleBids.get(
					propArray[0] + propArray[1] - 1).getMyUndiscountedUtil();
		}
		maxThresArray2[3] = min;
	}

	public int[] getPropArray() {
		return propArray;
	}

	public int getPhase() {
		return phase;
	}

	public double getThreshold() {
		return threshold;
	}

	public int getMovesLeft() {
		return movesLeft;
	}

	/**
	 * Create all possible bids using a call to the recursive Cartestian product
	 * options generator.
	 */
	private void createAllBids() {
		List<Issue> issues = negotiationSession.getUtilitySpace().getDomain()
				.getIssues();

		ArrayList<IssueDiscrete> discreteIssues = new ArrayList<IssueDiscrete>();

		for (Issue issue : issues) {
			discreteIssues.add((IssueDiscrete) issue);
		}

		ArrayList<ArrayList<Pair<Integer, ValueDiscrete>>> result = generateAllBids(
				discreteIssues, 0);

		for (ArrayList<Pair<Integer, ValueDiscrete>> bidSet : result) {
			HashMap<Integer, Value> values = new HashMap<Integer, Value>();
			for (Pair<Integer, ValueDiscrete> pair : bidSet) {
				values.put(pair.getFirst(), pair.getSecond());
			}
			try {
				Bid bid = new Bid(negotiationSession.getUtilitySpace()
						.getDomain(), values);
				double utility = negotiationSession.getUtilitySpace()
						.getUtility(bid);
				possibleBids.add(new BidDetails(bid, utility, -1.0));
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
	}

	/**
	 * The recursive Cartestian product options generator. Generates all
	 * possible bids.
	 * 
	 * @param issueList
	 * @param i
	 *            , parameter used in the recursion
	 * @return a list of a list with pairs of integer (issue at stake) and a
	 *         value (the chosen option)
	 */
	private ArrayList<ArrayList<Pair<Integer, ValueDiscrete>>> generateAllBids(
			ArrayList<IssueDiscrete> issueList, int i) {

		// stop condition
		if (i == issueList.size()) {
			// return a list with an empty list
			ArrayList<ArrayList<Pair<Integer, ValueDiscrete>>> result = new ArrayList<ArrayList<Pair<Integer, ValueDiscrete>>>();
			result.add(new ArrayList<Pair<Integer, ValueDiscrete>>());
			return result;
		}

		ArrayList<ArrayList<Pair<Integer, ValueDiscrete>>> result = new ArrayList<ArrayList<Pair<Integer, ValueDiscrete>>>();
		ArrayList<ArrayList<Pair<Integer, ValueDiscrete>>> recursive = generateAllBids(
				issueList, i + 1); // recursive call

		// for each element of the first list of input
		for (int j = 0; j < issueList.get(i).getValues().size(); j++) {
			// add the element to all combinations obtained for the rest of the
			// lists
			for (int k = 0; k < recursive.size(); k++) {
				// copy a combination from recursive
				ArrayList<Pair<Integer, ValueDiscrete>> newList = new ArrayList<Pair<Integer, ValueDiscrete>>();
				for (Pair<Integer, ValueDiscrete> set : recursive.get(k)) {
					newList.add(set);
				}
				// add element of the first list
				ValueDiscrete value = issueList.get(i).getValues().get(j);
				int issueNr = issueList.get(i).getNumber();
				newList.add(new Pair<Integer, ValueDiscrete>(issueNr, value));

				// add new combination to result
				result.add(newList);
			}
		}
		return result;
	}
}