package agents.anac.y2011.TheNegotiator;

import java.util.ArrayList;

import genius.core.bidding.BidDetails;
import genius.core.timeline.TimeLineInfo;

/**
 * The TimeManager class is used for time-related functions.
 * 
 * @author Alex Dirkzwager, Mark Hendrikx, Julian de Ruiter
 */
public class TimeManager {

	// used to store the timeline of the negotiation
	private TimeLineInfo timeline;
	// total time steps considered
	private final double totalTime = 180;
	// times when phases end for a non discount domain
	private final double[] endPhases1 = { 140.0 / totalTime, 35.0 / totalTime, 5.0 / totalTime };
	private final double maxThresArray1[] = { 0, 0, 0, 0 };
	private final double[] endPhases2 = { 0, 0, 0 };
	private final double maxThresArray2[] = { 0, 0, 0, 0 };
	// collection of bids
	private BidsCollection bidsCollection;

	// array for storing how many bids are in a certain threshold interval
	private final int propArray[] = { 0, 0, 0 };
	// queue holding last 30 lapsed time
	private Queue queue = new Queue();
	// queue size
	private int queueSize = 15;
	// last time
	double lastTime = 0;

	double discount;

	/**
	 * Creates a TimeManager-object which stores the timeline of the
	 * negotiation.
	 *
	 * @param timeline
	 *            of the negotiation
	 */
	public TimeManager(TimeLineInfo timeline, double discount, BidsCollection bidsCollection) {
		this.timeline = timeline;
		this.bidsCollection = bidsCollection;
		this.discount = discount;

		if (this.discount >= 1.0) {
			this.discount = 0; // compatibility with old discount method
		}

		// no discounts
		calculateEndPhaseThresholds();

		// discounts
		if (this.discount != 0) {
			calculatePropArray();
			calculateEndPhases();
		}
	}

	private void calculateEndPhaseThresholds() {
		maxThresArray1[0] = bidsCollection.getPossibleBids().get(0).getMyUndiscountedUtil();
		maxThresArray1[3] = bidsCollection.getPossibleBids().get(bidsCollection.getPossibleBids().size() - 1)
				.getMyUndiscountedUtil();
		double range = maxThresArray1[0] - maxThresArray1[3];
		maxThresArray1[1] = maxThresArray1[0] - ((1.0 / 8) * range);
		maxThresArray1[2] = maxThresArray1[0] - ((3.0 / 8) * range);
	}

	/**
	 * Returns the current phase of the negotiation.
	 * 
	 * @return phase of the negotiation
	 */
	public int getPhase(double time) {

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
		return phase;
	}

	/**
	 * Returns the time dependent threshold which specifies how good a bid of an
	 * opponent should be to be accepted. This threshold is also used as a
	 * minimum for the utility of the bid of our agent.
	 * 
	 * @return threshold
	 */
	public double getThreshold(double time) {
		int phase = getPhase(time);
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
			threshold = maxThresArray[0] - ((time - discountActive) / (endPhases[0] - discountActive))
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
		default:
			ErrorLogger.log("Unknown phase: " + phase);
			break;
		}
		return threshold;
	}

	public int getMovesLeft() {
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
			movesLeft = (int) Math.floor((1.0 - timeline.getTime()) / (total / (double) queueSize));
		}
		return movesLeft;
	}

	/**
	 * Calculate how many possible bids are within a certain threshold interval.
	 * This is done for all the bins (phases).
	 */
	public void calculatePropArray() {
		ArrayList<BidDetails> posBids = bidsCollection.getPossibleBids();

		double max = getThreshold(discount); // 0.0001 is just to be sure :)
		double min = bidsCollection.getPossibleBids().get(bidsCollection.getPossibleBids().size() - 1)
				.getMyUndiscountedUtil();
		double range = max - min;
		double rangeStep = range / 3;

		for (int i = 0; i < posBids.size(); i++) {
			double util = posBids.get(i).getMyUndiscountedUtil();

			// calculate if a utility of a bid is within a certain interval.
			// Intervals should be calculated!!!!
			if (util >= max - rangeStep && util <= max) {
				propArray[0]++;
			} else if (util >= max - 2 * rangeStep && util < max - rangeStep) {
				propArray[1]++;
			} else if (util >= max - 3 * rangeStep && util < max - 2 * rangeStep) {
				propArray[2]++;
			}
		}
		// find the maximum possible utility within a bin (plus the lowest bid
		// of the last bin)
		ArrayList<BidDetails> bidsCol = bidsCollection.getPossibleBids();
		maxThresArray2[0] = max;
		if (propArray[0] == 0) {
			maxThresArray2[1] = bidsCol.get(0).getMyUndiscountedUtil();
		} else {
			maxThresArray2[1] = bidsCol.get(propArray[0] - 1).getMyUndiscountedUtil(); // -1
																						// to
																						// correct
																						// for
																						// array
																						// offset
																						// of
																						// zero
		}
		if (propArray[0] + propArray[1] - 1 >= 0) {
			maxThresArray2[2] = bidsCol.get(propArray[0] + propArray[1] - 1).getMyUndiscountedUtil();
		} else {
			maxThresArray2[2] = bidsCol.get(0).getMyUndiscountedUtil();
		}
		maxThresArray2[3] = min;
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

		endPhases2[0] = discount + (((double) propArray[0] / (double) sum) * (1 - discount));
		endPhases2[1] = (((double) propArray[1] / (double) sum) * (1 - discount));
		endPhases2[2] = (((double) propArray[2] / (double) sum) * (1 - discount));
	}

	public double getTime() {
		return timeline.getTime();
	}
}