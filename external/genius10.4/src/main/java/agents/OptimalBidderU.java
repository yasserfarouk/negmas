/** 
 * 	OptimalBidderU: using the optimal stopping rule (cutoffs) for bidding
 *  @author rafik		
 **/

package agents;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.TreeMap;

import genius.core.Agent;
import genius.core.Bid;
import genius.core.actions.Action;
import genius.core.actions.Offer;
import genius.core.issue.Issue;
import genius.core.issue.Value;
import genius.core.timeline.DiscreteTimeline;

public abstract class OptimalBidderU extends Agent {
	private static double rv = -1.0;
	protected static int partitions;
	private static int ownTotalSessions;
	protected static HashMap<Integer, Value> values;
	private static ArrayList<Double> bids, utilities;
	protected static Issue pie;
	private Action actionOfPartner = null;

	/**
	 * computation of the bid for round j
	 * 
	 * @param round
	 *            j
	 * @return bid value
	 **/
	public abstract double bid(int j);

	public abstract double utility(int j);

	/**
	 * depending on the agent's utility space, the reservation value will have
	 * to be acquired in a specific manner
	 * 
	 * @param double arg
	 * @return double rv
	 * @throws Exception
	 **/
	public abstract double getReservationValue(double arg) throws Exception;

	/**
	 * Init is called when a next session starts with the same opponent.
	 **/
	public void init() {
		try {
			ownTotalSessions = (getTotalRounds() - 1) / 2;
			pie = utilitySpace.getDomain().getIssues().get(0); // unique issue

			print("=====================================================================");
			print("   OwntotalSessions = " + ownTotalSessions);
			print("   issue name = " + pie);
			print("   issue type = " + pie.getType());

			rv = getReservationValue(rv); // sets rvB

			bids = new ArrayList<Double>(ownTotalSessions);
			utilities = new ArrayList<Double>(ownTotalSessions);

			for (int i = 0; i < ownTotalSessions; i++) {
				bids.add(bid(i + 1));
				utilities.add(utility(i + 1));
			}
			print(" OwntotalSessions = " + ownTotalSessions);

			for (int i = 0; i < ownTotalSessions; i++)
				print(" \t B[" + i + "] = " + bids.get(i) + " \t U[" + i
						+ "] = " + utilities.get(i));

			print("\n=====================================================================");
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	@Override
	public String getVersion() {
		return "2.0 (Genius 4.2)";
	}

	@Override
	public String getName() {
		return "OptimalBidderU";
	}

	public void ReceiveMessage(Action opponentAction) {
		actionOfPartner = opponentAction;
	}

	public Action chooseAction() {
		Action action = null;
		try {
			if (actionOfPartner == null) {
				action = chooseOptimalBidAction();
			}
			if (actionOfPartner instanceof Offer) {
				action = chooseOptimalBidAction();
			}
		} catch (Exception e) {
			print("Exception in ChooseAction:" + e.getMessage());
		}
		return action;
	}

	/**
	 * Wrapper for getOptimalBid, for convenience
	 * 
	 * @return new Bid()
	 **/
	private Action chooseOptimalBidAction() {
		Bid nextBid = null;
		try {
			nextBid = getOptimalBid();
		} catch (Exception e) {
			print("Problem with received bid:" + e.getMessage()
					+ ". cancelling bidding");
		}
		return (new Offer(getAgentID(), nextBid));
	}

	// discrete rounds' methods
	public int getRound() {
		return ((DiscreteTimeline) timeline).getRound();
	}

	public int getRoundsLeft() {
		return ((DiscreteTimeline) timeline).getRoundsLeft();
	}

	public int getOwnRoundsLeft() {
		return ((DiscreteTimeline) timeline).getOwnRoundsLeft();
	}

	public int getTotalRounds() {
		return ((DiscreteTimeline) timeline).getTotalRounds();
	}

	public double getTotalTime() {
		return ((DiscreteTimeline) timeline).getTotalTime();
	}

	// trace
	void print(String s) {
		System.out.println("############  " + s);
	}

	/**
	 * 
	 *
	 */
	private Bid getOptimalBid() throws Exception {
		print("############   B's  ####################################");
		print(" Round         = " + getRound());
		print(" RoundsLeft    = " + getRoundsLeft());
		print(" OwnRoundsLeft = " + getOwnRoundsLeft());
		print(" TotalRounds   = " + getTotalRounds());
		print(" TotalTime     = " + getTotalTime());

		double min = 1.0;
		Value optValue = null;

		Double targetUtility = utilities.get(getOwnRoundsLeft()); // cutoff
																	// utility :
																	// U[roundsleft]
		double targetBid = targetUtility + rv; // Finding the x for which u_B(x)
												// = targetUtility
		print(" targetBidToBid  = " + targetBid);

		for (Integer key : new TreeMap<Integer, Value>(values).keySet()) {
			// find the closest bid to targetBidToBid
			double piePartition = (double) key / partitions;
			if (Math.abs(targetBid - piePartition) < min) {
				min = Math.abs(targetBid - piePartition);
				optValue = values.get(key);
			}
		}

		HashMap<Integer, Value> optVals = new HashMap<Integer, Value>();
		optVals.put(pie.getNumber(), optValue);
		return new Bid(utilitySpace.getDomain(), optVals); // optimal bid

	}
}

// End 