/** 
 * 	OptimalBidder: using the optimal stopping rule (cutoffs) for bidding
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

/**
 * framework for optimal bidder. This agent never accepts. Tim commented:
 * OptimalBidder beslist alleen over de concessies, maar kan natuurlijk goed
 * worden gecombineerd met acceptatiecomponenten (optimal stopping bv) en
 * leercomponenten (simple frequency learning bv). Het leren is belangrijk,
 * omdat de OptimalBidder alleen een utility target oplevert. Samen met een
 * leercomponent kan daar een goed Pareto-optimaal bod bij worden gevonden, maar
 * dit ligt aan de toepassing.
 *
 */
public abstract class OptimalBidder extends Agent {
	protected double rv = -1.0;
	protected static int partitions;
	protected static int ownTotalRounds;
	protected static HashMap<Integer, Value> values;
	private static ArrayList<Double> bids;
	protected static Issue pie;
	private Action actionOfPartner = null;

	/**
	 * computation of the bid for round j. FIXME this is not a bid. Maybe it's a
	 * target utility of the bid to place?
	 * 
	 * @param round
	 *            the number of rounds left
	 * @return utility for a bid to be placed when given number of rounds are
	 *         left.
	 **/
	public abstract double bid(int j);

	/**
	 * Implementors should fill the values list when this is called.
	 * 
	 * @throws Exception
	 **/
	public abstract void getValues() throws Exception;

	/**
	 * Init is called when a next session starts with the same opponent.
	 **/
	@Override
	public void init() {
		try {
			ownTotalRounds = (getTotalRounds() - 1) / 2;
			pie = utilitySpace.getDomain().getIssues().get(0); // unique issue

			print("=====================================================================");
			print("   ownTotalRounds  = " + ownTotalRounds);
			print("   issue name      = " + pie);
			print("   issue type      = " + pie.getType());

			getValues(); // setting all the values

			rv = utilitySpace.getReservationValue();

			print("   Reservation value = " + rv);

			bids = new ArrayList<Double>(ownTotalRounds);

			for (int i = 0; i < ownTotalRounds; i++)
				bids.add(bid(i + 1));

			print("   Bids : ");

			for (int i = 0; i < ownTotalRounds; i++)
				print("\tB[" + i + "] = " + bids.get(i));

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
		return "OptimalBidder";
	}

	@Override
	public void ReceiveMessage(Action opponentAction) {
		actionOfPartner = opponentAction;
	}

	@Override
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
			print("Problem with received bid: <" + e.getMessage()
					+ ">. Cancelling bidding");
			System.out.println("\t\t\t\tErrrrr!   => " + nextBid);
			throw new IllegalStateException("internal failure", e);
		}

		return (new Offer(getAgentID(), nextBid));
	}

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
		int roundsleft = 0;
		Value optValue = null;

		print(" bids.size = " + bids.size());
		print(" getOwnRoundsLeft = " + getOwnRoundsLeft());

		TreeMap<Integer, Value> T = new TreeMap<Integer, Value>(values);

		print(" T.size = " + T.size());

		for (Integer key : T.keySet()) {
			roundsleft = getOwnRoundsLeft();

			Double targetBid = bids.get(roundsleft);
			double piePartition = (double) key / partitions;

			if (Math.abs(targetBid - piePartition) < min) {
				min = Math.abs(targetBid - piePartition);
				optValue = values.get(key);
			}
		}

		HashMap<Integer, Value> optVals = new HashMap<Integer, Value>();
		optVals.put(pie.getNumber(), optValue);
		Bid ToBid = new Bid(utilitySpace.getDomain(), optVals);

		print(" ToBid = " + ToBid);

		return ToBid; // optimal bid

	}

	@Override
	public String getDescription() {
		return "using the optimal stopping rule (cutoffs) for bidding";
	}

}
