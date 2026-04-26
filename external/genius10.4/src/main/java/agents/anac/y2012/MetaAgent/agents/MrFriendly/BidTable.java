package agents.anac.y2012.MetaAgent.agents.MrFriendly;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map.Entry;

import genius.core.Agent;
import genius.core.Bid;
import genius.core.actions.Action;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.IssueInteger;
import genius.core.issue.IssueReal;
import genius.core.issue.Value;
import genius.core.issue.ValueDiscrete;
import genius.core.issue.ValueInteger;
import genius.core.issue.ValueReal;
import genius.core.utility.AbstractUtilitySpace;

import java.util.Random;
import java.util.Set;

public class BidTable {

	/**
	 * Agent object representing 'us'
	 */
	private Agent agent;

	/**
	 * utilitySpace which we need here
	 */
	private AbstractUtilitySpace utilitySpace;

	/**
	 * Contains all bids that we still consider, mapped to an index integer.
	 * Note that we remove the bids from this list if we offer them. However, if
	 * the list is empty, we start all over again and the list will be full
	 * again.
	 */
	private HashMap<Integer, Bid> bidTable;

	/**
	 * Contains the top TOP_BIDS_PERCENTAGE % of our bids
	 */
	private ArrayList<Bid> topXBids;

	/**
	 * Contains our utilities for the bids, mapped to the same index as the
	 * bids.
	 */
	private HashMap<Integer, Double> utilityTable;

	/**
	 * Contains estimated opponent utility for the bids, mapped to the same
	 * index as the bids.
	 */
	private HashMap<Integer, Double> estimatedOpponentUtilityTable;

	/**
	 * Counts to serve as the index for bidTable and utilityTable
	 */
	private int counter;

	/**
	 * The (dynamic) minimum utility that we want to accept from and offer to
	 * our opponent. This starts off the same value as our reservation value,
	 * but updates (increases) as we receive bids with higher utilities.
	 */
	private double minimumBidUtility;

	/**
	 * Holds the best bid that our opponent did so far.
	 */
	private Bid bestOpponentBid;

	/**
	 * The model of the opponent, given by the Group6 agent.
	 */
	private OpponentModel opponentModel;

	/**
	 * Object that keeps a history of the bids (mine and his).
	 */
	private BidHistoryTracker bidHistoryTracker;

	/**
	 * Scale on which the utility of the next bid you offer goes down
	 */
	private static final double OFFERED_UTIL_DECAY = 0.99;

	/**
	 * The percentage that determines how many bids belong to the top best bids.
	 */
	private static final double TOP_BIDS_PERCENTAGE = 0.005;

	public BidTable(Agent a, AbstractUtilitySpace us, double mbu,
			OpponentModel opponentModel) {
		agent = a;
		utilitySpace = us;
		bestOpponentBid = null;
		setMinimumBidUtility(mbu);
		this.opponentModel = opponentModel;

		bidTable = new HashMap<Integer, Bid>();
		topXBids = new ArrayList<Bid>();
		utilityTable = new HashMap<Integer, Double>();
		estimatedOpponentUtilityTable = new HashMap<Integer, Double>();

		counter = 0;
		bidHistoryTracker = new BidHistoryTracker();

		this.initializeTables();
	}

	/**
	 * Initializes tables, calls private methods that fill bid tables and
	 * utility tables
	 */
	private void initializeTables() {
		this.fillBidsArray();
		this.fillTopXHash();
	}

	/**
	 * Sets the minimum bid utility
	 * 
	 * @param mbu
	 *            double
	 */
	public void setMinimumBidUtility(double mbu) {
		if (mbu <= 1.05 && mbu >= 0)
			minimumBidUtility = mbu; // The environment allows utilities higher
										// than 1.
		else
			minimumBidUtility = 0;
	}

	/**
	 * Returns the minimum bid utility
	 * 
	 * @return double
	 */
	public double getMinimumBidUtility() {
		return minimumBidUtility;
	}

	/**
	 * Returns true iff we have offered the given bid before
	 * 
	 * @param bid
	 *            Bid
	 * @return boolean
	 */
	public boolean weHaveOfferedThisBefore(Bid bid) {
		return bidHistoryTracker.bidAlreadyDoneByMyself(bid);
	}

	/**
	 * Gets a random bid from the top TOP_BIDS_PERCENTAGE % of our list. If
	 * TOP_BIDS_PERCENTAGE % is less than 2 bids it chooses random from the top
	 * 2.
	 * 
	 * @return Bid
	 */
	public Bid getBestBid() {
		checkIfBidTableIsNotEmpty();
		Random rnd = new Random();
		int index = rnd.nextInt(topXBids.size());
		if (index > -1)
			return topXBids.get(index);

		// if topFivePercentBids doesnt have any elements for whatever reason,
		// just return the best possible bid
		return bidTable.get(this.getIndexByUtility(Collections.max(utilityTable
				.values())));
	}

	/**
	 * Get a best bid, using the estimated model of our opponent to try to find
	 * a bid that we both like
	 * 
	 * @param model
	 *            OpponentModel
	 * @return Bid
	 */
	public Bid getBestBidUsingModel() {
		checkIfBidTableIsNotEmpty();

		// If the OpponentModel isn't initialized properly, then just give back
		// the best bid.
		if (!opponentModel.isProperlyInitialized()) {
			return this.getBestBid();
		}

		// Idea:
		// Look to the (estimated) preferences of the opponent.
		// Select the (unoffered) bids from the list with a utility higher than
		// our desired value.
		// Return the one that maximizes the correspondence to the opponent
		// preferences.

		if (getLastOwnBid() == null) {
			// our first bid; choose best one - note, this is impossible
			return bidTable.get(this.getIndexByUtility(Collections
					.max(utilityTable.values())));
		}

		int indexBestBid = getIndexParetoBid();
		if (indexBestBid == -1) {
			// apparently there were no adequate bids found, or something is
			// wrong with the tables..
			// just try our old trick then

			// interpolate between our minimum expectation and our last bid
			double hi = agent.getUtility(getLastOwnBid());
			double lo = minimumBidUtility;
			double desiredValue = lo + (OFFERED_UTIL_DECAY * (hi - lo)); // our
																			// next
																			// bid
																			// should
																			// be
																			// around
																			// this
																			// value
			return bidTable.get(this.getIndexAroundUtility(desiredValue, 0.01));
		}

		return bidTable.get(indexBestBid);
	}

	/**
	 * Searches for a bid that is good for us and also good for our opponent (as
	 * far as we know), in other words, search for a bid that we expect to be on
	 * or close to the Pareto optimal frontier
	 * 
	 * @return int
	 */
	private int getIndexParetoBid() {

		// interpolate to slowly give in
		double hi = agent.getUtility(getLastOwnBid());
		double lo = minimumBidUtility;

		double desiredValue = lo + (OFFERED_UTIL_DECAY * (hi - lo)); // our next
																		// bid
																		// should
																		// be
																		// around
																		// this
																		// value
		if (hi < lo)
			desiredValue = lo; // if lo somehow surpasses hi, we just look for
								// lo

		// look for any Bid that gives us at least our desired value, and at the
		// same time maximizes opponents utility
		Set<Entry<Integer, Double>> my_util_entryset = utilityTable.entrySet();
		double maxOpponentUtility = 0;
		int index = -1;
		for (Entry<Integer, Double> entry : my_util_entryset) {
			if (entry.getValue() >= desiredValue
					&& estimatedOpponentUtilityTable.get(entry.getKey()) > maxOpponentUtility) {
				index = entry.getKey();
				maxOpponentUtility = estimatedOpponentUtilityTable.get(index);
			}
		}

		return index;
	}

	/**
	 * Updates our estimated opponent utility table with our current Opponent
	 * Model and our estimations for its preference profile
	 */
	private void updateEstimatedOpponentUtilityTable() {
		Set<Entry<Integer, Bid>> entryset = bidTable.entrySet();
		for (Entry<Integer, Bid> entry : entryset) {
			estimatedOpponentUtilityTable.put(entry.getKey(),
					opponentModel.getEstimatedUtility(entry.getValue()));
		}
	}

	/**
	 * This method updates the bidTable in the sense that it iterates over all
	 * utilities (in utiltyTable) and removes the bids (in bidTable AND
	 * utilityTable) that have a utility below the minimumBidUtility. This is
	 * necessary, because the function getIndexAroundUtility is able to find
	 * bids that are below the minimumBidUtility. Now we remove them, the
	 * bidTable would get empty and checkIfBidTableIsNotEmpty() will refill the
	 * bidTabel (and utilityTable) and we don't have to offer bids with a
	 * utility below minimumBidUtility. This method is called if and only if the
	 * minimumBidUtility has changed as a result of an opponent bid.
	 */
	public void updateBidTable() {
		// We keep track of the indexes to delete, because we can't delete them
		// in the first for-loop.
		ArrayList<Integer> indexesToDelete = new ArrayList<Integer>();

		// Loop through all utilities. If a utility is lower than the
		// minimumBidUtility, remove the corresponding bid
		// from the bidTable and add the index to the indexesToDelete.
		Set<Entry<Integer, Double>> entrySet = utilityTable.entrySet();
		for (Entry<Integer, Double> entry : entrySet) {
			// If the current entry has a utility lower than the
			// minimumBidUtility, we shouldn't offer it any more,
			// so we remove it from the table.
			if (entry.getValue() < minimumBidUtility) {
				bidTable.remove(entry.getKey()); // Remove the Bid object from
													// the bidTable.
				indexesToDelete.add(entry.getKey()); // Put this index integer
														// in the table with
														// utility indexes to
														// delete.
			}
		}

		// Delete the utility/utilities from the utility table and the opponent
		// utility table if appropriate.
		for (Integer entry : indexesToDelete) {
			utilityTable.remove(entry);
			if (estimatedOpponentUtilityTable.size() > 0)
				estimatedOpponentUtilityTable.remove(entry); // if initiated,
																// also delete
																// from
																// opponents
																// utilitytable
		}
	}

	/**
	 * Removes given bid from utilityTable and also from bidTable
	 * 
	 * @param bid
	 *            Bid
	 */
	public void removeBid(Bid bid) {
		// Get index for this bid.
		int index = -1;
		for (Entry<Integer, Bid> entry : bidTable.entrySet()) {
			if (entry.getValue().equals(bid)) {
				index = entry.getKey();
				break;
			}
		}
		// Remove the bid from the bidTable and the utilityTable.
		if (index != -1) {
			Object bidTableRemove, utilityTableRemove, opponentTableRemove;
			bidTableRemove = bidTable.remove(index);
			utilityTableRemove = utilityTable.remove(index);
			opponentTableRemove = estimatedOpponentUtilityTable.remove(index);
			// The field counter is not lowered, because is has no function
			// after filling the bidTable.

			if (bidTableRemove == null) {
				System.out
						.println("Error: unable to remove bid from bidTable, because it is not found.");
			}
			if (utilityTableRemove == null) {
				System.out
						.println("Error: unable to remove the utility from utilityTable, because it is not found.");
			}
			if (opponentTableRemove == null) {
				System.out
						.println("Error: unable to remove the utility from estimatedOpponentUtilityTable, because it is not found.");
			}
		} else {
			System.out
					.println("Error: unable to remove bid, because it is not found.");
		}
	}

	/**
	 * Retrieves the opponent action, distributes it among several classes and
	 * updates some values.
	 * 
	 * @param action
	 *            The last action from the opponent.
	 */
	public void addOpponentAction(Action action) {
		// Give the action to the BidHistoryTracker.
		bidHistoryTracker.addOpponentAction(action);
		// Give the bid to the OpponentModel.
		opponentModel.addOpponentBid(getLastOpponentBid());
		// Update our own minimumBidUtility.
		updateMinimumBidUtility();
		// Update the estimated opponent utility table now that the
		// OpponentModel is updated.
		updateEstimatedOpponentUtilityTable();
		// Update the bestOpponentBid if necessary.
		if (bestOpponentBid == null
				|| agent.getUtility(getLastOpponentBid()) > agent
						.getUtility(bestOpponentBid)) {
			bestOpponentBid = getLastOpponentBid();
		}
	}

	/**
	 * This method recalculates the minimumBidUtility.
	 */
	private void updateMinimumBidUtility() {
		Bid justReceivedBid = getLastOpponentBid();
		if (justReceivedBid != null) { // We only do this if this isn't the
										// round that we start.
			double justReceivedBidUtility = agent.getUtility(justReceivedBid);
			if (justReceivedBidUtility > getMinimumBidUtility()) {
				setMinimumBidUtility(justReceivedBidUtility);
				updateBidTable();
			}
		}
	}

	/**
	 * Returns the key to the bid with utility utility
	 * 
	 * @param utility
	 * @return Integer: Hashmap key to the bid with utility utility
	 */
	private Integer getIndexByUtility(Double utility) {
		for (Entry<Integer, Double> entry : utilityTable.entrySet()) {
			if (entry.getValue().equals(utility)) {
				return entry.getKey();
			}
		}
		return -1;
	}

	/**
	 * Same as above, but now we are not sure that 'utility' actually appears in
	 * the hashmap so we just seek the nearest bid; first we try within 0.01
	 * margins, if we can't find anything, we increase the margins, and we keep
	 * doing that until we find a proper bid
	 * 
	 * @param utility
	 * @param margin
	 * @return
	 */
	private Integer getIndexAroundUtility(Double utility, Double margin) {
		for (Entry<Integer, Double> entry : utilityTable.entrySet()) {
			if (entry.getValue() >= (utility - margin)
					&& entry.getValue() <= (utility + margin)) {
				return entry.getKey();
			}
		}
		// in case we find nothing, try again with bigger margins
		return this.getIndexAroundUtility(utility, (margin * 2));
	}

	/**
	 * Gives the last Bid the opponent offered us.
	 * 
	 * @return the last bid of the opponent.
	 */
	public Bid getLastOpponentBid() {
		return bidHistoryTracker.getLastOpponentBid();
	}

	/**
	 * Gives the last Bid we offered.
	 * 
	 * @return the last bid we offered.
	 */
	public Bid getLastOwnBid() {
		return bidHistoryTracker.getLastOwnBid();
	}

	/**
	 * Gives the best bid the opponent has offered. This value is updated by the
	 * addOpponentAction(Action) method.
	 * 
	 * @return The best bid from the opponent.
	 */
	public Bid getBestOpponentBidSoFar() {
		return bestOpponentBid;
	}

	/**
	 * Returns the number of bids we have received from our opponent
	 * 
	 * @return int
	 */
	public int getNumberOfOpponentBids() {
		return bidHistoryTracker.getNumberOfOpponentBids();
	}

	/**
	 * Returns the number of consecutive bids in which this opponent has given
	 * us a previously unoffered bid
	 * 
	 * @return int
	 */
	public int getConsecutiveBidsDifferent() {
		return bidHistoryTracker.getConsecutiveBidsDifferent();
	}

	/**
	 * Adds bid to tracker
	 * 
	 * @param bid
	 *            Bid
	 */
	public void addOwnBid(Bid bid) {
		bidHistoryTracker.addOwnBid(bid);
	}

	/**
	 * This method refills the bidTable if it became empty. This is necessary,
	 * because we remove the bids we offered from the bidTable (in
	 * chooseAction()). When all possible bids (above the minimumBidUtility) are
	 * offered, bidTable is empty and it has to be refilled to enable the agent
	 * to do more offers.
	 */
	private void checkIfBidTableIsNotEmpty() {
		if (bidTable.isEmpty()) {
			initializeTables();
		}
	}

	/**
	 * Fill the array with all bids; exclude bids that are too expensive
	 * (>1200e) or too weak (utility <= reservation_value)
	 */
	private void fillBidsArray() {
		bidTable = new HashMap<Integer, Bid>();
		utilityTable = new HashMap<Integer, Double>();
		topXBids = new ArrayList<Bid>();
		counter = 0;
		List<Issue> issues = utilitySpace.getDomain().getIssues();
		HashMap<Integer, Value> valueset = new HashMap<Integer, Value>();
		recursiveArrayFiller(issues, 0, valueset);
	}

	/**
	 * Recursive method to fill a hashmap with all possible bids.
	 * 
	 * @param issues
	 *            ArrayList of all Issue objects in this domain
	 * @param depth
	 *            The depth of recursion (should never be higher than the number
	 *            of issues)
	 * @param valueset
	 *            Hashmap containing the chosen values so far, for the first n
	 *            (depth) issues
	 */
	private void recursiveArrayFiller(List<Issue> issues, int depth,
			HashMap<Integer, Value> valueset) {
		if (depth < issues.size()) {
			Issue theIssue = issues.get(depth);
			switch (theIssue.getType()) {
			case DISCRETE:
				IssueDiscrete theIssueDiscrete = (IssueDiscrete) theIssue;
				List<ValueDiscrete> vals = theIssueDiscrete.getValues();
				for (ValueDiscrete v : vals) {
					valueset.put(theIssueDiscrete.getNumber(), v);
					recursiveArrayFiller(issues, (depth + 1), valueset);
				}
				break;
			case REAL:
				// TODO test REAL and INTEGER types
				IssueReal theIssueReal = (IssueReal) theIssue;
				double lo = theIssueReal.getLowerBound();
				double hi = theIssueReal.getUpperBound();
				int steps = theIssueReal.getNumberOfDiscretizationSteps();
				for (double i = lo; i <= hi; i += ((hi - lo) / steps)) {
					valueset.put(theIssueReal.getNumber(), new ValueReal(i));
					recursiveArrayFiller(issues, (depth + 1), valueset);
				}
				break;
			case INTEGER:
				IssueInteger theIssueInteger = (IssueInteger) theIssue;
				for (int i = theIssueInteger.getLowerBound(); i <= theIssueInteger
						.getUpperBound(); i++) {
					valueset.put(theIssueInteger.getNumber(), new ValueInteger(
							i));
					recursiveArrayFiller(issues, (depth + 1), valueset);
				}
				break;
			}
		} else if (depth == issues.size()) {
			try {
				@SuppressWarnings("unchecked")
				HashMap<Integer, Value> valuesetClone = (HashMap<Integer, Value>) valueset
						.clone();
				Bid bid = new Bid(utilitySpace.getDomain(), valuesetClone);
				if (agent.getUtility(bid) > minimumBidUtility) {
					bidTable.put(counter, bid);
					utilityTable.put(counter, agent.getUtility(bid));
					counter++;
				}
			} catch (Exception e) {
				System.out
						.println("Exception in "
								+ getClass().getName()
								+ " while creating a new Bid object: not all issues in the domain are assigned a value.\n"
								+ e.getStackTrace());
			}
		} else {
			System.out
					.println("OOPS! SOMETHING WENT WRONG! depth can never be higher than issues.size()");
		}
	}

	@SuppressWarnings("unchecked")
	private void fillTopXHash() {
		topXBids = new ArrayList<Bid>();
		// calculate and retrieve the best TOP_BIDS_PERCENTAGE% bids we have
		ArrayList as = new ArrayList(bidTable.entrySet());
		// total number of bids in our topTOP_BIDS_PERCENTAGE%, but at least 4,
		// unless we only have less than 4 possible bids (very unlikely)
		int number = Math.max(
				(int) Math.ceil(bidTable.size() * TOP_BIDS_PERCENTAGE),
				Math.min(bidTable.size(), 4));

		// sort all bids in an arraylist on utility
		Collections.sort(as, new Comparator<Entry>() {
			public int compare(Entry e1, Entry e2) {
				int iFirst = (Integer) e1.getKey();
				int iSecond = (Integer) e2.getKey();
				Double utilFirst = utilityTable.get(iFirst);
				Double utilSecond = utilityTable.get(iSecond);
				return utilSecond.compareTo(utilFirst);
			}
		});

		// fill arraylist with top TOP_BIDS_PERCENTAGE % bids
		for (int i = 0; i < number; i++) {
			topXBids.add((Bid) ((Entry) as.get(i)).getValue());
		}

		int size = topXBids.size();
		// System.out.println("Top "+TOP_BIDS_PERCENTAGE*100+"% size: "+size+". Min util: "+agent.getUtility(topXBids.get(size-1))+". Max util: "+agent.getUtility(topXBids.get(0)));
	}

	/**
	 * Returns true if we are stalling (ie 10 consecutive non-unique bids)
	 * 
	 * @return boolean
	 */
	public boolean weAreStalling() {
		return this.bidHistoryTracker.getOurStallingCoefficient() > 10;
	}
}
