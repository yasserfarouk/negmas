package genius.core.analysis;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

/**
 * Class which stores the Pareto-frontier. This class can be used to easily add
 * bids to the Pareto-frontier when calculating it.
 * 
 * @author Tim Baarslag
 */
public class ParetoFrontier implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 576939761881501811L;
	/** List holding the Pareto-frontier bids. */
	private List<BidPoint> frontier;

	/**
	 * Create an empty list to store the Pareto-frontier.
	 */
	public ParetoFrontier() {
		frontier = new ArrayList<BidPoint>();
	}

	/**
	 * Determines if a bid should be added to the Pareto-frontier. A bid is
	 * added when it strictly dominates a bid in the Pareto-frontier OR when it
	 * is equal to a bid in the Pareto-frontier.
	 * 
	 * @param bidpoint
	 *            bid to be added to the Pareto-frontier.
	 */
	public void mergeIntoFrontier(BidPoint bidpoint) {
		for (BidPoint f : frontier) {
			if (bidpoint.isStrictlyDominatedBy(f)) {
				// the bidpoint is dominated, and therefore not added
				return;
			}
			if (f.isStrictlyDominatedBy(bidpoint)) {
				// the bidpoint dominates a Pareto-bid, which is therefore
				// replaced
				merge(bidpoint, f);
				return;
			}
		}
		// the bid must be equal to an existing bid, therefore add it
		frontier.add(bidpoint);
	}

	/**
	 * Adds a bid to the Pareto frontier
	 * 
	 * @param newParetoBid
	 *            new Pareto bid.
	 * @param dominatedParetoBid
	 *            old Pareto bid which must now be removed.
	 */
	private void merge(BidPoint newParetoBid, BidPoint dominatedParetoBid) {
		// remove dominated bid.
		frontier.remove(dominatedParetoBid);
		// there might still be more bids which should be removed.
		ArrayList<BidPoint> toBeRemoved = new ArrayList<BidPoint>();
		for (BidPoint f : frontier) {
			if (f.isStrictlyDominatedBy(newParetoBid)) // delete dominated
														// frontier points
				toBeRemoved.add(f);
		}
		frontier.removeAll(toBeRemoved);
		frontier.add(newParetoBid);
	}

	/**
	 * Returns true if the given BidPoint is not part of the Pareto-frontier.
	 * 
	 * @param bid
	 *            to be checked if it is Pareto-optimal.
	 * @return true if NOT pareto-optimal bid.
	 */
	public boolean isBelowFrontier(BidPoint bid) {
		for (BidPoint f : this.getFrontier())
			if (bid.isStrictlyDominatedBy(f))
				return true;
		return false;
	}

	/**
	 * Order the bids based on the utility for agent A.
	 */
	public void sort() {
		Collections.sort(frontier, new Comparator<BidPoint>() {
			@Override
			public int compare(BidPoint x, BidPoint y) {
				if (x.getUtilityA() < y.getUtilityA())
					return -1;
				else if (x.getUtilityA() > y.getUtilityA())
					return 1;
				else
					return 0;
			}
		});
	}

	/**
	 * Returns the Pareto-optimal frontier.
	 * 
	 * @return Pareto-optimal frontier.
	 */
	public List<BidPoint> getFrontier() {
		return frontier;
	}
}