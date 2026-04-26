package genius.core.analysis;

import java.io.Serializable;
import java.util.Arrays;

import genius.core.Bid;

/**
 * A BidPoint is a tuple which contains the utility of a particular bid for each
 * agent.
 * 
 * @author Tim Baarslag, Dmytro Tykhonov, Mark Hendrikx
 */
public class BidPoint implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 6111098452770804678L;
	/** Bid of which the utilities are shown. */
	private Bid bid;
	/** Array holding the utility of the bid for each agent. */
	private Double[] utility;

	/**
	 * Create a BidPoint by given the bid and the tuple of utilities for that
	 * bid.
	 * 
	 * @param bid
	 *            from which the utilities are stored.
	 * @param utility
	 *            tuple of utilities of the bid.
	 */
	public BidPoint(Bid bid, Double... utility) {
		this.bid = bid;
		this.utility = utility.clone();
	}

	/**
	 * @return string representation of the object.
	 */
	@Override
	public String toString() {
		String result = "BidPoint [" + bid + ",";
		for (int i = 0; i < utility.length; i++) {
			result += "util" + i + "[" + utility[i] + "],";
		}
		return result;
	}

	/**
	 * @param obj
	 *            object to which this object is compared.
	 * @return true if this object is equal to the given object.
	 */
	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		BidPoint other = (BidPoint) obj;
		if (bid == null) {
			if (other.bid != null)
				return false;
		} else if (!bid.equals(other.bid))
			return false;
		if (this.utility.length != other.utility.length) {
			return false;
		}

		for (int i = 0; i < this.utility.length; i++) {
			if (!this.utility[i].equals(other.utility[i])) {
				return false;
			}
		}
		return true;
	}

	/**
	 * @return hashcode of this object.
	 */
	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + ((bid == null) ? 0 : bid.hashCode());
		result = prime * result + Arrays.hashCode(utility);
		return result;
	}

	/**
	 * Bid from which the utilities are represented. This bid may be null to
	 * save memory.
	 * 
	 * @return bid which utilities are represented.
	 */
	public Bid getBid() {
		return bid;
	}

	/**
	 * Returns the utility of the bid for the i'th agent (agent A = 0, etc.).
	 * 
	 * @param index
	 *            of the agent of which the utility should be returned.
	 * @return utility of the bid for the i'th agent.
	 */
	public Double getUtility(int index) {
		return utility[index];
	}

	/**
	 * Returns the utility of the bid for agent A.
	 * 
	 * @return utility for agent A.
	 */
	public Double getUtilityA() {
		return utility[0];
	}

	/**
	 * Returns the utility of the bid for agent B.
	 * 
	 * @return utility for agent B.
	 */
	public Double getUtilityB() {
		return utility[1];
	}

	/**
	 * Returns true if this BidPoint is strictly dominated by another BidPoint.
	 * A BidPoint is dominated when the utility of the other bid for at least
	 * one agent is higher and equal for the other agent.
	 * 
	 * @param other
	 *            BidPoint.
	 * @return true if "other" dominates "this".
	 */
	public boolean isStrictlyDominatedBy(BidPoint other) {
		if (this == other) {
			return false;
		}

		boolean atleastOneBetter = false;

		for (int i = 0; i < utility.length; i++) {
			if (other.utility[i] >= this.utility[i]) {
				if (other.utility[i] > this.utility[i]) {
					atleastOneBetter = true;
				}
			} else {
				return false;
			}
		}
		return atleastOneBetter;
	}

	/**
	 * Returns the distance between this BidPoint and another BidPoint. sqrt((Tx
	 * - Ox) ^ 2 + (Ty - Oy) ^ 2 + ...).
	 * 
	 * @param other
	 *            bidpoint to which the distance is calculated.
	 * @return distance to the given bidpoint.
	 */
	public double getDistance(BidPoint other) {
		double sum = 0;
		for (int i = 0; i < utility.length; i++) {
			sum += Math.pow(this.utility[i] - other.utility[i], 2);
		}
		return Math.sqrt(sum);
	}

	/**
	 * 
	 * @return sum of utilities of all parties for this bid
	 */
	public double getSocialWelfare() {
		double sum = 0;
		for (double util : utility) {
			sum += util;
		}
		return sum;
	}
}