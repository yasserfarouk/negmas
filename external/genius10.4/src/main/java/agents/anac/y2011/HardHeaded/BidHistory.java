package agents.anac.y2011.HardHeaded;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map.Entry;

import genius.core.Bid;
import genius.core.issue.Issue;
import genius.core.issue.ValueDiscrete;
import genius.core.utility.AdditiveUtilitySpace;
import genius.core.utility.UtilitySpace;

/**
 * Keeps track of all bids exchanged by both the agent and its opponent. Also
 * has some tools that work on history of exchanged bids.
 * 
 */
public class BidHistory {
	private List<Entry<Double, Bid>> myBids;
	private List<Bid> opponentBids;
	private UtilitySpace utilitySpace;

	/**
	 * BidHistory class constructor.
	 * 
	 * @param utilSpace
	 *            a {@link AdditiveUtilitySpace} to be set for utility
	 *            calculations of stored bids.
	 * @return An object of this type used to keep track of exchanged bids.
	 */
	public BidHistory(UtilitySpace utilSpace) {

		utilitySpace = utilSpace;
		myBids = new ArrayList<Entry<Double, Bid>>();
		opponentBids = new ArrayList<Bid>();

	}

	/**
	 * Adds a new bid {@link Entry} to the end of agent's own bids.
	 * 
	 * @param pBid
	 *            passed bid entry
	 */
	public void addMyBid(Entry<Double, Bid> pBid) {
		if (pBid == null)
			throw new IllegalArgumentException("pBid can't be null.");
		myBids.add(pBid);

	}

	/**
	 * @return the size (number) of offers already made
	 */
	public int getMyBidCount() {
		return myBids.size();
	}

	/**
	 * retrieves a bid {@link Entry} from the agent's bid list
	 * 
	 * @param pIndex
	 *            index of the bid
	 * @return a bid from the list
	 */
	public Entry<Double, Bid> getMyBid(int pIndex) {
		return myBids.get(pIndex);
	}

	/**
	 * retrieves last bid {@link Entry} from the agent's bid list
	 * 
	 * @return a bid from the list
	 */
	public Entry<Double, Bid> getMyLastBid() {
		Entry<Double, Bid> result = null;
		if (getMyBidCount() > 0) {
			result = myBids.get(getMyBidCount() - 1);
		}
		return result;
	}

	/**
	 * Adds a new bid {@link Entry} to the end of oppenent's bids.
	 * 
	 * @param pBid
	 *            passed bid entry
	 */
	public void addOpponentBid(Bid pBid) {
		if (pBid == null)
			throw new IllegalArgumentException("vBid can't be null.");
		opponentBids.add(pBid);
	}

	/**
	 * @return the number of bids the opponent has made
	 */
	public int getOpponentBidCount() {
		return opponentBids.size();
	}

	/**
	 * retrieves a bid from the opponent's bid list
	 * 
	 * @param pIndex
	 *            index of the bid
	 * @return a bid from the list
	 */
	public Bid getOpponentBid(int pIndex) {
		return opponentBids.get(pIndex);
	}

	/**
	 * retrieves last bid from the opponent's bid list
	 * 
	 * @return a bid from the list
	 */
	public Bid getOpponentLastBid() {
		Bid result = null;
		if (getOpponentBidCount() > 0) {
			result = opponentBids.get(getOpponentBidCount() - 1);
		}
		return result;
	}

	/**
	 * retrieves second last bid from the opponent's bid list
	 * 
	 * @return a bid from the list
	 */
	public Bid getOpponentSecondLastBid() {
		Bid result = null;
		if (getOpponentBidCount() > 1) {
			result = opponentBids.get(getOpponentBidCount() - 2);
		}
		return result;
	}

	/**
	 * receives two bids as arguments and returns a {@link HashMap} that
	 * contains for each issue whether or not its value is different between the
	 * two bids.
	 * 
	 * @param first
	 * @param second
	 * @return a {@link HashMap} with keys equal to issue IDs and with values 1
	 *         if different issue value observed and 0 if not.
	 */
	public HashMap<Integer, Integer> BidDifference(Bid first, Bid second) {

		HashMap<Integer, Integer> diff = new HashMap<Integer, Integer>();
		try {
			for (Issue i : utilitySpace.getDomain().getIssues()) {
				diff.put(i.getNumber(), (((ValueDiscrete) first.getValue(i
						.getNumber())).equals((ValueDiscrete) second.getValue(i
						.getNumber()))) ? 0 : 1);
			}
		} catch (Exception ex) {
			ex.printStackTrace();
		}

		return diff;
	}

	/**
	 * For the last two bids of the opponent returns a {@link HashMap} that
	 * contains for each issue whether or not its value is different between the
	 * two bids.
	 * 
	 * @return a {@link HashMap} with keys equal to issue IDs and with values 1
	 *         if different issue value observed and 0 if not.
	 */
	public HashMap<Integer, Integer> BidDifferenceofOpponentsLastTwo() {

		if (getOpponentBidCount() < 2)
			throw new ArrayIndexOutOfBoundsException();
		return BidDifference(getOpponentLastBid(), getOpponentSecondLastBid());
	}

}
