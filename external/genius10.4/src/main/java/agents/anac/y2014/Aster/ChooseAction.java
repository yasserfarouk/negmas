package agents.anac.y2014.Aster;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.Random;

import genius.core.Bid;
import genius.core.utility.AbstractUtilitySpace;

public class ChooseAction {
	private AbstractUtilitySpace utilitySpace;
	private Random rnd;
	private static final int NUMBER_ITERATIONS = 1000; // ç¹°ã‚Šè¿”ã�—å›žæ•°
	private static final double EPSILON = 0.3D;

	public ChooseAction(AbstractUtilitySpace utilitySpace) {
		this.utilitySpace = utilitySpace;
		this.rnd = new Random();
	}

	/**
	 * Select Next Bid (for MultiList)
	 *
	 */
	public Bid nextOfferingBid(double bidTarget,
			ArrayList<Bid> selectMyBidList, ArrayList<Bid> opponentBidList)
			throws Exception {
		Bid nextBid = null;
		double nextBidUtil = 0.0;
		int iteration = 0;

		do {
			if (Math.random() > EPSILON) {
				nextBid = selectWeightedBid(opponentBidList);
			} else {
				nextBid = selectRandomBid(selectMyBidList,
						selectMyBidList.size());
			}
			nextBidUtil = utilitySpace.getUtility(nextBid);
		} while ((nextBidUtil < bidTarget) && (++iteration < NUMBER_ITERATIONS));

		return nextBid;
	}

	/**
	 * SelectionNextBid
	 *
	 */
	public Bid nextOfferingBid(double bidTarget, ArrayList<Bid> bidList,
			boolean select_flag) throws Exception {
		Bid nextBid = null;
		double nextBidUtil = 0.0;
		int iteration = 0;
		int listSize = bidList.size();

		if (listSize == 1) {
			return bidList.get(0);
		}

		do {
			if (select_flag) {
				nextBid = selectRandomBid(bidList, listSize);
			} else {
				nextBid = selectWeightedBid(bidList);
			}
			nextBidUtil = utilitySpace.getUtility(nextBid);
		} while ((nextBidUtil < bidTarget) && (++iteration < NUMBER_ITERATIONS));

		return nextBid;
	}

	// SelectNextBid(Weighted)
	private Bid selectWeightedBid(ArrayList<Bid> bidList) {
		int totalUtil = 0;
		int index = 0;
		int listSize = bidList.size();

		if (listSize == 1) {
			return bidList.get(0);
		}

		try {
			for (Iterator<Bid> it = bidList.iterator(); it.hasNext();) {
				int bidUtility = (int) (utilitySpace.getUtility(it.next()) * 100);
				totalUtil += bidUtility;
			}

			int rndint = rnd.nextInt(totalUtil);

			for (Iterator<Bid> it = bidList.iterator(); it.hasNext(); index++) {
				int bidUtility = (int) (utilitySpace.getUtility(it.next()) * 100);
				if (rndint < bidUtility) {
					return bidList.get(index);
				} else {
					rndint -= bidUtility;
				}
			}
		} catch (Exception e) {
			e.printStackTrace();
		}

		// ifFailure
		return selectRandomBid(bidList, bidList.size());
	}

	// SelectNextBid(Random)
	private Bid selectRandomBid(ArrayList<Bid> bidList, int listSize) {
		int rand = rnd.nextInt(listSize);
		return bidList.get(rand);
	}
}