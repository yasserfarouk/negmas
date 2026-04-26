package agents.anac.y2016.ngent;

/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
import java.util.ArrayList;

import genius.core.Bid;
import genius.core.utility.UtilitySpace;

/**
 *
 * @author s1155032495
 */
public class OwnBidHistory {
	private ArrayList<Bid> BidHistory;
	private Bid minBidInHistory;

	public OwnBidHistory() {
		BidHistory = new ArrayList<Bid>();
	}

	protected void addBid(Bid bid, UtilitySpace utilitySpace) {
		if (BidHistory.indexOf(bid) == -1) {
			BidHistory.add(bid);
		}
		try {
			if (BidHistory.size() == 1) {
				this.minBidInHistory = BidHistory.get(0);
			} else {
				if (utilitySpace.getUtility(bid) < utilitySpace
						.getUtility(this.minBidInHistory)) {
					this.minBidInHistory = bid;
				}
			}
		} catch (Exception e) {
			System.out.println("error in addBid method of OwnBidHistory class"
					+ e.getMessage());
		}
	}

	protected Bid GetMinBidInHistory() {

		return this.minBidInHistory;
	}

	protected Bid getLastBid() {
		if (BidHistory.size() >= 1) {
			return BidHistory.get(BidHistory.size() - 1);
		} else {
			return null;
		}
	}

	protected int numOfBidsProposed() {
		return BidHistory.size();
	}

	protected Bid chooseLowestBidInHistory(UtilitySpace utilitySpace) {
		double minUtility = 100;
		Bid minBid = null;
		try {
			for (Bid bid : BidHistory) {
				if (utilitySpace.getUtility(bid) < minUtility) {
					minUtility = utilitySpace.getUtility(bid);
					minBid = bid;
				}
			}
		} catch (Exception e) {
			System.out.println("Exception in chooseLowestBidInHistory");
		}
		return minBid;
	}
}
