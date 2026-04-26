package agents.anac.y2014.BraveCat.necessaryClasses;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import genius.core.Bid;
import genius.core.bidding.BidDetails;
import genius.core.bidding.BidDetailsSorterTime;
import genius.core.bidding.BidDetailsSorterUtility;
import genius.core.utility.AbstractUtilitySpace;
import genius.core.utility.AdditiveUtilitySpace;

public class BidHistory implements Serializable {
	private List<Bid> opponentNLastBidHistory;
	public int numberOfTotalBidsInBidHistory = 20;
	public int numberOfUniqueBidsInBidHistory = 0;
	private static final long serialVersionUID = 1663962498632353562L;
	private List<BidDetails> bidList;
	private final boolean TEST_EQUIVALENCE = false;

	public BidHistory(List<BidDetails> bids) {
		this.bidList = bids;
		opponentNLastBidHistory = new ArrayList();
	}

	public BidHistory() {
		this.bidList = new ArrayList();
		opponentNLastBidHistory = new ArrayList();
	}

	public BidHistory filterBetweenTime(double t1, double t2) {
		return filterBetween(0.0D, 1.1D, t1, t2);
	}

	public BidHistory filterBetweenUtility(double minU, double maxU) {
		if (minU == maxU) {
			return filterUtility(minU);
		}
		BidHistory bidHistory = new BidHistory();
		for (BidDetails b : this.bidList) {
			if ((minU < b.getMyUndiscountedUtil())
					&& (b.getMyUndiscountedUtil() <= maxU))
				bidHistory.add(b);
		}
		return bidHistory;
	}

	public BidHistory filterBetween(double minU, double maxU, double minT,
			double maxT) {
		BidHistory bidHistory = new BidHistory();
		for (BidDetails b : this.bidList) {
			if ((minU < b.getMyUndiscountedUtil())
					&& (b.getMyUndiscountedUtil() <= maxU)
					&& (minT < b.getTime()) && (b.getTime() <= maxT))
				bidHistory.add(b);
		}
		return bidHistory;
	}

	public BidHistory filterUtility(double u) {
		BidHistory bidHistory = new BidHistory();
		for (BidDetails b : this.bidList)
			if (b.getMyUndiscountedUtil() == u)
				bidHistory.add(b);
		return bidHistory;
	}

	public BidHistory discountedFilterBetween(double minU, double maxU,
			double minT, double maxT, AdditiveUtilitySpace utilSpace) {
		BidHistory bidHistory = new BidHistory();
		for (BidDetails b : this.bidList) {
			if ((minU < utilSpace.getUtilityWithDiscount(b.getBid(),
					b.getTime()))
					&& (utilSpace.getUtilityWithDiscount(b.getBid(),
							b.getTime()) <= maxU)
					&& (minT < b.getTime())
					&& (b.getTime() <= maxT))
				bidHistory.add(b);
		}
		return bidHistory;
	}

	public void add(BidDetails bid) {
		if (!this.opponentNLastBidHistory.contains(bid.getBid()))
			numberOfUniqueBidsInBidHistory++;
		this.opponentNLastBidHistory.add(bid.getBid());
		if (this.opponentNLastBidHistory.size() == numberOfTotalBidsInBidHistory) {
			Bid tempBid = this.opponentNLastBidHistory.remove(0);
			if (!this.opponentNLastBidHistory.contains(tempBid))
				numberOfUniqueBidsInBidHistory--;
		}
		this.bidList.add(bid);
	}

	public List<BidDetails> getHistory() {
		return this.bidList;
	}

	public BidDetails getLastBidDetails() {
		BidDetails bid = null;
		if (this.bidList.size() > 0) {
			bid = (BidDetails) this.bidList.get(this.bidList.size() - 1);
		}
		return bid;
	}

	public Bid getLastBid() {
		BidDetails lastBidDetails = getLastBidDetails();
		if (lastBidDetails == null)
			return null;
		return lastBidDetails.getBid();
	}

	public BidDetails getFirstBidDetails() {
		return (BidDetails) this.bidList.get(0);
	}

	public BidDetails getBestBidDetails() {
		double max = (-1.0D / 0.0D);
		BidDetails bestBid = null;
		for (BidDetails b : this.bidList) {
			double utility = b.getMyUndiscountedUtil();
			if (utility >= max) {
				max = utility;
				bestBid = b;
			}
		}
		return bestBid;
	}

	public BidDetails getBestDiscountedBidDetails(AbstractUtilitySpace util) {
		double max = (-1.0D / 0.0D);
		BidDetails bestBid = null;
		for (BidDetails b : this.bidList) {
			double discountedUtility = util.getUtilityWithDiscount(b.getBid(),
					b.getTime());
			if (discountedUtility >= max) {
				max = discountedUtility;
				bestBid = b;
			}
		}
		return bestBid;
	}

	public BidDetails getWorstBidDetails() {
		double min = (1.0D / 0.0D);
		BidDetails worstBid = null;
		for (BidDetails b : this.bidList) {
			double utility = b.getMyUndiscountedUtil();
			if (utility < min) {
				min = utility;
				worstBid = b;
			}
		}
		return worstBid;
	}

	public List<BidDetails> getNBestBids(int count) {
		List result = new ArrayList();
		List sortedOpponentBids = new ArrayList(this.bidList);

		Collections.sort(sortedOpponentBids, new BidDetailsSorterUtility());

		for (int i = 0; (i < count) && (i < sortedOpponentBids.size()); i++) {
			result.add((BidDetails) sortedOpponentBids.get(i));
		}

		return result;
	}

	public int size() {
		return this.bidList.size();
	}

	public double getAverageUtility() {
		int size = size();
		if (size == 0)
			return 0.0D;
		double totalUtil = 0.0D;
		for (BidDetails bid : this.bidList) {
			totalUtil = bid.getMyUndiscountedUtil();
		}
		return totalUtil / size;
	}

	public double getAverageDiscountedUtility(AdditiveUtilitySpace utilSpace) {
		int size = size();
		if (size == 0)
			return 0.0D;
		double totalUtil = 0.0D;
		for (BidDetails bid : this.bidList) {
			totalUtil = utilSpace.getUtilityWithDiscount(bid.getBid(),
					bid.getTime());
		}
		return totalUtil / size;
	}

	public BidHistory sortToUtility() {
		BidHistory sortedHistory = this;
		Collections.sort(sortedHistory.getHistory(),
				new BidDetailsSorterUtility());
		return sortedHistory;
	}

	public BidHistory sortToTime() {
		BidHistory sortedHistory = this;
		Collections
				.sort(sortedHistory.getHistory(), new BidDetailsSorterTime());
		return sortedHistory;
	}

	public BidDetails getRandom() {
		return getRandom(new Random());
	}

	public BidDetails getRandom(Random rand) {
		int size = size();
		if (size == 0)
			return null;
		int index = rand.nextInt(size);
		return (BidDetails) this.bidList.get(index);
	}

	public boolean isEmpty() {
		return this.bidList.isEmpty();
	}
}