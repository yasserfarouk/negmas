package agents.anac.y2012.OMACagent;

import java.util.ArrayList;
import java.util.List;

import genius.core.Bid;
import genius.core.issue.Issue;
import genius.core.utility.AdditiveUtilitySpace;

//ver. 1.06
public class TimeBidHistory {
	private List<Bid> fMyBids;
	private List<Bid> fOpponentBids;
	public List<Double> fTimes; // new, record the time of receiving
								// counter-offers

	public int curIndex = 0; // new, refer to the current position of opp bid
								// history
	public int curLength = 0; // new, the size of current bid window
	private double discount = 0;
	protected AdditiveUtilitySpace fUtilitySpace;
	public double est_t = 0;
	public double est_u = 0;
	public double maxU = -1;
	public int pMaxIndex = 0;
	public Bid bestOppBid = null;
	public double[] maxBlock; // record max
	public List<Integer> newMC; // times where new maximum concession is given

	public TimeBidHistory(AdditiveUtilitySpace pUtilitySpace, double dis) {
		discount = dis;
		fMyBids = new ArrayList<Bid>();
		fTimes = new ArrayList<Double>(); // new
		newMC = new ArrayList<Integer>();
		this.fUtilitySpace = pUtilitySpace;

		maxBlock = new double[100];
		for (int i = 0; i < 100; i++) {
			maxBlock[i] = 0.0;
		}

	}

	/*
	 * Add our own bid to the list
	 */
	public void addMyBid(Bid pBid) {
		if (pBid == null)
			throw new IllegalArgumentException("pBid can't be null.");
		fMyBids.add(pBid);

	}

	/*
	 * returns the size (number) of offers already made
	 */
	public int getMyBidCount() {
		return fMyBids.size();
	}

	/*
	 * returns a bid from the list
	 */
	public Bid getMyBid(int pIndex) {
		return fMyBids.get(pIndex);
	}

	/*
	 * returns the last offer made
	 */
	public Bid getMyLastBid() {
		Bid result = null;
		if (getMyBidCount() > 0) {
			result = fMyBids.get(getMyBidCount() - 1);
		}
		return result;
	}

	/*
	 * returns true if a bid has already been made before
	 */
	public boolean isInsideMyBids(Bid a) {
		boolean result = false;
		for (int i = 0; i < getMyBidCount(); i++) {
			if (a.equals(getMyBid(i))) {
				result = true;
			}
		}
		return result;
	}

	/*
	 * add the bid the oppponent to his list
	 */
	public void addOpponentBidnTime(double oppU, Bid pBid, double time) {
		double undisOppU = oppU / Math.pow(discount, time);
		double nTime = time;

		if (pBid == null)
			throw new IllegalArgumentException("vBid can't be null.");

		fTimes.add(time);

		if (undisOppU > maxU) {
			maxU = undisOppU; // prabably not useful in some sense
			// pMaxIndex = getOpponentBidCount()-1;
			pMaxIndex = fTimes.size() - 1;
			bestOppBid = pBid;
			newMC.add(pMaxIndex);
		}

		if (nTime >= 1.0)
			nTime = 0.99999;

		if (maxBlock[(int) Math.floor(nTime * 100)] < undisOppU)
			maxBlock[(int) Math.floor(nTime * 100)] = undisOppU;

	}

	public double[] getTimeBlockList() {
		return maxBlock;
	}

	/*
	 * returns the number of bids the opponent has made
	 */
	public int getOpponentBidCount() {
		return fOpponentBids.size();
	}

	/*
	 * returns the bid at a given index
	 */
	public Bid getOpponentBid(int pIndex) {
		return fOpponentBids.get(pIndex);
	}

	/*
	 * returns the opponents' last bid
	 */
	public Bid getOpponentLastBid() {
		Bid result = null;
		if (getOpponentBidCount() > 0) {
			result = fOpponentBids.get(getOpponentBidCount() - 1);
		}
		return result;
	}

	public double getMyUtility(Bid b) {
		try {
			return this.fUtilitySpace.getUtility(b);
		} catch (Exception e) {
			return 0;
		}
	}

	public double getOpponentUtility(Bid b) {
		try {
			return this.fUtilitySpace.getUtility(b);
		} catch (Exception e) {
			return 0;
		}
	}

	/*
	 * returns the list of issues in the domain
	 */
	public List<Issue> getIssues() {
		return this.fUtilitySpace.getDomain().getIssues();
	}

	public double getFeaMC(double time) {
		int len = newMC.size();
		double dif = 1.0;

		if (len >= 3) {
			dif = fTimes.get(newMC.get(len - 1))
					- fTimes.get(newMC.get(len - 3));
			dif = dif / 2.0;
		} else if (len >= 2) {
			dif = fTimes.get(newMC.get(len - 1))
					- fTimes.get(newMC.get(len - 2));
		} else {
			dif = 0D;
		}

		// newmark("a-dif.txt",""+dif+","+len+","+time);
		return dif;
	}

	public double getMCtime() {
		if (newMC.size() == 0)
			return 0.0;

		return fTimes.get(newMC.get(newMC.size() - 1));
	}

}
