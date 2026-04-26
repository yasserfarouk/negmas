package agents.anac.y2013.MetaAgent.portfolio.thenegotiatorreloaded;

import genius.core.Bid;


/**
 * The BidDetails class is used to store a bid with it's corresponding utility and time it was offered.
 * In this way constant re-computation of the utility values is avoided.
 * 
 * @author Tim Baarslag, Alex Dirkzwager, Mark Hendrikx
 */
public class BidDetails implements Comparable<BidDetails>{

	/** the bid of an agent */
	private Bid bid;
	/** the utility corresponding to the bid */
	private double myUndiscountedUtil;
	/** time the bid was offered (so the discounted utility can be calculated at that time) */
	private double time;
	
	/**
	 * Creates a BidDetails-object which stores a bid with it's corresponding
	 * utility.
	 * 
	 * @param bid of an agent
	 * @param undiscounted utility of the bid
	 */
	public BidDetails(Bid bid, double myUndiscountedUtil) {
		this.bid = bid;
		this.myUndiscountedUtil = myUndiscountedUtil;
	}
	
	/**
	 * Creates a BidDetails-object which stores a bid with it's corresponding
	 * utility and the time it was offered.
	 * 
	 * @param bid of an agent
	 * @param myUndiscountedUtil of the bid
	 * @param time of offering
	 */
	public BidDetails(Bid bid, double myUndiscountedUtil, double time) {
		this.bid = bid;
		this.myUndiscountedUtil = myUndiscountedUtil;
		this.time = time;
	}
	
	public Bid getBid() {
		return bid;
	}
	
	public void setBid(Bid bid) {
		this.bid = bid;
	}
	
	public double getMyUndiscountedUtil() {
		return myUndiscountedUtil;
	}
	
	public void setMyUndiscountedUtil(double utility) {
		this.myUndiscountedUtil = utility;
	}
	
	
	public double getTime(){
		return time;
	}
	
	public void setTime(double t){
		time = t;
	}
	
	@Override
	public String toString()
	{
		return "(u=" + myUndiscountedUtil + ", t=" + time + ")";
	}
	
	/**
	 * A comperator for BidDetails which order the bids in
	 * reverse natural order of utility.
	 * 
	 * @param another utbid
	 */
	public int compareTo(BidDetails utbid) {
		double otherUtil = utbid.getMyUndiscountedUtil();
		
		int value = 0;
		if (this.myUndiscountedUtil < otherUtil) {
			value = 1;
		} else if (this.myUndiscountedUtil > otherUtil) {
			value = -1;
		}
		return value;
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + ((bid == null) ? 0 : bid.hashCode());
		long temp;
		temp = Double.doubleToLongBits(myUndiscountedUtil);
		result = prime * result + (int) (temp ^ (temp >>> 32));
		temp = Double.doubleToLongBits(time);
		result = prime * result + (int) (temp ^ (temp >>> 32));
		return result;
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		BidDetails other = (BidDetails) obj;
		if (bid == null) {
			if (other.bid != null)
				return false;
		} else if (!bid.equals(other.bid))
			return false;
		if (Double.doubleToLongBits(myUndiscountedUtil) != Double
				.doubleToLongBits(other.myUndiscountedUtil))
			return false;
		if (Double.doubleToLongBits(time) != Double
				.doubleToLongBits(other.time))
			return false;
		return true;
	}
}