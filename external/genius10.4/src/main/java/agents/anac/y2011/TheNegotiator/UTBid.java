package agents.anac.y2011.TheNegotiator;

import genius.core.Bid;

/**
 * The UTBid class is used to store a bid with it's corresponding utility.
 * In this way constant recomputation of the utility values is avoided.
 * 
 * @author Alex Dirkzwager, Mark Hendrikx, Julian de Ruiter
 */
public class UTBid implements Comparable<UTBid>{

	// the bid of an agent
	private Bid bid;
	// the utility corresponding to the bid
	private double utility;
	//indicates if the bid has already been offered
	private boolean alreadyOffered;
	
	/**
	 * Creates a UTBid-object which stores a bid with it's corresponding
	 * utility.
	 * 
	 * @param bid of an agent
	 * @param utility of the bid
	 */
	public UTBid(Bid bid, double utility) {
		this.bid = bid;
		this.utility = utility;
		alreadyOffered = false;
	}
	
	/**
	 * Method which returns the bid.
	 * 
	 * @return bid
	 */
	public Bid getBid() {
		return bid;
	}
	
	/**
	 * Method which sets the bid.
	 * 
	 * @param bid
	 */
	public void setBid(Bid bid) {
		this.bid = bid;
	}
	
	/**
	 * Method which returns the utility.
	 * 
	 * @return utility
	 */
	public double getUtility() {
		return utility;
	}
	
	/**
	 * Method which sets the utility.
	 * 
	 * @param utility
	 */
	public void setUtility(double utility) {
		this.utility = utility;
	}
	
	/**
	 * checks whether the bid has already been made
	 * @return boolean
	 */
	public boolean getAlreadyOffered(){
		return alreadyOffered;
	}
	
	/**
	 * sets the the bid as offered or not
	 * @param offered
	 */
	public void setAlreadyOffered(boolean offered){
		alreadyOffered = offered;
	}
	
	/**
	 * compareTo is used to compare UTbids. The comparision is made 
	 * in such a way that the result is in reverse natural order.
	 * 
	 * @param another utbid
	 */
	public int compareTo(UTBid utbid) {
		double otherUtil = utbid.getUtility();
		
		int value = 0;
		if (this.utility < otherUtil) {
			value = 1;
		} else if (this.utility > otherUtil) {
			value = -1;
		}
		return value;
	}
}