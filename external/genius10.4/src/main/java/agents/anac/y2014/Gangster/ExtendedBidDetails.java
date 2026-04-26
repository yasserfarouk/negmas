package agents.anac.y2014.Gangster;

import java.util.Comparator;

import genius.core.Bid;
import genius.core.bidding.BidDetails;

/**
 * appart from the bid also stores our utility, the closest bid from the opponent and the distance to this closest bid.
 * 
 * 
 * @author ddejonge
 *
 */
class ExtendedBidDetails{

	static int numCreated = 0;
	
	int bidID;  //this is needed as a tie-breaker for the camparators, because we can't put two objects in a TreeSet which compare equally.
				//IDEA: instead of using a counter to identify this object, we could use the underlying Bid to identify it. In that way we prevent storing identical bids multiple times.
	
	BidDetails bidDetails;
	
	Bid closestOpponentBid = null;
	int distanceToOpponent = Integer.MAX_VALUE; 				//Distance between my bid and the closest opponent bid.

	Bid ourClosestBid = null;
	int diversity = Integer.MAX_VALUE;
	
	
	public ExtendedBidDetails(BidDetails bidDetails) {
		this.bidDetails = bidDetails;
		this.bidID = numCreated++;
	}
	
	
	/**
	 * checks if the given bid is the closest one and if yes, sets it as the closest bid, and sets the distance.
	 * 
	 * @param opponentBid
	 * @throws Exception
	 */
	void setClosestOpponentBidMaybe(Bid opponentBid) throws Exception{

		int dist = Utils.calculateManhattanDistance(bidDetails.getBid(), opponentBid);
		if(dist < this.distanceToOpponent){
			this.distanceToOpponent = dist;
			this.closestOpponentBid = opponentBid;
		}
		
	}
	
	
	/**
	 * Sets the given bid as the closest one, and sets the distance.
	 * WARNING: doesn't check if it really is the closest bid!
	 * 
	 * @param opponentBid
	 * @throws Exception
	 */
	void setClosestOpponentBid(Bid opponentBid, int distance) throws Exception{
		this.distanceToOpponent = distance;
		closestOpponentBid = opponentBid;
	}

	
	
	void setOurClosestBidMaybe(Bid ourBid) throws Exception{
	
		int dist = Utils.calculateManhattanDistance(this.bidDetails.getBid(), ourBid);
		if(dist < this.diversity){
			this.diversity = dist;
			this.ourClosestBid = ourBid;
		}
		
	}
	
	
	void setOurClosestBid(Bid ourBid, int diversity) throws Exception{
		this.diversity = diversity;
		this.ourClosestBid = ourBid;
	}
	
	double getMyUndiscountedUtil(){
		return this.bidDetails.getMyUndiscountedUtil();
	}
	
	
	static ExtendedBidDetails dominator(ExtendedBidDetails b1, ExtendedBidDetails b2){
		
		if(b1.bidDetails.getMyUndiscountedUtil() > b2.bidDetails.getMyUndiscountedUtil() && b1.distanceToOpponent < b2.distanceToOpponent){
			return b1;
		}
		
		if(b2.bidDetails.getMyUndiscountedUtil() > b1.bidDetails.getMyUndiscountedUtil() && b2.distanceToOpponent < b1.distanceToOpponent){
			return b2;
		}
		
		return null;
	}

	
	public static Comparator<ExtendedBidDetails> UtilityComparator = new Comparator<ExtendedBidDetails>() {

		public int compare(ExtendedBidDetails bid1, ExtendedBidDetails bid2) {
			
			int diff = Double.compare(bid1.getMyUndiscountedUtil(), bid2.getMyUndiscountedUtil());
			if(diff == 0){
				//this is needed as a tie-breaker for the camparators, because we can't put two objects in a TreeSet which compare equally.
				return bid1.bidDetails.getBid().toString().compareTo(bid2.bidDetails.getBid().toString());
			}
			
			return diff;
		}

	};
	
	public static Comparator<ExtendedBidDetails> DistanceComparator = new Comparator<ExtendedBidDetails>() {
		public int compare(ExtendedBidDetails bid1, ExtendedBidDetails bid2) {
			
			int diff = bid1.distanceToOpponent - bid2.distanceToOpponent;
			if(diff == 0){
				 //this is needed as a tie-breaker for the camparators, because we can't put two objects in a TreeSet which compare equally.
				return bid1.bidDetails.getBid().toString().compareTo(bid2.bidDetails.getBid().toString());
			}
			return diff;
		}
	};

	public static Comparator<ExtendedBidDetails> DiversityComparator = new Comparator<ExtendedBidDetails>() {
		public int compare(ExtendedBidDetails bid1, ExtendedBidDetails bid2) {
			
			int diff = bid1.diversity - bid2.diversity;
			if(diff == 0){
				//this is needed as a tie-breaker for the camparators, because we can't put two objects in a TreeSet which compare equally.
				return bid1.bidDetails.getBid().toString().compareTo(bid2.bidDetails.getBid().toString());
			}
			return diff;
		}
	};
	
	
	public static Comparator<ExtendedBidDetails> IDComparator = new Comparator<ExtendedBidDetails>() {
		public int compare(ExtendedBidDetails bid1, ExtendedBidDetails bid2) {
			return bid1.bidDetails.getBid().toString().compareTo(bid2.bidDetails.getBid().toString());
		}
	};
	
	
	public boolean equals(Object o){
		
		if(o instanceof ExtendedBidDetails){
			return this.bidDetails.getBid().toString().equals(((ExtendedBidDetails)o).bidDetails.getBid().toString());
		}
		
		return false;
	}
}
