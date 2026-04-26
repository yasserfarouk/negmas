package agents.anac.y2013.TMFAgent;

import genius.core.Bid;

public class ComparableBid implements Comparable<ComparableBid>{
	public double utility;
	public Bid bid;
	
	public ComparableBid (Bid b,double u){
		bid = new Bid (b);
		utility = u;
	}
	
	public int compareTo(ComparableBid other){
		if(other.utility < utility)
			return -1;
		else if(other.utility > utility)
			return 1;
		else
			return 0;
	}
	
	public String toString(){
		return utility + ": " + bid.toString();
	}
}