package genius.core.uncertainty;

import genius.core.Bid;
import genius.core.bidding.BidDetails;
import genius.core.bidding.BidDetailsSorterUtility;

	/**
	 * @author Dimitrios Tsimpoukis
	 *
	 * Helper class, any object of which depicts the comparison of two negotiation of two outcomes.
	 * bid1 < bid2 is coded by comparisonResult = -1.
	 *  
	 * 
	 */

public class OutcomeComparison {

	private Bid bid1;
	private Bid bid2;
	private int comparisonResult;
	
	public OutcomeComparison (BidDetails bid1, BidDetails bid2) {
		this.bid1 = bid1.getBid();
		this.bid2 = bid2.getBid();		
		this.comparisonResult = (new BidDetailsSorterUtility()).compare(bid1 , bid2);		
	}
	
	public OutcomeComparison (Bid bid1, Bid bid2, int comparisonResult) 
	{
		this.bid1 = bid1;
		this.bid2 = bid2;		
		this.comparisonResult = comparisonResult;		
	}

	
	public Bid getBid1() {
		return bid1;
	}
	
	public Bid getBid2() {
		return bid2;
	}
	

	public int getComparisonResult() {
		return comparisonResult;
	}
	
	public void setComparisonResult(int comparisonResult) {
		this.comparisonResult = comparisonResult;
	}
	
	@Override
	public String toString() {
		String s;
		if (comparisonResult == 1) {
			s = bid2.toString() + " is preferred over the outcome " +bid1.toString();
		}
		else if (comparisonResult == -1) {
			s = bid1.toString() + " is preferred over the outcome " +bid2.toString();
		}
		else  if (comparisonResult == 0){
			s = "Both bids are of equal utility";
		}
		else s = "No Comparison";
		return s;
	}
	
}


