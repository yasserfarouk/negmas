package negotiator.boaframework.opponentmodel.agentsmith;
import java.util.Comparator;

import genius.core.Bid;
import genius.core.bidding.BidDetails;
import genius.core.utility.AdditiveUtilitySpace;

/*
 * The BidComparator compares two given bids. This class is used to slowly descent to the opponent while
 * staying above the given threshold (OWN_UTILITY_WEIGHT), which determines how much he values his
 * own interest or wants to give in a lot to the opponent to make him 'happy'.
 */
public class SmithBidComparator implements Comparator<BidDetails> {

	private AdditiveUtilitySpace mySpace;
	private AdditiveUtilitySpace opponentSpace;
	
	
	public SmithBidComparator(AdditiveUtilitySpace mySpace, AdditiveUtilitySpace opponentSpace) {
		this.mySpace = mySpace;
		this.opponentSpace = opponentSpace;
	}
	
	/*
	 * returns 1 if his own bid is better than the opponents, -1 otherwise
	 */
	public int compare(BidDetails b1, BidDetails b2) {
		return getMeasure(b2.getBid()) > getMeasure(b1.getBid()) ? -1 : 1;
	}
	
	/*
	 * returns a double that represents the value of a value of a bid, taking into account both the agents 
	 * own and opponents' utility. 
	 */
	public double getMeasure(Bid b1) {
		try {
			double a = (1 - this.mySpace.getUtility(b1)); 
			double b = (1 - this.opponentSpace.getUtility(b1));
			
			double alpha = Math.atan(b/a);
			
			return a + b + (0.5*Math.PI / alpha) * 0.5*Math.PI;
		} catch (Exception e) {
			e.printStackTrace();
			return 0;
		}
	}
}