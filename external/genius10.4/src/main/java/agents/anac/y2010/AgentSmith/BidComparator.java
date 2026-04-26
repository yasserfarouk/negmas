package agents.anac.y2010.AgentSmith;
import java.util.Comparator;

import genius.core.Bid;

/*
 * The BidComparator compares two given bids. This class is used to slowly descent to the opponent while
 * staying above the given threshold (OWN_UTILITY_WEIGHT), which determines how much he values his
 * own interest or wants to give in a lot to the opponent to make him 'happy'.
 */
public class BidComparator implements Comparator<Bid> {
	private PreferenceProfileManager fPreferenceProfile;
	
	
	
	public BidComparator(PreferenceProfileManager pProfile) {
		this.fPreferenceProfile = pProfile;
	}
	
	/*
	 * returns 1 if his own bid is better than the opponents, -1 otherwise
	 */
	public int compare(Bid b1, Bid b2) {
		return getMeasure(b2) > getMeasure(b1) ? -1 : 1;
	}
	
	/*
	 * returns a double that represents the value of a value of a bid, taking into account both the agents 
	 * own and opponents' utility. 
	 */
	public double getMeasure(Bid b1) {
		double a = (1 - this.fPreferenceProfile.getMyUtility(b1)); 
		double b = (1 - this.fPreferenceProfile.getOpponentUtility(b1));
		
		double alpha = Math.atan(b/a);
		
		return a + b + (0.5*Math.PI / alpha) * 0.5*Math.PI;
	}
	
}
