package negotiator.boaframework.sharedagentstate.anac2011;

import java.util.ArrayList;
import java.util.Collections;

import genius.core.bidding.BidDetails;
import genius.core.boaframework.NegotiationSession;
import genius.core.boaframework.SharedAgentState;

/**
 * This is the shared code of the acceptance condition and bidding strategy of ANAC 2011 HardHeaded.
 * The code was taken from the ANAC2011 HardHeaded and adapted to work within the BOA framework.
 * 
 * @author Mark Hendrikx
 */
public class HardHeadedSAS extends SharedAgentState{

	private NegotiationSession negotiationSession;
	private double lowestYetUtility;
	
	public HardHeadedSAS(NegotiationSession negoSession) {
		negotiationSession = negoSession;

		NAME = "HardHeaded";
		lowestYetUtility = 1;
	}
	
	public double getLowestYetUtility() {
		return lowestYetUtility;
	}
	
	public void setLowestYetUtility (double util) {
		
		lowestYetUtility = util;
	}
	
	public double getLowestDiscountedUtilityYet(){
		double lowestYetDiscountedUtility;
		if (negotiationSession.getOwnBidHistory().getHistory().size() > 0) {
			ArrayList<BidDetails> sortedOwnBids = (ArrayList<BidDetails>) negotiationSession.getOwnBidHistory().getHistory();
			Collections.sort(sortedOwnBids);
			BidDetails bid = sortedOwnBids.get(sortedOwnBids.size() - 1);
			lowestYetDiscountedUtility =  negotiationSession.getDiscountedUtility(bid.getBid(), bid.getTime());
			if(bid.getTime() == -1.0)
				System.out.print("error bid of bid is -1");
		} else {
			lowestYetDiscountedUtility = 1;
		}
		return lowestYetDiscountedUtility;

	}
	
	
	public double getLowestUtilityYet(){
		double lowestYetUtility;
		if (negotiationSession.getOwnBidHistory().getHistory().size() > 0) {
			ArrayList<BidDetails> sortedOwnBids = (ArrayList<BidDetails>) negotiationSession.getOwnBidHistory().getHistory();
			Collections.sort(sortedOwnBids);
			lowestYetUtility =  sortedOwnBids.get(sortedOwnBids.size() - 1).getMyUndiscountedUtil();
		} else {
			lowestYetUtility = 1;
		}
		return lowestYetUtility;

	}
}
