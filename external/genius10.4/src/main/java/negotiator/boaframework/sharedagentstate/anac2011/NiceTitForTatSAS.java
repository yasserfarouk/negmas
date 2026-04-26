package negotiator.boaframework.sharedagentstate.anac2011;

import java.util.ArrayList;

import genius.core.bidding.BidDetails;
import genius.core.boaframework.NegotiationSession;
import genius.core.boaframework.SharedAgentState;

/**
 * This is the shared code of the acceptance condition and bidding strategy of ANAC 2011 Nice Tit for Tat.
 * The code was taken from the ANAC2011 Nice Tit for Tat and adapted to work within the BOA framework.
 * 
 * @author Mark Hendrikx
 */
public class NiceTitForTatSAS extends SharedAgentState{
	
	private NegotiationSession negotiationSession;
	
	public NiceTitForTatSAS (NegotiationSession negoSession) {
		negotiationSession = negoSession;
		NAME = "NiceTitForTat";
	}
	
	public ArrayList<BidDetails> filterBetween(double minU, double maxU, double minT, double maxT) {
		ArrayList<BidDetails> filterBids = new ArrayList<BidDetails>();
		for (BidDetails b : negotiationSession.getOpponentBidHistory().getHistory()) {
			if (minU < b.getMyUndiscountedUtil() &&
					b.getMyUndiscountedUtil() <= maxU &&
					minT < b.getTime() &&
					b.getTime() <= maxT)
				filterBids.add(b);
		}
		return filterBids;	
	}
	
	public ArrayList<BidDetails> discountedFilterBetween(double minU, double maxU, double minT, double maxT) {
		ArrayList<BidDetails> filterBids = new ArrayList<BidDetails>();
		for (BidDetails b : negotiationSession.getOpponentBidHistory().getHistory()) {
			if (minU < negotiationSession.getDiscountedUtility(b.getBid(), b.getTime()) &&
					negotiationSession.getDiscountedUtility(b.getBid(), b.getTime()) <= maxU &&
					minT < b.getTime() &&
					b.getTime() <= maxT)
				filterBids.add(b);
		}
		return filterBids;	
	}
	
	
	public ArrayList<BidDetails> filterBetweenTime(double minT, double maxT){
		return filterBetween(0,1, minT, maxT);
	}
	
	public boolean isDomainBig() {
		negotiationSession.getUtilitySpace().getDomain().getNumberOfPossibleBids();
		return	negotiationSession.getUtilitySpace().getDomain().getNumberOfPossibleBids() > 10000;
	}
}
