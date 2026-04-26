package agents.anac.y2015.TUDMixedStrategyAgent;

import java.util.ArrayList;

import genius.core.AgentID;
import genius.core.Bid;

public class  Strategy {
	// Just Static methods
	private Strategy() {
		super();
	}
	/*
	 * Generate the next Bid based on the Utility value for this round.
	*/
	public static Bid calculateMyBid(TUDMixedStrategyAgent info) {
		double U = nextBidUtility(info);
		// Get a List of bids with this utility or more 10%
		ArrayList<Bid> bidsInRange = BidGenerator.getBidsInRange(info.getPossibleBids(), U, U*1.1, info);
		if(bidsInRange.isEmpty())
			return null;
		// Select the best bid for our opponents
		return bidSelect(bidsInRange,info.getAgentUtilsList());
	}
	
	// Receives a List of bids, and a list of agentUtils, selects the bid that maximizes the minimum utility over all agents 
	public static Bid bidSelect(ArrayList<Bid> bids, ArrayList<AgentUtils> agentUtils) {
		Bid bestBid =null;
		double bestBidUtil=0;
		for(int i=0; i<bids.size();i++) {
			double lowestUtil=1;
			for(int j=0;j<agentUtils.size();j++) {
				double currentUtil =agentUtils.get(j).getAgentUtil(bids.get(i));
				if(currentUtil<lowestUtil)
					lowestUtil=currentUtil;
			}
			if(bestBidUtil<lowestUtil) {
				bestBid=bids.get(i);
				bestBidUtil=lowestUtil;
			}
		}
		if(bestBid==null)
			bestBid=bids.get(0);
		return bestBid;	
	}
	
	
	// Returns the target utility of our Bid on this round-
	public static double nextBidUtility(TUDMixedStrategyAgent info) {
		return  1- ((double) info.getRoundN())/info.roundDeadline()*(1-info.getUtilitySpace().getReservationValue());
		
	}
	
	// Returns true if we should accept.
	public static boolean acceptingConditions(TUDMixedStrategyAgent info) {
		// Check if the offer we received has higher utility than our target utility
		if(nextBidUtility(info)>lastOfferUtility(info)) {
			return false;
		}
		// check if we received an offer that has larger utility than this one
		else if(lastOfferUtility(info)<biggestReceivedOffer(info)) {
			return false;
		}
		return true;
	}
	
	// Returns true if we should offer a previously unaccepted offer.
	public static boolean offerPreviousOffer(TUDMixedStrategyAgent info) {
		if(nextBidUtility(info)<biggestReceivedOffer(info)) {
			return true;
		}
		return false;
	}
	
	// Returns the biggest received offer that we haven't resent	
	public static Bid bestPreviousBid(TUDMixedStrategyAgent info) {
		Bid biggestBid = null;
		double biggestUtil=0;
		if(info.getPartylist().isEmpty())
			return null;
		for(int i=0; i<info.getPartylist().size();i++) {
			AgentID agent = info.getPartylist().get(i);
			for(int j=0;j<info.getBidhistory(agent).getHistory().size();j++) {
				if(!info.getAlreadyProposed().contains(info.getBidhistory(agent).getHistory().get(j).getBid())){
					if(biggestUtil<info.getBidhistory(agent).getHistory().get(j).getMyUndiscountedUtil()) {
						biggestBid = info.getBidhistory(agent).getHistory().get(j).getBid();
						biggestUtil= info.getBidhistory(agent).getHistory().get(j).getMyUndiscountedUtil();
					}
				}
			}
		}
		return biggestBid;
	}

	// Returns the utility of the biggest received offer, that hasn't been proposed by us.	
	public static double biggestReceivedOffer(TUDMixedStrategyAgent info) {

		double biggestUtil=0;
		if(info.getPartylist().isEmpty())
			return 0;
		for(int i=0; i<info.getPartylist().size();i++) {
			AgentID agent = info.getPartylist().get(i);
			for(int j=0;j<info.getBidhistory(agent).getHistory().size();j++) {
				if(!info.getAlreadyProposed().contains(info.getBidhistory(agent).getHistory().get(j).getBid())){
					if(biggestUtil<info.getBidhistory(agent).getHistory().get(j).getMyUndiscountedUtil()) {
						biggestUtil= info.getBidhistory(agent).getHistory().get(j).getMyUndiscountedUtil();
					}
				}
			}
		}
		return biggestUtil;

	}

	
	
	// Returns the our Utility for the current offer

	public static double lastOfferUtility(TUDMixedStrategyAgent info) {
		return info.getUtility(info.getLastbid());
	}
	
	
	
}
