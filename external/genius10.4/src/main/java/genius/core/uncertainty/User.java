package genius.core.uncertainty;

import java.util.ArrayList;
import java.util.List;


import genius.core.Bid;
import genius.core.bidding.BidDetails;
import genius.core.bidding.BidDetailsSorterUtility;
import genius.core.boaframework.NegotiationSession;
import genius.core.utility.AbstractUtilitySpace;
import genius.core.utility.UncertainAdditiveUtilitySpace;

/**
 * This class intends to be the gateway for more dynamic negotiations under preference uncertainty. 
 * In exchange of a cost, it should provide an agent with more information about the true utility space.
 * 
 * Author: Adel Magra
 */

public class User {
	
	/**
	 * The underlying true utility space that is not accessible from the agent's perspective.
	 */
	private UncertainAdditiveUtilitySpace utilspace;
	
	/**
	 * The total bother cost inflicted to the user from the elicitation actions of the agent.
	 */
	private double elicitationBother;
	
	public User(UncertainAdditiveUtilitySpace utilspace) {
		this.utilspace = utilspace;
		this.elicitationBother = 0;
	}
	/**
	 * This function allows an agent to update its user model with a bid against a cost.
	 * @param bid: bid that we wish to add at its correct place in the ranking of the user model
	 * @param usermodel: current user model
	 * @return an updated user model with the bid added to it
	 */
	
	public UserModel elicitRank(Bid bid, UserModel userModel){
		
		BidRanking currentRanking = userModel.getBidRanking();
		List<Bid> currentOrder = currentRanking.getBidOrder();
		List<Bid> newOrder = new ArrayList<Bid>();
		for( int i=0; i<currentOrder.size(); i++) {
			newOrder.add(currentOrder.get(i));
		}
		
		//In the case where the bid is already in the user model, just return the current user model
		if(newOrder.contains(bid))
			return userModel;
		
		//General Case
		elicitationBother += this.getElicitationCost();
		BidDetails newBid = new BidDetails(bid, utilspace.getUtility(bid));
		
		//bid will never be the max or min bid, because both are in every user model
		//So we only take care of the case where minBid < bid < maxBid
		for(int i=0; i<= newOrder.size()-1; i++){
			Bid iBid = newOrder.get(i);
			BidDetails currentBid = new BidDetails(iBid,utilspace.getUtility(iBid));
			int comparResult = (new BidDetailsSorterUtility()).compare(newBid, currentBid);
			if(comparResult == 1) {  // bid < iBid
				newOrder.add(i,bid);
				BidRanking newRank = new BidRanking(newOrder, userModel.getBidRanking().getLowUtility(), userModel.getBidRanking().getHighUtility());
				UserModel newModel = new UserModel(newRank);
				return newModel;
			}
		}
		//In case something fails
		System.out.println("Couldn't update user model upon request");
		return userModel; 
	}
	
	/**
	 * Gives back the cost of eliciting the user.
	 * @return elicitation cost.
	 */
	public double getElicitationCost() {
		return utilspace.getElicitationCost();
	}
	
	/**
	 * Gives back the Total Bother cost inflicted to the user
	 * @return elicitationBother
	 */
	public double getTotalBother() {
		return elicitationBother;
	}
	
}
