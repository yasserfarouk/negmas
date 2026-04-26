package agents.anac.y2012.MetaAgent.agents.MrFriendly;

import java.util.ArrayList;

import genius.core.Bid;
import genius.core.actions.Action;
import genius.core.actions.Offer;

public class BidHistoryTracker {
	/**
	 * The list of bids from the opponent.
	 */
	private ArrayList<Bid> bidListOpponent;
	/**
	 * The list of bids from myself.
	 */
	private ArrayList<Bid> bidListSelf;
	
	/**
	 * counts the number of consecutive bids our opponent does that are
	 * new to us
	 */
	private int consecutiveBidsDifferent;
	
	/**
	 * Keeps track of the number of consecutive bids we have done that are non-unique
	 */
	private int ourStallingCoefficient;
	
	/**
	 * The constructor that just creates the empty ArrayLists.
	 */
	public BidHistoryTracker(){
		this.bidListOpponent=new ArrayList<Bid>();
		this.bidListSelf=new ArrayList<Bid>();
		this.consecutiveBidsDifferent = 0;
		this.ourStallingCoefficient = 0;
	}
	
	/**
	 * This function accepts an Action and saves the Bid (if the Action is an Offer).
	 * 
	 * @param action The opponent's action.
	 */
	public void addOpponentAction(Action action){
		if(action instanceof Offer){ // We ignore Accepts, EndNegotiations and nulls.
			Offer offer=(Offer)action;
			Bid bid=offer.getBid();
			
			//check wether we have received this bid before, in which case we reset the
			//counter that counts how many consecutive unique bids we've gotten. otherwise, it
			//increments it.
			if(this.bidAlreadyDoneByOpponent(bid)){
				this.consecutiveBidsDifferent = 0;
			}else{
				this.consecutiveBidsDifferent++;
			}
			bidListOpponent.add(bid);
		}
	}
	
	/**
	 * This function accepts a bid and saves it as a bid of the agent.
	 * @param bid The bid of myself.
	 */
	public void addOwnBid(Bid bid){
		if(this.bidAlreadyDoneByMyself(bid)){
			this.ourStallingCoefficient++;
		}else{
			this.ourStallingCoefficient = 0;
		}
		
		if(bid != null){ // We ignore nulls.
			bidListSelf.add(bid);
		}
	}
	
	/**
	 * Give the last bid offered by the opponent.
	 * @return The last offered by the opponent.
	 */
	public Bid getLastOpponentBid(){
		return getLastBidOf(bidListOpponent);
	}
	
	/**
	 * Gives the last bid we did ourselves.
	 * @return The last bid we did ourselves.
	 */
	public Bid getLastOwnBid(){
		return getLastBidOf(bidListSelf);
	}
	
	/**
	 * Get the last bid of an ArrayList<Bid>.
	 * @param list The list.
	 * @return The last bid.
	 */
	private Bid getLastBidOf(ArrayList<Bid> list){
		Bid result=null;
		if(!list.isEmpty()){
			result=list.get(list.size()-1);
		}
		return result;
	}
	
	/**
	 * Returns true iff we have offered the parameter bid ourself
	 * 
	 * @param bid Bid
	 * @return boolean
	 */
	public boolean bidAlreadyDoneByMyself(Bid bid){
		return bidListSelf.contains(bid);
	}
	
	/**
	 * Returns true iff our opponent has offered the parameter bid before
	 * 
	 * @param bid
	 * @return boolean
	 */
	private boolean bidAlreadyDoneByOpponent(Bid bid){
		return bidListOpponent.contains(bid);
	}

	/**
	 * Returns the number of bids we have received from our opponent
	 * 
	 * @return int
	 */
	public int getNumberOfOpponentBids(){
		return bidListOpponent.size(); 
	}
	
	/**
	 * Returns the number of consecutive bids in which this opponent has given
	 * us a previously unoffered bid
	 * 
	 * @return int
	 */
	public int getConsecutiveBidsDifferent(){
		return consecutiveBidsDifferent;
	}
	
	/**
	 * Returns the number of consecutive bids we have done that were non-unique
	 * 
	 * @return int
	 */
	public int getOurStallingCoefficient(){
		return ourStallingCoefficient;
	}
}
