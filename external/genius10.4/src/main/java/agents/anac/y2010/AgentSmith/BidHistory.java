package agents.anac.y2010.AgentSmith;

import java.util.ArrayList;
import java.util.List;

import genius.core.Bid;

/*
 * The storage point for the bidhistory of the agent and the opponent
 */

public class BidHistory {
	
	private List<Bid> fMyBids;
	private List<Bid> fOpponentBids;
	private List<IBidHistoryListener> fListeners;
	
	public BidHistory() {
		fMyBids = new ArrayList<Bid>();
		fOpponentBids = new ArrayList<Bid>();
		fListeners = new ArrayList<IBidHistoryListener>();
	}
	
	/*
	 * Add our own bid to the list
	 */
	public void addMyBid(Bid pBid) {
		if (pBid == null)
			throw new IllegalArgumentException("vBid can't be null.");
		fMyBids.add(pBid);
		for (IBidHistoryListener listener : fListeners) {
			listener.myBidAdded(this, pBid);
		}
	}
	
	/*
	 * returns the size (number) of offers already made
	 */
	public int getMyBidCount() {
		return fMyBids.size();
	}
	
	/*
	 * returns a bid from the list
	 */
	public Bid getMyBid(int pIndex) {
		return fMyBids.get(pIndex);
	}
	
	/*
	 * returns the last offer made
	 */
	public Bid getMyLastBid() {
		Bid result = null;
		if(getMyBidCount() > 0){
			result = fMyBids.get(getMyBidCount()-1);
		}
		return result;		
	}
	
	/*
	 * returns true if a bid has already been made before
	 */
	public boolean isInsideMyBids(Bid a){
		boolean result = false;
		for(int i = 0; i < getMyBidCount(); i++){
			if(a.equals(getMyBid(i))){
				result = true;
			}
		}
		return result;
	}
	
	/*
	 * add the bid the oppponent to his list
	 */
	public void addOpponentBid(Bid pBid) {
		if (pBid == null)
			throw new IllegalArgumentException("vBid can't be null.");
		fOpponentBids.add(pBid);
		for (IBidHistoryListener listener : fListeners) {
			listener.opponentBidAdded(this, pBid);
		}
	}
	
	/*
	 * returns the number of bids the opponent has made
	 */
	public int getOpponentBidCount() {
		return fOpponentBids.size();
	}
	
	/*
	 * returns the bid at a given index
	 */
	public Bid getOpponentBid(int pIndex) {
		return fOpponentBids.get(pIndex);
	}
	
	/*
	 * returns the opponents' last bid
	 */
	public Bid getOpponentLastBid() {
		Bid result = null;
		if(getOpponentBidCount() > 0){
			result = fOpponentBids.get(getOpponentBidCount()-1);
		}
		return result;
	}	
	
	/*
	 * add a given listener to the list
	 */
	public void addListener(IBidHistoryListener pListener) {
		fListeners.add(pListener);
	}
	
	/*
	 * remove a given listener from the list
	 */
	public void removeListener(IBidHistoryListener pListener) {
		fListeners.remove(pListener);
	}
	
}
