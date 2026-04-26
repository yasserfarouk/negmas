package agents.anac.y2011.ValueModelAgent;

import java.util.ArrayList;
import java.util.Collections;

public class BidList {
	public ArrayList<BidWrapper> bids;
	public BidList(){
		bids = new ArrayList<BidWrapper>();
	}
	public void sortByOurUtil(){
		Collections.sort(bids, bids.get(0).new OurUtilityComperator());
	}
	public void sortByOpponentUtil(ValueModeler model){
		for(BidWrapper bid: bids){
			try{
				bid.theirUtility = 1-model.utilityLoss(bid.bid).getDecrease();
			}
			catch(Exception e){
				e.printStackTrace();
			}
		}
		Collections.sort(bids, bids.get(0).new OpponentUtilityComperator());
	}
	//returns false if not new
	public boolean addIfNew(BidWrapper newBid){
		
		for(int i=0;i<bids.size();i++){
			if(bids.get(i).bid.equals(newBid.bid)){
				bids.get(i).lastSentBid = newBid.lastSentBid;
				return false;
			}
		}
		bids.add(newBid);
		
		return true;
	}
}
