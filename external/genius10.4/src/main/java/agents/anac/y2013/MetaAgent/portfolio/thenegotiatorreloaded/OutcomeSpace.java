package agents.anac.y2013.MetaAgent.portfolio.thenegotiatorreloaded;

import java.util.ArrayList;
import java.util.List;

import genius.core.Bid;
import genius.core.BidIterator;
import genius.core.utility.AdditiveUtilitySpace;

/**
 * This class generates the complete outcome space and is therefore
 * Useful if someone wants to quickly implement an agent.
 * Note that while this outcomespace is faster upon initialization,
 * the sorted outcomespace class is faster during the negotiation.
 * 
 * @author Alex Dirkzwager, Mark Hendrikx
 */
public class OutcomeSpace {
	
	/** Reference to the utility space */
	protected AdditiveUtilitySpace utilitySpace;
	/** List of all possible bids in the domain */
	protected List<BidDetails> allBids = new ArrayList<BidDetails>();
	
	public OutcomeSpace() { }
	
	public OutcomeSpace(AdditiveUtilitySpace utilSpace) {
		this.utilitySpace = utilSpace;
		generateAllBids(utilSpace);
	}
	
	/**
	 * Generates all the possible bids in the domain
	 * 
	 * @param utilSpace
	 */
	public void generateAllBids(AdditiveUtilitySpace utilSpace) {
		
		BidIterator iter = new BidIterator(utilSpace.getDomain());
		while (iter.hasNext()) {
			Bid bid = iter.next();
			try {
				BidDetails BidDetails = new BidDetails(bid, utilSpace.getUtility(bid));
				allBids.add(BidDetails);
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
	}
	
	/**
	 * @return list of all possible bids
	 */
	public List<BidDetails> getAllOutcomes(){
		return allBids;
	}
	
	
	/**
	 * gets a list of bids (from possibleBids) that have a utility within the
	 * given range.
	 * 
	 * @param range
	 * @return list of BidDetails
	 */
	public List<BidDetails> getBidsinRange (Range r){
		ArrayList<BidDetails> result = new ArrayList<BidDetails>();
		double upperbound = r.getUpperbound();
		double lowerbound = r.getLowerbound();
		
		for(BidDetails bid: allBids){
			if (bid.getMyUndiscountedUtil() > lowerbound && bid.getMyUndiscountedUtil() < upperbound){
				result.add(bid);
			}
		}
		return result;
	}
	
	/**
	 * gets a list of bids (from possibleBids) that have a utility between the range
	 * @param range
	 * @return list of BidDetails
	 */
	public List<BidDetails> getBidsinDiscountedRange (Range r, double time){
		ArrayList<BidDetails> result = new ArrayList<BidDetails>();
		double upperbound = r.getUpperbound();
		double lowerbound = r.getLowerbound();
		

		for(BidDetails bid: allBids){
			if (utilitySpace.getUtilityWithDiscount(bid.getBid(), time) > lowerbound && utilitySpace.getUtilityWithDiscount(bid.getBid(), time) < upperbound){
				result.add(bid);
			}
		}
		return result;
	}
	
	/**
	 * gets a BidDetails which is closest to the give utility
	 * @param utility
	 * @return BidDetails
	 */
	public BidDetails getBidNearUtility(double utility) {
		BidDetails closesBid = null;
		double closesDistance = 1;
		for(BidDetails bid : allBids){
			if(Math.abs(bid.getMyUndiscountedUtil()-utility) < closesDistance) {
				closesBid = bid;
				closesDistance = Math.abs(bid.getMyUndiscountedUtil()-utility);
			}
		}
		return closesBid;
	}
	
	/**
	 * gets a BidDetails which is closest to the give utility
	 * @param utility
	 * @return BidDetails
	 */
	public BidDetails getBidNearDiscountedUtility(double utility, double time) {
		BidDetails closestBid = null;
		double closestDistance = 1;
		for(BidDetails bid : allBids){
			if(Math.abs(utilitySpace.getUtilityWithDiscount(bid.getBid(), time)-utility) < closestDistance) {
				closestBid = bid;
				closestDistance = Math.abs(bid.getMyUndiscountedUtil()-utility);
			}
		}
		return closestBid;
	}	
	
	public BidDetails getMaxBidPossible(){
		BidDetails maxBid = allBids.get(0);
		for(BidDetails bid : allBids){
			if(bid.getMyUndiscountedUtil() > maxBid.getMyUndiscountedUtil()) {
				maxBid = bid;
			}
		}
		return maxBid;
	}
	
	public BidDetails getMinBidPossible(){
		BidDetails minBid = allBids.get(0);
		for(BidDetails bid : allBids){
			if(bid.getMyUndiscountedUtil() < minBid.getMyUndiscountedUtil()) {
				minBid = bid;
			}
		}
		return minBid;
	}
}
