package agents.anac.y2013.MetaAgent.portfolio.thenegotiatorreloaded;

import java.util.Collections;
import java.util.List;

import genius.core.utility.AdditiveUtilitySpace;

/**
 * This class is an OutcomeSpace but with a sorted list of BidDetails based on the utility
 * Methods have been optimized to work with a sorted list.
 * Useful if someone wants to quickly implement an agent.
 * 
 * @author Alex Dirkzwager
 */
public class SortedOutcomeSpace extends OutcomeSpace {

	public SortedOutcomeSpace(AdditiveUtilitySpace utilSpace) {
		generateAllBids(utilSpace);
		utilitySpace = utilSpace;
		Collections.sort(allBids, new BidDetailsSorterUtility());
	}
	
	/**
	 * gets a list of BidDetails that have a utility within the range
	 * @param r
	 * @return A list of BidDetails
	 */
	@Override
	public List<BidDetails> getBidsinRange(Range r){
		//get upperbound index
		int upperboundIndex = searchIndexWith(r.getUpperbound());
		//get lowerbound index
		int lowerboundIndex = searchIndexWith(r.getLowerbound());
		
		//test upperbound index element is under upperbound
		if(!(allBids.get(upperboundIndex).getMyUndiscountedUtil()<=r.getUpperbound())){
			upperboundIndex++;
		}
		//test lowerbound index element is under lowerbound
		if(!(allBids.get(lowerboundIndex).getMyUndiscountedUtil()>=r.getLowerbound())){
			lowerboundIndex++;
		}

		return allBids.subList(upperboundIndex, lowerboundIndex+1);

	}
	
	/**
	 * Gets a BidDetails which is close to the utility
	 * @param utility
	 * @return BidDetails
	 */
	@Override
	public BidDetails getBidNearUtility(double utility){
		int index = searchIndexWith(utility);
		int newIndex = -1;
		double closestDistance = Math.abs(allBids.get(index).getMyUndiscountedUtil() - utility);
		
		//checks if the BidDetails above the selected is closer to the targetUtility
		if(index > 0 && Math.abs(allBids.get(index-1).getMyUndiscountedUtil() - utility) < closestDistance) {
			newIndex = index -1;
			closestDistance = Math.abs(allBids.get(index-1).getMyUndiscountedUtil() - utility);
		}
		
		//checks if the BidDetails below the selected is closer to the targetUtility
		if(index + 1 < allBids.size() && Math.abs(allBids.get(index+1).getMyUndiscountedUtil() - utility)<closestDistance){
			newIndex = index+1;
			closestDistance = Math.abs(allBids.get(index+1).getMyUndiscountedUtil() - utility);
		}
		else 
				newIndex = index;
		return allBids.get(newIndex);
		
	}
	
	/**
	 * Binary search of a BidDetails with a particular value
	 * if there is no BidDetails with the exact value gives the last index because this is the closest BidDetails to the value
	 * @param value
	 * @return index
	 */
	public int searchIndexWith(double value){
		int middle = -1;
		int low = 0;
		int high = allBids.size() - 1;
		int lastMiddle = 0;
		while(lastMiddle != middle) {
			lastMiddle = middle;
			middle = (low + high) / 2;
			if(allBids.get(middle).getMyUndiscountedUtil() == value) {
				return middle;
			}
			if(allBids.get(middle).getMyUndiscountedUtil() < value) {
				high = middle;
			}
			if(allBids.get(middle).getMyUndiscountedUtil() > value) {
				low = middle;
			}
		}
		return middle;
	}
	
	@Override
	public BidDetails getMaxBidPossible(){
		return allBids.get(0);
	}
	
	public BidDetails getMinBidPossible() {
		return allBids.get(allBids.size() - 1);
	}
}