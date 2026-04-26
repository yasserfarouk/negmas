package agents.anac.y2014.Gangster;

import java.util.ArrayList;
import java.util.List;
import java.util.PriorityQueue;
import java.util.TreeSet;

import genius.core.Bid;
import genius.core.bidding.BidDetails;

class BidStorage {
	
	int MAX_CAPACITY;			//maximum size to avoid decrease in performance.
	int MAX_SIZE_AFTER_CLEANING;  // 2 times the minimum size we need to maintain good results.
	int expectedAmountOfProposals; 
	
	int MAX_SIZE_NOT_SELFISH_ENOUGH = 5000;
	int MAX_SIZE_NOT_SELFISH_ENOUGH_AFTER_CLEANING = 1000;
	
	int NUM_BINS_FOR_PROPOSED_TO_US = 100;
	
	//these two sets store the samples found by the searches and which are selfish enough. These two lists should always contain exactly the same elements.
	private TreeSet<ExtendedBidDetails> sortedByDiversity = new TreeSet(ExtendedBidDetails.DiversityComparator);	
	private TreeSet<ExtendedBidDetails> sortedByDistance = new TreeSet(ExtendedBidDetails.DistanceComparator);
	
	private TreeSet<ExtendedBidDetails> notSelfishEnough = new TreeSet(ExtendedBidDetails.UtilityComparator); //although priority queue is faster, it allows multiple copies of the same element, which we don't want.
	
	//these lists store the proposals made
	private ArrayList<Bid> proposedByOpponent;
	private ArrayList<Bid> proposedByUs;
	
	PriorityQueue<BidDetails>[] reproposableBids;
	TreeSet<BidDetails> reproposableBidsMain;
	
	private double targetUtility;
	
	public BidStorage(int maxCapacity, int maxSizeAfterCleaning, int expectedAmountOfProposals) {
		this.MAX_CAPACITY = maxCapacity;
		this.MAX_SIZE_AFTER_CLEANING = maxSizeAfterCleaning;
		
		proposedByOpponent = new ArrayList(expectedAmountOfProposals);
		proposedByUs = new ArrayList(expectedAmountOfProposals);
		reproposableBids = new PriorityQueue[NUM_BINS_FOR_PROPOSED_TO_US];
		reproposableBidsMain = new TreeSet<BidDetails>();
	}
	
	
	
	
	/**
	 * Add incoming bid from the opponent.
	 * 
	 * @param opponentBid
	 * @throws Exception 
	 */
	void addOpponentBid(BidDetails bd) throws Exception{
		
		proposedByOpponent.add(bd.getBid());
		
		//if the list gets too large, clean up.
		if(proposedByOpponent.size() > 3000){
			ArrayList<Bid> newList = new ArrayList(3500);
			for(int i=proposedByOpponent.size()/2; i<proposedByOpponent.size(); i++){
				newList.add(proposedByOpponent.get(i));
				proposedByOpponent = newList;
			}
			
		}
		
		
		
		//now we should recalculate the distances of samples stored in the list sortedByOpponentDistance to keep the sorting correct.
		for(ExtendedBidDetails ebd : sortedByDiversity){ //we loop over sortedByDiversity because it contains exactly the same elements as sortedByOpponentDistance
			
			//check if the new opponent bid is closer to the current sample than the closest opponent bid currently set in the sample.		
			int dist = Utils.calculateManhattanDistance(ebd.bidDetails.getBid(), bd.getBid());
			if(dist < ebd.distanceToOpponent){
				
				//remove the ebd from the treeset
				sortedByDistance.remove(ebd);
				
				//change the opponent distance
				ebd.setClosestOpponentBid(bd.getBid(), dist);				
				
				//re-insert ebd into the treeSet
				sortedByDistance.add(ebd);
			}
			
			
		}
		
		int i = (int)Math.floor(bd.getTime() * ((double)reproposableBids.length));
		if(i == reproposableBids.length){
			i--; //security measure.
		}
		
		if(reproposableBids[i] == null){
			reproposableBids[i] = new PriorityQueue<BidDetails>();
		}
		
		if(reproposableBids[i].size() > 0){
			BidDetails old_first = (BidDetails)reproposableBids[i].peek();
			if(bd.getMyUndiscountedUtil() > old_first.getMyUndiscountedUtil()){
				reproposableBidsMain.remove(old_first);
				reproposableBidsMain.add(bd);
			}
		}else{
			reproposableBidsMain.add(bd);
		}
		reproposableBids[i].add(bd);

		
	}
	
	
	void addBidProposedByUs(Bid ourBid) throws Exception{
		
		proposedByUs.add(ourBid);
		
		//if the list gets too large, clean up.
		if(proposedByUs.size() > 3000){
			ArrayList<Bid> newList = new ArrayList(3500);
			for(int i=proposedByUs.size()/2; i<proposedByUs.size(); i++){
				newList.add(proposedByUs.get(i));
				proposedByUs = newList;
			}
		}
		
		//now we should recalculate the diversity of all stored samples.
		for(ExtendedBidDetails ebd : sortedByDistance){ //we loop over sortedByOpponentDistance because it contains exactly the same elements as sortedByOpponentDistance
			
			int diversity = Utils.calculateManhattanDistance(ebd.bidDetails.getBid(), ourBid);
			if(diversity < ebd.diversity){
				
				//remove the ebd from the treeset
				sortedByDiversity.remove(ebd);
				
				//change the opponent distance
				ebd.setOurClosestBid(ourBid, diversity);			
				
				//re-insert ebd into the treeSet
				sortedByDiversity.add(ebd);
			}
		}
		
		
	}
	
	
	
	void setTargetUtility(double targetUtility) throws Exception{
		
		this.targetUtility = targetUtility;
		
		if(notSelfishEnough.size() == 0){
			return;
		}
		
		ExtendedBidDetails ebd = notSelfishEnough.last();
		double util = ebd.getMyUndiscountedUtil();
		while(util > targetUtility){

			notSelfishEnough.remove(ebd);
				
			//recalculate distance and diversity of ebd.
			for(Bid opponentBid : proposedByOpponent){
				ebd.setClosestOpponentBidMaybe(opponentBid);
			}
			for(Bid ourBid : proposedByUs){
				ebd.setOurClosestBidMaybe(ourBid);
			}
				
			sortedByDistance.add(ebd);
			sortedByDiversity.add(ebd);
			
			if(notSelfishEnough.size() == 0){
				break;
			}
			
			ebd = notSelfishEnough.last();
			util = ebd.getMyUndiscountedUtil();
		}
		
		
		
	}
	
	
	
	//add all bids found by genetic algorithm
	void addAll(List<BidDetails> bids, boolean foundByLocalSearch) throws Exception{
		for(BidDetails bd : bids){
			add(bd, foundByLocalSearch);
		}
		
	}
	
	//add a bid found by genetic algorithm
	void add(BidDetails sample, boolean foundByLocalSearch) throws Exception{
		
		//wrap the sample in an extended data structure
		ExtendedBidDetails ebd = new ExtendedBidDetails(sample);
		
		//store the sample, but only if we haven't already proposed it before.
		if(ebd.diversity > 0){
			
			if(ebd.getMyUndiscountedUtil() > targetUtility){
				
				
				//find the opponent bid that was closest to this sample.
				for(Bid opponentBid : proposedByOpponent){
					ebd.setClosestOpponentBidMaybe(opponentBid);
				}
				sortedByDistance.add(ebd);
				
				
				
				//find our bid that was closest to this sample.
				for(Bid ourBid : proposedByUs){
					ebd.setOurClosestBidMaybe(ourBid);
				}
				sortedByDiversity.add(ebd);
				
				

			}else{
				notSelfishEnough.add(ebd);
			}
			
			
			if(notSelfishEnough.size() > MAX_SIZE_NOT_SELFISH_ENOUGH){
				cleanUpNotSelfishEnough();
			}
			
			
			//if we have stored too many elements we should throw away some.
			if(sortedByDiversity.size() > MAX_CAPACITY){
				cleanUp();
			}
			
			
		}
		

	}
	
	

	
	//This method is called when we have to many bids stored.
	//throws away a predefined part of the stored samples.
	void cleanUp(){
		
		int n = MAX_SIZE_AFTER_CLEANING / 2;
		TreeSet<ExtendedBidDetails> savedBids = new TreeSet(ExtendedBidDetails.IDComparator);
		
		//save the n best elements of both lists..
		ExtendedBidDetails nextElement;
		nextElement = sortedByDiversity.last();
		for(int i=0; i<n; i++){
			savedBids.add(nextElement);
			nextElement = sortedByDiversity.lower(nextElement);
		}
		
		
		nextElement = sortedByDistance.last();
		for(int i=0; i<n; i++){
			savedBids.add(nextElement);
			nextElement = sortedByDistance.lower(nextElement);
		}
		
		sortedByDiversity.clear();
		sortedByDiversity.addAll(savedBids);
		
		
		sortedByDistance.clear();
		sortedByDistance.addAll(savedBids);

	}
	
	
	void cleanUpNotSelfishEnough(){
		
		ArrayList<ExtendedBidDetails> savedBids = new ArrayList();
		
		//save the n best elements of both lists..
		ExtendedBidDetails nextElement;
		nextElement = notSelfishEnough.last();
		for(int i=0; i<MAX_SIZE_NOT_SELFISH_ENOUGH_AFTER_CLEANING; i++){
			savedBids.add(nextElement);
			nextElement = notSelfishEnough.lower(nextElement);
		}
		
		notSelfishEnough.clear();
		notSelfishEnough.addAll(savedBids);
		
	}
	
	
	boolean weHaveSelfishEnoughBids(){
		return sortedByDiversity.size() > 0;
	}
	

	/**
	 * Returns the next bid to propose.
	 * Returns null if we don't have any bid that is selfish enough.
	 * 
	 * @param ourTargetUtility
	 * @param maxDistance
	 * @return
	 */	
	ExtendedBidDetails getNext(double ourTargetUtility, int maxDistance){
		
		//do we have any bids that are selfish enough?
		// if no: return null
		// if yes:  do we have any bids that are altruistic enough and selfish enough?
		//   if yes: get the set of bids that are altreuistic enough and selfish enough, and return the most altruistic bid
		//   if no:  get the set of bids that are selfish enough, and return the bid with highest diversity among those
		
		
		if(sortedByDistance.size() == 0){
			return null;
		}
		
		
		ExtendedBidDetails next = sortedByDistance.last();
		if(next.distanceToOpponent < maxDistance){
			return next;
		}
		
		next = sortedByDiversity.last();
		return next;
		
	}
	
	void removeBid(ExtendedBidDetails ebd){
		sortedByDiversity.remove(ebd);
		sortedByDistance.remove(ebd);
	}
	
	
	
	int firstNonNullBucket = 0;
	
	
	/**
	 * Returns the best bid in the past time window proposed to us.
	 * 
	 * @param time
	 * @param ourTargetUtility
	 * @return
	 */
	BidDetails getReproposableBid(double time, double ourTargetUtility, double timeWindow){
		
		
		double minTime = time - timeWindow;

		
		int j = (int) Math.floor(minTime * (double)reproposableBids.length);
		for(int k=firstNonNullBucket; k<j; k++){
			reproposableBids[k] = null;
		}
		if(j>0){
			firstNonNullBucket = j;
		}
		
		BidDetails possibleProposal = null;
		while(reproposableBidsMain.size() > 0){
			
			possibleProposal = reproposableBidsMain.first();
			
			//first test if utility is high enough
			if(possibleProposal.getMyUndiscountedUtil() < ourTargetUtility){
				return null; //even the best opponent bid doesn't have enough utility
			}
			
			//then test if it is inside our time window.
			if(possibleProposal.getTime() > minTime){
				
				//if yes, we should return it. but fist 
				//remove it from the main list, as well as from its bucket
				//get the new best element from the bucket and add it to the main list.
				
				reproposableBidsMain.remove(possibleProposal);
				
				int i = (int) Math.floor(possibleProposal.getTime() * (double)reproposableBids.length);
				if(reproposableBids[i] != null){
					
					reproposableBids[i].remove(possibleProposal);
				
					if(reproposableBids[i].size() > 0){
						reproposableBidsMain.add((BidDetails)reproposableBids[i].peek());
					}
				}
				return possibleProposal;
				
			}else{
				
				//if the bid is not inside the time window, then just remove it.
				reproposableBidsMain.remove(possibleProposal);
			}
		}
		
		return null;
	}
	
	/**
	 *This method should be called when we do not expect the discounted utilities offered by the opponent to increase.
	 * Sets the target to the next expected utility u_i.
	 *
	 */
	double getBestOfferedUtility(double time, double timeWindow){
		
		if(reproposableBidsMain.size() == 0){
			return 0;
		}
		
		BidDetails best = reproposableBidsMain.first();
		while(best != null && best.getTime() < (time-timeWindow)){
			best = reproposableBidsMain.higher(best);
		}
		
		if(best == null){
			return 0;
		}
		
		return best.getMyUndiscountedUtil();
		
	}
	
	
	int size(){
		return sortedByDiversity.size() + notSelfishEnough.size();
	}

	//Returns the number of bids stored that are selfish enough.
	int getNumSelfishBids(){
		return sortedByDiversity.size();
	}
	
	boolean testConsistency(){
		return sortedByDiversity.size() == sortedByDistance.size();
	}
}
