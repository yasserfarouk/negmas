package negotiator.boaframework.sharedagentstate.anac2010;

import java.util.LinkedList;
import java.util.Random;

import genius.core.bidding.BidDetails;
import genius.core.boaframework.NegotiationSession;
import genius.core.boaframework.SharedAgentState;

/**
 * This is the shared code of the acceptance condition and bidding strategy of ANAC 2010 Yushu.
 * The code was taken from the ANAC2010 Yushu and adapted to work within the BOA framework.
 * 
 * @author Mark Hendrikx
 */
public class YushuSAS extends SharedAgentState {

	private NegotiationSession negotiationSession;
	private double previousTime;
	private double totalOppBidUtil = 0;
	private double eagerness = 1.2; 
	private double minimumBidUtil;
	private LinkedList<BidDetails> bestTenBids;
	private double highPosUtil; //the highest utility that can be achieved
	private double roundleft;
	private double acceptableUtil;
	private LinkedList<Double> timeBetweenRounds;
	private BidDetails suggestBid;
	private double targetUtil;
	private Random random100;
	private final boolean TEST_EQUIVALENCE = false;

	public YushuSAS (NegotiationSession negoSession) {
		negotiationSession = negoSession;
		NAME = "Yushu";
		
		totalOppBidUtil = 0;
		previousTime = 0;
		minimumBidUtil = 0.95;
		bestTenBids = new LinkedList<BidDetails>();
		timeBetweenRounds = new LinkedList<Double>();
		try {
			highPosUtil = negoSession.getUtilitySpace().getUtility(negoSession.getUtilitySpace().getMaxUtilityBid());
		} catch (Exception e) {
			e.printStackTrace();
		}
		acceptableUtil = 1.0;
    	if (TEST_EQUIVALENCE) {
    		random100 = new Random(100);
    	} else {
    		random100 = new Random();
    	}
	}
	
	public void updateBelief(BidDetails opponentBid){
		totalOppBidUtil += opponentBid.getMyUndiscountedUtil();
		if(bestTenBids.size()==0)
			bestTenBids.add(opponentBid);
		else{
			LinkedList<BidDetails> newlist = new LinkedList<BidDetails>();
			
			if(opponentBid.getMyUndiscountedUtil() > bestTenBids.getFirst().getMyUndiscountedUtil()){
				newlist.add(opponentBid);
				for(int j=0; j < bestTenBids.size(); j++)
					newlist.add(bestTenBids.get(j));
				bestTenBids = newlist;
			}
			else if(opponentBid.getMyUndiscountedUtil() <= bestTenBids.getLast().getMyUndiscountedUtil())
				bestTenBids.add(opponentBid);
			else{
				for(int i=1;i<bestTenBids.size();i++){
					if((opponentBid.getMyUndiscountedUtil() <= bestTenBids.get(i-1).getMyUndiscountedUtil()) &
							(opponentBid.getMyUndiscountedUtil() > bestTenBids.get(i).getMyUndiscountedUtil())){
						for(int j=0;j<i;j++)
							newlist.add(bestTenBids.get(j));
						newlist.add(opponentBid);
						for(int j=i;j<bestTenBids.size();j++)
							newlist.add(bestTenBids.get(j));
						break;
					}
				}
				bestTenBids=newlist;	 
			}
		}
		if(bestTenBids.size() > 10)
			bestTenBids.removeLast();
	}
	
	public double calculateTargetUtility() {
		timeBetweenRounds.add(negotiationSession.getTime() - previousTime);
		previousTime = negotiationSession.getTime();
		
		double tround = Math.max(averResT(), averLastTResT(3));
		double lefttime = 1 - negotiationSession.getTime();
		roundleft = lefttime / tround;
		
		
		
		if(roundleft > 6.7) {
			minimumBidUtil = 0.93 * highPosUtil;
		} else if (roundleft>5) {
			minimumBidUtil = 0.90 * highPosUtil;
		} else if (lefttime> 3 * tround) {
			minimumBidUtil = 0.86 * highPosUtil;
		} else if (lefttime>2.3*tround) {
			minimumBidUtil =0.8 * highPosUtil;
		} else {
			minimumBidUtil =0.6 * highPosUtil;
		}
		if(lefttime<15*tround) {
			acceptableUtil = 0.92 * highPosUtil;
		} else {
			acceptableUtil = 0.96 * highPosUtil;
		}
		
		// consider the domain competition
		double averopu=0, averopui=0;
		if (negotiationSession.getOpponentBidHistory().size() > 0) {
			averopui = totalOppBidUtil / negotiationSession.getOpponentBidHistory().size();
		}
		averopu=Math.max(0.30, averopui);
				
		double rte=20+(1-averopu)/0.10*20;
		if((lefttime<rte*tround) && (negotiationSession.getOpponentBidHistory().size()>3) && (averopu<0.75)) {
			minimumBidUtil = minimumBidUtil -(0.75-averopu) / 2.5;
		}
		
		minimumBidUtil =Math.max(0.50, minimumBidUtil); // no less than 0.5
		double time= negotiationSession.getTime();

		minimumBidUtil = minimumBidUtil * (Math.min(0.75, averopu) / 3 + 0.75);
		minimumBidUtil =Math.max(minimumBidUtil, averopu);
		double targetuti = highPosUtil -(highPosUtil - minimumBidUtil)*Math.pow(time, eagerness);

		if(lefttime<1.6*tround) {
			suggestBid = this.bestTenBids.getFirst();
			
			this.targetUtil = targetuti;
			return targetuti;
		}
		
		//consider op's best past offer
		if (lefttime > 50 * tround) {
			targetuti=Math.max(targetuti, bestTenBids.getFirst().getMyUndiscountedUtil() * 1.001);
		}
		if (((lefttime < 10 * tround) & (bestTenBids.getFirst().getMyUndiscountedUtil() > targetuti * 0.95)) | 
				bestTenBids.getFirst().getMyUndiscountedUtil() >= targetuti) {
			double newtargetuti=targetuti;
			if ((lefttime < 10 * tround) & (bestTenBids.getFirst().getMyUndiscountedUtil() > targetuti * 0.95)) {
				newtargetuti=targetuti * 0.95;
			}

			//check whether the best op bid was offered in the last 4 rouds
			boolean offered=false;
			int length = Math.min(negotiationSession.getOwnBidHistory().size(), 4);

			for(int i = negotiationSession.getOwnBidHistory().size() - 1; i >= negotiationSession.getOwnBidHistory().size() - length; i--){
				if (negotiationSession.getOwnBidHistory().getHistory().get(i).getBid().equals(bestTenBids.getFirst().getBid())) {
					offered=true;
				}
			}
			if(offered){
				LinkedList<BidDetails> candidates = new LinkedList<BidDetails>();
				for(int b=0; b < bestTenBids.size(); b++)
					if(bestTenBids.get(b).getMyUndiscountedUtil() >= newtargetuti)
						candidates.add(bestTenBids.get(b));
				int indexc = (int)(random100.nextDouble() * candidates.size());
				suggestBid = candidates.get(indexc);
			}
			else {
				suggestBid = bestTenBids.getFirst();
			}
			targetuti=newtargetuti;
		}
		
		targetUtil = targetuti;
		return targetuti;
	}

	public double averResT(){
		if (timeBetweenRounds.size()==0) {
			return 0;
		}
		double total=0;
		for(int i=0; i < timeBetweenRounds.size();i++)
			total = total + timeBetweenRounds.get(i);
		return total / timeBetweenRounds.size();
	}

	public double averLastTResT(int length){
		if(timeBetweenRounds.size()<length)
			return 0;
		double total=0;
		for(int i = (timeBetweenRounds.size()-1); i > timeBetweenRounds.size() - length - 1; i--)
			total = total + timeBetweenRounds.get(i);
		return total/length;
	}
	
	public BidDetails getSuggestBid() {
		return suggestBid;
	}
	
	public double getRoundLeft() {
		return roundleft;
	}
	
	public LinkedList<BidDetails> getBestTenBids(){
		return bestTenBids;
	}
	
	public double getMinimumBidUtil() {
		return minimumBidUtil;
	}
	
	public double getTargetUtil(){
		return targetUtil;
	}
	
	public double getAcceptableUtil() {
		return acceptableUtil;
	}
	
	public void setPreviousTime(double time) {
		previousTime = time;
	}
}
