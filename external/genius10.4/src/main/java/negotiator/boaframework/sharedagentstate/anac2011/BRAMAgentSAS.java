package negotiator.boaframework.sharedagentstate.anac2011;
import genius.core.Bid;
import genius.core.boaframework.NegotiationSession;
import genius.core.boaframework.SharedAgentState;

/**
 * This is the shared code of the acceptance condition and bidding strategy of ANAC 2011 BRAMAagent.
 * The code was taken from the ANAC2011 BRAMAagent and adapted to work within the BOA framework.
 * 
 * @author Mark Hendrikx
 */
public class BRAMAgentSAS extends SharedAgentState{

	private NegotiationSession negotiationSession;
	
	//The threshold will be calculated as percentage of the required utility depending of the elapsed time 
	private final double THRESHOLD_PERC_FLEXIBILITY_1 = 0.07;
	private final double THRESHOLD_PERC_FLEXIBILITY_2 = 0.15;
	private final double THRESHOLD_PERC_FLEXIBILITY_3 = 0.3;
	private final double THRESHOLD_PERC_FLEXIBILITY_4 = 0.8;
	private double threshold = 0;
	
	public BRAMAgentSAS (NegotiationSession negoSession) {
		negotiationSession = negoSession;
		NAME = "BRAMAgent";
	}
	
	
	public double getThreshold() {
		return threshold;
	}
	
	
	/**
	 * This function calculates the threshold.
	 * It takes into consideration the time that passed from the beginning of the game.
	 * As time goes by, the agent becoming more flexible to the offers that it is willing to accept.
	 * 
	 * @return - the threshold
	 */
	public double getNewThreshold(Bid minBid, Bid maxBid) {
		double time = negotiationSession.getTime();
		double minUtil = negotiationSession.getUtilitySpace().getUtilityWithDiscount(minBid, time);
		double maxUtil = negotiationSession.getUtilitySpace().getUtilityWithDiscount(maxBid, time);
		double thresholdBestBidDiscount = 0.0;
		
		if(negotiationSession.getTimeline().getTime() < 60.0 / 180.0) {
			thresholdBestBidDiscount =   maxUtil - (maxUtil-minUtil)* THRESHOLD_PERC_FLEXIBILITY_1;
			
		} else if(negotiationSession.getTimeline().getTime() < 150.0 / 180.0) {
			thresholdBestBidDiscount =  maxUtil - (maxUtil-minUtil) * THRESHOLD_PERC_FLEXIBILITY_2;
		}
		else if (negotiationSession.getTimeline().getTime() < 175.0 / 180.0)
			thresholdBestBidDiscount =  maxUtil - (maxUtil-minUtil)* THRESHOLD_PERC_FLEXIBILITY_3;
		else {
			thresholdBestBidDiscount =  maxUtil - (maxUtil-minUtil)* THRESHOLD_PERC_FLEXIBILITY_4; 
		}
		
		threshold = thresholdBestBidDiscount;
		return thresholdBestBidDiscount;
	}
}
