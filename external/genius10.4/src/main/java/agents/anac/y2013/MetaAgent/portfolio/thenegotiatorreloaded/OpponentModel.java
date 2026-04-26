package agents.anac.y2013.MetaAgent.portfolio.thenegotiatorreloaded;

import java.util.HashMap;

import genius.core.Bid;
import genius.core.protocol.BilateralAtomicNegotiationSession;
import genius.core.utility.AdditiveUtilitySpace;

/**
 * This is the abstract class for the agents Opponent Model.
 * 
 * @author Alex Dirkzwager, Mark Hendrikx
 */
public abstract class OpponentModel {
	
	protected NegotiationSession negotiationSession;
	protected AdditiveUtilitySpace opponentUtilitySpace;
	private boolean cleared;
	
	public void init(NegotiationSession domainKnow, HashMap<String, Double> parameters) throws Exception {
		negotiationSession = domainKnow;
		opponentUtilitySpace = new AdditiveUtilitySpace(domainKnow.getUtilitySpace());
		cleared = false;
	}
	
	public void init(NegotiationSession domainKnow) {
		negotiationSession = domainKnow;
		opponentUtilitySpace = new AdditiveUtilitySpace(domainKnow.getUtilitySpace());
	}

	public abstract void updateModel(Bid opponentBid);
	
	/**
	 * Determines the utility of the opponent according to the OpponentModel
	 * @param Bid
	 * @return Utility of the bid
	 */
	public double getBidEvaluation(Bid b){
		try {
			return opponentUtilitySpace.getUtility(b);
		} catch (Exception e) {
			e.printStackTrace();
		}
		return -1;
	}
	
	/**
	 * Determines the discounted utility of the opponent according to the OpponentModel
	 * @param Bid
	 * @param Time
	 * @return
	 */
	public double getDiscountedBidEvaluation(Bid b, double time){
		return opponentUtilitySpace.getUtilityWithDiscount(b, time);
	}
	
	public AdditiveUtilitySpace getOpponentUtilitySpace(){
		return opponentUtilitySpace;
	}

	public void setOpponentUtilitySpace(BilateralAtomicNegotiationSession fNegotiation) { 
		
	}
	
	public void cleanUp() {
		negotiationSession = null;
		cleared = true;
	}

	public boolean isCleared() {
		return cleared;
	}
}
