package agents.anac.y2013.MetaAgent.portfolio.thenegotiatorreloaded;

import java.util.HashMap;

/**
 * This is an abstract class for the agents offering strategy
 * 
 * @author Alex Dirkzwager, Mark Hendrikx
 * @version 15-12-11
 */
public abstract class OfferingStrategy { 
	/** The next bid the agent plans to present */
	protected BidDetails nextBid;
	/** Reference to the negotiation session */
	protected NegotiationSession negotiationSession;
	/** Reference to the opponent model */
	protected OpponentModel opponentModel;
	/** Reference to the opponent model strategy */
	protected OMStrategy omStrategy;
	/** Reference to helper class used if there are dependencies between
	 * the acceptance condition an offering strategy  */
	protected SharedAgentState helper;
	
	/**
	 * Initializes the offering strategy. If parameters are used,
	 * this method should be overridden.
	 * 
	 * @param negotiationSession
	 * @param parameters
	 */
	public void init(NegotiationSession negotiationSession, OpponentModel opponentModel, 
						OMStrategy omStrategy, HashMap<String, Double> parameters) throws Exception {
		this.negotiationSession = negotiationSession;
		this.opponentModel = opponentModel;
		this.omStrategy = omStrategy;
	}
	
	/**
	 * determines the first bid to be offered by the agent
	 * @return UTBid the beginBid
	 */
	public abstract BidDetails determineOpeningBid();

	
	/**
	 * determines the next bid the agent will offer to the opponent
	 * @return UTBid the nextBid
	 */
	public abstract BidDetails determineNextBid();
		
	
	public BidDetails getNextBid(){
		return nextBid;
	}
	
	public void setNextBid(BidDetails counterBid) {
		nextBid = counterBid;
	}
	
	public SharedAgentState getHelper() {
		return helper;
	}
}