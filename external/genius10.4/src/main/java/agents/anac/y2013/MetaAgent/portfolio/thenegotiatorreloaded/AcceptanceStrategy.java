package agents.anac.y2013.MetaAgent.portfolio.thenegotiatorreloaded;

import java.util.HashMap;

/**
 * This is an abstract class for the agents acceptance strategy.
 * 
 * @author Alex Dirkzwager, Mark Hendrikx
 */
public abstract class  AcceptanceStrategy {
	
	/** Reference to the object which holds all information about the negotiation */
	protected NegotiationSession negotiationSession;
	/** Reference to the offering strategy  */
	protected OfferingStrategy offeringStrategy;
	/** Reference to the helper-object, which is used when there is overlap between
	 * the acceptance condition and offering strategy */
	protected SharedAgentState helper;
	
	/**
	 * Standard initialize method to be called after using the empty constructor.
	 * Most of the time this method should be overridden for usage by the decoupled
	 * framework.
	 * 
	 * @param reference to the negotiation session
	 * @param strat
	 * @param parameters
	 * @throws Exception
	 */
	public void init(NegotiationSession negotiationSession, OfferingStrategy offeringStrategy,
						HashMap<String, Double> parameters) throws Exception {
		this.negotiationSession = negotiationSession;
		this.offeringStrategy = offeringStrategy;
	}
	
	public String printParameters(){
		return"";
	}
	
	/**
	 * Determines the either to accept and offer or not.
	 * @return true if accept
	 */
	public abstract Actions determineAcceptability();
}