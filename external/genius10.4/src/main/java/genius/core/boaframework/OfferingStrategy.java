package genius.core.boaframework;

import java.io.Serializable;
import java.util.Map;

import genius.core.bidding.BidDetails;

/**
 * Describes a bidding strategy of an agent of the BOA framework.
 * 
 * Tim Baarslag, Koen Hindriks, Mark Hendrikx, Alex Dirkzwager and Catholijn M.
 * Jonker. Decoupling Negotiating Agents to Explore the Space of Negotiation
 * Strategies
 * 
 * @author Alex Dirkzwager, Mark Hendrikx
 * @author W.Pasman: this now extends {@link BOA} to unite all BOA components.
 * @version 15-12-11
 */
public abstract class OfferingStrategy extends BOA {
	/** The next bid the agent plans to present */
	protected BidDetails nextBid;
	/** Reference to the opponent model */
	protected OpponentModel opponentModel;
	/** Reference to the opponent model strategy */
	protected OMStrategy omStrategy;
	/**
	 * Reference to helper class used if there are dependencies between the
	 * acceptance condition an offering strategy
	 */
	protected SharedAgentState helper;
	/** Boolean to see if endNegotiation is called */
	protected boolean endNegotiation;

	/**
	 * Initializes the offering strategy. If parameters are used, this method
	 * should be overridden.
	 * 
	 * @param negotiationSession
	 *            state of the negotiation.
	 * @param opponentModel
	 *            opponent model which may be used.
	 * @param omStrategy
	 *            opponent model strategy which may be used.
	 * @param parameters
	 *            optional parameters for the offering strategy.
	 * @throws Exception
	 *             if the offering strategy fails to initialize.
	 */
	public void init(NegotiationSession negotiationSession, OpponentModel opponentModel, OMStrategy omStrategy,
			Map<String, Double> parameters) throws Exception {
		super.init(negotiationSession, parameters);
		this.opponentModel = opponentModel;
		this.omStrategy = omStrategy;
		this.endNegotiation = false;
	}

	/**
	 * Determines the first bid to be offered by the agent
	 * 
	 * @return the opening bid of the agent.
	 */
	public abstract BidDetails determineOpeningBid();

	/**
	 * Determines the next bid the agent will offer to the opponent
	 * 
	 * @return bid to offer to the opponent.
	 */
	public abstract BidDetails determineNextBid();

	/**
	 * @return next bid to be offered to the opponent.
	 */
	public BidDetails getNextBid() {
		return nextBid;
	}

	/**
	 * Set the next bid of the agent. This method is automatically called by the
	 * BOA framework.
	 * 
	 * @param nextBid
	 *            to offer to the opponent.
	 */
	public void setNextBid(BidDetails nextBid) {
		this.nextBid = nextBid;
	}

	/**
	 * Return the Helper-object. A helper is used to hold the code shared
	 * between the offering strategy and acceptance strategy. A good design does
	 * not require a helper.
	 * 
	 * @return helper with shared code.
	 */
	public SharedAgentState getHelper() {
		return helper;
	}

	/**
	 * @return true if the negotiation should be ended.
	 */
	public boolean isEndNegotiation() {
		return endNegotiation;
	}

	@Override
	public final void storeData(Serializable object) {
		negotiationSession.setData(BoaType.BIDDINGSTRATEGY, object);
	}

	@Override
	public final Serializable loadData() {
		return negotiationSession.getData(BoaType.BIDDINGSTRATEGY);
	}

}