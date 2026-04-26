package genius.core.boaframework;

import java.io.Serializable;
import java.util.Map;

import genius.core.protocol.BilateralAtomicNegotiationSession;

/**
 * Describes an acceptance strategy of an agent of the BOA framework.
 * 
 * Tim Baarslag, Koen Hindriks, Mark Hendrikx, Alex Dirkzwager and Catholijn M.
 * Jonker. Decoupling Negotiating Agents to Explore the Space of Negotiation
 * Strategies
 * 
 * @author Alex Dirkzwager, Mark Hendrikx
 */
public abstract class AcceptanceStrategy extends BOA {

	/** Reference to the offering strategy. */
	protected OfferingStrategy offeringStrategy;
	/**
	 * Reference to the helper-object, which is used when there is overlap
	 * between the acceptance condition and offering strategy.
	 */
	protected SharedAgentState helper;
	/** Reference to opponnent model of agent. */
	protected OpponentModel opponentModel;

	/**
	 * Standard initialize method to be called after using the empty
	 * constructor. Most of the time this method should be overridden for usage
	 * by the decoupled framework.
	 * 
	 * @param negotiationSession
	 *            state of the negotiation.
	 * @param offeringStrategy
	 *            of the agent.
	 * @param parameters
	 *            of the acceptance strategy.
	 * @throws Exception
	 *             thrown when initializing the acceptance strategy fails.
	 */
	public void init(NegotiationSession negotiationSession, OfferingStrategy offeringStrategy,
			OpponentModel opponentModel, Map<String, Double> parameters) throws Exception {
		super.init(negotiationSession, parameters);
		this.offeringStrategy = offeringStrategy;
		this.opponentModel = opponentModel;
	}

	/**
	 * @return string representation of the parameters supplied to the model.
	 */
	public String printParameters() {
		return "";
	}

	/**
	 * Method which may be overwritten to get access to the opponent's
	 * utilityspace in an experimental setup.
	 * 
	 * @param fNegotiation
	 *            reference to negotiation setting.
	 */
	public void setOpponentUtilitySpace(BilateralAtomicNegotiationSession fNegotiation) {
	}

	/**
	 * Determines to either to either accept or reject the opponent's bid or
	 * even quit the negotiation.
	 * 
	 * @return one of three possible actions: Actions.Accept, Actions.Reject,
	 *         Actions.Break.
	 */
	public abstract Actions determineAcceptability();

	@Override
	public final void storeData(Serializable object) {
		negotiationSession.setData(BoaType.ACCEPTANCESTRATEGY, object);
	}

	@Override
	public final Serializable loadData() {
		return negotiationSession.getData(BoaType.ACCEPTANCESTRATEGY);
	}

	/**
	 * Method which states if the current acceptance strategy is the
	 * Multi-Acceptance Strategy. This method should always return false, except
	 * for the MAC.
	 * 
	 * @return if AC is MAC.
	 */
	public boolean isMAC() {
		return false;
	}

}