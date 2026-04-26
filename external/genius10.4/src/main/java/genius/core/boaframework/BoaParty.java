package genius.core.boaframework;

import java.util.List;

import java.util.Map;

import genius.core.AgentID;
import genius.core.Bid;
import genius.core.NegotiationResult;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.EndNegotiation;
import genius.core.actions.Offer;
import genius.core.bidding.BidDetails;
import genius.core.parties.AbstractNegotiationParty;
import genius.core.parties.NegotiationInfo;
import genius.core.parties.NegotiationParty;
import genius.core.persistent.PersistentDataType;
import genius.core.repository.boa.BoaPartyRepItem;

/**
 * This class is used to convert a BOA party to a real agent.
 * 
 * This class is special in that it is constructed directly by a
 * {@link BoaPartyRepItem}.
 * <p>
 * You can extend this directly to build your own BoaParty from your BOA
 * components. To do this, override the {@link #init(NegotiationInfo)} call like
 * this:<br>
 * 
 * <pre>
 * <code> 
 * &#64;Override
 * public void init(NegotiationInfo info) {
 * 	.. setup your_ac, acparams etc...
 * 	configure(your_ac, acparams, your_os,osparams, your_om, omparams,
 *		your_oms, omsparams); 
 * 	super.init(info);
 * }  </code>
 * </pre>
 * 
 * And of course you should do the usual override's of {@link NegotiationParty}
 * functions.
 * <p>
 * For more information, see: Baarslag T., Hindriks K.V., Hendrikx M.,
 * Dirkzwager A., Jonker C.M. Decoupling Negotiating Agents to Explore the Space
 * of Negotiation Strategies. Proceedings of The Fifth International Workshop on
 * Agent-based Complex Automated Negotiations (ACAN 2012), 2012.
 * https://homepages.cwi.nl/~baarslag/pub/Decoupling_Negotiating_Agents_to_Explore_the_Space_of_Negotiation_Strategies_ACAN_2012.pdf
 * 
 */
@SuppressWarnings("serial")
public abstract class BoaParty extends AbstractNegotiationParty {
	/** Decides when to accept */
	protected AcceptanceStrategy acceptConditions;
	/** Decides what to offer */
	protected OfferingStrategy offeringStrategy;
	/** Approximates the utility of a bid for the opponent */
	protected OpponentModel opponentModel;
	/** Selects which bid to send when using an opponent model */
	protected OMStrategy omStrategy;

	// init params for all components.
	private Map<String, Double> acParams;
	private Map<String, Double> osParams;
	private Map<String, Double> omParams;
	private Map<String, Double> omsParams;

	/** Links to the negotiation domain */
	protected NegotiationSession negotiationSession;
	/** Contains the space of possible bids */
	protected OutcomeSpace outcomeSpace;
	private Bid oppBid;

	/**
	 * Stores all relevant values for initializing the components. Will be used
	 * when init is called. This has therefore to be called either as first call
	 * in init, or before the call to init.
	 * 
	 * @param ac
	 *            {@link AcceptanceStrategy}
	 * @param acParams
	 *            a Map<String,Double> containing the init parameters for the
	 *            AcceptanceStrategy
	 * @param os
	 *            {@link OfferingStrategy}
	 * @param osParams
	 *            a Map<String,Double> containing the init parameters for the
	 *            OfferingStrategy
	 * @param om
	 *            {@link OpponentModel}
	 * @param omParams
	 *            a Map<String,Double> containing the init parameters for the
	 *            OpponentModel
	 * @param oms
	 *            {@link OMStrategy}
	 * @param omsParams
	 *            a Map<String,Double> containing the init parameters for the
	 *            OMStrategy
	 */
	public BoaParty configure(AcceptanceStrategy ac,
			Map<String, Double> acParams, OfferingStrategy os,
			Map<String, Double> osParams, OpponentModel om,
			Map<String, Double> omParams, OMStrategy oms,
			Map<String, Double> omsParams) {
		acceptConditions = ac;
		this.acParams = acParams;
		offeringStrategy = os;
		this.osParams = osParams;
		opponentModel = om;
		this.omParams = omParams;
		omStrategy = oms;
		this.omsParams = omsParams;

		return this;
	}

	@Override
	public void init(NegotiationInfo info) {
		// Note that utility space is estimated here in case of uncertainty
		super.init(info);

		SessionData sessionData = null;
		if (info.getPersistentData()
				.getPersistentDataType() == PersistentDataType.SERIALIZABLE) {
			sessionData = (SessionData) info.getPersistentData().get();
		}
		if (sessionData == null) {
			sessionData = new SessionData();
		}
		negotiationSession = new NegotiationSession(sessionData, utilitySpace,
				timeline, null, info.getUserModel(), info.getUser());
		initStrategies();
	}

	/**
	 * Follows the BOA flow for receiving an offer: 1. store the details of the
	 * received bid. 2. tell the {@link OpponentModel} to update itself based on
	 * the received bid.
	 */
	@Override
	public void receiveMessage(AgentID sender, Action opponentAction) {
		// 1. if the opponent made a bid
		if (opponentAction instanceof Offer) {
			oppBid = ((Offer) opponentAction).getBid();
			// 2. store the opponent's trace
			try {
				BidDetails opponentBid = new BidDetails(oppBid,
						negotiationSession.getUtilitySpace().getUtility(oppBid),
						negotiationSession.getTime());
				negotiationSession.getOpponentBidHistory().add(opponentBid);
			} catch (Exception e) {
				e.printStackTrace();
			}
			// 3. if there is an opponent model, update it using the opponent's
			// bid
			if (opponentModel != null && !(opponentModel instanceof NoModel)) {
				if (omStrategy.canUpdateOM()) {
					opponentModel.updateModel(oppBid);
				} else {
					if (!opponentModel.isCleared()) {
						opponentModel.cleanUp();
					}
				}
			}
		}
	}

	/**
	 * Follows the BOA flow for making an offer: 1. ask the
	 * {@link OfferingStrategy} for a bid. 2. ask the {@link AcceptanceStrategy}
	 * if the opponent bid needs to be accepted instead.
	 */
	@Override
	public Action chooseAction(List<Class<? extends Action>> possibleActions) {

		BidDetails bid;

		// if our history is empty, then make an opening bid
		if (negotiationSession.getOwnBidHistory().getHistory().isEmpty()) {
			bid = offeringStrategy.determineOpeningBid();
		} else {
			// else make a normal bid
			bid = offeringStrategy.determineNextBid();
			if (offeringStrategy.isEndNegotiation()) {
				return new EndNegotiation(getPartyId());
			}
		}

		// if the offering strategy made a mistake and didn't set a bid: accept
		if (bid == null) {
			System.out.println("Error in code, null bid was given");
			return new Accept(getPartyId(), oppBid);
		} else {
			offeringStrategy.setNextBid(bid);
		}

		// check if the opponent bid should be accepted
		Actions decision = Actions.Reject;
		if (!negotiationSession.getOpponentBidHistory().getHistory()
				.isEmpty()) {
			decision = acceptConditions.determineAcceptability();
		}

		// check if the agent decided to break off the negotiation
		if (decision.equals(Actions.Break)) {
			System.out.println("send EndNegotiation");
			return new EndNegotiation(getPartyId());
		}
		// if agent does not accept, it offers the counter bid
		if (decision.equals(Actions.Reject)) {
			negotiationSession.getOwnBidHistory().add(bid);
			return new Offer(getPartyId(), bid.getBid());
		} else {
			return new Accept(getPartyId(), oppBid);
		}
	}

	/**
	 * Method that first calls the endSession method of each component to
	 * receiveMessage the session data and then stores the session data if it is
	 * not empty and is changed.
	 */
	public void endSession(NegotiationResult result) 
	{
		offeringStrategy.endSession(result);
		acceptConditions.endSession(result);
		opponentModel.endSession(result);
		SessionData savedData = negotiationSession.getSessionData();
		if (!savedData.isEmpty() && savedData.isChanged()) {
			savedData.changesCommitted();
			getData().put(savedData);
		}
	}

	/**
	 * Clears the agent's variables.
	 */
	public void cleanUp() {
		offeringStrategy = null;
		acceptConditions = null;
		omStrategy = null;
		opponentModel = null;
		outcomeSpace = null;
		negotiationSession = null;
	}

	/**
	 * Returns the offering strategy of the agent.
	 * 
	 * @return offeringstrategy of the agent.
	 */
	public OfferingStrategy getOfferingStrategy() {
		return offeringStrategy;
	}

	/**
	 * Returns the opponent model of the agent.
	 * 
	 * @return opponent model of the agent.
	 */
	public OpponentModel getOpponentModel() {
		return opponentModel;
	}

	/**
	 * Returns the acceptance strategy of the agent.
	 * 
	 * @return acceptance strategy of the agent.
	 */
	public AcceptanceStrategy getAcceptanceStrategy() {
		return acceptConditions;
	}

	private void initStrategies() {
		// init the components.
		try {
			opponentModel.init(negotiationSession, omParams);
			// opponentModel.setOpponentUtilitySpace((BilateralAtomicNegotiationSession)fNegotiation);
			omStrategy.init(negotiationSession, opponentModel, omsParams);
			offeringStrategy.init(negotiationSession, opponentModel, omStrategy,
					osParams);
			acceptConditions.init(negotiationSession, offeringStrategy,
					opponentModel, acParams);
			// acceptConditions.setOpponentUtilitySpace(fNegotiation);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}