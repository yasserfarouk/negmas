package agents.anac.y2013.MetaAgent.portfolio.thenegotiatorreloaded;

import genius.core.Agent;
import genius.core.Bid;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.EndNegotiation;
import genius.core.actions.Offer;

/**
 * This class describes a basic decoupled agent. The TheDecoupledAgent class
 * extends this class and sets the required parameters.
 * 
 * @author Alex Dirkzwager
 */
public abstract class BOAagent extends Agent {

	/** when to accept */
	protected AcceptanceStrategy acceptConditions;
	/** link to domain */
	protected NegotiationSession negotiationSession;
	/** what to offer */
	protected OfferingStrategy offeringStrategy;
	/** used to determine the utility of a bid for the opponent */
	protected OpponentModel opponentModel;
	/** which bid to select using an opponent model */
	protected OMStrategy omStrategy;
	/** space of possible bids */
	protected OutcomeSpace outcomeSpace;
	private Bid oppBid;

	public void init() {
		super.init();
		negotiationSession = new NegotiationSession(utilitySpace, timeline);
		agentSetup();
	}

	/**
	 * Method used to setup the agent. The method is called directly after
	 * initialization of the agent.
	 */
	public abstract void agentSetup();

	/**
	 * Set the components of the decoupled agent.
	 * 
	 * @param ac
	 *            the acceptance strategy
	 * @param os
	 *            the offering strategy
	 * @param om
	 *            the opponent model
	 * @param oms
	 *            the opponent model strategy
	 */
	public void setDecoupledComponents(AcceptanceStrategy ac,
			OfferingStrategy os, OpponentModel om, OMStrategy oms) {
		acceptConditions = ac;
		offeringStrategy = os;
		opponentModel = om;
		omStrategy = oms;
	}

	@Override
	public String getVersion() {
		return "1.0";
	}

	public abstract String getName();

	/**
	 * Store the actions made by a partner. First store the bid in the history,
	 * then receiveMessage the opponent model.
	 * 
	 * @param Action
	 *            by opponent in current turn
	 */
	public void ReceiveMessage(Action opponentAction) {
		if (opponentAction instanceof Offer) {
			oppBid = ((Offer) opponentAction).getBid();
			try {
				BidDetails opponentBid = new BidDetails(
						oppBid,
						negotiationSession.getUtilitySpace().getUtility(oppBid),
						negotiationSession.getTime());
				negotiationSession.getOpponentBidHistory().add(opponentBid);
			} catch (Exception e) {
				e.printStackTrace();
			}
			if (opponentModel != null && !(opponentModel instanceof NullModel)) {
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
	 * Choose an action to perform.
	 * 
	 * @return Action the agent performs
	 */
	@Override
	public Action chooseAction() {

		BidDetails bid;

		// if our history is empty, then make an opening bid
		if (negotiationSession.getOwnBidHistory().getHistory().isEmpty()) {
			bid = offeringStrategy.determineOpeningBid();
		} else {
			// else make a normal bid
			bid = offeringStrategy.determineNextBid();
		}

		// if the offering strategy made a mistake and didn't set a bid: accept
		if (bid == null) {
			System.out.println("Error in code, null bid was given");
			return new Accept(this.getAgentID(), oppBid);
		} else {
			offeringStrategy.setNextBid(bid);
		}

		// check if the opponent bid should be accepted
		Actions decision = Actions.Reject;
		if (!negotiationSession.getOpponentBidHistory().getHistory().isEmpty()) {
			decision = acceptConditions.determineAcceptability();
		} else {
		}

		if (decision.equals(Actions.Break)) {
			return new EndNegotiation(this.getAgentID());
		}
		// if agent does not accept, it offers the counter bid
		if (decision.equals(Actions.Reject)) {
			negotiationSession.getOwnBidHistory().add(bid);
			return new Offer(this.getAgentID(), bid.getBid());
		} else {
			return new Accept(this.getAgentID(), oppBid);
		}
	}

	public OpponentModel getOpponentModel() {
		return opponentModel;
	}

	/**
	 * Clear the agent's variables.
	 */
	public void cleanUp() {
		offeringStrategy = null;
		acceptConditions = null;
		omStrategy = null;
		opponentModel = null;
		outcomeSpace = null;
		negotiationSession = null;
	}
}