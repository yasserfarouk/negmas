package agents.anac.y2014.BraveCat.necessaryClasses;

import java.io.Serializable;
import java.util.ArrayList;

import agents.anac.y2014.BraveCat.AcceptanceStrategies.AcceptanceStrategy;
import agents.anac.y2014.BraveCat.OfferingStrategies.OfferingStrategy;
import agents.anac.y2014.BraveCat.OpponentModelStrategies.OMStrategy;
import agents.anac.y2014.BraveCat.OpponentModels.NoModel;
import agents.anac.y2014.BraveCat.OpponentModels.OpponentModel;
import genius.core.Agent;
import genius.core.Bid;
import genius.core.NegotiationResult;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.EndNegotiation;
import genius.core.actions.Offer;
import genius.core.bidding.BidDetails;
import genius.core.boaframework.Actions;
import genius.core.boaframework.OutcomeSpace;
import genius.core.boaframework.SessionData;
import genius.core.misc.Pair;

public abstract class BOAagent extends Agent {
	// Added variables.
	public int numberOfReceivedBids = 0;
	public int numberOfEvaluatedBids = 0;

	protected AcceptanceStrategy acceptConditions;
	protected OfferingStrategy offeringStrategy;
	protected OpponentModel opponentModel;
	public NegotiationSession negotiationSession;
	protected OMStrategy omStrategy;
	public ArrayList<Pair<Bid, String>> savedOutcomes;
	protected OutcomeSpace outcomeSpace;
	private Bid oppBid;

	@Override
	public void init() {
		super.init();
		Serializable storedData = loadSessionData();
		SessionData sessionData;
		if (storedData == null)
			sessionData = new SessionData();
		else {
			sessionData = (SessionData) storedData;
		}
		this.negotiationSession = new NegotiationSession(sessionData,
				this.utilitySpace, this.timeline);
		agentSetup();
	}

	public abstract void agentSetup();

	public void setDecoupledComponents(AcceptanceStrategy ac,
			OfferingStrategy os, OpponentModel om, OMStrategy oms) {
		this.acceptConditions = ac;
		this.offeringStrategy = os;
		this.opponentModel = om;
		this.omStrategy = oms;
	}

	@Override
	public abstract String getName();

	@Override
	public void ReceiveMessage(Action opponentAction) {
		if ((opponentAction instanceof Offer)) {
			numberOfReceivedBids++;
			oppBid = ((Offer) opponentAction).getBid();
			try {
				BidDetails opponentBid = new BidDetails(oppBid,
						this.negotiationSession.getUtilitySpace().getUtility(
								oppBid), this.negotiationSession.getTime());
				this.negotiationSession.getOpponentBidHistory()
						.add(opponentBid);
			} catch (Exception e) {
			}

			if ((this.opponentModel != null)
					&& (!(this.opponentModel instanceof NoModel)))
				if (this.omStrategy.canUpdateOM()) {
					this.opponentModel.updateModel(oppBid);
				} else if (!this.opponentModel.isCleared())
					this.opponentModel.cleanUp();
		}
	}

	@Override
	public Action chooseAction() {
		BidDetails bid = null;
		if (this.negotiationSession.getOwnBidHistory().getHistory().isEmpty()) {
			bid = this.offeringStrategy.determineOpeningBid();
		} else {
			try {
				bid = this.offeringStrategy.determineNextBid();
			} catch (Exception ex) {
				System.out
						.println("Exception occured while determining the next bid in the chooseAction function!");
			}
			if (this.offeringStrategy.isEndNegotiation()) {
				return new EndNegotiation(getAgentID());
			}

		}

		if (bid == null) {
			System.out.println("Error in code, null bid was given");
			return new Accept(getAgentID(), oppBid);
		}
		this.offeringStrategy.setNextBid(bid);

		Actions decision = Actions.Reject;
		if (!this.negotiationSession.getOpponentBidHistory().getHistory()
				.isEmpty()) {
			decision = this.acceptConditions.determineAcceptability();
		}

		if (decision.equals(Actions.Break)) {
			System.out.println("send EndNegotiation");
			return new EndNegotiation(getAgentID());
		}

		if (decision.equals(Actions.Reject)) {
			try {
				this.negotiationSession.getOwnBidHistory().add(
						new BidDetails(bid.getBid(), this.negotiationSession
								.getUtilitySpace().getUtility(bid.getBid()),
								this.negotiationSession.getTime()));
			} catch (Exception ex) {
			}
			return new Offer(getAgentID(), bid.getBid());
		}
		return new Accept(getAgentID(), oppBid);
	}

	public OfferingStrategy getOfferingStrategy() {
		return this.offeringStrategy;
	}

	public OpponentModel getOpponentModel() {
		return this.opponentModel;
	}

	public AcceptanceStrategy getAcceptanceStrategy() {
		return this.acceptConditions;
	}

	@Override
	public void endSession(NegotiationResult result) {
		this.offeringStrategy.endSession(result);
		this.acceptConditions.endSession(result);
		this.opponentModel.endSession(result);
		SessionData savedData = this.negotiationSession.getSessionData();
		if ((!savedData.isEmpty()) && (savedData.isChanged())) {
			savedData.changesCommitted();
			saveSessionData(savedData);
		}
	}

	public void cleanUp() {
		this.offeringStrategy = null;
		this.acceptConditions = null;
		this.omStrategy = null;
		this.opponentModel = null;
		this.outcomeSpace = null;
		this.negotiationSession = null;
	}

	public NegotiationSession ReturnnegotiationSession() {
		return this.negotiationSession;
	}
}