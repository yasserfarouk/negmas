package agents.anac.y2014.BraveCat.OfferingStrategies;

import java.io.Serializable;
import java.util.HashMap;

import agents.anac.y2014.BraveCat.OpponentModelStrategies.OMStrategy;
import agents.anac.y2014.BraveCat.OpponentModels.OpponentModel;
import agents.anac.y2014.BraveCat.necessaryClasses.NegotiationSession;
import agents.anac.y2014.BraveCat.necessaryClasses.Schedular;
import genius.core.NegotiationResult;
import genius.core.bidding.BidDetails;
import genius.core.boaframework.BoaType;
import genius.core.boaframework.SharedAgentState;

public abstract class OfferingStrategy {
	protected Schedular schedular;
	protected BidDetails nextBid;
	protected NegotiationSession negotiationSession;
	protected OpponentModel opponentModel;
	protected OMStrategy omStrategy;
	protected SharedAgentState helper;
	protected boolean endNegotiation;

	public void init(NegotiationSession negotiationSession, OpponentModel opponentModel, OMStrategy omStrategy,
			HashMap<String, Double> parameters) throws Exception {
		this.negotiationSession = negotiationSession;
		this.opponentModel = opponentModel;
		this.omStrategy = omStrategy;
		this.endNegotiation = false;
		this.schedular = new Schedular(negotiationSession);
	}

	public abstract BidDetails determineOpeningBid();

	public abstract BidDetails determineNextBid();

	public BidDetails getNextBid() {
		return this.nextBid;
	}

	public void setNextBid(BidDetails nextBid) {
		this.nextBid = nextBid;
	}

	public SharedAgentState getHelper() {
		return this.helper;
	}

	public boolean isEndNegotiation() {
		return this.endNegotiation;
	}

	public final void storeData(Serializable object) {
		this.negotiationSession.setData(BoaType.BIDDINGSTRATEGY, object);
	}

	public final Serializable loadData() {
		return this.negotiationSession.getData(BoaType.BIDDINGSTRATEGY);
	}

	public void endSession(NegotiationResult result) {
	}

	public abstract String GetName();
}