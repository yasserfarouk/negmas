package agents.anac.y2014.BraveCat.AcceptanceStrategies;

import java.io.Serializable;
import java.util.HashMap;

import agents.anac.y2014.BraveCat.OfferingStrategies.OfferingStrategy;
import agents.anac.y2014.BraveCat.OpponentModels.OpponentModel;
import agents.anac.y2014.BraveCat.necessaryClasses.NegotiationSession;
import agents.anac.y2014.BraveCat.necessaryClasses.Schedular;
import genius.core.NegotiationResult;
import genius.core.boaframework.Actions;
import genius.core.boaframework.BoaType;
import genius.core.boaframework.SharedAgentState;
import genius.core.protocol.BilateralAtomicNegotiationSession;

public abstract class AcceptanceStrategy {
	protected Schedular schedular;
	protected NegotiationSession negotiationSession;
	protected OfferingStrategy offeringStrategy;
	protected SharedAgentState helper;
	protected OpponentModel opponentModel;

	public void init(NegotiationSession negotiationSession, OfferingStrategy offeringStrategy,
			OpponentModel opponentModel, HashMap<String, Double> parameters) throws Exception {
		this.negotiationSession = negotiationSession;
		this.offeringStrategy = offeringStrategy;
		this.opponentModel = opponentModel;
		this.schedular = new Schedular(negotiationSession);
	}

	public String printParameters() {
		return "";
	}

	public void setOpponentUtilitySpace(BilateralAtomicNegotiationSession fNegotiation) {
	}

	public abstract Actions determineAcceptability();

	public final void storeData(Serializable object) {
		this.negotiationSession.setData(BoaType.ACCEPTANCESTRATEGY, object);
	}

	public final Serializable loadData() {
		return this.negotiationSession.getData(BoaType.ACCEPTANCESTRATEGY);
	}

	public void endSession(NegotiationResult result) {
	}

	public boolean isMAC() {
		return false;
	}
}