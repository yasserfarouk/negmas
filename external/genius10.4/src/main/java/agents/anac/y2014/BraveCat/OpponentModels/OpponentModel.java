package agents.anac.y2014.BraveCat.OpponentModels;

import java.io.Serializable;
import java.util.HashMap;

import agents.anac.y2014.BraveCat.necessaryClasses.NegotiationSession;
import genius.core.Bid;
import genius.core.NegotiationResult;
import genius.core.bidding.BidDetails;
import genius.core.boaframework.BoaType;
import genius.core.issue.Issue;
import genius.core.protocol.BilateralAtomicNegotiationSession;
import genius.core.utility.AbstractUtilitySpace;
import genius.core.utility.AdditiveUtilitySpace;
import genius.core.utility.UTILITYSPACETYPE;

public abstract class OpponentModel {
	protected NegotiationSession negotiationSession;
	protected AbstractUtilitySpace opponentUtilitySpace;
	private boolean cleared;

	public void init(NegotiationSession negotiationSession, HashMap<String, Double> parameters,
			UTILITYSPACETYPE utilitySpaceType) throws Exception {
		this.negotiationSession = negotiationSession;
		this.opponentUtilitySpace = (AbstractUtilitySpace) negotiationSession.getUtilitySpace().copy();
		this.cleared = false;
	}

	public void init(NegotiationSession negotiationSession) throws Exception {
		this.negotiationSession = negotiationSession;
		this.cleared = false;
	}

	public void init(NegotiationSession negotiationSession, UTILITYSPACETYPE utilitySpaceType) throws Exception {
		this.negotiationSession = negotiationSession;
		this.opponentUtilitySpace = (AbstractUtilitySpace) negotiationSession.getUtilitySpace().copy();
		this.cleared = false;
	}

	public void updateModel(Bid opponentBid) {
		updateModel(opponentBid, this.negotiationSession.getTime());
	}

	public abstract void updateModel(Bid paramBid, double paramDouble);

	public double getBidEvaluation(Bid bid) throws Exception {
		try {
			return this.opponentUtilitySpace.getUtility(bid);
		} catch (Exception e) {
		}
		return -1.0D;
	}

	public double getBidEvaluation(BidDetails bid) throws Exception {
		return 0;
	}

	public double getRealBidEvaluation(Bid bid) throws Exception {
		return 0;
	}

	public AbstractUtilitySpace getOpponentUtilitySpace() {
		return this.opponentUtilitySpace;
	}

	public void setOpponentUtilitySpace(BilateralAtomicNegotiationSession fNegotiation) {
	}

	public void setOpponentUtilitySpace(AdditiveUtilitySpace opponentUtilitySpace) {
	}

	public double getWeight(Issue issue) {
		if (opponentUtilitySpace instanceof AdditiveUtilitySpace) {
			return ((AdditiveUtilitySpace) this.opponentUtilitySpace).getWeight(issue.getNumber());
		}
		return 0;
	}

	public double[] getIssueWeights() {
		double[] estimatedIssueWeights = new double[this.negotiationSession.getUtilitySpace().getDomain().getIssues()
				.size()];
		int i = 0;
		for (Issue issue : this.negotiationSession.getUtilitySpace().getDomain().getIssues()) {
			estimatedIssueWeights[i] = getWeight(issue);
			i++;
		}
		return estimatedIssueWeights;
	}

	public void cleanUp() {
		this.negotiationSession = null;
		this.cleared = true;
	}

	public boolean isCleared() {
		return this.cleared;
	}

	public String getName() {
		return "Default";
	}

	public final void storeData(Serializable object) {
		this.negotiationSession.setData(BoaType.OPPONENTMODEL, object);
	}

	public final Serializable loadData() {
		return this.negotiationSession.getData(BoaType.OPPONENTMODEL);
	}

	public void endSession(NegotiationResult result) {
	}
}