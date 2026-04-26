package agents.anac.y2014.BraveCat.AcceptanceStrategies;

import java.util.HashMap;

import agents.anac.y2014.BraveCat.OfferingStrategies.OfferingStrategy;
import agents.anac.y2014.BraveCat.OpponentModels.OpponentModel;
import agents.anac.y2014.BraveCat.necessaryClasses.NegotiationSession;
import genius.core.boaframework.Actions;

public class AC_Next extends AcceptanceStrategy {
	private double a;
	private double b;

	public AC_Next() {
	}

	public AC_Next(NegotiationSession negoSession, OfferingStrategy strat, double alpha, double beta) {
		this.negotiationSession = negoSession;
		this.offeringStrategy = strat;
		this.a = alpha;
		this.b = beta;
	}

	@Override
	public void init(NegotiationSession negoSession, OfferingStrategy strat, OpponentModel opponentModel,
			HashMap<String, Double> parameters) throws Exception {
		this.negotiationSession = negoSession;
		this.offeringStrategy = strat;

		if ((parameters.get("a") != null) || (parameters.get("b") != null)) {
			this.a = ((Double) parameters.get("a")).doubleValue();
			this.b = ((Double) parameters.get("b")).doubleValue();
		} else {
			this.a = 1.0D;
			this.b = 0.0D;
		}
	}

	@Override
	public String printParameters() {
		String str = "[a: " + this.a + " b: " + this.b + "]";
		return str;
	}

	@Override
	public Actions determineAcceptability() {
		double nextMyBidUtil = this.offeringStrategy.getNextBid().getMyUndiscountedUtil();
		double lastOpponentBidUtil = this.negotiationSession.getOpponentBidHistory().getLastBidDetails()
				.getMyUndiscountedUtil();

		if (this.a * lastOpponentBidUtil + this.b >= nextMyBidUtil) {
			return Actions.Accept;
		}
		return Actions.Reject;
	}
}