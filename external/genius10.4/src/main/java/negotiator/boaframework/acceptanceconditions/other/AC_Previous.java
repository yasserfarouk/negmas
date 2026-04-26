package negotiator.boaframework.acceptanceconditions.other;

import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import genius.core.boaframework.AcceptanceStrategy;
import genius.core.boaframework.Actions;
import genius.core.boaframework.BOAparameter;
import genius.core.boaframework.NegotiationSession;
import genius.core.boaframework.OfferingStrategy;
import genius.core.boaframework.OpponentModel;

/**
 * This Acceptance Condition accepts an opponent bid if the utility is higher
 * than the previous bid the agent made.
 * 
 * Decoupling Negotiating Agents to Explore the Space of Negotiation Strategies
 * T. Baarslag, K. Hindriks, M. Hendrikx, A. Dirkzwager, C.M. Jonker
 * 
 * @author Alex Dirkzwager, Mark Hendrikx
 */
public class AC_Previous extends AcceptanceStrategy {

	private double a;
	private double b;

	/**
	 * Empty constructor for the BOA framework.
	 */
	public AC_Previous() {
	}

	public AC_Previous(NegotiationSession negoSession, double alpha, double beta) {
		this.negotiationSession = negoSession;
		this.a = alpha;
		this.b = beta;
	}

	@Override
	public void init(NegotiationSession negoSession, OfferingStrategy strat, OpponentModel opponentModel,
			Map<String, Double> parameters) throws Exception {
		this.negotiationSession = negoSession;
		if (parameters.get("a") != null || parameters.get("b") != null) {
			a = parameters.get("a");
			b = parameters.get("b");
		} else {
			throw new Exception("Parameters were not set.");
		}
	}

	@Override
	public String printParameters() {
		String str = "[a: " + a + " b: " + b + "]";
		return str;
	}

	@Override
	public Actions determineAcceptability() {
		if (!negotiationSession.getOwnBidHistory().getHistory().isEmpty()) {
			double opponentBidUtil = negotiationSession.getOpponentBidHistory().getLastBidDetails()
					.getMyUndiscountedUtil();
			double ownLastBidUtil = negotiationSession.getOwnBidHistory().getLastBidDetails().getMyUndiscountedUtil();
			if (a * opponentBidUtil + b >= ownLastBidUtil) {
				return Actions.Accept;
			}
		}
		return Actions.Reject;

	}

	@Override
	public Set<BOAparameter> getParameterSpec() {

		Set<BOAparameter> set = new HashSet<BOAparameter>();
		set.add(new BOAparameter("a", 1.0,
				"Accept when the opponent's utility * a + b is greater than the utility of our previous bid"));
		set.add(new BOAparameter("b", 0.0,
				"Accept when the opponent's utility * a + b is greater than the utility of our previous bid"));

		return set;
	}

	@Override
	public String getName() {
		return "Other - Previous";
	}
}