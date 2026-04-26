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
 * This is the decoupled Acceptance Conditions Based on Tim Baarslag's paper on
 * Acceptance Conditions: "Acceptance Conditions in Automated Negotiation"
 * 
 * This Acceptance Condition accepts a bid if the utility of the opponent's bid
 * plus a constant (the gap) is higher than the utility of the next bid. The
 * acceptance condition is a restricted version of AC_next.
 * 
 * Decoupling Negotiating Agents to Explore the Space of Negotiation Strategies
 * T. Baarslag, K. Hindriks, M. Hendrikx, A. Dirkzwager, C.M. Jonker
 * 
 * @author Alex Dirkzwager
 */
public class AC_Gap extends AcceptanceStrategy {

	private double constant;

	/**
	 * Empty constructor for the BOA framework.
	 */
	public AC_Gap() {
	}

	public AC_Gap(NegotiationSession negoSession, double c) {
		this.negotiationSession = negoSession;
		this.constant = c;
	}

	@Override
	public void init(NegotiationSession negoSession, OfferingStrategy strat, OpponentModel opponentModel,
			Map<String, Double> parameters) throws Exception {
		this.negotiationSession = negoSession;
		if (parameters.get("c") != null) {
			constant = parameters.get("c");
		} else {
			throw new Exception("Constant \"c\" for the threshold was not set.");
		}
	}

	@Override
	public String printParameters() {
		return "[c: " + constant + "]";
	}

	@Override
	public Actions determineAcceptability() {
		if (!negotiationSession.getOwnBidHistory().getHistory().isEmpty()) {
			double opponentBidUtil = negotiationSession.getOpponentBidHistory().getLastBidDetails()
					.getMyUndiscountedUtil();
			double prevMyBidUtil = negotiationSession.getOwnBidHistory().getLastBidDetails().getMyUndiscountedUtil();

			if (opponentBidUtil + constant >= prevMyBidUtil) {
				return Actions.Accept;
			}
		}
		return Actions.Reject;
	}

	@Override
	public Set<BOAparameter> getParameterSpec() {
		Set<BOAparameter> set = new HashSet<BOAparameter>();
		set.add(new BOAparameter("c", 0.01,
				"Accept when opponent's bid utility plus constant c is greater than utility of previous offer"));
		return set;
	}

	@Override
	public String getName() {
		return "Other - Gap";
	}
}