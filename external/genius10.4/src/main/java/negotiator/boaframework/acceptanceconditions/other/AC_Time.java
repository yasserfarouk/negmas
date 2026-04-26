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
 * This Acceptance Condition accept an opponent bid after a certain time has
 * passed
 * 
 * Decoupling Negotiating Agents to Explore the Space of Negotiation Strategies
 * T. Baarslag, K. Hindriks, M. Hendrikx, A. Dirkzwager, C.M. Jonker
 * 
 * @author Alex Dirkzwager, Mark Hendrikx
 * @version 15-12-11
 */
public class AC_Time extends AcceptanceStrategy {

	private double constant;

	/**
	 * Empty constructor for the BOA framework.
	 */
	public AC_Time() {
	}

	public AC_Time(NegotiationSession negoSession, double c) {
		this.negotiationSession = negoSession;
		this.constant = c;
	}

	@Override
	public void init(NegotiationSession negoSession, OfferingStrategy strat, OpponentModel opponentModel,
			Map<String, Double> parameters) throws Exception {
		this.negotiationSession = negoSession;
		if (parameters.get("t") != null) {
			constant = parameters.get("t");
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
		if (negotiationSession.getTime() > constant) {
			return Actions.Accept;
		}
		return Actions.Reject;
	}

	@Override
	public Set<BOAparameter> getParameterSpec() {

		Set<BOAparameter> set = new HashSet<BOAparameter>();
		set.add(new BOAparameter("t", 0.99, "If time greater than t, then accept"));

		return set;
	}

	@Override
	public String getName() {
		return "Other - Time";
	}
}