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
 * This acceptance condition uses two versions of AC_next. The used AC_next
 * depends on if the discount of the domain is (non)-negligible. The parameter e
 * determines when a domain is marked as discounted or not.
 * 
 * Decoupling Negotiating Agents to Explore the Space of Negotiation Strategies
 * T. Baarslag, K. Hindriks, M. Hendrikx, A. Dirkzwager, C.M. Jonker
 * 
 * @author Alex Dirkzwager, Mark Hendrikx
 */
public class AC_CombiV4 extends AcceptanceStrategy {

	private double a;
	private double b;
	private double c;
	private double d;
	private double e;
	private boolean discountedDomain;

	/**
	 * Empty constructor for the BOA framework.
	 */
	public AC_CombiV4() {
	}

	public AC_CombiV4(NegotiationSession negoSession, OfferingStrategy strat, double a, double b, double c, double d,
			double e) {

		this.negotiationSession = negoSession;
		this.offeringStrategy = strat;
		this.a = a;
		this.b = b;
		this.c = c;
		this.d = d;
		this.e = e;

		if (negotiationSession.getDiscountFactor() < 0.00001 || negotiationSession.getDiscountFactor() > e) {
			discountedDomain = false;
		} else {
			discountedDomain = true;
		}
	}

	@Override
	public void init(NegotiationSession negoSession, OfferingStrategy strat, OpponentModel opponentModel,
			Map<String, Double> parameters) throws Exception {
		this.negotiationSession = negoSession;
		this.offeringStrategy = strat;
		if (parameters.get("a") != null || parameters.get("b") != null
				|| parameters.get("c") != null && parameters.get("d") != null && parameters.get("e") != null) {
			a = parameters.get("a");
			b = parameters.get("b");
			c = parameters.get("c");
			d = parameters.get("d");
			e = parameters.get("e");
			if (negotiationSession.getDiscountFactor() < 0.00001 || negotiationSession.getDiscountFactor() > e) {
				discountedDomain = false;
			} else {
				discountedDomain = true;
			}
		} else {
			throw new Exception("Paramaters were not correctly set");
		}
	}

	@Override
	public String printParameters() {
		return "[a: " + a + " b: " + b + " c: " + c + " d: " + d + " e: " + e + "]";
	}

	@Override
	public Actions determineAcceptability() {
		double nextMyBidUtil = offeringStrategy.getNextBid().getMyUndiscountedUtil();
		double lastOpponentBidUtil = negotiationSession.getOpponentBidHistory().getLastBidDetails()
				.getMyUndiscountedUtil();

		double target = 0;
		if (!discountedDomain) {
			// no discount mode
			target = a * lastOpponentBidUtil + b;
		} else {
			// discount mode
			target = c * lastOpponentBidUtil + d;
		}
		if (target > 1.0) {
			target = 1.0;
		}

		if (target >= nextMyBidUtil) {
			return Actions.Accept;
		}

		return Actions.Reject;
	}

	@Override
	public Set<BOAparameter> getParameterSpec() {

		Set<BOAparameter> set = new HashSet<BOAparameter>();
		set.add(new BOAparameter("a", 1.0, "Multiplier"));
		set.add(new BOAparameter("b", 0.0, "Constant"));
		set.add(new BOAparameter("c", 1.0, "Multiplier discount"));
		set.add(new BOAparameter("d", 0.0, "Constant discount"));
		set.add(new BOAparameter("e", 0.8, "Threshold discount"));

		return set;
	}

	@Override
	public String getName() {
		return "Other - CombiV4";
	}
}
