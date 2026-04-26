package agents.anac.y2019.harddealer;

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
 * This Acceptance Condition will accept an opponent bid if the utility is
 * higher than the acceptance boundary or higher than the bid the agent is 
 * ready to present.
 */
public class HardDealer_AS extends AcceptanceStrategy {
	private double a;
	private double b;
	private double reservationValue;
	private double k;
	/** Concession factor */
	private double e;

	/**
	 * Empty constructor for the BOA framework.
	 */
	public HardDealer_AS() {
	}

	public double f(double t) {
		// First 10% of session: Slowly come up from bids with util .9
		if (t < .1)
		{
			return 20*(t-0.1)*(t-0.1);
		}
		else
		{
			if (e == 0)
			{
				return k;
			}
			double ft = k + (1 - k) * Math.pow(t, 1.0 / e);
			return ft;
		}

	}

	public double p(double t) {
		return this.reservationValue + (1 - this.reservationValue) * (1 - f(t));
	}

	/** 
	 * Method directly called after creating the party which is used to 
	 * initialize the component.
	 */
	@Override
	public void init(NegotiationSession negoSession, OfferingStrategy strat,
			OpponentModel opponentModel, Map<String, Double> parameters)
			throws Exception {
		this.negotiationSession = negoSession;
		this.offeringStrategy = strat;
		this.e = (1.8) / negotiationSession.getTimeline().getTotalTime(); 
				
		if (parameters.get("a") != null || parameters.get("b") != null) {
			a = parameters.get("a");
			b = parameters.get("b");
		} else {
			a = 1;
			b = 0;
		}
		this.reservationValue = negotiationSession.getUtilitySpace().getReservationValue();
		
		if (parameters.get("k") != null)
			this.k = parameters.get("k");
		else
			this.k = 0;
	}

	@Override
	public String printParameters() {
		String str = "[a: " + a + " b: " + b + "]";
		return str;
	}
	
	
	/** 
	 * Method which determines if the party should accept the opponent�s bid.
	 */ 
	@Override
	public Actions determineAcceptability() {
		double normalizedTime = negotiationSession.getTime(); // Normalized time
		
		//CURVE Accepting
		double acceptationValue = p(normalizedTime);

		double nextMyBidUtil = offeringStrategy.getNextBid()
				.getMyUndiscountedUtil(); // The next bid to propose
		double lastOpponentBidUtil = negotiationSession.getOpponentBidHistory()
				.getLastBidDetails().getMyUndiscountedUtil(); 
		if ((a * lastOpponentBidUtil + b >= nextMyBidUtil) || (a * lastOpponentBidUtil + b >= acceptationValue)) {
			return Actions.Accept;
		}
		// Otherwise reject
		return Actions.Reject;
	}

	@Override
	public Set<BOAparameter> getParameterSpec() {

		Set<BOAparameter> set = new HashSet<BOAparameter>();
		set.add(new BOAparameter("a", 1.0,
				"Accept when the opponent's utility * a + b is greater than "
				+ "the utility of our current bid or our acceptance boundary"));
		set.add(new BOAparameter("b", 0.0,
				"Accept when the opponent's utility * a + b is greater than "
				+ "the utility of our current bid or our acceptance boundary"));

		return set;
	}

	@Override
	public String getName() {
		return "HardDealer_AS";
	}
}
