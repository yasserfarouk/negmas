package negotiator.boaframework.offeringstrategy.anac2012.TheNegotiatorReloaded;

import genius.core.bidding.BidDetails;
import genius.core.boaframework.NegotiationSession;
import genius.core.boaframework.NoModel;
import genius.core.boaframework.OMStrategy;
import genius.core.boaframework.OpponentModel;
import genius.core.boaframework.SortedOutcomeSpace;

public class TimeDependentFunction {
	
	private NegotiationSession negoSession;
	private double e;
	private double k;
	private double Pmin;
	private double Pmax;
	private double utilityGoal = 1.0;
	private OpponentModel opponentModel;
	private OMStrategy oms;
	private SortedOutcomeSpace outcomespace;
	
	public TimeDependentFunction(NegotiationSession negoSession, OMStrategy oms, OpponentModel opponentModel) throws Exception {
		this.negoSession = negoSession; 
		this.outcomespace = new SortedOutcomeSpace(negoSession.getUtilitySpace());
		negoSession.setOutcomeSpace(outcomespace);
		this.opponentModel = opponentModel;
		this.oms = oms;
	}
	
	public BidDetails getNextBid(double e, double k, double min, double max) {
		this.e = e;
		this.k = k;
		this.Pmin = min;
		this.Pmax = max;
		
		double time = negoSession.getTime();

		utilityGoal = p(time);
		
		if (opponentModel instanceof NoModel) {
			return negoSession.getOutcomeSpace().getBidNearUtility(utilityGoal);
		} else {
			return oms.getBid(outcomespace, utilityGoal);
		}
	}
	
	/**
	 * From [1]:
	 * 
	 * A wide range of time dependent functions can be defined by varying the way in
	 * which f(t) is computed. However, functions must ensure that 0 <= f(t) <= 1,
	 * f(0) = k, and f(1) = 1.
	 * 
	 * That is, the offer will always be between the value range, 
	 * at the beginning it will give the initial constant and when the deadline is reached, it
	 * will offer the reservation value.
	 */
	private double f(double t) {
		double ft = k + (1 - k) * Math.pow(t, 1.0/e);
		return ft;
	}

	/**
	 * Makes sure the target utility with in the acceptable range according to the domain
	 * @param t
	 * @return double
	 */
	private double p(double t) {
		return Pmin + (Pmax - Pmin) * (1 - f(t));
	}
}