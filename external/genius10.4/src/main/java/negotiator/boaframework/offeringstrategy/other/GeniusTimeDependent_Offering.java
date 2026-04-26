package negotiator.boaframework.offeringstrategy.other;

import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import genius.core.bidding.BidDetails;
import genius.core.boaframework.BOAparameter;
import genius.core.boaframework.NegotiationSession;
import genius.core.boaframework.NoModel;
import genius.core.boaframework.OMStrategy;
import genius.core.boaframework.OfferingStrategy;
import genius.core.boaframework.OpponentModel;
import genius.core.boaframework.SortedOutcomeSpace;

/**
 * The problem with the default TDT, is that when the negotiation starts, some
 * time has already passed. For small domains this amount is negligible.
 * However, for big domains this often results in the first offer not being the
 * best possible offer.
 * 
 * @author Mark Hendrikx
 */
public class GeniusTimeDependent_Offering extends OfferingStrategy {

	private double startTime;
	/**
	 * k \in [0, 1]. For k = 0 the agent starts with a bid of maximum utility
	 */
	private double k;
	/** Maximum target utility */
	private double Pmax;
	/** Minimum target utility */
	private double Pmin;
	/** Concession factor */
	private double e;
	/** Outcome space */
	SortedOutcomeSpace outcomespace;

	/**
	 * Empty constructor used for reflexion. Note this constructor assumes that
	 * init is called next.
	 */
	public GeniusTimeDependent_Offering() {
	}

	public GeniusTimeDependent_Offering(NegotiationSession negoSession, OpponentModel model, OMStrategy oms, double e,
			double k, double max, double min) {
		this.e = e;
		this.k = k;
		this.Pmax = max;
		this.Pmin = min;
		this.negotiationSession = negoSession;
		outcomespace = new SortedOutcomeSpace(negotiationSession.getUtilitySpace());
		negotiationSession.setOutcomeSpace(outcomespace);
		this.opponentModel = model;
		this.omStrategy = oms;
	}

	@Override
	public void init(NegotiationSession negoSession, OpponentModel model, OMStrategy oms,
			Map<String, Double> parameters) throws Exception {
		if (parameters.get("e") != null) {
			this.negotiationSession = negoSession;

			outcomespace = new SortedOutcomeSpace(negotiationSession.getUtilitySpace());
			negotiationSession.setOutcomeSpace(outcomespace);

			this.e = parameters.get("e");

			if (parameters.get("k") != null)
				this.k = parameters.get("k");
			else
				this.k = 0;

			if (parameters.get("min") != null)
				this.Pmin = parameters.get("min");
			else
				this.Pmin = negoSession.getMinBidinDomain().getMyUndiscountedUtil();

			if (parameters.get("max") != null) {
				Pmax = parameters.get("max");
			} else {
				BidDetails maxBid = negoSession.getMaxBidinDomain();
				Pmax = maxBid.getMyUndiscountedUtil();
			}

			this.opponentModel = model;
			this.omStrategy = oms;
		} else {
			throw new Exception("Constant \"e\" for the concession speed was not set.");
		}
	}

	@Override
	public BidDetails determineOpeningBid() {
		startTime = negotiationSession.getTime();
		return determineNextBid();
	}

	/**
	 * Simple offering strategy which retrieves the target utility and finds
	 */
	@Override
	public BidDetails determineNextBid() {
		double time = negotiationSession.getTime();
		double utilityGoal;
		utilityGoal = p(time);

		// if there is no opponent model available
		if (opponentModel instanceof NoModel) {
			nextBid = negotiationSession.getOutcomeSpace().getBidNearUtility(utilityGoal);
		} else {
			nextBid = omStrategy.getBid(outcomespace, utilityGoal);
		}
		return nextBid;
	}

	/**
	 * From [1]:
	 * 
	 * A wide range of time dependent functions can be defined by varying the
	 * way in which f(t) is computed. However, functions must ensure that 0 <=
	 * f(t) <= 1, f(0) = k, and f(1) = 1.
	 * 
	 * That is, the offer will always be between the value range, at the
	 * beginning it will give the initial constant and when the deadline is
	 * reached, it will offer the reservation value.
	 */
	public double f(double t) {
		double ft = k + (1.0 - k) * Math.pow((t - startTime) / (1.0 - startTime), 1.0 / e);
		return ft;
	}

	/**
	 * Makes sure the target utility with in the acceptable range according to
	 * the domain
	 * 
	 * @param t
	 * @return double
	 */
	public double p(double t) {
		return Pmin + (Pmax - Pmin) * (1.0 - f(t));
	}

	@Override
	public Set<BOAparameter> getParameterSpec() {
		Set<BOAparameter> set = new HashSet<BOAparameter>();
		set.add(new BOAparameter("e", 1.0, "Concession rate"));
		set.add(new BOAparameter("k", 0.0, "Offset"));
		set.add(new BOAparameter("min", 0.0, "Minimum utility"));
		set.add(new BOAparameter("max", 0.99, "Maximum utility"));

		return set;
	}

	@Override
	public String getName() {
		return "Other - Genius TimeDependent Offering";
	}
}