package agents.anac.y2019.harddealer;

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
 * We have used the TimedependentAgent class to implement our bidding strategy.
 * Below are the original creators, the code is edited by Sven Hendrikx.
 */
/**
 * This is an abstract class used to implement a TimeDependentAgent Strategy
 * adapted from [1] [1] S. Shaheen Fatima Michael Wooldridge Nicholas R.
 * Jennings Optimal Negotiation Strategies for Agents with Incomplete
 * Information http://eprints.ecs.soton.ac.uk/6151/1/atal01.pdf
 *
 * The default strategy was extended to enable the usage of opponent models.
 */
public class HardDealer_BS extends OfferingStrategy {

	/**
	 * k in [0, 1]. For k = 0 the agent starts with a bid of maximum utility
	 */
	private double reservationValue;
	private double k;
	/** Maximum target utility */
	private double Pmax;
	/** Minimum target utility */
	private double Pmin;
	/** Concession factor */
	private double e;
	/** Outcome space */
	private SortedOutcomeSpace outcomespace;
	/**
	 * Method which initializes the agent by setting all parameters. The
	 * parameter "e" is the only parameter which is required.
	 */
	@Override
	public void init(NegotiationSession negoSession, OpponentModel model, OMStrategy oms,
			Map<String, Double> parameters) throws Exception {
		super.init(negoSession, parameters);
		if (parameters.get("e") != null) {
			this.negotiationSession = negoSession;

			outcomespace = new SortedOutcomeSpace(negotiationSession.getUtilitySpace());

			negotiationSession.setOutcomeSpace(outcomespace);
			
			// We initialize "e" as a function of the negotiation length since different lengths require different concession rates.
			// This will be overwritten if the the timeline is of type Time
			// for more info see report section on bidding strategy
			this.e = (1.8) / negotiationSession.getTimeline().getTotalTime();
			this.reservationValue = negotiationSession.getUtilitySpace().getReservationValue();
			
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
		return determineNextBid();
	}

	/**
	 * Simple offering strategy which retrieves the target utility and looks for
	 * the nearest bid if no opponent model is specified. If an opponent model
	 * is specified, then the agent return a bid according to the opponent model
	 * strategy.
	 */
	@Override
	public BidDetails determineNextBid() {		
		double time = negotiationSession.getTime();
		double TargetValue;
		
		if (p(time) < reservationValue)
			TargetValue = reservationValue;
		else
			TargetValue = p(time);
		
		// In case there is no model, just be time dependent
		if (opponentModel instanceof NoModel) {
			nextBid = negotiationSession.getOutcomeSpace().getBidNearUtility(TargetValue);
		} else {
			nextBid = omStrategy.getBid(outcomespace, TargetValue);
		}
		
		return nextBid;
	}

	public double f(double t) {
		// First 10% of session: Slowly come up from bids with util .9 to allow the opponent to model us.
		if (t < .1)
		{
			return 20*(t-0.1)*(t-0.1);
		}
		else
		{
			if (e == 0)
				return k;
			
			double ft = k + (1 - k) * Math.pow(t, 1.0 / e);
			return ft;
		}

	}

	public double p(double t) {
		return Pmin + (Pmax - Pmin) * (1 - f(t));
			
	}
	
	public NegotiationSession getNegotiationSession() {
		return negotiationSession;
	}

	@Override
	public Set<BOAparameter> getParameterSpec() {
		Set<BOAparameter> set = new HashSet<BOAparameter>();
		set.add(new BOAparameter("e", this.e, "Concession rate"));
		set.add(new BOAparameter("k", this.k, "Offset"));
		set.add(new BOAparameter("min", this.Pmin, "Minimum utility"));
		set.add(new BOAparameter("max", this.Pmax, "Maximum utility"));

		return set;
	}

	@Override
	public String getName() {
		return "HardDealer_BS";
	}
}
