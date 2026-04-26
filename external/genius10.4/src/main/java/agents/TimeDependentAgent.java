package agents;

import java.util.List;

import agents.anac.y2011.Nice_Tit_for_Tat.BilateralAgent;
import genius.core.Bid;
import genius.core.bidding.BidDetails;
import genius.core.boaframework.SortedOutcomeSpace;

/**
 * Boulware/Conceder tactics, by Tim Baarslag, adapted from [1]. Adapted by Mark
 * Hendrikx to use the SortedOutcomeSpace instead of BidHistory.
 * 
 * [1] S. Shaheen Fatima Michael Wooldridge Nicholas R. Jennings Optimal
 * Negotiation Strategies for Agents with Incomplete Information
 * http://eprints.ecs.soton.ac.uk/6151/1/atal01.pdf
 * 
 * @author Tim Baarslag, Mark Hendrikx
 */
public abstract class TimeDependentAgent extends BilateralAgent {
	/** k \in [0, 1]. For k = 0 the agent starts with a bid of maximum utility */
	private static final double k = 0;

	private SortedOutcomeSpace outcomeSpace;
	private double Pmax;
	private double Pmin;

	/**
	 * Depending on the value of e, extreme sets show clearly different patterns
	 * of behaviour [1]:
	 * 
	 * 1. Boulware: For this strategy e < 1 and the initial offer is maintained
	 * till time is almost exhausted, when the agent concedes up to its
	 * reservation value.
	 * 
	 * 2. Conceder: For this strategy e > 1 and the agent goes to its
	 * reservation value very quickly.
	 * 
	 * 3. When e = 1, the price is increased linearly.
	 * 
	 * 4. When e = 0, the agent plays hardball.
	 */
	public abstract double getE();

	@Override
	public String getName() {
		return "Time Dependent Agent";
	}

	@Override
	public String getVersion() {
		return "1.1";
	}

	/**
	 * Sets Pmax to the highest obtainable utility, en Pmin to the lowest
	 * obtainable utility above the reservation value.
	 */
	@Override
	public void init() {
		super.init();
		initFields();
	}

	/**
	 * Initialize our fields.
	 */
	protected void initFields() {
		outcomeSpace = new SortedOutcomeSpace(utilitySpace);
		List<BidDetails> allOutcomes = outcomeSpace.getAllOutcomes();
		double maximumUtilityInScenario = allOutcomes.get(0)
				.getMyUndiscountedUtil();
		double mininimumUtilityInScenario = allOutcomes.get(
				allOutcomes.size() - 1).getMyUndiscountedUtil();
		double rv = utilitySpace.getReservationValueUndiscounted();
		Pmax = maximumUtilityInScenario;
		Pmin = Math.max(mininimumUtilityInScenario, rv);
		// log("Pmin = " + Pmin);
		// log("Pmax = " + Pmax);

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
	 * 
	 * For e = 0 (special case), it will behave as a Hardliner.
	 */
	public double f(double t) {
		if (getE() == 0)
			return k;
		double ft = k + (1 - k) * Math.pow(t, 1 / getE());
		// log("f(t) = " + ft);
		return ft;
	}

	public double p(double t) {
		return Pmin + (Pmax - Pmin) * (1 - f(t));
	}

	/**
	 * Does not care about opponent's utility!
	 */
	public Bid pickBidOfUtility(double utility) {
		return outcomeSpace.getBidNearUtility(utility).getBid();
	}

	public Bid makeBid() {
		double time = timeline.getTime();
		double utilityGoal = p(time);
		Bid b = pickBidOfUtility(utilityGoal);

		// double foundUtility = getUtility(b);
		// if (getE() != 0)
		// log("[e=" + getE() + "] t = " + round2(time) + ". Aiming for " +
		// round2(utilityGoal) + ". Found bid of utility " +
		// round2(foundUtility));
		return b;
	}

	@Override
	public Bid chooseCounterBid() {
		return makeBid();
	}

	@Override
	public Bid chooseFirstCounterBid() {
		return makeBid();
	}

	@Override
	public Bid chooseOpeningBid() {
		return makeBid();
	}

	@Override
	public boolean isAcceptable(Bid plannedBid) {
		Bid opponentLastBid = getOpponentLastBid();
		// is acnext(1, 0);
		if (getUndiscountedUtility(opponentLastBid) >= getUndiscountedUtility(plannedBid))
			return true;
		return false;
	}
}
