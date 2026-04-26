package negotiator.boaframework.acceptanceconditions.anac2012;

import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import genius.core.BidHistory;
import genius.core.boaframework.AcceptanceStrategy;
import genius.core.boaframework.Actions;
import genius.core.boaframework.BOAparameter;
import genius.core.boaframework.NegotiationSession;
import genius.core.boaframework.OfferingStrategy;
import genius.core.boaframework.OpponentModel;

/**
 * Uses an extended implementation of AC_next_discounted as discussed in
 * "Decoupling Negotiating Agents to Explore the Space of Negotiation
 * Strategies" . The actual conditions vary based on the value of the
 * reservation value and discount.
 * 
 * @author Alex Dirkzwager, Mark Hendrikx
 */
public class AC_TheNegotiatorReloaded extends AcceptanceStrategy {

	// multiplication constant of AC_next in no discount mode
	private double aNext;
	// addition constant of AC_next in no discount mode
	private double bNext;
	// multiplication constant AC_next in discount mode
	private double aNextDiscount;
	// addition constant of AC_next in discount mode
	private double bNextDiscount;
	// time after which a switch occurs from AC_next to AC_maxinwindow
	private double time;
	// accept when the utility of the opponent's bid is above a constant
	private double constant;
	// if the discount can be ignored or not
	private boolean discountMode = false;
	// if the discount is high
	private boolean highDiscount = false;
	// do not ignore discount when discount is not 0 and lower than this
	// constant
	private final double DISCOUNT_CONSTANT = 0.95;
	// enables the usage of a panic phase in which everything is accepted above
	// constant
	private final boolean PANIC_PHASE = true;

	public AC_TheNegotiatorReloaded() {
	}

	/**
	 * Initializes the acceptance condition.
	 * 
	 * @param negoSession
	 *            negotiation environment
	 * @param strat
	 *            offering strategy of agent
	 * @param a
	 *            multiplication factor of AC_next in no-discount mode
	 * @param b
	 *            addition factor of AC_next in no-discount mode
	 * @param ad
	 *            multiplication factor of AC_next in discount mode
	 * @param bd
	 *            addition factor of AC_next in disount mode
	 * @param c
	 *            constant above which a bid should always be accepted
	 * @param t
	 *            time when AC_next changes in AC_maxinwindow in no-discount
	 *            mode
	 */
	public AC_TheNegotiatorReloaded(NegotiationSession negoSession, OfferingStrategy strat, double a, double b,
			double ad, double bd, double c, double t) {
		this.aNext = a;
		this.bNext = b;
		this.aNextDiscount = ad;
		this.bNextDiscount = bd;
		this.constant = c;
		this.time = t;

		initializeAgent(negoSession, strat);
	}

	/**
	 * Uses the parameters given by the decoupled framework to initialize the
	 * agent.
	 */
	@Override
	public void init(NegotiationSession negoSession, OfferingStrategy strat, OpponentModel opponentModel,
			Map<String, Double> parameters) throws Exception {
		if (parameters.get("a") != null || parameters.get("b") != null || parameters.get("ad") != null
				|| parameters.get("bd") != null || parameters.get("c") != null || parameters.get("t") != null) {
			aNext = parameters.get("a");
			bNext = parameters.get("b");
			aNextDiscount = parameters.get("ad");
			bNextDiscount = parameters.get("bd");
			constant = parameters.get("c");
			time = parameters.get("t");
		} else {
			throw new Exception("Parameters were not correctly set");
		}
		initializeAgent(negoSession, strat);
		System.out.println("Domain size: " + negoSession.getDomain().getNumberOfPossibleBids());
	}

	/**
	 * Helper method to initialize the agent. Determines if the discount can be
	 * neglected.
	 * 
	 * @param negotiationSession
	 *            negotiation environment
	 * @param strat
	 *            offeringstrategy of the agent
	 */
	private void initializeAgent(NegotiationSession negotiationSession, OfferingStrategy strat) {
		this.negotiationSession = negotiationSession;
		this.offeringStrategy = strat;
		if (negotiationSession.getDiscountFactor() > 0.001
				&& negotiationSession.getDiscountFactor() < DISCOUNT_CONSTANT) {
			discountMode = true;
			if (negotiationSession.getDiscountFactor() <= 0.4) {
				highDiscount = true;
			}
		}
	}

	/**
	 * String representation of the acceptance condition used by the decoupled
	 * framework.
	 */
	@Override
	public String printParameters() {
		return "[a: " + aNext + " b: " + bNext + "ad: " + aNextDiscount + " bd: " + bNextDiscount + " time: " + time
				+ " constant: " + constant + "]";
	}

	/**
	 * Determines if the opponent's bid should be accepted, rejected, or the
	 * negotation should be broken down.
	 */
	@Override
	public Actions determineAcceptability() {

		double now = negotiationSession.getTime();
		double nextMyBidUtil = offeringStrategy.getNextBid().getMyUndiscountedUtil();
		double nextMyBidDiscountedUtil = negotiationSession.getUtilitySpace()
				.getUtilityWithDiscount(offeringStrategy.getNextBid().getBid(), now);
		double lastOpponentBidUtil = negotiationSession.getOpponentBidHistory().getLastBidDetails()
				.getMyUndiscountedUtil();
		double currentRV = negotiationSession.getUtilitySpace().getReservationValueWithDiscount(now);

		if (nextMyBidDiscountedUtil <= currentRV) {
			return Actions.Break;
		}

		if (lastOpponentBidUtil >= constant) {
			return Actions.Accept;
		}

		// NO DISCOUNT scenario
		if (!discountMode) {

			// AC_Next && AC_Constant
			if (aNext * lastOpponentBidUtil + bNext >= nextMyBidUtil) {
				return Actions.Accept;
			}
		} else { // DISCOUNT mode
			if (highDiscount && currentRV > 0.85) {
				return Actions.Break;
			}
			if (aNextDiscount * lastOpponentBidUtil + bNextDiscount >= nextMyBidUtil) {
				return Actions.Accept;
			}
		}

		// PANIC phase: AC_CombiMaxinWindow
		if (PANIC_PHASE && discountMode && now > time) {

			double window = 1 - now;
			BidHistory recentBids = negotiationSession.getOpponentBidHistory().filterBetweenTime(now - window, now);
			double max;

			if (recentBids.size() > 0) {
				max = recentBids.getBestBidDetails().getMyUndiscountedUtil();
			} else {
				max = 0.5;
			}

			double expectedUtilOfWaitingForABetterBid = max;

			if (lastOpponentBidUtil >= expectedUtilOfWaitingForABetterBid) {
				return Actions.Accept;
			}
		}
		return Actions.Reject;
	}

	@Override
	public Set<BOAparameter> getParameterSpec() {
		Set<BOAparameter> set = new HashSet<BOAparameter>();
		set.add(new BOAparameter("a", 1.0, "Scaling factor for AC_next when no discount"));
		set.add(new BOAparameter("b", 0.0, "Constant factor for AC_next when no discount"));
		set.add(new BOAparameter("ad", 1.05, "Scaling factor for AC_next when discount"));
		set.add(new BOAparameter("bd", 0.0, "Constant factor for AC_next when discount"));
		set.add(new BOAparameter("c", 0.98, "Constant specifying utility threshold"));
		set.add(new BOAparameter("t", 0.99, "Time after which AC_max is used"));

		return set;
	}

	@Override
	public String getName() {
		return "2012 - TheNegotiatorReloaded";
	}
}