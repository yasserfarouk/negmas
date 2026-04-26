package negotiator.boaframework.offeringstrategy.anac2010;

import java.util.HashSet;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import genius.core.Bid;
import genius.core.bidding.BidDetails;
import genius.core.boaframework.BOAparameter;
import genius.core.boaframework.NegotiationSession;
import genius.core.boaframework.NoModel;
import genius.core.boaframework.OMStrategy;
import genius.core.boaframework.OfferingStrategy;
import genius.core.boaframework.OpponentModel;
import genius.core.boaframework.SortedOutcomeSpace;
import genius.core.misc.Range;
import negotiator.boaframework.opponentmodel.DefaultModel;

/**
 * This is the decoupled Offering Strategy for IAMCrazyHaggler (ANAC2010). The
 * code was taken from the ANAC2010 IAMCrazyHaggler and adapted to work within
 * the BOA framework.
 * 
 * An opponent model was added, which selects the best set of bids for the
 * opponent with a utility above 0.9.
 * 
 * DEFAULT OM: None
 * 
 * Decoupling Negotiating Agents to Explore the Space of Negotiation Strategies
 * T. Baarslag, K. Hindriks, M. Hendrikx, A. Dirkzwager, C.M. Jonker
 * 
 * @author Alex Dirkzwager, Mark Hendrikx
 */
public class IAMCrazyHaggler_Offering extends OfferingStrategy {

	private double breakoff = 0.9;
	private Random random100;
	private final boolean TEST_EQUIVALENCE = false;

	/**
	 * Empty constructor called by BOA framework.
	 */
	public IAMCrazyHaggler_Offering() {
	}

	@Override
	public void init(NegotiationSession domainKnow, OpponentModel model, OMStrategy oms, Map<String, Double> parameters)
			throws Exception {
		initializeAgent(domainKnow, model, oms);
		if (parameters != null && parameters.get("b") != null) {
			breakoff = parameters.get("b");
		}
	}

	private void initializeAgent(NegotiationSession negoSession, OpponentModel model, OMStrategy oms) {
		if (model instanceof DefaultModel) {
			model = new NoModel();
		}
		this.negotiationSession = negoSession;
		this.opponentModel = model;
		this.omStrategy = oms;
		if (negoSession.getDiscountFactor() == 0) {
			breakoff = 0.95;
		}
		if (TEST_EQUIVALENCE) {
			random100 = new Random(100);
		} else {
			random100 = new Random();
		}

		if (!(opponentModel instanceof NoModel)) {
			SortedOutcomeSpace space = new SortedOutcomeSpace(negotiationSession.getUtilitySpace());
			negotiationSession.setOutcomeSpace(space);
		}
	}

	@Override
	public BidDetails determineOpeningBid() {
		return determineNextBid();
	}

	@Override
	public BidDetails determineNextBid() {
		if (opponentModel instanceof NoModel) {
			Bid bid = null;
			try {
				do {
					bid = negotiationSession.getUtilitySpace().getDomain().getRandomBid(random100);
				} while (negotiationSession.getUtilitySpace().getUtility(bid) <= breakoff);
				nextBid = new BidDetails(bid, negotiationSession.getUtilitySpace().getUtility(bid));
			} catch (Exception e) {
				e.printStackTrace();
			}
		} else {
			nextBid = omStrategy.getBid(negotiationSession.getOutcomeSpace(), new Range(breakoff, 1.1));
		}
		return nextBid;
	}

	@Override
	public Set<BOAparameter> getParameterSpec() {
		Set<BOAparameter> set = new HashSet<BOAparameter>();
		set.add(new BOAparameter("b", 0.9, "Minimum utility"));

		return set;
	}

	@Override
	public String getName() {
		return "2010 - IAMCrazyHaggler";
	}
}