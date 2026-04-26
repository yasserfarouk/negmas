package negotiator.boaframework.acceptanceconditions.anac2012;

import java.util.Map;

import genius.core.Bid;
import genius.core.boaframework.AcceptanceStrategy;
import genius.core.boaframework.Actions;
import genius.core.boaframework.NegotiationSession;
import genius.core.boaframework.OfferingStrategy;
import genius.core.boaframework.OpponentModel;
import genius.core.utility.AdditiveUtilitySpace;

/**
 * This is the decoupled Acceptance Condition from OMACagent (ANAC2012). The
 * code was taken from the ANAC2012 OMACagent and adapted to work within the BOA
 * framework.
 * 
 * Decoupling Negotiating Agents to Explore the Space of Negotiation Strategies
 * T. Baarslag, K. Hindriks, M. Hendrikx, A. Dirkzwager, C.M. Jonker
 *
 * @author Alex Dirkzwager
 * @version 31/10/12
 */
public class AC_OMACagent extends AcceptanceStrategy {

	private double discount = 1.0;
	private AdditiveUtilitySpace utilitySpace;
	public double discountThreshold = 0.845D;

	public AC_OMACagent() {
	}

	public AC_OMACagent(NegotiationSession negoSession, OfferingStrategy strat) throws Exception {
		init(negoSession, strat, null, null);
	}

	@Override
	public void init(NegotiationSession negoSession, OfferingStrategy strat, OpponentModel opponentModel,
			Map<String, Double> parameters) throws Exception {
		this.negotiationSession = negoSession;
		offeringStrategy = strat;
		utilitySpace = (AdditiveUtilitySpace) negoSession.getUtilitySpace();

		if (utilitySpace.getDiscountFactor() <= 1D && utilitySpace.getDiscountFactor() > 0D)
			discount = utilitySpace.getDiscountFactor();

	}

	@Override
	public Actions determineAcceptability() {
		Bid partnerBid = negotiationSession.getOpponentBidHistory().getLastBid();
		double time = negotiationSession.getTime();
		if (discount < discountThreshold) {
			if (bidAlreadyMade(partnerBid)) {
				System.out.println("Decoupled accept1");
				return Actions.Accept;
			}
		} else if (time > 0.97) {
			// System.out.println("Decoupled bidAlreadyMade: " +
			// bidAlreadyMade(partnerBid));
			if (bidAlreadyMade(partnerBid)) {
				System.out.println("Decoupled accept2");
				return Actions.Accept;

			}
		}
		double myOfferedUtil = negotiationSession.getDiscountedUtility(offeringStrategy.getNextBid().getBid(), time);
		double offeredUtilFromOpponent = negotiationSession
				.getDiscountedUtility(negotiationSession.getOpponentBidHistory().getLastBid(), time);
		// accept under certain conditions
		try {
			// System.out.println("Decoupled accept3");

			if (isAcceptable(offeredUtilFromOpponent, myOfferedUtil, time, partnerBid))
				return Actions.Accept;
		} catch (Exception e) {
			e.printStackTrace();
		}

		return Actions.Reject;
	}

	private boolean isAcceptable(double offeredUtilFromOpponent, double myOfferedUtil, double time, Bid oppBid)
			throws Exception {

		if (offeredUtilFromOpponent >= myOfferedUtil) {
			return true;
		}

		return false;
	}

	public boolean bidAlreadyMade(Bid a) {
		boolean result = false;
		for (int i = 0; i < negotiationSession.getOwnBidHistory().size(); i++) {
			if (a.equals(negotiationSession.getOwnBidHistory().getHistory().get(i).getBid())) {
				result = true;
			}
		}
		return result;
	}

	@Override
	public String getName() {
		return "2012 - OMACagent";
	}
}
