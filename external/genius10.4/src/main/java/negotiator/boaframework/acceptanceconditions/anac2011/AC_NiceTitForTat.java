package negotiator.boaframework.acceptanceconditions.anac2011;

import java.util.ArrayList;
import java.util.Map;

import genius.core.BidHistory;
import genius.core.bidding.BidDetails;
import genius.core.boaframework.AcceptanceStrategy;
import genius.core.boaframework.Actions;
import genius.core.boaframework.NegotiationSession;
import genius.core.boaframework.OfferingStrategy;
import genius.core.boaframework.OpponentModel;
import negotiator.boaframework.sharedagentstate.anac2011.NiceTitForTatSAS;

/**
 * This is the decoupled Acceptance Conditions for NiceTitForTat (ANAC2011). The
 * code was taken from the ANAC2011 NiceTitForTat and adapted to work within the
 * BOA framework.
 * 
 * Decoupling Negotiating Agents to Explore the Space of Negotiation Strategies
 * T. Baarslag, K. Hindriks, M. Hendrikx, A. Dirkzwager, C.M. Jonker
 * 
 * @author Alex Dirkzwager, Mark Hendrikx
 */
public class AC_NiceTitForTat extends AcceptanceStrategy {

	/**
	 * Empty constructor for the BOA framework.
	 */
	public AC_NiceTitForTat() {
	}

	public AC_NiceTitForTat(NegotiationSession negoSession, OfferingStrategy strat) throws Exception {
		init(negoSession, strat, null, null);
	}

	@Override
	public void init(NegotiationSession negoSession, OfferingStrategy strat, OpponentModel opponentModel,
			Map<String, Double> parameters) throws Exception {
		this.negotiationSession = negoSession;
		this.offeringStrategy = strat;

		// checking if offeringStrategy helper is a NiceTitForTatHelper
		if (offeringStrategy.getHelper() == null || (!offeringStrategy.getHelper().getName().equals("NiceTitForTat"))) {
			helper = new NiceTitForTatSAS(negotiationSession);
		} else {
			helper = (NiceTitForTatSAS) offeringStrategy.getHelper();
		}
	}

	@Override
	public Actions determineAcceptability() {

		if (isAcceptable()) {
			BidDetails opponentBid = negotiationSession.getOpponentBidHistory().getLastBidDetails();

			double offeredUtility = opponentBid.getMyUndiscountedUtil();
			double now = negotiationSession.getTime();
			double timeLeft = 1 - now;

			BidHistory recentBids = negotiationSession.getOpponentBidHistory().filterBetweenTime(now - timeLeft, now);
			int expectedBids = recentBids.size();

			BidDetails bestBid = negotiationSession.getOpponentBidHistory().getBestBidDetails();
			double bestBidUtility = bestBid.getMyUndiscountedUtil();

			// we expect to see more bids, that are strictly better than what we
			// are about to offer
			if (!(expectedBids > 1 && bestBidUtility > offeredUtility)) {
				return Actions.Accept;
			}
		}
		return Actions.Reject;
		// NOTE: I deleted the code which would give the best bid instead of
		// false!
	}

	public boolean isAcceptable() {
		double time = negotiationSession.getTime();
		BidDetails opponentBid = negotiationSession.getOpponentBidHistory().getLastBidDetails();
		BidDetails myNextBid = offeringStrategy.getNextBid();

		// AC_NEXT
		try {
			if (negotiationSession.getUtilitySpace().getUtility(opponentBid.getBid()) >= myNextBid
					.getMyUndiscountedUtil()) {
				return true;
			}
		} catch (Exception e) {
			e.printStackTrace();
		}

		if (time < 0.98)
			return false;

		double offeredUndiscountedUtility = opponentBid.getMyUndiscountedUtil();
		double now = time;
		double timeLeft = 1 - now;

		// if we will still see a lot of bids (more than enoughBidsToCome), we
		// do not accept
		ArrayList<BidDetails> recentBids = ((NiceTitForTatSAS) helper).filterBetweenTime(now - timeLeft, now);
		int recentBidsSize = recentBids.size();
		int enoughBidsToCome = 10;
		if (((NiceTitForTatSAS) helper).isDomainBig())
			enoughBidsToCome = 40;
		if (recentBidsSize > enoughBidsToCome) {
			return false;
		}

		double window = timeLeft;
		ArrayList<BidDetails> recentBetterBids = ((NiceTitForTatSAS) helper).filterBetween(offeredUndiscountedUtility,
				1, now - window, now);
		int n = recentBetterBids.size();
		double p = timeLeft / window;
		if (p > 1)
			p = 1;

		double pAllMiss = Math.pow(1 - p, n);
		if (n == 0)
			pAllMiss = 1;
		double pAtLeastOneHit = 1 - pAllMiss;

		double avg = getAverageUtility(recentBetterBids);

		double expectedUtilOfWaitingForABetterBid = pAtLeastOneHit * avg;

		if (offeredUndiscountedUtility > expectedUtilOfWaitingForABetterBid)
			return true;
		return false;
	}

	public double getAverageUtility(ArrayList<BidDetails> list) {
		int size = list.size();
		if (size == 0)
			return 0;
		double totalUtil = 0;
		for (BidDetails b : list)
			totalUtil += b.getMyUndiscountedUtil();
		return totalUtil / size;
	}

	@Override
	public String getName() {
		return "2011 - NiceTitForTat";
	}

}