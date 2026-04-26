package negotiator.boaframework.acceptanceconditions.anac2010;

import java.util.Map;

import genius.core.bidding.BidDetails;
import genius.core.boaframework.AcceptanceStrategy;
import genius.core.boaframework.Actions;
import genius.core.boaframework.NegotiationSession;
import genius.core.boaframework.OfferingStrategy;
import genius.core.boaframework.OpponentModel;
import negotiator.boaframework.sharedagentstate.anac2010.YushuSAS;

/**
 * This is the decoupled Acceptance Conditions for Yushu (ANAC2010). The code
 * was taken from the ANAC2010 Yushu and adapted to work within the BOA
 * framework.
 * 
 * Decoupling Negotiating Agents to Explore the Space of Negotiation Strategies
 * T. Baarslag, K. Hindriks, M. Hendrikx, A. Dirkzwager, C.M. Jonker
 * 
 * @author Alex Dirkzwager, Mark Hendrikx
 */
public class AC_Yushu extends AcceptanceStrategy {

	private double roundleft;

	/**
	 * Empty constructor for the BOA framework.
	 */
	public AC_Yushu() {
	}

	public AC_Yushu(NegotiationSession negoSession, OfferingStrategy strat) throws Exception {
		init(negoSession, strat, null, null);
	}

	@Override
	public void init(NegotiationSession negoSession, OfferingStrategy strat, OpponentModel opponentModel,
			Map<String, Double> parameters) throws Exception {
		this.negotiationSession = negoSession;
		offeringStrategy = strat;

		// checking if offeringStrategy helper is a YushuHelper
		if (offeringStrategy.getHelper() == null || (!offeringStrategy.getHelper().getName().equals("Yushu"))) {
			helper = new YushuSAS(negotiationSession);
		} else {
			helper = (YushuSAS) offeringStrategy.getHelper();
		}
	}

	@Override
	public Actions determineAcceptability() {

		BidDetails opponentBid = negotiationSession.getOpponentBidHistory().getLastBidDetails();
		BidDetails myBid = negotiationSession.getOwnBidHistory().getLastBidDetails();
		roundleft = ((YushuSAS) helper).getRoundLeft();
		if (offeringStrategy.getHelper() == null || (!offeringStrategy.getHelper().getName().equals("Yushu"))) {
			((YushuSAS) helper).updateBelief(opponentBid);
		}

		BidDetails suggestedBid = ((YushuSAS) helper).getSuggestBid();
		double acceptableUtil = ((YushuSAS) helper).getAcceptableUtil();

		if (opponentBid != null) {
			double targetUtil;
			if (myBid == null) {
				targetUtil = 1;
			} else if (offeringStrategy.getHelper() != null && offeringStrategy.getHelper().getName().equals("Yushu")) {
				targetUtil = ((YushuSAS) helper).getTargetUtil();
			} else {
				targetUtil = ((YushuSAS) helper).calculateTargetUtility();
			}
			if ((opponentBid.getMyUndiscountedUtil() >= targetUtil)
					| (opponentBid.getMyUndiscountedUtil() >= acceptableUtil)) {
				if (suggestedBid == null || roundleft < 8) {
					return Actions.Accept;
				}
				if ((suggestedBid != null) && (roundleft > 8)) {
					if (suggestedBid.getMyUndiscountedUtil() <= opponentBid.getMyUndiscountedUtil()) {
						return Actions.Accept;
					}
				}
			}
		}
		return Actions.Reject;
	}

	@Override
	public String getName() {
		return "2010 - Yushu";
	}
}