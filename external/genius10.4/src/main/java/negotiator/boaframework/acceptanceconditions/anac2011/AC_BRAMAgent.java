package negotiator.boaframework.acceptanceconditions.anac2011;

import java.util.Map;

import genius.core.Bid;
import genius.core.bidding.BidDetails;
import genius.core.boaframework.AcceptanceStrategy;
import genius.core.boaframework.Actions;
import genius.core.boaframework.NegotiationSession;
import genius.core.boaframework.OfferingStrategy;
import genius.core.boaframework.OpponentModel;
import negotiator.boaframework.sharedagentstate.anac2011.BRAMAgentSAS;

/**
 * This is the decoupled Acceptance Conditions from the BRAMAgent (ANAC2011).
 * The code was taken from the ANAC2011 BRAMAgent and adapted to work within the
 * BOA framework.
 * 
 * Decoupling Negotiating Agents to Explore the Space of Negotiation Strategies
 * T. Baarslag, K. Hindriks, M. Hendrikx, A. Dirkzwager, C.M. Jonker
 * 
 * @author Alex Dirkzwager, Mark Hendrikx
 * @version 26/12/11
 */
public class AC_BRAMAgent extends AcceptanceStrategy {

	private boolean activeHelper = false;
	private BidDetails bestBid;
	private Bid worstBid;

	/**
	 * Empty constructor for the BOA framework.
	 */
	public AC_BRAMAgent() {
	}

	public AC_BRAMAgent(NegotiationSession negoSession, OfferingStrategy strat) throws Exception {
		init(negoSession, strat, null, null);
	}

	@Override
	public void init(NegotiationSession negoSession, OfferingStrategy strat, OpponentModel opponentModel,
			Map<String, Double> parameters) throws Exception {
		this.negotiationSession = negoSession;
		this.offeringStrategy = strat;
		bestBid = negoSession.getMaxBidinDomain();
		worstBid = negoSession.getUtilitySpace().getMinUtilityBid();

		// checking if offeringStrategy SAS is a BRAMAgentSAS
		if (offeringStrategy.getHelper() == null || (!offeringStrategy.getHelper().getName().equals("BRAMAgent"))) {
			helper = new BRAMAgentSAS(negotiationSession);
			activeHelper = true;
		} else {
			helper = (BRAMAgentSAS) offeringStrategy.getHelper();
		}
	}

	@Override
	public Actions determineAcceptability() {
		double offeredUtility = negotiationSession.getUtilitySpace().getUtilityWithDiscount(
				negotiationSession.getOpponentBidHistory().getLastBidDetails().getBid(), negotiationSession.getTime());
		double threshold;

		if (activeHelper) {
			threshold = ((BRAMAgentSAS) helper).getNewThreshold(worstBid, bestBid.getBid());
		} else {
			threshold = ((BRAMAgentSAS) helper).getThreshold();// Update the
																// threshold
																// according to
																// the discount
																// factor
		}

		// If the utility of the bid that we received from the opponent is
		// larger than the threshold that we ready to accept,
		// we will accept the offer OR if the opponent bid is greater than what
		// we are about to offer.
		double nextBidDiscounted = negotiationSession.getUtilitySpace()
				.getUtilityWithDiscount(offeringStrategy.getNextBid().getBid(), negotiationSession.getTime());
		if ((offeredUtility >= threshold)
				|| (offeringStrategy.getNextBid() != null && offeredUtility >= nextBidDiscounted)) {
			return Actions.Accept;
		}
		return Actions.Reject;
	}

	@Override
	public String getName() {
		return "2011 - BRAMAgent";
	}
}
