package negotiator.boaframework.acceptanceconditions.anac2012;

import java.util.Map;

import genius.core.Bid;
import genius.core.bidding.BidDetails;
import genius.core.boaframework.AcceptanceStrategy;
import genius.core.boaframework.Actions;
import genius.core.boaframework.NegotiationSession;
import genius.core.boaframework.OfferingStrategy;
import genius.core.boaframework.OpponentModel;
import genius.core.utility.AdditiveUtilitySpace;
import negotiator.boaframework.sharedagentstate.anac2011.BRAMAgentSAS;

/**
 * This is the decoupled Acceptance Condition from BRAMAgent2 (ANAC2012). The
 * code was taken from the ANAC2012 BRAMAgent2 and adapted to work within the
 * BOA framework.
 * 
 * Decoupling Negotiating Agents to Explore the Space of Negotiation Strategies
 * T. Baarslag, K. Hindriks, M. Hendrikx, A. Dirkzwager, C.M. Jonker
 *
 * @author Alex Dirkzwager
 * @version 31/10/12
 */
public class AC_BRAMAgent2 extends AcceptanceStrategy {

	private boolean activeHelper = false;
	private BidDetails bestBid;
	private Bid worstBid;
	private AdditiveUtilitySpace utilitySpace;

	public AC_BRAMAgent2() {
	}

	public AC_BRAMAgent2(NegotiationSession negoSession, OfferingStrategy strat) throws Exception {
		init(negoSession, strat, null, null);
	}

	@Override
	public void init(NegotiationSession negoSession, OfferingStrategy strat, OpponentModel opponentModel,
			Map<String, Double> parameters) throws Exception {
		this.negotiationSession = negoSession;
		this.offeringStrategy = strat;
		bestBid = negoSession.getMaxBidinDomain();
		worstBid = negoSession.getUtilitySpace().getMinUtilityBid();
		utilitySpace = (AdditiveUtilitySpace) negoSession.getUtilitySpace();

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
		double offeredUtility = utilitySpace.getUtilityWithDiscount(
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

		if (offeredUtility >= threshold)// If the utility of the bid that we
										// received from the opponent
			// is larger than the threshold that we ready to accept,
			// we will accept the offer
			return Actions.Accept;

		if (/* ( bidToOffer != null ) && */(offeredUtility >= utilitySpace
				.getUtilityWithDiscount(offeringStrategy.getNextBid().getBid(), negotiationSession.getTime()))) {
			return Actions.Accept;
		}

		/* (bidToOffer == null )|| */
		if (((offeredUtility < this.utilitySpace.getReservationValueWithDiscount(negotiationSession.getTimeline()))
				&& (negotiationSession.getTime() > 177.0 / 180.0))
				&& (this.utilitySpace.getReservationValueWithDiscount(
						negotiationSession.getTimeline()) > this.utilitySpace.getUtilityWithDiscount(
								offeringStrategy.getNextBid().getBid(), negotiationSession.getTimeline()))) {
			return Actions.Break;
		}

		return Actions.Reject;
	}

	@Override
	public String getName() {
		return "2012 - BRAMAgent2";
	}

}
