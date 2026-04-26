package negotiator.boaframework.acceptanceconditions.anac2011;

import java.util.Map;

import genius.core.bidding.BidDetails;
import genius.core.boaframework.AcceptanceStrategy;
import genius.core.boaframework.Actions;
import genius.core.boaframework.NegotiationSession;
import genius.core.boaframework.OfferingStrategy;
import genius.core.boaframework.OpponentModel;
import negotiator.boaframework.sharedagentstate.anac2011.HardHeadedSAS;

/**
 * This is the decoupled Acceptance Conditions for HardHeaded (ANAC2011). The
 * code was taken from the ANAC2011 HardHeaded and adapted to work within the
 * BOA framework.
 * 
 * Decoupling Negotiating Agents to Explore the Space of Negotiation Strategies
 * T. Baarslag, K. Hindriks, M. Hendrikx, A. Dirkzwager, C.M. Jonker
 * 
 * @author Alex Dirkzwager, Mark Hendrikx
 */
public class AC_HardHeaded extends AcceptanceStrategy {

	private boolean activeHelper = false;
	private double prevUtil = 1.0;
	private double lowestOfferBidUtil = 1.0;

	/**
	 * Empty constructor for the BOA framework.
	 */
	public AC_HardHeaded() {
	}

	public AC_HardHeaded(NegotiationSession negoSession, OfferingStrategy strat) throws Exception {
		init(negoSession, strat, null, null);

	}

	@Override
	public void init(NegotiationSession negoSession, OfferingStrategy strat, OpponentModel opponentModel,
			Map<String, Double> parameters) throws Exception {
		this.negotiationSession = negoSession;
		this.offeringStrategy = strat;

		if (offeringStrategy.getHelper() == null || (!offeringStrategy.getHelper().getName().equals("HardHeaded"))) {
			helper = new HardHeadedSAS(negoSession);
			activeHelper = true;
		} else {
			helper = (HardHeadedSAS) offeringStrategy.getHelper();
		}
	}

	@Override
	public Actions determineAcceptability() {
		BidDetails opponentsLastBid = negotiationSession.getOpponentBidHistory().getLastBidDetails();
		// if opponent's suggested bid is better than the one we just selected,
		// then accept it

		if (activeHelper) {
			prevUtil = ((HardHeadedSAS) helper).getLowestUtilityYet();
		} else {
			if (lowestOfferBidUtil != prevUtil) {
				prevUtil = lowestOfferBidUtil;
			}
			lowestOfferBidUtil = ((HardHeadedSAS) helper).getLowestYetUtility();
		}

		double util = 0;
		try {
			util = offeringStrategy.getNextBid().getMyUndiscountedUtil();
		} catch (Exception e) {
			e.printStackTrace();
		}

		if (opponentsLastBid != null && (opponentsLastBid.getMyUndiscountedUtil() > prevUtil
				|| util <= opponentsLastBid.getMyUndiscountedUtil())) {
			return Actions.Accept;
		}
		return Actions.Reject;
	}

	@Override
	public String getName() {
		return "2011 - HardHeaded";
	}
}
