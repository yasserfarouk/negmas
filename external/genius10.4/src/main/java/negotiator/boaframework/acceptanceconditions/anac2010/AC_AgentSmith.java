package negotiator.boaframework.acceptanceconditions.anac2010;

import java.util.Map;

import genius.core.bidding.BidDetails;
import genius.core.boaframework.AcceptanceStrategy;
import genius.core.boaframework.Actions;
import genius.core.boaframework.NegotiationSession;
import genius.core.boaframework.OfferingStrategy;
import genius.core.boaframework.OpponentModel;

/**
 * This is the decoupled Acceptance Condition from AgentSmith (ANAC2010). The
 * code was taken from the ANAC2010 AgentSmith and adapted to work within the
 * BOA framework.
 * 
 * Decoupling Negotiating Agents to Explore the Space of Negotiation Strategies
 * T. Baarslag, K. Hindriks, M. Hendrikx, A. Dirkzwager, C.M. Jonker
 *
 * @author Alex Dirkzwager, Mark Hendrikx
 * @version 26/12/11
 */
public class AC_AgentSmith extends AcceptanceStrategy {

	private final double ACCEPT_MARGIN = 0.9;

	/**
	 * Empty constructor for the BOA framework.
	 */
	public AC_AgentSmith() {
	}

	public AC_AgentSmith(NegotiationSession negoSession, OfferingStrategy strat) throws Exception {
		init(negoSession, strat, null, null);
	}

	@Override
	public void init(NegotiationSession negoSession, OfferingStrategy strat, OpponentModel opponentModel,
			Map<String, Double> parameters) throws Exception {
		this.negotiationSession = negoSession;
	}

	@Override
	public Actions determineAcceptability() {
		BidDetails lastOpponentBid = negotiationSession.getOpponentBidHistory().getLastBidDetails();
		BidDetails lastOwnBid = negotiationSession.getOwnBidHistory().getLastBidDetails();

		if (lastOpponentBid != null && lastOwnBid != null) {
			// accepts if the opponentLastBid is higher than the constant
			// acceptMargin OR
			// the utility of the opponents' bid is higher then ours -> accept!
			if (lastOpponentBid.getMyUndiscountedUtil() > ACCEPT_MARGIN
					|| lastOpponentBid.getMyUndiscountedUtil() >= lastOwnBid.getMyUndiscountedUtil()) {
				return Actions.Accept;
			}
		}
		return Actions.Reject;
	}

	@Override
	public String getName() {
		return "2010 - AgentSmith";
	}
}