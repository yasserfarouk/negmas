package negotiator.boaframework.acceptanceconditions.anac2010;

import java.util.Map;

import genius.core.boaframework.AcceptanceStrategy;
import genius.core.boaframework.Actions;
import genius.core.boaframework.NegotiationSession;
import genius.core.boaframework.OfferingStrategy;
import genius.core.boaframework.OpponentModel;

/**
 * This is the decoupled Acceptance Condition from AgentFSEGA (ANAC2010). The
 * code was taken from the ANAC2010 AgentFSEGA and adapted to work within the
 * BOA framework.
 * 
 * Decoupling Negotiating Agents to Explore the Space of Negotiation Strategies
 * T. Baarslag, K. Hindriks, M. Hendrikx, A. Dirkzwager, C.M. Jonker
 * 
 * @author Alex Dirkzwager, Mark Hendrikx
 * @version 26/12/11
 */
public class AC_AgentFSEGA extends AcceptanceStrategy {
	private double maxUtilInDomain;

	/**
	 * Empty constructor for the BOA framework.
	 */
	public AC_AgentFSEGA() {
	}

	public AC_AgentFSEGA(NegotiationSession negoSession, OfferingStrategy strat) throws Exception {
		init(negoSession, strat, null, null);
	}

	@Override
	public void init(NegotiationSession negoSession, OfferingStrategy strat, OpponentModel opponentModel,
			Map<String, Double> parameters) throws Exception {
		this.negotiationSession = negoSession;
		this.offeringStrategy = strat;
		maxUtilInDomain = negotiationSession.getMaxBidinDomain().getMyUndiscountedUtil();
	}

	@Override
	public Actions determineAcceptability() {

		if (negotiationSession.getOpponentBidHistory().getHistory().isEmpty()
				|| negotiationSession.getOwnBidHistory().getHistory().isEmpty()) {
			return Actions.Reject;
		} else {
			double lastOpponentBidUtil = negotiationSession.getOpponentBidHistory().getLastBidDetails()
					.getMyUndiscountedUtil();
			double lastMyBidUtil = negotiationSession.getOwnBidHistory().getLastBidDetails().getMyUndiscountedUtil();
			double nextMyBidUtil = offeringStrategy.getNextBid().getMyUndiscountedUtil();

			if ((lastOpponentBidUtil * 1.03 >= lastMyBidUtil) || (lastOpponentBidUtil > nextMyBidUtil)
					|| (lastOpponentBidUtil == maxUtilInDomain)) {
				return Actions.Accept;
			}
		}
		return Actions.Reject;
	}

	@Override
	public String getName() {
		return "2010 - AgentFSEGA";
	}
}